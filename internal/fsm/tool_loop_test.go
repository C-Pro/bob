package fsm

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"

	openai "github.com/sashabaranov/go-openai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type mockLLMClient struct {
	handler func(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error)
	calls   int32
}

func (m *mockLLMClient) CreateChatCompletion(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error) {
	atomic.AddInt32(&m.calls, 1)
	if m.handler != nil {
		return m.handler(ctx, req)
	}
	return nil, errors.New("no mock handler configured")
}

func TestToolLoop_TextOnlyResponse(t *testing.T) {
	db := setupTestDB(t)
	store := NewStore(db)
	ctx := context.Background()

	llm := &mockLLMClient{
		handler: func(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error) {
			return &openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role:    openai.ChatMessageRoleAssistant,
							Content: "Hello, I am Bob!",
						},
					},
				},
			}, nil
		},
	}

	invoker := ToolInvokerFunc(func(ctx context.Context, name, argsJSON string) (string, error) {
		return "", errors.New("should not be called")
	})

	executor := NewStepExecutor(invoker, store)
	runner := NewToolLoopRunner(llm, executor, "test-model")

	contextJSON, err := EncodeMessages([]openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleUser, Content: "Hello!"},
	})
	require.NoError(t, err)

	run := &FSMRun{
		ID:            "run_text_only",
		ChatID:        "chat_1",
		UserID:        "user_1",
		FSMType:       FSMTypeToolLoop,
		Status:        RunStatusPending,
		CurrentState:  StateInit,
		Iteration:     0,
		MaxIterations: 10,
		ContextJSON:   contextJSON,
	}
	require.NoError(t, store.CreateRun(ctx, run))

	err = runner.Execute(ctx, run, store, nil, "test-model")
	require.NoError(t, err)

	assert.Equal(t, RunStatusCompleted, run.Status)
	assert.Equal(t, StateCompleted, run.CurrentState)
	assert.Equal(t, "Hello, I am Bob!", run.ResultJSON)
	assert.Equal(t, 0, run.Iteration)
	assert.Equal(t, int32(1), atomic.LoadInt32(&llm.calls))

	// Verify persistence in SQLite
	persisted, err := store.GetRun(ctx, run.ID)
	require.NoError(t, err)
	assert.Equal(t, RunStatusCompleted, persisted.Status)
	assert.Equal(t, "Hello, I am Bob!", persisted.ResultJSON)
}

func TestToolLoop_SingleToolCallExecution(t *testing.T) {
	db := setupTestDB(t)
	store := NewStore(db)
	ctx := context.Background()

	var toolCallsMade int32

	llm := &mockLLMClient{
		handler: func(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error) {
			// Turn 1: request tool call
			if len(req.Messages) == 1 {
				return &openai.ChatCompletionResponse{
					Choices: []openai.ChatCompletionChoice{
						{
							Message: openai.ChatCompletionMessage{
								Role: openai.ChatMessageRoleAssistant,
								ToolCalls: []openai.ToolCall{
									{
										ID:   "call_search_1",
										Type: openai.ToolTypeFunction,
										Function: openai.FunctionCall{
											Name:      "web_search",
											Arguments: `{"query": "golang 1.26"}`,
										},
									},
								},
							},
						},
					},
				}, nil
			}

			// Turn 2: verify tool result was appended
			require.GreaterOrEqual(t, len(req.Messages), 3)
			lastMsg := req.Messages[len(req.Messages)-1]
			assert.Equal(t, openai.ChatMessageRoleTool, lastMsg.Role)
			assert.Equal(t, "call_search_1", lastMsg.ToolCallID)
			assert.Contains(t, lastMsg.Content, "Go 1.26 released")

			return &openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role:    openai.ChatMessageRoleAssistant,
							Content: "Go 1.26 has been released with major updates.",
						},
					},
				},
			}, nil
		},
	}

	invoker := ToolInvokerFunc(func(ctx context.Context, name, argsJSON string) (string, error) {
		atomic.AddInt32(&toolCallsMade, 1)
		assert.Equal(t, "web_search", name)
		return `{"result": "Go 1.26 released"}`, nil
	})

	executor := NewStepExecutor(invoker, store)
	runner := NewToolLoopRunner(llm, executor, "test-model")

	contextJSON, err := EncodeMessages([]openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleUser, Content: "What is new in Go?"},
	})
	require.NoError(t, err)

	run := &FSMRun{
		ID:            "run_single_tool",
		ChatID:        "chat_1",
		UserID:        "user_1",
		FSMType:       FSMTypeToolLoop,
		Status:        RunStatusRunning,
		CurrentState:  StateInit,
		Iteration:     0,
		MaxIterations: 5,
		ContextJSON:   contextJSON,
	}
	require.NoError(t, store.CreateRun(ctx, run))

	tools := []openai.Tool{
		{
			Type: openai.ToolTypeFunction,
			Function: &openai.FunctionDefinition{
				Name: "web_search",
			},
		},
	}

	err = runner.Execute(ctx, run, store, tools, "test-model")
	require.NoError(t, err)

	assert.Equal(t, RunStatusCompleted, run.Status)
	assert.Equal(t, StateCompleted, run.CurrentState)
	assert.Equal(t, "Go 1.26 has been released with major updates.", run.ResultJSON)
	assert.Equal(t, 1, run.Iteration)
	assert.Equal(t, int32(1), atomic.LoadInt32(&toolCallsMade))

	// Verify steps persisted in SQLite
	steps, err := store.ListStepsByIteration(ctx, run.ID, 1)
	require.NoError(t, err)
	require.Len(t, steps, 1)
	assert.Equal(t, StepStatusCompleted, steps[0].Status)
	assert.Equal(t, "web_search", steps[0].ToolName)
	assert.Equal(t, `{"result": "Go 1.26 released"}`, steps[0].ResultJSON)
}

func TestToolLoop_ParallelToolCallsExecution(t *testing.T) {
	db := setupTestDB(t)
	store := NewStore(db)
	ctx := context.Background()

	var executedTools []string
	var execMu sync.Mutex

	llm := &mockLLMClient{
		handler: func(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error) {
			if len(req.Messages) == 1 {
				// Return two read-only tool calls in parallel
				return &openai.ChatCompletionResponse{
					Choices: []openai.ChatCompletionChoice{
						{
							Message: openai.ChatCompletionMessage{
								Role: openai.ChatMessageRoleAssistant,
								ToolCalls: []openai.ToolCall{
									{
										ID:   "call_s1",
										Type: openai.ToolTypeFunction,
										Function: openai.FunctionCall{
											Name:      "web_search",
											Arguments: `{"q":"a"}`,
										},
									},
									{
										ID:   "call_s2",
										Type: openai.ToolTypeFunction,
										Function: openai.FunctionCall{
											Name:      "web_fetch",
											Arguments: `{"url":"b"}`,
										},
									},
								},
							},
						},
					},
				}, nil
			}

			// Final completion
			return &openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role:    openai.ChatMessageRoleAssistant,
							Content: "Parallel tools finished.",
						},
					},
				},
			}, nil
		},
	}

	invoker := ToolInvokerFunc(func(ctx context.Context, name, argsJSON string) (string, error) {
		execMu.Lock()
		executedTools = append(executedTools, name)
		execMu.Unlock()
		return fmt.Sprintf(`{"tool": %q}`, name), nil
	})

	executor := NewStepExecutor(invoker, store)
	runner := NewToolLoopRunner(llm, executor, "test-model")

	contextJSON, err := EncodeMessages([]openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleUser, Content: "Look up two things"},
	})
	require.NoError(t, err)

	run := &FSMRun{
		ID:            "run_parallel_tools",
		ChatID:        "chat_1",
		FSMType:       FSMTypeToolLoop,
		Status:        RunStatusRunning,
		CurrentState:  StateInit,
		Iteration:     0,
		MaxIterations: 5,
		ContextJSON:   contextJSON,
	}
	require.NoError(t, store.CreateRun(ctx, run))

	err = runner.Execute(ctx, run, store, nil, "test-model")
	require.NoError(t, err)

	assert.Equal(t, RunStatusCompleted, run.Status)
	assert.Equal(t, "Parallel tools finished.", run.ResultJSON)

	// Verify both were executed and classified as parallel
	execMu.Lock()
	assert.Len(t, executedTools, 2)
	execMu.Unlock()

	steps, err := store.ListStepsByIteration(ctx, run.ID, 1)
	require.NoError(t, err)
	require.Len(t, steps, 2)
	for _, s := range steps {
		assert.Equal(t, ExecutionModeParallel, s.ExecutionMode)
		assert.Equal(t, StepStatusCompleted, s.Status)
	}
}

func TestToolLoop_MaxIterationsSynthesis(t *testing.T) {
	db := setupTestDB(t)
	store := NewStore(db)
	ctx := context.Background()

	llm := &mockLLMClient{
		handler: func(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error) {
			// If tools are nil, it's the synthesis prompt!
			if req.Tools == nil {
				lastMsg := req.Messages[len(req.Messages)-1]
				assert.Equal(t, openai.ChatMessageRoleUser, lastMsg.Role)
				assert.Equal(t, SynthesisPrompt, lastMsg.Content)
				return &openai.ChatCompletionResponse{
					Choices: []openai.ChatCompletionChoice{
						{
							Message: openai.ChatCompletionMessage{
								Role:    openai.ChatMessageRoleAssistant,
								Content: "Synthesized summary after max iterations reached.",
							},
						},
					},
				}, nil
			}

			// Keep requesting tool calls
			return &openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role: openai.ChatMessageRoleAssistant,
							ToolCalls: []openai.ToolCall{
								{
									ID:   fmt.Sprintf("call_%d", len(req.Messages)),
									Type: openai.ToolTypeFunction,
									Function: openai.FunctionCall{
										Name:      "web_search",
										Arguments: `{"q":"loop"}`,
									},
								},
							},
						},
					},
				},
			}, nil
		},
	}

	invoker := ToolInvokerFunc(func(ctx context.Context, name, argsJSON string) (string, error) {
		return `{"ok":true}`, nil
	})

	executor := NewStepExecutor(invoker, store)
	runner := NewToolLoopRunner(llm, executor, "test-model")

	contextJSON, err := EncodeMessages([]openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleUser, Content: "Keep searching"},
	})
	require.NoError(t, err)

	// Max iterations set to 2
	run := &FSMRun{
		ID:            "run_max_iter",
		ChatID:        "chat_1",
		FSMType:       FSMTypeToolLoop,
		Status:        RunStatusRunning,
		CurrentState:  StateInit,
		Iteration:     0,
		MaxIterations: 2,
		ContextJSON:   contextJSON,
	}
	require.NoError(t, store.CreateRun(ctx, run))

	tools := []openai.Tool{
		{Type: openai.ToolTypeFunction, Function: &openai.FunctionDefinition{Name: "web_search"}},
	}

	err = runner.Execute(ctx, run, store, tools, "test-model")
	require.NoError(t, err)

	assert.Equal(t, RunStatusCompleted, run.Status)
	assert.Equal(t, StateCompleted, run.CurrentState)
	// Exactly 2 tool iterations were executed before synthesis
	assert.Equal(t, 2, run.Iteration)
}

func TestToolLoop_ToolExecutionFailureRecordedAndFedBack(t *testing.T) {
	db := setupTestDB(t)
	store := NewStore(db)
	ctx := context.Background()

	llm := &mockLLMClient{
		handler: func(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error) {
			if len(req.Messages) == 1 {
				return &openai.ChatCompletionResponse{
					Choices: []openai.ChatCompletionChoice{
						{
							Message: openai.ChatCompletionMessage{
								Role: openai.ChatMessageRoleAssistant,
								ToolCalls: []openai.ToolCall{
									{
										ID:   "call_fail",
										Type: openai.ToolTypeFunction,
										Function: openai.FunctionCall{
											Name:      "web_search",
											Arguments: `{"bad":"args"}`,
										},
									},
								},
							},
						},
					},
				}, nil
			}

			// Verify error was fed back as tool output
			lastMsg := req.Messages[len(req.Messages)-1]
			assert.Equal(t, openai.ChatMessageRoleTool, lastMsg.Role)
			assert.Contains(t, lastMsg.Content, "invalid tool argument")

			return &openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role:    openai.ChatMessageRoleAssistant,
							Content: "Search failed, acknowledging error.",
						},
					},
				},
			}, nil
		},
	}

	invoker := ToolInvokerFunc(func(ctx context.Context, name, argsJSON string) (string, error) {
		return "", errors.New("invalid tool argument")
	})

	executor := NewStepExecutor(invoker, store)
	runner := NewToolLoopRunner(llm, executor, "test-model")

	contextJSON, err := EncodeMessages([]openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleUser, Content: "Fail tool"},
	})
	require.NoError(t, err)

	run := &FSMRun{
		ID:            "run_tool_fail",
		ChatID:        "chat_1",
		FSMType:       FSMTypeToolLoop,
		Status:        RunStatusRunning,
		CurrentState:  StateInit,
		Iteration:     0,
		MaxIterations: 5,
		ContextJSON:   contextJSON,
	}
	require.NoError(t, store.CreateRun(ctx, run))

	err = runner.Execute(ctx, run, store, nil, "test-model")
	require.NoError(t, err)

	assert.Equal(t, RunStatusCompleted, run.Status)
	assert.Equal(t, "Search failed, acknowledging error.", run.ResultJSON)

	// Step in SQLite has failed status and error text
	steps, err := store.ListStepsByIteration(ctx, run.ID, 1)
	require.NoError(t, err)
	require.Len(t, steps, 1)
	assert.Equal(t, StepStatusFailed, steps[0].Status)
	assert.Contains(t, steps[0].ErrorText, "invalid tool argument")
}

func TestEngine_RunToolLoop_WithTransitionCallback(t *testing.T) {
	db := setupTestDB(t)
	s := NewStore(db)
	storeProv := NewStaticStoreProvider(s)

	llmClient := &mockLLMClient{
		handler: func(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error) {
			return &openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role:    openai.ChatMessageRoleAssistant,
							Content: "Hello world!",
						},
					},
				},
			}, nil
		},
	}

	engine := NewEngine(storeProv, llmClient, nil)

	var observedStates []RunState
	var mu sync.Mutex

	res, err := engine.RunToolLoop(context.Background(), ToolLoopRequest{
		ChatID: "townhall",
		UserID: "user1",
		IsDM:   false,
		Messages: []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleUser, Content: "Hello"},
		},
		OnTransition: func(state RunState, run *FSMRun) {
			mu.Lock()
			defer mu.Unlock()
			observedStates = append(observedStates, state)
		},
	})
	require.NoError(t, err)
	assert.Equal(t, "Hello world!", res.Content)

	mu.Lock()
	defer mu.Unlock()
	assert.Contains(t, observedStates, StateInit)
	assert.Contains(t, observedStates, StateLLMRequest)
	assert.Contains(t, observedStates, StateCompleted)
}
