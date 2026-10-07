package fsm

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"

	"bob/internal/agentapi"

	openai "github.com/sashabaranov/go-openai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestToolAuthority_NilAndEmptyToolset_ZeroToolsAdvertised verifies that nil or empty toolsets
// advertise 0 tools to the LLM and reject unexpected tool calls immediately without dispatch.
func TestToolAuthority_NilAndEmptyToolset_ZeroToolsAdvertised(t *testing.T) {
	db := setupTestDB(t)
	store := NewStore(db)
	ctx := context.Background()

	var advertisedToolsCount int
	var invokerCalls int32

	invoker := ToolInvokerFunc(func(ctx context.Context, name, argsJSON string) (string, error) {
		atomic.AddInt32(&invokerCalls, 1)
		return "unexpected", nil
	})

	llm := &mockLLMClient{
		handler: func(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error) {
			advertisedToolsCount = len(req.Tools)
			// Malicious/hallucinated model returns a tool call when none were advertised
			return &openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role: openai.ChatMessageRoleAssistant,
							ToolCalls: []openai.ToolCall{
								{
									ID:   "call_hallucinated_1",
									Type: openai.ToolTypeFunction,
									Function: openai.FunctionCall{
										Name:      "web_search",
										Arguments: `{"query":"golang"}`,
									},
								},
							},
						},
					},
				},
			}, nil
		},
	}

	runner := NewToolLoopRunner(llm, NewStepExecutor(invoker), "test-model")
	contextJSON, err := EncodeMessages([]openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleUser, Content: "Hello"},
	})
	require.NoError(t, err)

	run := &FSMRun{
		ID:            "run_auth_zero_tools",
		ChatID:        "chat_1",
		FSMType:       FSMTypeToolLoop,
		Status:        RunStatusRunning,
		CurrentState:  StateInit,
		Iteration:     0,
		MaxIterations: 5,
		ContextJSON:   contextJSON,
	}
	require.NoError(t, store.CreateRun(ctx, run))

	// Execute with nil toolset
	err = runner.Execute(ctx, run, store, nil, "test-model")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "unauthorized tool call: model called \"web_search\" when no tools were authorized")
	assert.Equal(t, 0, advertisedToolsCount)
	assert.Equal(t, int32(0), atomic.LoadInt32(&invokerCalls))
	assert.Equal(t, RunStatusFailed, run.Status)
	assert.Equal(t, StateFailed, run.CurrentState)
}

// TestToolAuthority_EngineGlobalProviderDoesNotAuthorizeWithoutRequestTools verifies that
// global Engine tool definition providers do not leak tools to RunToolLoop when request tools are omitted.
func TestToolAuthority_EngineGlobalProviderDoesNotAuthorizeWithoutRequestTools(t *testing.T) {
	db := setupTestDB(t)
	store := NewStore(db)
	storeProvider := NewStaticStoreProvider(store)
	ctx := context.Background()

	var invokerCalls int32
	invoker := ToolInvokerFunc(func(ctx context.Context, name, argsJSON string) (string, error) {
		atomic.AddInt32(&invokerCalls, 1)
		return `{"result":"ok"}`, nil
	})

	provider := ToolDefinitionProviderFunc(func(ctx context.Context, chatID string, isDM bool) []openai.Tool {
		return []openai.Tool{
			{Type: openai.ToolTypeFunction, Function: &openai.FunctionDefinition{Name: "global_tool"}},
		}
	})

	llm := &mockLLMClient{
		handler: func(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error) {
			assert.Empty(t, req.Tools, "engine must not advertise global provider tools when request does not authorize them")
			return &openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role: openai.ChatMessageRoleAssistant,
							ToolCalls: []openai.ToolCall{
								{
									ID:   "call_unauth_global",
									Type: openai.ToolTypeFunction,
									Function: openai.FunctionCall{
										Name:      "global_tool",
										Arguments: `{}`,
									},
								},
							},
						},
					},
				},
			}, nil
		},
	}

	engine := NewEngine(storeProvider, llm, invoker, WithToolDefinitionProvider(provider), WithDefaultModel("test-model"))

	req := ToolLoopRequest{
		ChatID: "chat_1",
		UserID: "user_1",
		Messages: []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleUser, Content: "Run tool"},
		},
		// Neither Toolset nor Tools supplied -> strictly 0 tools authorized
	}

	res, err := engine.RunToolLoop(ctx, req)
	require.Error(t, err)
	assert.Nil(t, res)
	assert.Contains(t, err.Error(), "unauthorized tool call: model called \"global_tool\" when no tools were authorized")
	assert.Equal(t, int32(0), atomic.LoadInt32(&invokerCalls))
}

// TestToolAuthority_ExplicitEmptyToolsetOverridesLegacyTools verifies that passing an explicit
// Toolset (even with 0 definitions) takes precedence over req.Tools.
func TestToolAuthority_ExplicitEmptyToolsetOverridesLegacyTools(t *testing.T) {
	db := setupTestDB(t)
	store := NewStore(db)
	storeProvider := NewStaticStoreProvider(store)
	ctx := context.Background()

	invoker := ToolInvokerFunc(func(ctx context.Context, name, argsJSON string) (string, error) {
		return "ok", nil
	})

	llm := &mockLLMClient{
		handler: func(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error) {
			assert.Empty(t, req.Tools)
			return &openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role:    openai.ChatMessageRoleAssistant,
							Content: "No tools available.",
						},
					},
				},
			}, nil
		},
	}

	engine := NewEngine(storeProvider, llm, invoker, WithDefaultModel("test-model"))

	emptyToolset := NewLegacyToolset(nil, nil)
	req := ToolLoopRequest{
		ChatID:  "chat_1",
		UserID:  "user_1",
		Toolset: emptyToolset,
		Tools: []openai.Tool{
			{Type: openai.ToolTypeFunction, Function: &openai.FunctionDefinition{Name: "legacy_tool"}},
		},
		Messages: []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleUser, Content: "Hello"},
		},
	}

	res, err := engine.RunToolLoop(ctx, req)
	require.NoError(t, err)
	require.NotNil(t, res)
	assert.Equal(t, "No tools available.", res.Content)
}

// mockCustomToolset implements agentapi.Toolset for testing.
type mockCustomToolset struct {
	defs    []agentapi.ToolDefinition
	execFn  func(ctx context.Context, call agentapi.ToolCall) (agentapi.ToolResult, error)
}

func (m *mockCustomToolset) Definitions(ctx context.Context) ([]agentapi.ToolDefinition, error) {
	return m.defs, nil
}

func (m *mockCustomToolset) Execute(ctx context.Context, call agentapi.ToolCall) (agentapi.ToolResult, error) {
	if m.execFn != nil {
		return m.execFn(ctx, call)
	}
	return agentapi.ToolResult{Content: "default"}, nil
}

// TestToolAuthority_FrontendReadOnlyClassification verifies that StepExecutionMode classification
// is driven strictly by the frontend-declared ReadOnly flag on ToolDefinition.
func TestToolAuthority_FrontendReadOnlyClassification(t *testing.T) {
	ctx := context.Background()

	toolA := agentapi.ToolDefinition{
		Schema: openai.Tool{
			Type: openai.ToolTypeFunction,
			Function: &openai.FunctionDefinition{Name: "tool_a"},
		},
		ReadOnly: true,
	}
	toolB := agentapi.ToolDefinition{
		Schema: openai.Tool{
			Type: openai.ToolTypeFunction,
			Function: &openai.FunctionDefinition{Name: "tool_b"},
		},
		ReadOnly: true,
	}

	parallelToolset := &mockCustomToolset{defs: []agentapi.ToolDefinition{toolA, toolB}}
	bParallel, err := NewToolBinding(ctx, parallelToolset)
	require.NoError(t, err)

	steps := []*FSMStep{
		{ToolName: "tool_a"},
		{ToolName: "tool_b"},
	}
	assert.Equal(t, ExecutionModeParallel, bParallel.ClassifyStepExecutionMode(steps))

	// If tool B is mutating (ReadOnly: false), execution mode must be sequential
	toolBMutating := toolB
	toolBMutating.ReadOnly = false
	sequentialToolset := &mockCustomToolset{defs: []agentapi.ToolDefinition{toolA, toolBMutating}}
	bSequential, err := NewToolBinding(ctx, sequentialToolset)
	require.NoError(t, err)

	assert.Equal(t, ExecutionModeSequential, bSequential.ClassifyStepExecutionMode(steps))
}

// TestToolAuthority_ConcurrentRuns_IsolatedToolsets verifies that two concurrent runs
// with the same tool name use their own run-scoped toolset implementations without cross-talk.
func TestToolAuthority_ConcurrentRuns_IsolatedToolsets(t *testing.T) {
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
										ID:   "call_scoped",
										Type: openai.ToolTypeFunction,
										Function: openai.FunctionCall{
											Name:      "fetch_data",
											Arguments: `{}`,
										},
									},
								},
							},
						},
					},
				}, nil
			}
			return &openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role:    openai.ChatMessageRoleAssistant,
							Content: "Done",
						},
					},
				},
			}, nil
		},
	}

	runner := NewToolLoopRunner(llm, NewStepExecutor(nil), "test-model")

	var wg sync.WaitGroup
	wg.Add(2)

	runTurn := func(runID, expectedOutput string) {
		defer wg.Done()

		contextJSON, err := EncodeMessages([]openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleUser, Content: "Fetch"},
		})
		require.NoError(t, err)

		run := &FSMRun{
			ID:            runID,
			ChatID:        runID + "_chat",
			FSMType:       FSMTypeToolLoop,
			Status:        RunStatusRunning,
			CurrentState:  StateInit,
			Iteration:     0,
			MaxIterations: 5,
			ContextJSON:   contextJSON,
		}
		require.NoError(t, store.CreateRun(ctx, run))

		scopedToolset := &mockCustomToolset{
			defs: []agentapi.ToolDefinition{
				{
					Schema: openai.Tool{
						Type: openai.ToolTypeFunction,
						Function: &openai.FunctionDefinition{Name: "fetch_data"},
					},
					ReadOnly: true,
				},
			},
			execFn: func(ctx context.Context, call agentapi.ToolCall) (agentapi.ToolResult, error) {
				return agentapi.ToolResult{Content: expectedOutput}, nil
			},
		}

		err = runner.Execute(ctx, run, store, scopedToolset, "test-model")
		require.NoError(t, err)
		assert.Equal(t, RunStatusCompleted, run.Status)

		steps, err := store.ListStepsByRun(ctx, runID)
		require.NoError(t, err)
		require.Len(t, steps, 1)

		decoded, err := DecodeStoredToolResult(steps[0].ResultJSON)
		require.NoError(t, err)
		assert.Equal(t, expectedOutput, decoded.Content)
	}

	go runTurn("run_scope_alpha", "alpha_data_123")
	go runTurn("run_scope_beta", "beta_data_456")

	wg.Wait()
}

// TestToolAuthority_PreExistingStepUnauthorized verifies that if a run has pre-existing steps
// in the database referencing unauthorized tools, execution fails before dispatch.
func TestToolAuthority_PreExistingStepUnauthorized(t *testing.T) {
	db := setupTestDB(t)
	store := NewStore(db)
	ctx := context.Background()

	run := &FSMRun{
		ID:            "run_unauth_step",
		ChatID:        "chat_1",
		FSMType:       FSMTypeToolLoop,
		Status:        RunStatusRunning,
		CurrentState:  StateExecuteSteps,
		Iteration:     1,
		MaxIterations: 5,
		ContextJSON:   "[]",
	}
	require.NoError(t, store.CreateRun(ctx, run))

	// Pre-create step with an unauthorized tool name
	step := FSMStep{
		ID:          "step_unauth",
		RunID:       run.ID,
		Iteration:   1,
		ToolName:    "forbidden_tool",
		ToolCallID:  "call_forbid",
		Status:      StepStatusPending,
		MaxAttempts: 1,
	}
	require.NoError(t, store.CreateSteps(ctx, []FSMStep{step}))

	// Runner authorized only for "safe_tool"
	safeToolset := &mockCustomToolset{
		defs: []agentapi.ToolDefinition{
			{
				Schema: openai.Tool{
					Type: openai.ToolTypeFunction,
					Function: &openai.FunctionDefinition{Name: "safe_tool"},
				},
			},
		},
	}

	runner := NewToolLoopRunner(nil, NewStepExecutor(nil), "test-model")
	err := runner.Execute(ctx, run, store, safeToolset, "test-model")
	require.Error(t, err)
	assert.Contains(t, err.Error(), `step step_unauth uses unauthorized tool "forbidden_tool"`)
	assert.Equal(t, RunStatusFailed, run.Status)
	assert.Equal(t, StateFailed, run.CurrentState)
}
