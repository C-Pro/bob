package fsm

import (
	"context"
	"fmt"
	"strings"
	"testing"

	openai "github.com/sashabaranov/go-openai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDetectLoop_ConsecutiveIdenticalCalls(t *testing.T) {
	cfg := DefaultLoopGuardConfig()

	callA := openai.ToolCall{
		ID:   "call_1",
		Type: openai.ToolTypeFunction,
		Function: openai.FunctionCall{
			Name:      "sandbox_exec",
			Arguments: `{"command": "ls -la"}`,
		},
	}

	history := []FSMStep{
		{
			ID:         "step_1",
			Iteration:  1,
			ToolName:   "sandbox_exec",
			ArgsJSON:   `{"command":"ls -la"}`,
			ResultJSON: "file1\nfile2",
			Status:     StepStatusCompleted,
		},
		{
			ID:         "step_2",
			Iteration:  2,
			ToolName:   "sandbox_exec",
			ArgsJSON:   `{"command":"ls -la"}`,
			ResultJSON: "file1\nfile2",
			Status:     StepStatusCompleted,
		},
	}

	// 1. Strike 1: 3rd identical call with unchanged output triggers Intercept
	verdict := DetectLoop(history, []openai.ToolCall{callA}, cfg)
	assert.Equal(t, LoopActionIntercept, verdict.Action)
	assert.Contains(t, verdict.Reason, "consecutive times")
	assert.Contains(t, verdict.NoticeByCall[0], LoopGuardNoticePrefix)

	// 2. Strike 2: If strike 1 already occurred, repeating it triggers ForceSynthesis
	historyWithStrike1 := append(history, FSMStep{
		ID:         "step_3",
		Iteration:  3,
		ToolName:   "sandbox_exec",
		ArgsJSON:   `{"command":"ls -la"}`,
		ResultJSON: FormatLoopGuardNotice("sandbox_exec", `{"command":"ls -la"}`),
		ErrorText:  LoopGuardBlockedErrorText,
		Status:     StepStatusCompleted,
	})

	verdict2 := DetectLoop(historyWithStrike1, []openai.ToolCall{callA}, cfg)
	assert.Equal(t, LoopActionForceSynthesis, verdict2.Action)
	assert.Contains(t, verdict2.Reason, "repeated after being blocked")
}

func TestDetectLoop_CyclicSequence(t *testing.T) {
	cfg := DefaultLoopGuardConfig()

	callA := openai.ToolCall{
		ID:   "call_a",
		Type: openai.ToolTypeFunction,
		Function: openai.FunctionCall{
			Name:      "sandbox_exec",
			Arguments: `{"command": "ls -la extracted_images"}`,
		},
	}
	callB := openai.ToolCall{
		ID:   "call_b",
		Type: openai.ToolTypeFunction,
		Function: openai.FunctionCall{
			Name:      "sandbox_exec",
			Arguments: `{"command": "ls -la dejavu_pkg"}`,
		},
	}
	callC := openai.ToolCall{
		ID:   "call_c",
		Type: openai.ToolTypeFunction,
		Function: openai.FunctionCall{
			Name:      "sandbox_exec",
			Arguments: `{"command": "python3 -c import reportlab"}`,
		},
	}

	// Cycle of length 3: [A, B, C] run twice in history
	history := []FSMStep{
		{ID: "s1", Iteration: 1, ToolName: "sandbox_exec", ArgsJSON: `{"command":"ls -la extracted_images"}`, ResultJSON: "img1", Status: StepStatusCompleted},
		{ID: "s2", Iteration: 2, ToolName: "sandbox_exec", ArgsJSON: `{"command":"ls -la dejavu_pkg"}`, ResultJSON: "font1", Status: StepStatusCompleted},
		{ID: "s3", Iteration: 3, ToolName: "sandbox_exec", ArgsJSON: `{"command":"python3 -c import reportlab"}`, ResultJSON: "5.1.0", Status: StepStatusCompleted},
		{ID: "s4", Iteration: 4, ToolName: "sandbox_exec", ArgsJSON: `{"command":"ls -la extracted_images"}`, ResultJSON: "img1", Status: StepStatusCompleted},
		{ID: "s5", Iteration: 5, ToolName: "sandbox_exec", ArgsJSON: `{"command":"ls -la dejavu_pkg"}`, ResultJSON: "font1", Status: StepStatusCompleted},
		{ID: "s6", Iteration: 6, ToolName: "sandbox_exec", ArgsJSON: `{"command":"python3 -c import reportlab"}`, ResultJSON: "5.1.0", Status: StepStatusCompleted},
	}

	// Strike 1: candidate call proposing next cycle block [A, B, C]
	verdict := DetectLoop(history, []openai.ToolCall{callA, callB, callC}, cfg)
	assert.Equal(t, LoopActionIntercept, verdict.Action)
	assert.Contains(t, verdict.Reason, "cyclic sequence of 3 tool calls repeated 2 times")
	assert.Contains(t, verdict.NoticeByCall[0], LoopGuardNoticePrefix)
	assert.Contains(t, verdict.NoticeByCall[1], LoopGuardNoticePrefix)
	assert.Contains(t, verdict.NoticeByCall[2], LoopGuardNoticePrefix)

	// Strike 2: Repeating after prior strike forces synthesis
	historyWithStrike := append(history, FSMStep{
		ID:         "s7",
		Iteration:  7,
		ToolName:   "sandbox_exec",
		ArgsJSON:   `{"command":"ls -la extracted_images"}`,
		ResultJSON: FormatLoopGuardNotice("sandbox_exec", `{"command":"ls -la extracted_images"}`),
		ErrorText:  LoopGuardBlockedErrorText,
		Status:     StepStatusCompleted,
	})

	verdictStrike2 := DetectLoop(historyWithStrike, []openai.ToolCall{callA, callB, callC}, cfg)
	assert.Equal(t, LoopActionForceSynthesis, verdictStrike2.Action)
}

func TestDetectLoop_NonLoopingOperationsAllowed(t *testing.T) {
	cfg := DefaultLoopGuardConfig()

	// Calling different commands
	history := []FSMStep{
		{ID: "s1", Iteration: 1, ToolName: "sandbox_exec", ArgsJSON: `{"command":"ls -la"}`, ResultJSON: "out", Status: StepStatusCompleted},
		{ID: "s2", Iteration: 2, ToolName: "sandbox_exec", ArgsJSON: `{"command":"python build.py"}`, ResultJSON: "ok", Status: StepStatusCompleted},
	}

	newCall := openai.ToolCall{
		ID:   "call_3",
		Type: openai.ToolTypeFunction,
		Function: openai.FunctionCall{
			Name:      "sandbox_exec",
			Arguments: `{"command": "python test.py"}`,
		},
	}

	verdict := DetectLoop(history, []openai.ToolCall{newCall}, cfg)
	assert.Equal(t, LoopActionNone, verdict.Action)
}

func TestDetectLoop_DisabledConfig(t *testing.T) {
	cfg := DefaultLoopGuardConfig()
	cfg.Enabled = false

	call := openai.ToolCall{
		ID:   "call_1",
		Type: openai.ToolTypeFunction,
		Function: openai.FunctionCall{
			Name:      "tool",
			Arguments: `{}`,
		},
	}
	history := []FSMStep{
		{ID: "s1", ToolName: "tool", ArgsJSON: `{}`},
		{ID: "s2", ToolName: "tool", ArgsJSON: `{}`},
		{ID: "s3", ToolName: "tool", ArgsJSON: `{}`},
	}

	verdict := DetectLoop(history, []openai.ToolCall{call}, cfg)
	assert.Equal(t, LoopActionNone, verdict.Action)
}

func TestToolLoopRunner_LoopGuardEndToEnd_InterceptThenForceSynthesis(t *testing.T) {
	db := setupTestDB(t)
	store := NewStore(db)
	ctx := context.Background()

	var llmCalls int
	var executedTools []string

	llm := &mockLLMClient{
		handler: func(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error) {
			llmCalls++
			// If in synthesis state, tools are nil:
			if len(req.Tools) == 0 {
				return &openai.ChatCompletionResponse{
					Choices: []openai.ChatCompletionChoice{
						{
							Message: openai.ChatCompletionMessage{
								Role:    openai.ChatMessageRoleAssistant,
								Content: "Final synthesized answer after loop detection.",
							},
						},
					},
				}, nil
			}

			// Propose the exact same tool call every iteration
			return &openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role: openai.ChatMessageRoleAssistant,
							ToolCalls: []openai.ToolCall{
								{
									ID:   fmt.Sprintf("call_%d", llmCalls),
									Type: openai.ToolTypeFunction,
									Function: openai.FunctionCall{
										Name:      "test_cmd",
										Arguments: `{"arg":"same"}`,
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
		executedTools = append(executedTools, fmt.Sprintf("%s:%s", name, argsJSON))
		return "tool output unchanged", nil
	})

	runner := NewToolLoopRunner(llm, NewStepExecutor(invoker), "test-model")
	runner.SetCompactionConfig(CompactionConfig{Enabled: false})
	runner.SetLoopGuardConfig(LoopGuardConfig{
		Enabled:             true,
		MaxConsecutiveCalls: 3,
		MaxCycleRepetitions: 2,
		MaxCycleLength:      3,
	})

	ctxJSON, err := EncodeMessages([]openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleUser, Content: "Do something"},
	})
	require.NoError(t, err)

	run := &FSMRun{
		ID:            "run_loop_guard_e2e",
		ChatID:        "chat_lg",
		UserID:        "user_lg",
		FSMType:       FSMTypeToolLoop,
		Status:        RunStatusPending,
		CurrentState:  StateInit,
		MaxIterations: 20, // would run 20 times if not guarded
		ContextJSON:   ctxJSON,
	}
	require.NoError(t, store.CreateRun(ctx, run))

	tools := []openai.Tool{
		{Type: openai.ToolTypeFunction, Function: &openai.FunctionDefinition{Name: "test_cmd"}},
	}
	toolset := NewLegacyToolset(tools, invoker)

	err = runner.Execute(ctx, run, store, toolset, "test-model")
	require.NoError(t, err)

	// Verify run reached StateCompleted
	assert.Equal(t, StateCompleted, run.CurrentState)
	assert.Equal(t, RunStatusCompleted, run.Status)
	assert.Equal(t, "Final synthesized answer after loop detection.", run.ResultJSON)

	// The run stopped well before MaxIterations=20!
	assert.True(t, run.Iteration < 6, "run should have halted early due to loop guard, got iteration %d", run.Iteration)

	// Tool should only have been actually executed 2 times (strike 1 intercepted the 3rd, strike 2 forced synthesis on 4th)
	assert.Equal(t, 2, len(executedTools), "tool should not be invoked after strike 1 intercept")

	// Verify that history contains the intercepted step with LoopGuardNoticePrefix
	steps, err := store.ListStepsByRun(ctx, run.ID)
	require.NoError(t, err)

	var hasBlockedStep bool
	for _, s := range steps {
		if s.ErrorText == LoopGuardBlockedErrorText {
			hasBlockedStep = true
			res, _ := DecodeStoredToolResult(s.ResultJSON)
			assert.Contains(t, res.Content, LoopGuardNoticePrefix)
		}
	}
	assert.True(t, hasBlockedStep, "store must contain at least one step intercepted by loop guard")
}

func TestToolLoopRunner_LoopGuardRecoversIfModelBreaksLoop(t *testing.T) {
	db := setupTestDB(t)
	store := NewStore(db)
	ctx := context.Background()

	var llmCalls int
	var executedTools []string

	llm := &mockLLMClient{
		handler: func(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error) {
			llmCalls++
			// Iterations 1, 2, 3 call same command
			// Iteration 4 sees the LoopGuardNoticePrefix in context, breaks out and calls action_cmd!
			// Iteration 5 finishes with final message
			if llmCalls <= 3 {
				return &openai.ChatCompletionResponse{
					Choices: []openai.ChatCompletionChoice{
						{
							Message: openai.ChatCompletionMessage{
								Role: openai.ChatMessageRoleAssistant,
								ToolCalls: []openai.ToolCall{
									{
										ID:   fmt.Sprintf("call_%d", llmCalls),
										Type: openai.ToolTypeFunction,
										Function: openai.FunctionCall{
											Name:      "inspect_cmd",
											Arguments: `{"cmd":"ls"}`,
										},
									},
								},
							},
						},
					},
				}, nil
			}

			if llmCalls == 4 {
				// Verify the LLM context actually received the loop guard notice!
				hasNoticeInPrompt := false
				for _, m := range req.Messages {
					if strings.Contains(m.Content, LoopGuardNoticePrefix) {
						hasNoticeInPrompt = true
						break
					}
				}
				assert.True(t, hasNoticeInPrompt, "model prompt should receive loop guard notice")

				return &openai.ChatCompletionResponse{
					Choices: []openai.ChatCompletionChoice{
						{
							Message: openai.ChatCompletionMessage{
								Role: openai.ChatMessageRoleAssistant,
								ToolCalls: []openai.ToolCall{
									{
										ID:   "call_breakout",
										Type: openai.ToolTypeFunction,
										Function: openai.FunctionCall{
											Name:      "action_cmd",
											Arguments: `{"action":"write_file"}`,
										},
									},
								},
							},
						},
					},
				}, nil
			}

			// Iteration 5: Complete
			return &openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role:    openai.ChatMessageRoleAssistant,
							Content: "Task completed successfully after taking action.",
						},
					},
				},
			}, nil
		},
	}

	invoker := ToolInvokerFunc(func(ctx context.Context, name, argsJSON string) (string, error) {
		executedTools = append(executedTools, name)
		return "ok", nil
	})

	runner := NewToolLoopRunner(llm, NewStepExecutor(invoker), "test-model")
	runner.SetCompactionConfig(CompactionConfig{Enabled: false})
	runner.SetLoopGuardConfig(LoopGuardConfig{
		Enabled:             true,
		MaxConsecutiveCalls: 3,
		MaxCycleRepetitions: 2,
		MaxCycleLength:      3,
	})

	ctxJSON, err := EncodeMessages([]openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleUser, Content: "Build the report"},
	})
	require.NoError(t, err)

	run := &FSMRun{
		ID:            "run_loop_recover",
		ChatID:        "chat_recover",
		UserID:        "user_recover",
		FSMType:       FSMTypeToolLoop,
		Status:        RunStatusPending,
		CurrentState:  StateInit,
		MaxIterations: 10,
		ContextJSON:   ctxJSON,
	}
	require.NoError(t, store.CreateRun(ctx, run))

	tools := []openai.Tool{
		{Type: openai.ToolTypeFunction, Function: &openai.FunctionDefinition{Name: "inspect_cmd"}},
		{Type: openai.ToolTypeFunction, Function: &openai.FunctionDefinition{Name: "action_cmd"}},
	}
	toolset := NewLegacyToolset(tools, invoker)

	err = runner.Execute(ctx, run, store, toolset, "test-model")
	require.NoError(t, err)

	assert.Equal(t, StateCompleted, run.CurrentState)
	assert.Equal(t, "Task completed successfully after taking action.", run.ResultJSON)

	// action_cmd should have run!
	assert.Contains(t, executedTools, "action_cmd")
}
