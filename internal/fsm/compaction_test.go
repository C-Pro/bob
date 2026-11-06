package fsm

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"bob/internal/trajectory"

	openai "github.com/sashabaranov/go-openai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestToolLoop_RubricTriggeredCompactionAndViewConstruction(t *testing.T) {
	db := setupTestDB(t)
	store := NewStore(db)
	ctx := context.Background()

	var mu sync.Mutex
	var capturedIter4Messages []openai.ChatCompletionMessage
	var probeCalls int32

	llm := &mockLLMClient{
		handler: func(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error) {
			// Detect rubric probe request by system prompt
			if len(req.Messages) > 0 && strings.Contains(req.Messages[0].Content, "SelfCompact") {
				atomic.AddInt32(&probeCalls, 1)
				probeResp := CompactionProbeResult{
					ShouldCompact: true,
					Reason:        "Iterations 1 and 2 completed initial repository discovery",
					State: &CompactedTaskState{
						OriginalObjective:    "Find the database version and run tests",
						ImmutableConstraints: []string{"Do not delete raw steps"},
						VerifiedFindings: []VerifiedFinding{
							{
								Summary:       "FSM schema version is 4",
								SourceStepIDs: []string{"run_rubric_1_i1_s0", "run_rubric_1_i2_s0"},
							},
						},
						DecisionsAndRejected: []string{"Confirmed per-chat SQLite schema"},
						CurrentSubgoal:       "Finish verification after iteration 3",
						NextIntendedActions:  []string{"Provide final summary"},
					},
				}
				rawJSON, err := json.Marshal(probeResp)
				require.NoError(t, err)
				return &openai.ChatCompletionResponse{
					Choices: []openai.ChatCompletionChoice{
						{
							Message: openai.ChatCompletionMessage{
								Role:    openai.ChatMessageRoleAssistant,
								Content: "```json\n" + string(rawJSON) + "\n```",
							},
						},
					},
				}, nil
			}

			// Normal tool loop requests
			// Count how many assistant tool-call turns have happened by checking current state or messages
			for _, m := range req.Messages {
				if m.Role == openai.ChatMessageRoleSystem && strings.Contains(m.Content, "[Compacted Task State v1 | Iterations 1..2]") {
					mu.Lock()
					capturedIter4Messages = append([]openai.ChatCompletionMessage(nil), req.Messages...)
					mu.Unlock()
					return &openai.ChatCompletionResponse{
						Choices: []openai.ChatCompletionChoice{
							{
								Message: openai.ChatCompletionMessage{
									Role:    openai.ChatMessageRoleAssistant,
									Content: "Verification complete: schema v4 and tests pass.",
								},
							},
						},
					}, nil
				}
			}

			// Determine next iteration number from tool messages present
			toolCount := 0
			for _, m := range req.Messages {
				if m.Role == openai.ChatMessageRoleTool {
					toolCount++
				}
			}
			nextIter := toolCount + 1
			return &openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role: openai.ChatMessageRoleAssistant,
							ToolCalls: []openai.ToolCall{
								{
									ID:   fmt.Sprintf("call_iter_%d", nextIter),
									Type: openai.ToolTypeFunction,
									Function: openai.FunctionCall{
										Name:      "inspect_step",
										Arguments: fmt.Sprintf(`{"step":%d}`, nextIter),
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
		return fmt.Sprintf(`{"observation":"output for %s %s"}`, argsJSON, strings.Repeat("detail_", 80)), nil
	})

	runner := NewToolLoopRunner(llm, NewStepExecutor(invoker), "test-model")
	runner.SetCompactionConfig(CompactionConfig{
		Enabled:            true,
		MinIterations:      3,
		IntervalIterations: 3,
		TailIterations:     1,
		SoftThresholdBytes: 65536,
		HardThresholdBytes: 262144,
	})

	initialMsgs := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleSystem, Content: "Pinned system instructions"},
		{Role: openai.ChatMessageRoleUser, Content: "Earlier user greeting"},
		{Role: openai.ChatMessageRoleAssistant, Content: "Earlier assistant reply"},
		{Role: openai.ChatMessageRoleUser, Content: "Find the database version and run tests"},
	}
	contextJSON, err := EncodeMessages(initialMsgs)
	require.NoError(t, err)

	run := &FSMRun{
		ID:            "run_rubric_1",
		ChatID:        "chat_1",
		UserID:        "user_1",
		FSMType:       FSMTypeToolLoop,
		Status:        RunStatusPending,
		CurrentState:  StateInit,
		MaxIterations: 10,
		ContextJSON:   contextJSON,
	}
	require.NoError(t, store.CreateRun(ctx, run))

	tools := []openai.Tool{
		{Type: openai.ToolTypeFunction, Function: &openai.FunctionDefinition{Name: "inspect_step"}},
	}

	require.NoError(t, runner.Execute(ctx, run, store, NewLegacyToolset(tools, invoker), "test-model"))
	assert.Equal(t, RunStatusCompleted, run.Status)
	assert.Equal(t, "Verification complete: schema v4 and tests pass.", run.ResultJSON)
	assert.Equal(t, int32(1), atomic.LoadInt32(&probeCalls))

	// Verify compacted snapshot in SQLite
	latestComp, err := store.GetLatestCompaction(ctx, run.ID)
	require.NoError(t, err)
	require.NotNil(t, latestComp)
	assert.Equal(t, 1, latestComp.Version)
	assert.Equal(t, 0, latestComp.ParentVersion)
	assert.Equal(t, 2, latestComp.CompactedUpToIter)
	assert.Equal(t, CompactionVerdictProbedCompact, latestComp.RubricVerdict)
	assert.Equal(t, []string{"run_rubric_1_i1_s0", "run_rubric_1_i2_s0"}, latestComp.CompactedStepIDs)

	// Verify reconstructed messages sent on iteration 4 LLM request:
	// [0..3]: pinned prefix (system, prior user, prior assistant, current user objective)
	// [4]: rendered CompactedTaskState covering iterations 1..2
	// [5]: raw tail iteration 3 assistant tool call (call_iter_3)
	// [6]: raw tail iteration 3 tool result (call_iter_3)
	mu.Lock()
	defer mu.Unlock()
	require.Len(t, capturedIter4Messages, 7)
	assert.Equal(t, "Pinned system instructions", capturedIter4Messages[0].Content)
	assert.Equal(t, "Earlier user greeting", capturedIter4Messages[1].Content)
	assert.Equal(t, "Earlier assistant reply", capturedIter4Messages[2].Content)
	assert.Equal(t, "Find the database version and run tests", capturedIter4Messages[3].Content)
	assert.Contains(t, capturedIter4Messages[4].Content, "[Compacted Task State v1 | Iterations 1..2]")
	assert.Contains(t, capturedIter4Messages[4].Content, "FSM schema version is 4 [source_steps: run_rubric_1_i1_s0, run_rubric_1_i2_s0]")
	require.Len(t, capturedIter4Messages[5].ToolCalls, 1)
	assert.Equal(t, "call_iter_3", capturedIter4Messages[5].ToolCalls[0].ID)
	assert.Equal(t, openai.ChatMessageRoleTool, capturedIter4Messages[6].Role)
	assert.Equal(t, "call_iter_3", capturedIter4Messages[6].ToolCallID)

	// Raw steps for iterations 1..3 remain untouched in fsm_steps
	allSteps, err := store.ListStepsByRun(ctx, run.ID)
	require.NoError(t, err)
	assert.Len(t, allSteps, 3)
}

func TestToolLoop_RubricProbeSkipRecordsEventWithoutCompacting(t *testing.T) {
	db := setupTestDB(t)
	store := NewStore(db)
	ctx := context.Background()

	var probeCalls int32
	invoker := ToolInvokerFunc(func(ctx context.Context, name, argsJSON string) (string, error) {
		return `{"ok":true}`, nil
	})
	runner := NewToolLoopRunner(&mockLLMClient{
		handler: func(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error) {
			toolCount := 0
			for _, m := range req.Messages {
				if m.Role == openai.ChatMessageRoleTool {
					toolCount++
				}
			}
			if toolCount >= 3 {
				return &openai.ChatCompletionResponse{
					Choices: []openai.ChatCompletionChoice{
						{Message: openai.ChatCompletionMessage{Role: openai.ChatMessageRoleAssistant, Content: "Done after 3 tools"}},
					},
				}, nil
			}
			nextIter := toolCount + 1
			return &openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role: openai.ChatMessageRoleAssistant,
							ToolCalls: []openai.ToolCall{
								{
									ID:   fmt.Sprintf("call_%d", nextIter),
									Type: openai.ToolTypeFunction,
									Function: openai.FunctionCall{
										Name:      "inspect_step",
										Arguments: fmt.Sprintf(`{"step":%d}`, nextIter),
									},
								},
							},
						},
					},
				},
			}, nil
		},
	}, NewStepExecutor(invoker), "test-model")

	runner.SetCompactionConfig(DefaultCompactionConfig())
	runner.SetCompactionDecider(CompactionDeciderFunc(func(ctx context.Context, input CompactionProbeInput) (*CompactionProbeResult, error) {
		atomic.AddInt32(&probeCalls, 1)
		return &CompactionProbeResult{
			ShouldCompact: false,
			Reason:        "Midway through verification",
		}, nil
	}))

	ctxJSON, err := EncodeMessages([]openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleUser, Content: "Verify steps"},
	})
	require.NoError(t, err)

	run := &FSMRun{
		ID:            "run_probe_skip",
		ChatID:        "chat_1",
		UserID:        "user_1",
		FSMType:       FSMTypeToolLoop,
		Status:        RunStatusPending,
		CurrentState:  StateInit,
		MaxIterations: 10,
		ContextJSON:   ctxJSON,
	}
	require.NoError(t, store.CreateRun(ctx, run))

	tools := []openai.Tool{
		{Type: openai.ToolTypeFunction, Function: &openai.FunctionDefinition{Name: "inspect_step"}},
	}
	require.NoError(t, runner.Execute(ctx, run, store, NewLegacyToolset(tools, invoker), "test-model"))
	assert.Equal(t, int32(1), atomic.LoadInt32(&probeCalls))

	latest, err := store.GetLatestCompaction(ctx, run.ID)
	require.NoError(t, err)
	assert.Nil(t, latest)

	var payloadBytes []byte
	require.NoError(t, db.QueryRowContext(ctx, `
		SELECT payload FROM agent_trajectory_events
		WHERE run_id = ? AND event_type = 'compaction'
	`, run.ID).Scan(&payloadBytes))
	var evPayload trajectory.CompactionEventPayload
	require.NoError(t, json.Unmarshal(payloadBytes, &evPayload))
	assert.False(t, evPayload.ShouldCompact)
	assert.Equal(t, CompactionVerdictProbedSkip, evPayload.RubricVerdict)
	assert.Equal(t, "Midway through verification", evPayload.Reason)
}

func TestToolLoop_StuckLoopGuardSuppressesCompaction(t *testing.T) {
	db := setupTestDB(t)
	store := NewStore(db)
	ctx := context.Background()

	var probeCalls int32

	llm := &mockLLMClient{
		handler: func(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error) {
			toolCount := 0
			for _, m := range req.Messages {
				if m.Role == openai.ChatMessageRoleTool {
					toolCount++
				}
			}
			if toolCount >= 3 {
				return &openai.ChatCompletionResponse{
					Choices: []openai.ChatCompletionChoice{
						{Message: openai.ChatCompletionMessage{Role: openai.ChatMessageRoleAssistant, Content: "Stopping after repeated failure"}},
					},
				}, nil
			}
			nextIter := toolCount + 1
			// Iterations 2 and 3 execute identical failing tool call + arguments
			args := `{"cmd":"go test ./..."}`
			if nextIter == 1 {
				args = `{"cmd":"ls"}`
			}
			return &openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role: openai.ChatMessageRoleAssistant,
							ToolCalls: []openai.ToolCall{
								{
									ID:   fmt.Sprintf("call_fail_%d", nextIter),
									Type: openai.ToolTypeFunction,
									Function: openai.FunctionCall{
										Name:      "run_cmd",
										Arguments: args,
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
		if strings.Contains(argsJSON, "go test") {
			return "Exit Code: 1\n[stderr]\nFAIL", nil
		}
		return "Exit Code: 0\n[stdout]\nok", nil
	})

	runner := NewToolLoopRunner(llm, NewStepExecutor(invoker), "test-model")
	runner.SetCompactionConfig(CompactionConfig{
		Enabled:            true,
		MinIterations:      3,
		IntervalIterations: 3,
		TailIterations:     1,
		SoftThresholdBytes: 100, // Even below soft/hard thresholds, stuck guard suppresses compaction
		HardThresholdBytes: 200,
	})
	runner.SetCompactionDecider(CompactionDeciderFunc(func(ctx context.Context, input CompactionProbeInput) (*CompactionProbeResult, error) {
		atomic.AddInt32(&probeCalls, 1)
		return &CompactionProbeResult{ShouldCompact: true}, nil
	}))

	ctxJSON, err := EncodeMessages([]openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleUser, Content: "Run tests"},
	})
	require.NoError(t, err)

	run := &FSMRun{
		ID:            "run_stuck_guard",
		ChatID:        "chat_1",
		UserID:        "user_1",
		FSMType:       FSMTypeToolLoop,
		Status:        RunStatusPending,
		CurrentState:  StateInit,
		MaxIterations: 10,
		ContextJSON:   ctxJSON,
	}
	require.NoError(t, store.CreateRun(ctx, run))

	tools := []openai.Tool{
		{Type: openai.ToolTypeFunction, Function: &openai.FunctionDefinition{Name: "run_cmd"}},
	}
	require.NoError(t, runner.Execute(ctx, run, store, NewLegacyToolset(tools, invoker), "test-model"))

	// Probe must not have been called and no compaction saved because iterations 2 and 3 repeated an identical failing call
	assert.Equal(t, int32(0), atomic.LoadInt32(&probeCalls))
	latest, err := store.GetLatestCompaction(ctx, run.ID)
	require.NoError(t, err)
	assert.Nil(t, latest)
}

func TestToolLoop_HardByteThresholdBackstopForcedCompaction(t *testing.T) {
	db := setupTestDB(t)
	store := NewStore(db)
	ctx := context.Background()

	var probeCalls int32
	var sawCompactedOnIter3 atomic.Bool

	llm := &mockLLMClient{
		handler: func(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error) {
			for _, m := range req.Messages {
				if m.Role == openai.ChatMessageRoleSystem && strings.Contains(m.Content, "[Compacted Task State v1 | Iterations 1..1]") {
					sawCompactedOnIter3.Store(true)
					return &openai.ChatCompletionResponse{
						Choices: []openai.ChatCompletionChoice{
							{Message: openai.ChatCompletionMessage{Role: openai.ChatMessageRoleAssistant, Content: "Completed after forced backstop"}},
						},
					}, nil
				}
			}

			toolCount := 0
			for _, m := range req.Messages {
				if m.Role == openai.ChatMessageRoleTool {
					toolCount++
				}
			}
			nextIter := toolCount + 1
			return &openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role: openai.ChatMessageRoleAssistant,
							ToolCalls: []openai.ToolCall{
								{
									ID:   fmt.Sprintf("call_big_%d", nextIter),
									Type: openai.ToolTypeFunction,
									Function: openai.FunctionCall{
										Name:      "big_tool",
										Arguments: fmt.Sprintf(`{"iter":%d}`, nextIter),
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
		return strings.Repeat("X", 4096), nil
	})

	runner := NewToolLoopRunner(llm, NewStepExecutor(invoker), "test-model")
	runner.SetCompactionConfig(CompactionConfig{
		Enabled:            true,
		MinIterations:      2,
		IntervalIterations: 5,
		TailIterations:     1,
		SoftThresholdBytes: 2048,
		HardThresholdBytes: 4096,
	})
	runner.SetCompactionDecider(CompactionDeciderFunc(func(ctx context.Context, input CompactionProbeInput) (*CompactionProbeResult, error) {
		atomic.AddInt32(&probeCalls, 1)
		return &CompactionProbeResult{ShouldCompact: true}, nil
	}))

	ctxJSON, err := EncodeMessages([]openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleUser, Content: "Collect large tool outputs"},
	})
	require.NoError(t, err)

	run := &FSMRun{
		ID:            "run_hard_backstop",
		ChatID:        "chat_1",
		UserID:        "user_1",
		FSMType:       FSMTypeToolLoop,
		Status:        RunStatusPending,
		CurrentState:  StateInit,
		MaxIterations: 10,
		ContextJSON:   ctxJSON,
	}
	require.NoError(t, store.CreateRun(ctx, run))

	tools := []openai.Tool{
		{Type: openai.ToolTypeFunction, Function: &openai.FunctionDefinition{Name: "big_tool"}},
	}
	require.NoError(t, runner.Execute(ctx, run, store, NewLegacyToolset(tools, invoker), "test-model"))

	assert.True(t, sawCompactedOnIter3.Load())
	assert.Equal(t, int32(0), atomic.LoadInt32(&probeCalls), "hard backstop must not require rubric probe round-trip")

	latest, err := store.GetLatestCompaction(ctx, run.ID)
	require.NoError(t, err)
	require.NotNil(t, latest)
	assert.Equal(t, CompactionVerdictBackstopForced, latest.RubricVerdict)
	assert.Equal(t, 1, latest.CompactedUpToIter)
}

func TestEngine_RecoverRestoresLatestCompactedTaskState(t *testing.T) {
	db := setupTestDB(t)
	store := NewStore(db)
	ctx := context.Background()

	initialMsgs := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleSystem, Content: "System prompt"},
		{Role: openai.ChatMessageRoleUser, Content: "Recover me with compacted state"},
		{
			Role: openai.ChatMessageRoleAssistant,
			ToolCalls: []openai.ToolCall{
				{ID: "call_1", Type: openai.ToolTypeFunction, Function: openai.FunctionCall{Name: "read_data", Arguments: `{"n":1}`}},
			},
		},
		{
			Role: openai.ChatMessageRoleAssistant,
			ToolCalls: []openai.ToolCall{
				{ID: "call_2", Type: openai.ToolTypeFunction, Function: openai.FunctionCall{Name: "read_data", Arguments: `{"n":2}`}},
			},
		},
		{
			Role: openai.ChatMessageRoleAssistant,
			ToolCalls: []openai.ToolCall{
				{ID: "call_3", Type: openai.ToolTypeFunction, Function: openai.FunctionCall{Name: "read_data", Arguments: `{"n":3}`}},
			},
		},
	}
	ctxJSON, err := EncodeMessages(initialMsgs)
	require.NoError(t, err)

	now := time.Now().Unix()
	run := &FSMRun{
		ID:            "run_recover_compacted",
		ChatID:        "chat_recover",
		UserID:        "user_1",
		FSMType:       FSMTypeToolLoop,
		Status:        RunStatusRunning,
		CurrentState:  StateLLMRequest,
		Iteration:     3,
		MaxIterations: 10,
		ContextJSON:   ctxJSON,
		CreatedAt:     now,
		UpdatedAt:     now,
	}
	require.NoError(t, store.CreateRun(ctx, run))

	steps := []FSMStep{
		{ID: "s1", RunID: run.ID, Iteration: 1, StepIndex: 0, ToolName: "read_data", ToolCallID: "call_1", ArgsJSON: `{"n":1}`, ResultJSON: `{"content":"res1"}`, Status: StepStatusCompleted, CompletedAt: &now},
		{ID: "s2", RunID: run.ID, Iteration: 2, StepIndex: 0, ToolName: "read_data", ToolCallID: "call_2", ArgsJSON: `{"n":2}`, ResultJSON: `{"content":"res2"}`, Status: StepStatusCompleted, CompletedAt: &now},
		{ID: "s3", RunID: run.ID, Iteration: 3, StepIndex: 0, ToolName: "read_data", ToolCallID: "call_3", ArgsJSON: `{"n":3}`, ResultJSON: `{"content":"res3"}`, Status: StepStatusCompleted, CompletedAt: &now},
	}
	require.NoError(t, store.CreateSteps(ctx, steps))

	comp := &CompactedTaskState{
		Version:           1,
		ParentVersion:     0,
		RunID:             run.ID,
		CompactedUpToIter: 2,
		CompactedStepIDs:  []string{"s1", "s2"},
		OriginalObjective: "Recover me with compacted state",
		VerifiedFindings: []VerifiedFinding{
			{Summary: "Read items 1 and 2", SourceStepIDs: []string{"s1", "s2"}},
		},
		CurrentSubgoal: "Process item 3 and finish",
		RubricVerdict:  CompactionVerdictProbedCompact,
		Model:          "test-model",
	}
	require.NoError(t, store.SaveCompaction(ctx, comp))

	var recoveredMsgs []openai.ChatCompletionMessage
	var mu sync.Mutex
	doneCh := make(chan struct{})

	llm := &mockLLMClient{
		handler: func(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error) {
			mu.Lock()
			recoveredMsgs = append([]openai.ChatCompletionMessage(nil), req.Messages...)
			mu.Unlock()
			return &openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{Message: openai.ChatCompletionMessage{Role: openai.ChatMessageRoleAssistant, Content: "Recovered and finished!"}},
				},
			}, nil
		},
	}

	invoker := ToolInvokerFunc(func(ctx context.Context, name, argsJSON string) (string, error) {
		return `{}`, nil
	})
	engine := NewEngine(
		NewStaticStoreProvider(store),
		llm,
		invoker,
		WithDefaultModel("test-model"),
		WithCompactionConfig(DefaultCompactionConfig()),
		WithToolDefinitionProvider(ToolDefinitionProviderFunc(func(ctx context.Context, chatID string, isDM bool) []openai.Tool {
			return []openai.Tool{
				{Type: openai.ToolTypeFunction, Function: &openai.FunctionDefinition{Name: "read_data"}},
			}
		})),
		WithRunFinalizer(func(ctx context.Context, r *FSMRun) error {
			close(doneCh)
			return nil
		}),
	)

	require.NoError(t, engine.Recover(ctx))

	select {
	case <-doneCh:
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for recovered compacted run to finish")
	}
	engine.Stop()

	persisted, err := store.GetRun(ctx, run.ID)
	require.NoError(t, err)
	assert.Equal(t, RunStatusCompleted, persisted.Status)
	assert.Equal(t, "Recovered and finished!", persisted.ResultJSON)

	mu.Lock()
	defer mu.Unlock()
	// Expected messages on recovery:
	// [0]: System prompt
	// [1]: User ("Recover me with compacted state")
	// [2]: System ("[Compacted Task State v1 | Iterations 1..2]...")
	// [3]: Assistant (call_3)
	// [4]: Tool (call_3 -> res3)
	require.Len(t, recoveredMsgs, 5)
	assert.Equal(t, "System prompt", recoveredMsgs[0].Content)
	assert.Equal(t, "Recover me with compacted state", recoveredMsgs[1].Content)
	assert.Contains(t, recoveredMsgs[2].Content, "[Compacted Task State v1 | Iterations 1..2]")
	assert.Equal(t, "call_3", recoveredMsgs[3].ToolCalls[0].ID)
	assert.Equal(t, "call_3", recoveredMsgs[4].ToolCallID)
}

func TestBuildChatMessages_PreservesPriorToolExchangeInPinnedPrefix(t *testing.T) {
	db := setupTestDB(t)
	store := NewStore(db)
	ctx := context.Background()

	// Prior conversation history includes an earlier turn's assistant tool call + tool response
	// that is NOT part of this run's fsm_steps.
	initialMsgs := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleSystem, Content: "System instructions"},
		{Role: openai.ChatMessageRoleUser, Content: "Prior turn question"},
		{
			Role: openai.ChatMessageRoleAssistant,
			ToolCalls: []openai.ToolCall{
				{ID: "prior_turn_call_1", Type: openai.ToolTypeFunction, Function: openai.FunctionCall{Name: "old_tool", Arguments: `{}`}},
			},
		},
		{Role: openai.ChatMessageRoleTool, ToolCallID: "prior_turn_call_1", Content: `{"old":"result"}`},
		{Role: openai.ChatMessageRoleAssistant, Content: "Prior turn answer"},
		{Role: openai.ChatMessageRoleUser, Content: "Current turn objective"},
		{
			Role: openai.ChatMessageRoleAssistant,
			ToolCalls: []openai.ToolCall{
				{ID: "run_call_1", Type: openai.ToolTypeFunction, Function: openai.FunctionCall{Name: "curr_tool", Arguments: `{"i":1}`}},
			},
		},
		{
			Role: openai.ChatMessageRoleAssistant,
			ToolCalls: []openai.ToolCall{
				{ID: "run_call_2", Type: openai.ToolTypeFunction, Function: openai.FunctionCall{Name: "curr_tool", Arguments: `{"i":2}`}},
			},
		},
	}
	ctxJSON, err := EncodeMessages(initialMsgs)
	require.NoError(t, err)

	run := &FSMRun{
		ID:            "run_prior_tool_prefix",
		ChatID:        "chat_1",
		UserID:        "user_1",
		FSMType:       FSMTypeToolLoop,
		Status:        RunStatusRunning,
		CurrentState:  StateLLMRequest,
		Iteration:     2,
		MaxIterations: 10,
		ContextJSON:   ctxJSON,
	}
	require.NoError(t, store.CreateRun(ctx, run))

	now := time.Now().Unix()
	require.NoError(t, store.CreateSteps(ctx, []FSMStep{
		{ID: "s1", RunID: run.ID, Iteration: 1, StepIndex: 0, ToolName: "curr_tool", ToolCallID: "run_call_1", ArgsJSON: `{"i":1}`, ResultJSON: `{"content":"r1"}`, Status: StepStatusCompleted, CompletedAt: &now},
		{ID: "s2", RunID: run.ID, Iteration: 2, StepIndex: 0, ToolName: "curr_tool", ToolCallID: "run_call_2", ArgsJSON: `{"i":2}`, ResultJSON: `{"content":"r2"}`, Status: StepStatusCompleted, CompletedAt: &now},
	}))

	require.NoError(t, store.SaveCompaction(ctx, &CompactedTaskState{
		Version:           1,
		ParentVersion:     0,
		RunID:             run.ID,
		CompactedUpToIter: 1,
		CompactedStepIDs:  []string{"s1"},
		OriginalObjective: "Current turn objective",
		CurrentSubgoal:    "Finish iter 2",
	}))

	runner := NewToolLoopRunner(nil, nil, "test-model")
	reconstructed, err := runner.buildChatMessages(ctx, run, store)
	require.NoError(t, err)

	// [0..5]: full pinned prefix including prior_turn_call_1 and Current turn objective
	// [6]: CompactedTaskState v1 (inserted after Current turn objective, before run_call_1)
	// [7]: Assistant (run_call_2)
	// [8]: Tool (run_call_2)
	require.Len(t, reconstructed, 9)
	assert.Equal(t, "System instructions", reconstructed[0].Content)
	assert.Equal(t, "Prior turn question", reconstructed[1].Content)
	assert.Equal(t, "prior_turn_call_1", reconstructed[2].ToolCalls[0].ID)
	assert.Equal(t, "prior_turn_call_1", reconstructed[3].ToolCallID)
	assert.Equal(t, "Prior turn answer", reconstructed[4].Content)
	assert.Equal(t, "Current turn objective", reconstructed[5].Content)
	assert.Contains(t, reconstructed[6].Content, "[Compacted Task State v1 | Iterations 1..1]")
	assert.Equal(t, "run_call_2", reconstructed[7].ToolCalls[0].ID)
	assert.Equal(t, "run_call_2", reconstructed[8].ToolCallID)
}

func TestCompaction_ProbeBoundsAndOversizedStateRejection(t *testing.T) {
	db := setupTestDB(t)
	store := NewStore(db)
	ctx := context.Background()

	var capturedProbePrompt string
	var probeHadDeadline bool
	var probeMaxTokens int
	llm := &mockLLMClient{
		handler: func(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error) {
			if len(req.Messages) > 0 && strings.Contains(req.Messages[0].Content, "SelfCompact") {
				_, probeHadDeadline = ctx.Deadline()
				probeMaxTokens = req.MaxCompletionTokens
				capturedProbePrompt = req.Messages[1].Content
				// Return an oversized state (> maxCompactedStateBytes)
				oversizedResp := CompactionProbeResult{
					ShouldCompact: true,
					Reason:        "Oversized test",
					State: &CompactedTaskState{
						OriginalObjective: "Test oversized rejection",
						CurrentSubgoal:    strings.Repeat("Z", maxCompactedStateBytes+1024),
					},
				}
				b, _ := json.Marshal(oversizedResp)
				return &openai.ChatCompletionResponse{
					Choices: []openai.ChatCompletionChoice{
						{Message: openai.ChatCompletionMessage{Role: openai.ChatMessageRoleAssistant, Content: string(b)}},
					},
				}, nil
			}
			return nil, nil
		},
	}

	baseMsgs := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleUser, Content: "Test oversized rejection"},
		{Role: openai.ChatMessageRoleAssistant, ToolCalls: []openai.ToolCall{{ID: "c1", Type: openai.ToolTypeFunction, Function: openai.FunctionCall{Name: "t1", Arguments: "{}"}}}},
		{Role: openai.ChatMessageRoleAssistant, ToolCalls: []openai.ToolCall{{ID: "c2", Type: openai.ToolTypeFunction, Function: openai.FunctionCall{Name: "t1", Arguments: "{}"}}}},
		{Role: openai.ChatMessageRoleAssistant, ToolCalls: []openai.ToolCall{{ID: "c3", Type: openai.ToolTypeFunction, Function: openai.FunctionCall{Name: "t1", Arguments: "{}"}}}},
	}
	baseJSON, err := EncodeMessages(baseMsgs)
	require.NoError(t, err)

	run := &FSMRun{
		ID:            "run_bounds_test",
		ChatID:        "chat_1",
		UserID:        "user_1",
		FSMType:       FSMTypeToolLoop,
		Status:        RunStatusRunning,
		CurrentState:  StateExecuteSteps,
		Iteration:     3,
		MaxIterations: 10,
		ContextJSON:   baseJSON,
	}
	require.NoError(t, store.CreateRun(ctx, run))

	now := time.Now().Unix()
	hugeArgs := fmt.Sprintf(`{"blob":%q}`, strings.Repeat("A", 4096))
	require.NoError(t, store.CreateSteps(ctx, []FSMStep{
		{ID: "s1", RunID: run.ID, Iteration: 1, StepIndex: 0, ToolName: "t1", ToolCallID: "c1", ArgsJSON: hugeArgs, ResultJSON: `{"content":"ok"}`, Status: StepStatusCompleted, CompletedAt: &now},
		{ID: "s2", RunID: run.ID, Iteration: 2, StepIndex: 0, ToolName: "t1", ToolCallID: "c2", ArgsJSON: `{}`, ResultJSON: `{"content":"ok"}`, Status: StepStatusCompleted, CompletedAt: &now},
		{ID: "s3", RunID: run.ID, Iteration: 3, StepIndex: 0, ToolName: "t1", ToolCallID: "c3", ArgsJSON: `{}`, ResultJSON: `{"content":"ok"}`, Status: StepStatusCompleted, CompletedAt: &now},
	}))

	runner := NewToolLoopRunner(llm, nil, "test-model")
	runner.SetCompactionConfig(DefaultCompactionConfig())

	require.NoError(t, runner.maybeCompact(ctx, run, store, "test-model"))

	// Probe deadline, MaxTokens, and argument truncation must be enforced
	assert.True(t, probeHadDeadline, "probe request must carry a context deadline")
	assert.Equal(t, maxProbeCompletionToks, probeMaxTokens)
	assert.Contains(t, capturedProbePrompt, "...[truncated for probe]")
	assert.Less(t, len(capturedProbePrompt), 3000)

	// Oversized state must be rejected and recorded as a probe skip event
	latest, err := store.GetLatestCompaction(ctx, run.ID)
	require.NoError(t, err)
	assert.Nil(t, latest)

	var payloadBytes []byte
	require.NoError(t, db.QueryRowContext(ctx, `
		SELECT payload FROM agent_trajectory_events
		WHERE run_id = ? AND iteration = 3 AND event_type = 'compaction'
	`, run.ID).Scan(&payloadBytes))
	var ev trajectory.CompactionEventPayload
	require.NoError(t, json.Unmarshal(payloadBytes, &ev))
	assert.False(t, ev.ShouldCompact)
	assert.Equal(t, "probe_oversized_state", ev.Reason)

	// Next iteration (iter 4): test an expanding state (< 32 KiB, e.g. 4 KiB state replacing tiny steps)
	run.Iteration = 4
	require.NoError(t, store.CreateSteps(ctx, []FSMStep{
		{ID: "s4", RunID: run.ID, Iteration: 4, StepIndex: 0, ToolName: "t1", ToolCallID: "c4", ArgsJSON: `{}`, ResultJSON: `{"content":"ok"}`, Status: StepStatusCompleted, CompletedAt: &now},
	}))
	runner.SetCompactionDecider(CompactionDeciderFunc(func(ctx context.Context, input CompactionProbeInput) (*CompactionProbeResult, error) {
		return &CompactionProbeResult{
			ShouldCompact: true,
			Reason:        "Expanding state below 32 KiB",
			State: &CompactedTaskState{
				OriginalObjective: "Test expanding state",
				CurrentSubgoal:    strings.Repeat("E", 4096),
			},
		}, nil
	}))
	require.NoError(t, runner.maybeCompact(ctx, run, store, "test-model"))

	latestAfterExpand, err := store.GetLatestCompaction(ctx, run.ID)
	require.NoError(t, err)
	assert.Nil(t, latestAfterExpand)

	var expandPayload []byte
	require.NoError(t, db.QueryRowContext(ctx, `
		SELECT payload FROM agent_trajectory_events
		WHERE run_id = ? AND iteration = 4 AND event_type = 'compaction'
	`, run.ID).Scan(&expandPayload))
	var expandEv trajectory.CompactionEventPayload
	require.NoError(t, json.Unmarshal(expandPayload, &expandEv))
	assert.False(t, expandEv.ShouldCompact)
	assert.Equal(t, "compaction_did_not_shrink_context", expandEv.Reason)

	// Iteration 5: test forced backstop with an oversized previous state (> 32 KiB) being rejected safely
	_, err = db.ExecContext(ctx, `
		INSERT INTO fsm_compactions (
			id, run_id, version, parent_version, compacted_up_to_iter,
			rubric_verdict, state_json, model, input_bytes, output_bytes, created_at
		) VALUES ('forced_oversized_prev', ?, 1, 0, 3, 'probed_compact', ?, 'test-model', 40000, 35000, 1700000000)
	`, run.ID, fmt.Sprintf(`{"version":1,"run_id":%q,"compacted_up_to_iter":3,"current_subgoal":%q}`, run.ID, strings.Repeat("P", maxCompactedStateBytes+500)))
	require.NoError(t, err)

	run.Iteration = 5
	require.NoError(t, store.CreateSteps(ctx, []FSMStep{
		{ID: "s5", RunID: run.ID, Iteration: 5, StepIndex: 0, ToolName: "t1", ToolCallID: "c5", ArgsJSON: `{}`, ResultJSON: fmt.Sprintf(`{"content":%q}`, strings.Repeat("H", 10000)), Status: StepStatusCompleted, CompletedAt: &now},
	}))
	runner.SetCompactionConfig(CompactionConfig{
		Enabled:            true,
		MinIterations:      2,
		IntervalIterations: 2,
		TailIterations:     1,
		SoftThresholdBytes: 1024,
		HardThresholdBytes: 2048,
	})
	require.NoError(t, runner.maybeCompact(ctx, run, store, "test-model"))

	latestAfterOversizedForced, err := store.GetLatestCompaction(ctx, run.ID)
	require.NoError(t, err)
	require.NotNil(t, latestAfterOversizedForced)
	assert.Equal(t, 1, latestAfterOversizedForced.Version, "oversized forced backstop state must not save v2")
}

func TestCompaction_RubricThenForcedBackstopPreservesPlan(t *testing.T) {
	db := setupTestDB(t)
	store := NewStore(db)
	ctx := context.Background()

	initMsgs := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleUser, Content: "Multi-stage objective"},
		{Role: openai.ChatMessageRoleAssistant, ToolCalls: []openai.ToolCall{{ID: "c1", Type: openai.ToolTypeFunction, Function: openai.FunctionCall{Name: "t1", Arguments: "{}"}}}},
		{Role: openai.ChatMessageRoleAssistant, ToolCalls: []openai.ToolCall{{ID: "c2", Type: openai.ToolTypeFunction, Function: openai.FunctionCall{Name: "t1", Arguments: "{}"}}}},
		{Role: openai.ChatMessageRoleAssistant, ToolCalls: []openai.ToolCall{{ID: "c3", Type: openai.ToolTypeFunction, Function: openai.FunctionCall{Name: "t1", Arguments: "{}"}}}},
	}
	initJSON, err := EncodeMessages(initMsgs)
	require.NoError(t, err)

	run := &FSMRun{
		ID:            "run_rubric_then_backstop",
		ChatID:        "chat_1",
		UserID:        "user_1",
		FSMType:       FSMTypeToolLoop,
		Status:        RunStatusRunning,
		CurrentState:  StateExecuteSteps,
		Iteration:     3,
		MaxIterations: 10,
		ContextJSON:   initJSON,
	}
	require.NoError(t, store.CreateRun(ctx, run))

	now := time.Now().Unix()
	stepOut := fmt.Sprintf(`{"content":%q}`, strings.Repeat("step_output_", 60))
	require.NoError(t, store.CreateSteps(ctx, []FSMStep{
		{ID: "s1", RunID: run.ID, Iteration: 1, StepIndex: 0, ToolName: "t1", ToolCallID: "c1", ArgsJSON: `{}`, ResultJSON: stepOut, Status: StepStatusCompleted, CompletedAt: &now},
		{ID: "s2", RunID: run.ID, Iteration: 2, StepIndex: 0, ToolName: "t1", ToolCallID: "c2", ArgsJSON: `{}`, ResultJSON: stepOut, Status: StepStatusCompleted, CompletedAt: &now},
		{ID: "s3", RunID: run.ID, Iteration: 3, StepIndex: 0, ToolName: "t1", ToolCallID: "c3", ArgsJSON: `{}`, ResultJSON: stepOut, Status: StepStatusCompleted, CompletedAt: &now},
	}))

	runner := NewToolLoopRunner(nil, nil, "test-model")
	runner.SetCompactionConfig(CompactionConfig{
		Enabled:            true,
		MinIterations:      3,
		IntervalIterations: 2,
		TailIterations:     1,
		SoftThresholdBytes: 4096,
		HardThresholdBytes: 8192,
	})
	runner.SetCompactionDecider(CompactionDeciderFunc(func(ctx context.Context, input CompactionProbeInput) (*CompactionProbeResult, error) {
		return &CompactionProbeResult{
			ShouldCompact: true,
			Reason:        "Stage 1 done",
			State: &CompactedTaskState{
				OriginalObjective:   "Multi-stage objective",
				CurrentSubgoal:      "Deploy service and verify healthcheck",
				NextIntendedActions: []string{"Run deploy script", "Check /healthz"},
				VerifiedFindings: []VerifiedFinding{
					{Summary: "CRITICAL_DURABLE_FINDING: Build succeeded", SourceStepIDs: []string{"s1", "s2"}},
				},
			},
		}, nil
	}))

	// 1. Rubric compaction at iteration 3 (compacts 1..2)
	require.NoError(t, runner.maybeCompact(ctx, run, store, "test-model"))

	v1, err := store.GetLatestCompaction(ctx, run.ID)
	require.NoError(t, err)
	require.NotNil(t, v1)
	assert.Equal(t, 1, v1.Version)
	assert.Equal(t, 2, v1.CompactedUpToIter)

	// 2. Iteration 4 has 40 parallel steps (> 32 steps) and large output that triggers hard backstop (compacting up to iter 3)
	run.Iteration = 4
	var iter3ExtraCalls []openai.ToolCall
	var iter3ExtraSteps []FSMStep
	for idx := 1; idx <= 35; idx++ {
		callID := fmt.Sprintf("c3_extra_%d", idx)
		stepID := fmt.Sprintf("s3_extra_%d", idx)
		iter3ExtraCalls = append(iter3ExtraCalls, openai.ToolCall{
			ID:       callID,
			Type:     openai.ToolTypeFunction,
			Function: openai.FunctionCall{Name: "t1", Arguments: "{}"},
		})
		iter3ExtraSteps = append(iter3ExtraSteps, FSMStep{
			ID:          stepID,
			RunID:       run.ID,
			Iteration:   3,
			StepIndex:   idx,
			ToolName:    "t1",
			ToolCallID:  callID,
			ArgsJSON:    "{}",
			ResultJSON:  stepOut,
			Status:      StepStatusCompleted,
			CompletedAt: &now,
		})
	}
	baseMsgs := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleUser, Content: "Multi-stage objective"},
		{Role: openai.ChatMessageRoleAssistant, ToolCalls: []openai.ToolCall{{ID: "c1", Type: openai.ToolTypeFunction, Function: openai.FunctionCall{Name: "t1", Arguments: "{}"}}}},
		{Role: openai.ChatMessageRoleAssistant, ToolCalls: []openai.ToolCall{{ID: "c2", Type: openai.ToolTypeFunction, Function: openai.FunctionCall{Name: "t1", Arguments: "{}"}}}},
		{Role: openai.ChatMessageRoleAssistant, ToolCalls: append([]openai.ToolCall{{ID: "c3", Type: openai.ToolTypeFunction, Function: openai.FunctionCall{Name: "t1", Arguments: "{}"}}}, iter3ExtraCalls...)},
		{Role: openai.ChatMessageRoleAssistant, ToolCalls: []openai.ToolCall{{ID: "c4", Type: openai.ToolTypeFunction, Function: openai.FunctionCall{Name: "t1", Arguments: "{}"}}}},
	}
	run.ContextJSON, err = EncodeMessages(baseMsgs)
	require.NoError(t, err)
	require.NoError(t, store.CreateSteps(ctx, append(iter3ExtraSteps, FSMStep{
		ID: "s4", RunID: run.ID, Iteration: 4, StepIndex: 0, ToolName: "t1", ToolCallID: "c4", ArgsJSON: `{}`, ResultJSON: fmt.Sprintf(`{"content":%q}`, strings.Repeat("B", 10000)), Status: StepStatusCompleted, CompletedAt: &now,
	})))

	require.NoError(t, runner.maybeCompact(ctx, run, store, "test-model"))

	v2, err := store.GetLatestCompaction(ctx, run.ID)
	require.NoError(t, err)
	require.NotNil(t, v2)
	assert.Equal(t, 2, v2.Version)
	assert.Equal(t, 1, v2.ParentVersion)
	assert.Equal(t, 3, v2.CompactedUpToIter)
	assert.Equal(t, CompactionVerdictBackstopForced, v2.RubricVerdict)
	assert.Equal(t, "Deploy service and verify healthcheck", v2.CurrentSubgoal)
	assert.Equal(t, []string{"Run deploy script", "Check /healthz"}, v2.NextIntendedActions)
	require.Len(t, v2.VerifiedFindings, 2)
	assert.Equal(t, "CRITICAL_DURABLE_FINDING: Build succeeded", v2.VerifiedFindings[0].Summary, "prior durable findings must never be evicted")
	assert.Contains(t, v2.VerifiedFindings[1].Summary, "Consolidated 36 tool steps")
}

func TestCompaction_ProbeErrorEventAndReplayIdempotency(t *testing.T) {
	db := setupTestDB(t)
	store := NewStore(db)
	ctx := context.Background()

	run := &FSMRun{
		ID:            "run_probe_err_replay",
		ChatID:        "chat_1",
		UserID:        "user_1",
		FSMType:       FSMTypeToolLoop,
		Status:        RunStatusRunning,
		CurrentState:  StateExecuteSteps,
		Iteration:     3,
		MaxIterations: 10,
		ContextJSON:   `[{"role":"user","content":"Test replay idempotency"}]`,
	}
	require.NoError(t, store.CreateRun(ctx, run))

	now := time.Now().Unix()
	require.NoError(t, store.CreateSteps(ctx, []FSMStep{
		{ID: "s1", RunID: run.ID, Iteration: 1, StepIndex: 0, ToolName: "t1", ToolCallID: "c1", ArgsJSON: `{}`, ResultJSON: `{"content":"ok"}`, Status: StepStatusCompleted, CompletedAt: &now},
		{ID: "s2", RunID: run.ID, Iteration: 2, StepIndex: 0, ToolName: "t1", ToolCallID: "c2", ArgsJSON: `{}`, ResultJSON: `{"content":"ok"}`, Status: StepStatusCompleted, CompletedAt: &now},
		{ID: "s3", RunID: run.ID, Iteration: 3, StepIndex: 0, ToolName: "t1", ToolCallID: "c3", ArgsJSON: `{}`, ResultJSON: `{"content":"ok"}`, Status: StepStatusCompleted, CompletedAt: &now},
	}))

	var deciderCalls int32
	runner := NewToolLoopRunner(nil, nil, "test-model")
	runner.SetCompactionConfig(DefaultCompactionConfig())
	runner.SetCompactionDecider(CompactionDeciderFunc(func(ctx context.Context, input CompactionProbeInput) (*CompactionProbeResult, error) {
		atomic.AddInt32(&deciderCalls, 1)
		return nil, fmt.Errorf("simulated LLM timeout")
	}))

	// First call: probe fails, records probe_error skip event
	require.NoError(t, runner.maybeCompact(ctx, run, store, "test-model"))
	assert.Equal(t, int32(1), atomic.LoadInt32(&deciderCalls))

	hasEv, err := store.HasCompactionEventForIteration(ctx, run.ID, 3)
	require.NoError(t, err)
	assert.True(t, hasEv)

	// Second call on replay at the same iteration: must be a no-op without re-probing or failing
	require.NoError(t, runner.maybeCompact(ctx, run, store, "test-model"))
	assert.Equal(t, int32(1), atomic.LoadInt32(&deciderCalls))

	// Event write failure in handleExecuteSteps fails the run instead of being ignored
	run.Iteration = 4
	run.CurrentState = StateExecuteSteps
	require.NoError(t, store.UpdateRun(ctx, run))
	require.NoError(t, store.CreateSteps(ctx, []FSMStep{
		{ID: "s4", RunID: run.ID, Iteration: 4, StepIndex: 0, ToolName: "t1", ToolCallID: "c4", ArgsJSON: `{}`, ResultJSON: `{"content":"ok"}`, Status: StepStatusCompleted, CompletedAt: &now},
	}))
	_, err = db.ExecContext(ctx, `CREATE TRIGGER reject_trajectory_events_insert BEFORE INSERT ON agent_trajectory_events BEGIN SELECT RAISE(ABORT, 'event insert rejected'); END`)
	require.NoError(t, err)

	invoker := ToolInvokerFunc(func(ctx context.Context, name, argsJSON string) (string, error) {
		return "ok", nil
	})
	tools := []openai.Tool{{Type: openai.ToolTypeFunction, Function: &openai.FunctionDefinition{Name: "t1"}}}
	_, err = runner.handleExecuteSteps(ctx, run, store, NewLegacyToolset(tools, invoker), "test-model")
	require.Error(t, err)
	assert.Equal(t, RunStatusFailed, run.Status)
	assert.Equal(t, StateFailed, run.CurrentState)
	assert.Contains(t, run.ErrorText, "compaction persistence failure")
}

func TestToolLoop_HardBackstopBeforeFinalSynthesis(t *testing.T) {
	db := setupTestDB(t)
	store := NewStore(db)
	ctx := context.Background()

	var synthesisSawCompacted atomic.Bool
	llm := &mockLLMClient{
		handler: func(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error) {
			lastMsg := req.Messages[len(req.Messages)-1]
			if lastMsg.Role == openai.ChatMessageRoleUser && lastMsg.Content == SynthesisPrompt {
				for _, m := range req.Messages {
					if m.Role == openai.ChatMessageRoleSystem && strings.Contains(m.Content, "[Compacted Task State v1 | Iterations 1..1]") {
						synthesisSawCompacted.Store(true)
					}
				}
				return &openai.ChatCompletionResponse{
					Choices: []openai.ChatCompletionChoice{
						{Message: openai.ChatCompletionMessage{Role: openai.ChatMessageRoleAssistant, Content: "Synthesized after final iteration backstop"}},
					},
				}, nil
			}

			toolCount := 0
			for _, m := range req.Messages {
				if m.Role == openai.ChatMessageRoleTool {
					toolCount++
				}
			}
			nextIter := toolCount + 1
			return &openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role: openai.ChatMessageRoleAssistant,
							ToolCalls: []openai.ToolCall{
								{
									ID:   fmt.Sprintf("call_syn_%d", nextIter),
									Type: openai.ToolTypeFunction,
									Function: openai.FunctionCall{
										Name:      "big_tool",
										Arguments: fmt.Sprintf(`{"i":%d}`, nextIter),
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
		return strings.Repeat("Y", 4096), nil
	})

	runner := NewToolLoopRunner(llm, NewStepExecutor(invoker), "test-model")
	runner.SetCompactionConfig(CompactionConfig{
		Enabled:            true,
		MinIterations:      2,
		IntervalIterations: 5,
		TailIterations:     1,
		SoftThresholdBytes: 2048,
		HardThresholdBytes: 4096,
	})

	ctxJSON, err := EncodeMessages([]openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleUser, Content: "Run up to max iterations"},
	})
	require.NoError(t, err)

	run := &FSMRun{
		ID:            "run_syn_backstop",
		ChatID:        "chat_1",
		UserID:        "user_1",
		FSMType:       FSMTypeToolLoop,
		Status:        RunStatusPending,
		CurrentState:  StateInit,
		MaxIterations: 2, // Final iteration is iteration 2
		ContextJSON:   ctxJSON,
	}
	require.NoError(t, store.CreateRun(ctx, run))

	tools := []openai.Tool{
		{Type: openai.ToolTypeFunction, Function: &openai.FunctionDefinition{Name: "big_tool"}},
	}
	require.NoError(t, runner.Execute(ctx, run, store, NewLegacyToolset(tools, invoker), "test-model"))
	assert.Equal(t, RunStatusCompleted, run.Status)
	assert.True(t, synthesisSawCompacted.Load(), "final iteration before StateSynthesis must still trigger hard backstop")
}

func TestEvaluateCompactionTrigger_NonTerminalAndPairwiseStuckGuard(t *testing.T) {
	cfg := DefaultCompactionConfig()

	// Non-terminal PENDING or RUNNING step suppresses compaction even when HardThresholdBytes is exceeded
	stepsWithPending := []FSMStep{
		{ID: "s1", Iteration: 1, ToolName: "t1", ArgsJSON: `{}`, Status: StepStatusCompleted},
		{ID: "s2", Iteration: 2, ToolName: "t1", ArgsJSON: `{}`, Status: StepStatusCompleted},
		{ID: "s3", Iteration: 3, ToolName: "t1", ArgsJSON: `{}`, Status: StepStatusPending},
	}
	mode, _ := evaluateCompactionTrigger(cfg, 3, 0, cfg.HardThresholdBytes+1000, stepsWithPending)
	assert.Equal(t, compactionTriggerNone, mode)

	stepsWithRunning := []FSMStep{
		{ID: "s1", Iteration: 1, ToolName: "t1", ArgsJSON: `{}`, Status: StepStatusCompleted},
		{ID: "s2", Iteration: 2, ToolName: "t1", ArgsJSON: `{}`, Status: StepStatusCompleted},
		{ID: "s3", Iteration: 3, ToolName: "t1", ArgsJSON: `{}`, Status: StepStatusRunning},
	}
	modeRunning, _ := evaluateCompactionTrigger(cfg, 3, 0, cfg.HardThresholdBytes+1000, stepsWithRunning)
	assert.Equal(t, compactionTriggerNone, modeRunning)

	// Pairwise stuck guard: if iteration 2 had [callA(fail), callB(ok)] and iteration 3 had [callA(ok), callB(fail)],
	// different calls failed in each iteration so it is NOT a stuck repeating failure of the same call.
	stepsDiffFailures := []FSMStep{
		{ID: "s1", Iteration: 1, ToolName: "t0", ArgsJSON: `{}`, Status: StepStatusCompleted},
		{ID: "s2a", Iteration: 2, StepIndex: 0, ToolName: "tA", ArgsJSON: `{"x":1}`, Status: StepStatusFailed},
		{ID: "s2b", Iteration: 2, StepIndex: 1, ToolName: "tB", ArgsJSON: `{"y":2}`, Status: StepStatusCompleted},
		{ID: "s3a", Iteration: 3, StepIndex: 0, ToolName: "tA", ArgsJSON: `{"x":1}`, Status: StepStatusCompleted},
		{ID: "s3b", Iteration: 3, StepIndex: 1, ToolName: "tB", ArgsJSON: `{"y":2}`, Status: StepStatusFailed},
	}
	assert.False(t, isStuckRepeatingFailure(stepsDiffFailures, 3))
	mode2, target2 := evaluateCompactionTrigger(cfg, 3, 0, 1000, stepsDiffFailures)
	assert.Equal(t, compactionTriggerProbeRubric, mode2)
	assert.Equal(t, 2, target2)
}
