package fsm

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"bob/internal/tools"

	openai "github.com/sashabaranov/go-openai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestEngine_RunToolLoopSuccess(t *testing.T) {
	db := setupTestDB(t)
	store := NewStore(db)
	storeProvider := NewStaticStoreProvider(store)

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
										ID:   "call_1",
										Type: openai.ToolTypeFunction,
										Function: openai.FunctionCall{
											Name:      "web_search",
											Arguments: `{"query": "golang"}`,
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
							Content: "Golang is an open-source programming language.",
						},
					},
				},
			}, nil
		},
	}

	invoker := ToolInvokerFunc(func(ctx context.Context, name, argsJSON string) (string, error) {
		return `{"title": "Go", "snippet": "Go is great"}`, nil
	})

	engine := NewEngine(storeProvider, llm, invoker, WithDefaultModel("test-model"))

	req := ToolLoopRequest{
		ChatID: "townhall",
		UserID: "user_42",
		IsDM:   false,
		Messages: []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleUser, Content: "Tell me about Go"},
		},
	}

	res, err := engine.RunToolLoop(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, res)
	assert.NotEmpty(t, res.RunID)
	assert.Equal(t, "Golang is an open-source programming language.", res.Content)
	assert.Equal(t, 1, res.TotalIterations)

	// Verify run in store
	persisted, err := store.GetRun(context.Background(), res.RunID)
	require.NoError(t, err)
	assert.Equal(t, RunStatusCompleted, persisted.Status)
	assert.Equal(t, StateCompleted, persisted.CurrentState)
	assert.Equal(t, 10, persisted.MaxIterations) // Townhall default
}

func TestEngine_DefaultIterationLimits(t *testing.T) {
	db := setupTestDB(t)
	store := NewStore(db)
	storeProvider := NewStaticStoreProvider(store)

	llm := &mockLLMClient{
		handler: func(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error) {
			return &openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{Message: openai.ChatCompletionMessage{Role: "assistant", Content: "OK"}},
				},
			}, nil
		},
	}

	engine := NewEngine(storeProvider, llm, nil, WithDefaultModel("test-model"))

	// Townhall request: max_iterations defaults to 10
	resTH, err := engine.RunToolLoop(context.Background(), ToolLoopRequest{
		ChatID:   "townhall",
		IsDM:     false,
		Messages: []openai.ChatCompletionMessage{{Role: "user", Content: "hi"}},
	})
	require.NoError(t, err)
	runTH, err := store.GetRun(context.Background(), resTH.RunID)
	require.NoError(t, err)
	assert.Equal(t, 10, runTH.MaxIterations)

	// DM request: max_iterations defaults to 20
	resDM, err := engine.RunToolLoop(context.Background(), ToolLoopRequest{
		ChatID:   "dm_123",
		IsDM:     true,
		Messages: []openai.ChatCompletionMessage{{Role: "user", Content: "hi"}},
	})
	require.NoError(t, err)
	runDM, err := store.GetRun(context.Background(), resDM.RunID)
	require.NoError(t, err)
	assert.Equal(t, 20, runDM.MaxIterations)
}

func TestEngine_DelayedTransitionPoller(t *testing.T) {
	db := setupTestDB(t)
	store := NewStore(db)
	storeProvider := NewStaticStoreProvider(store)
	ctx := context.Background()

	var resumedCount int32

	// Custom runner to simulate a workflow that enters WAITING and resumes
	runner := &mockRunner{
		execFunc: func(ctx context.Context, run *FSMRun, s *Store, tools []openai.Tool, model string) error {
			if run.Iteration == 0 {
				run.Iteration = 1
				run.Status = RunStatusWaiting
				run.CurrentState = StateWaiting
				now := time.Now().Unix()
				run.ResumeAt = &now // Due immediately
				return s.UpdateRun(ctx, run)
			}
			// Turn 2 after waking up
			atomic.AddInt32(&resumedCount, 1)
			run.Status = RunStatusCompleted
			run.CurrentState = StateCompleted
			run.ResultJSON = "resumed successfully"
			return s.UpdateRun(ctx, run)
		},
	}

	fastPoll := 50 * time.Millisecond
	engine := NewEngine(storeProvider, nil, nil, WithPollInterval(fastPoll), WithDefaultModel("test-model"))
	engine.RegisterRunner(FSMTypeToolLoop, runner)

	// Create waiting run in SQLite
	now := time.Now().Unix()
	run := &FSMRun{
		ID:            "run_waiting_due",
		ChatID:        "chat_wait",
		FSMType:       FSMTypeToolLoop,
		Status:        RunStatusWaiting,
		CurrentState:  StateWaiting,
		Iteration:     1,
		MaxIterations: 5,
		ResumeAt:      &now,
		ContextJSON:   "[]",
	}
	require.NoError(t, store.CreateRun(ctx, run))

	// Start engine and background poller
	require.NoError(t, engine.Start(ctx))
	defer engine.Stop()

	// Wait for background poller to pick up due run
	require.Eventually(t, func() bool {
		r, err := store.GetRun(ctx, "run_waiting_due")
		if err != nil {
			return false
		}
		return r.Status == RunStatusCompleted && atomic.LoadInt32(&resumedCount) > 0
	}, 2*time.Second, 50*time.Millisecond)

	r, err := store.GetRun(ctx, "run_waiting_due")
	require.NoError(t, err)
	assert.Equal(t, RunStatusCompleted, r.Status)
	assert.Equal(t, "resumed successfully", r.ResultJSON)
}

func TestEngine_CrashRecoveryInterruptedRuns(t *testing.T) {
	db := setupTestDB(t)
	store := NewStore(db)
	storeProvider := NewStaticStoreProvider(store)
	ctx := context.Background()

	llm := &mockLLMClient{
		handler: func(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error) {
			return &openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role:    openai.ChatMessageRoleAssistant,
							Content: "Recovered and completed.",
						},
					},
				},
			}, nil
		},
	}

	engine := NewEngine(storeProvider, llm, nil, WithDefaultModel("test-model"))

	// Simulate runs that were mid-flight when process crashed
	contextJSON, err := EncodeMessages([]openai.ChatCompletionMessage{
		{Role: "user", Content: "continue task"},
	})
	require.NoError(t, err)

	runInterrupted := &FSMRun{
		ID:            "run_interrupted",
		ChatID:        "chat_crash",
		FSMType:       FSMTypeToolLoop,
		Status:        RunStatusRunning, // Left running during crash
		CurrentState:  StateLLMRequest,
		Iteration:     1,
		MaxIterations: 5,
		ContextJSON:   contextJSON,
	}
	require.NoError(t, store.CreateRun(ctx, runInterrupted))

	// Run Recover
	err = engine.Recover(ctx)
	require.NoError(t, err)

	// Wait for recovered execution to complete
	require.Eventually(t, func() bool {
		r, err := store.GetRun(ctx, "run_interrupted")
		if err != nil {
			return false
		}
		return r.Status == RunStatusCompleted
	}, 2*time.Second, 50*time.Millisecond)

	recovered, err := store.GetRun(ctx, "run_interrupted")
	require.NoError(t, err)
	assert.Equal(t, RunStatusCompleted, recovered.Status)
	assert.Equal(t, "Recovered and completed.", recovered.ResultJSON)
}

func TestEngine_RecoverySessionContext(t *testing.T) {
	db := setupTestDB(t)
	store := NewStore(db)
	storeProvider := NewStaticStoreProvider(store)
	ctx := context.Background()

	var capturedSession tools.ChatSessionContext
	runner := &mockRunner{
		execFunc: func(ctx context.Context, run *FSMRun, s *Store, toolsList []openai.Tool, model string) error {
			if sess, ok := tools.ChatSessionFromContext(ctx); ok {
				capturedSession = sess
			}
			run.Status = RunStatusCompleted
			return s.UpdateRun(ctx, run)
		},
	}

	engine := NewEngine(storeProvider, nil, nil)
	engine.RegisterRunner(FSMTypeToolLoop, runner)

	run := &FSMRun{
		ID:            "run_dm_recovery",
		ChatID:        "dm_chat_123",
		UserID:        "user_alice",
		IsDM:          true,
		FSMType:       FSMTypeToolLoop,
		Status:        RunStatusRunning,
		CurrentState:  StateExecuteSteps,
		Iteration:     1,
		MaxIterations: 20,
		ContextJSON:   "[]",
	}
	require.NoError(t, store.CreateRun(ctx, run))

	err := engine.Recover(ctx)
	require.NoError(t, err)

	require.Eventually(t, func() bool {
		r, err := store.GetRun(ctx, "run_dm_recovery")
		return err == nil && r.Status == RunStatusCompleted
	}, 2*time.Second, 50*time.Millisecond)

	assert.Equal(t, "dm_chat_123", capturedSession.ChatID)
	assert.Equal(t, "user_alice", capturedSession.UserID)
	assert.True(t, capturedSession.IsDM)
}

func TestEngine_RecoveryToolDefinitionProvider(t *testing.T) {
	db := setupTestDB(t)
	store := NewStore(db)
	storeProvider := NewStaticStoreProvider(store)
	ctx := context.Background()

	var capturedTools []openai.Tool
	runner := &mockRunner{
		execFunc: func(ctx context.Context, run *FSMRun, s *Store, toolsList []openai.Tool, model string) error {
			capturedTools = toolsList
			run.Status = RunStatusCompleted
			return s.UpdateRun(ctx, run)
		},
	}

	expectedTools := []openai.Tool{
		{Type: openai.ToolTypeFunction, Function: &openai.FunctionDefinition{Name: "custom_tool"}},
	}
	provider := ToolDefinitionProviderFunc(func(ctx context.Context, chatID string, isDM bool) []openai.Tool {
		if isDM {
			return expectedTools
		}
		return nil
	})

	engine := NewEngine(storeProvider, nil, nil, WithToolDefinitionProvider(provider))
	engine.RegisterRunner(FSMTypeToolLoop, runner)

	run := &FSMRun{
		ID:            "run_tool_recovery",
		ChatID:        "dm_chat_456",
		UserID:        "user_bob",
		IsDM:          true,
		FSMType:       FSMTypeToolLoop,
		Status:        RunStatusRunning,
		CurrentState:  StateExecuteSteps,
		Iteration:     1,
		MaxIterations: 20,
		ContextJSON:   "[]",
	}
	require.NoError(t, store.CreateRun(ctx, run))

	require.NoError(t, engine.Recover(ctx))

	require.Eventually(t, func() bool {
		r, err := store.GetRun(ctx, "run_tool_recovery")
		return err == nil && r.Status == RunStatusCompleted
	}, 2*time.Second, 50*time.Millisecond)

	require.Len(t, capturedTools, 1)
	assert.Equal(t, "custom_tool", capturedTools[0].Function.Name)
}

func TestEngine_ConcurrencyDeduplication(t *testing.T) {
	db := setupTestDB(t)
	store := NewStore(db)
	storeProvider := NewStaticStoreProvider(store)

	engine := NewEngine(storeProvider, nil, nil)

	cancelCalled := false
	cancelFunc := func() { cancelCalled = true }

	// First acquire succeeds
	assert.True(t, engine.acquireRun("run_concurrency", cancelFunc))
	assert.True(t, engine.isRunActive("run_concurrency"))

	// Second acquire fails
	assert.False(t, engine.acquireRun("run_concurrency", func() {}))

	// Release allows subsequent acquire
	engine.releaseRun("run_concurrency")
	assert.False(t, engine.isRunActive("run_concurrency"))
	assert.True(t, engine.acquireRun("run_concurrency", func() {}))
	engine.releaseRun("run_concurrency")
	assert.False(t, cancelCalled)
}

func TestEngine_LifecycleStartStop(t *testing.T) {
	db := setupTestDB(t)
	store := NewStore(db)
	storeProvider := NewStaticStoreProvider(store)

	engine := NewEngine(storeProvider, nil, nil, WithPollInterval(10*time.Millisecond), WithDefaultModel("test-model"))

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	require.NoError(t, engine.Start(ctx))
	// Safe to call stop multiple times
	engine.Stop()
	engine.Stop()
}

type mockRunner struct {
	execFunc func(ctx context.Context, run *FSMRun, store *Store, tools []openai.Tool, model string) error
}

func (m *mockRunner) Execute(ctx context.Context, run *FSMRun, store *Store, tools []openai.Tool, model string) error {
	if m.execFunc != nil {
		return m.execFunc(ctx, run, store, tools, model)
	}
	return nil
}

type mockResultSink struct {
	delivered []*FSMRun
	mu        sync.Mutex
}

func (s *mockResultSink) Deliver(ctx context.Context, run *FSMRun) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.delivered = append(s.delivered, run)
	return nil
}

func TestEngine_RecoveryResultSinkDelivery(t *testing.T) {
	db := setupTestDB(t)
	store := NewStore(db)
	storeProvider := NewStaticStoreProvider(store)
	ctx := context.Background()

	sink := &mockResultSink{}
	runner := &mockRunner{
		execFunc: func(ctx context.Context, run *FSMRun, s *Store, toolsList []openai.Tool, model string) error {
			run.Status = RunStatusCompleted
			run.ResultJSON = "Here is your recovered answer."
			return s.UpdateRun(ctx, run)
		},
	}

	engine := NewEngine(storeProvider, nil, nil, WithResultSink(sink))
	engine.RegisterRunner(FSMTypeToolLoop, runner)

	run := &FSMRun{
		ID:            "run_deliver_test",
		ChatID:        "chat_dm_deliver",
		UserID:        "user_bob",
		IsDM:          true,
		FSMType:       FSMTypeToolLoop,
		Status:        RunStatusRunning,
		CurrentState:  StateExecuteSteps,
		Iteration:     1,
		MaxIterations: 20,
		ContextJSON:   "[]",
	}
	require.NoError(t, store.CreateRun(ctx, run))

	require.NoError(t, engine.Recover(ctx))

	require.Eventually(t, func() bool {
		sink.mu.Lock()
		defer sink.mu.Unlock()
		return len(sink.delivered) > 0
	}, 2*time.Second, 50*time.Millisecond)

	sink.mu.Lock()
	defer sink.mu.Unlock()
	require.Len(t, sink.delivered, 1)
	assert.Equal(t, "run_deliver_test", sink.delivered[0].ID)
	assert.Equal(t, "chat_dm_deliver", sink.delivered[0].ChatID)
	assert.Equal(t, "Here is your recovered answer.", sink.delivered[0].ResultJSON)
}

func TestEngine_RunToolLoop_MaxWaitCyclesExceeded(t *testing.T) {
	db := setupTestDB(t)
	store := NewStore(db)
	storeProvider := NewStaticStoreProvider(store)
	ctx := context.Background()

	runner := &mockRunner{
		execFunc: func(ctx context.Context, run *FSMRun, s *Store, toolsList []openai.Tool, model string) error {
			now := time.Now().Unix() - 10
			run.Status = RunStatusWaiting
			run.ResumeAt = &now
			return s.UpdateRun(ctx, run)
		},
	}

	engine := NewEngine(storeProvider, nil, nil)
	engine.RegisterRunner(FSMTypeToolLoop, runner)

	req := ToolLoopRequest{
		RunID:  "run_wait_loop",
		ChatID: "chat_wait",
		UserID: "user_wait",
		IsDM:   true,
		Model:  "test-model",
		Messages: []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleUser, Content: "Hello"},
		},
	}

	res, err := engine.RunToolLoop(ctx, req)
	require.Error(t, err)
	assert.Nil(t, res)
	assert.Contains(t, err.Error(), "exceeded maximum wait cycles")

	persisted, err := store.GetRun(ctx, "run_wait_loop")
	require.NoError(t, err)
	assert.Equal(t, RunStatusFailed, persisted.Status)
	assert.Equal(t, StateFailed, persisted.CurrentState)
	assert.Equal(t, 10, persisted.WaitCycles)
	assert.Contains(t, persisted.ErrorText, "exceeded maximum wait cycles")
}

func TestEngine_RecoverySkipsTerminalFreshRun(t *testing.T) {
	db := setupTestDB(t)
	store := NewStore(db)
	storeProvider := NewStaticStoreProvider(store)
	ctx := context.Background()

	var execCount atomic.Int32
	runner := &mockRunner{
		execFunc: func(ctx context.Context, run *FSMRun, s *Store, toolsList []openai.Tool, model string) error {
			run.Status = RunStatusCompleted
			run.CurrentState = StateCompleted
			run.ResultJSON = "completed"
			err := s.UpdateRun(ctx, run)
			if err == nil {
				execCount.Add(1)
			}
			return err
		},
	}

	engine := NewEngine(storeProvider, nil, nil)
	engine.RegisterRunner(FSMTypeToolLoop, runner)

	run := &FSMRun{
		ID:            "run_fresh_check",
		ChatID:        "chat_test",
		FSMType:       FSMTypeToolLoop,
		Status:        RunStatusRunning,
		CurrentState:  StateExecuteSteps,
		Iteration:     1,
		MaxIterations: 5,
		ContextJSON:   "[]",
	}
	require.NoError(t, store.CreateRun(ctx, run))

	// Launch multiple concurrent Recover passes
	const passes = 10
	var wg sync.WaitGroup
	for i := 0; i < passes; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_ = engine.Recover(ctx)
		}()
	}
	wg.Wait()

	require.Eventually(t, func() bool {
		return execCount.Load() == 1
	}, 3*time.Second, 10*time.Millisecond)

	engine.Stop()

	// Exactly 1 execution should have occurred despite multiple concurrent passes
	assert.Equal(t, int32(1), execCount.Load(), "exactly one recovery pass should execute the run")
	persisted, err := store.GetRun(ctx, "run_fresh_check")
	require.NoError(t, err)
	assert.Equal(t, RunStatusCompleted, persisted.Status)
	assert.Equal(t, "completed", persisted.ResultJSON)
}

func TestEngine_PollDueWaitingRuns_SkipsTerminalFreshRun(t *testing.T) {
	db := setupTestDB(t)
	store := NewStore(db)
	storeProvider := NewStaticStoreProvider(store)
	ctx := context.Background()

	var execCount atomic.Int32
	runner := &mockRunner{
		execFunc: func(ctx context.Context, run *FSMRun, s *Store, toolsList []openai.Tool, model string) error {
			run.Status = RunStatusCompleted
			run.CurrentState = StateCompleted
			run.ResultJSON = "completed from poll"
			err := s.UpdateRun(ctx, run)
			if err == nil {
				execCount.Add(1)
			}
			return err
		},
	}

	engine := NewEngine(storeProvider, nil, nil)
	engine.RegisterRunner(FSMTypeToolLoop, runner)

	past := time.Now().Unix() - 10
	run := &FSMRun{
		ID:            "run_poll_race_check",
		ChatID:        "chat_test",
		FSMType:       FSMTypeToolLoop,
		Status:        RunStatusWaiting,
		CurrentState:  StateWaiting,
		ResumeAt:      &past,
		Iteration:     1,
		MaxIterations: 5,
		ContextJSON:   "[]",
	}
	require.NoError(t, store.CreateRun(ctx, run))

	// Launch multiple concurrent PollDueWaitingRuns passes
	const passes = 10
	var wg sync.WaitGroup
	for i := 0; i < passes; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_ = engine.PollDueWaitingRuns(ctx)
		}()
	}
	wg.Wait()

	require.Eventually(t, func() bool {
		return execCount.Load() == 1
	}, 3*time.Second, 10*time.Millisecond)

	engine.Stop()

	assert.Equal(t, int32(1), execCount.Load(), "exactly one poll pass should execute the run")
	persisted, err := store.GetRun(ctx, "run_poll_race_check")
	require.NoError(t, err)
	assert.Equal(t, RunStatusCompleted, persisted.Status)
	assert.Equal(t, "completed from poll", persisted.ResultJSON)
}

func TestEngine_Stop_ConcurrentWithSpawn(t *testing.T) {
	db := setupTestDB(t)
	store := NewStore(db)
	storeProvider := NewStaticStoreProvider(store)
	engine := NewEngine(storeProvider, nil, nil)

	var wg sync.WaitGroup
	stopSignal := make(chan struct{})

	// Spin up workers that repeatedly attempt to spawn tasks on engine
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-stopSignal:
					return
				default:
					engine.spawn(func() {
						time.Sleep(time.Microsecond)
					})
				}
			}
		}()
	}

	// Let workers run for a short duration
	time.Sleep(10 * time.Millisecond)

	// Stop the engine concurrently with active spawners
	engine.Stop()
	close(stopSignal)
	wg.Wait()

	// After Stop, further spawn calls must return false
	assert.False(t, engine.spawn(func() {}), "spawn must return false after engine is stopped")
}

func TestEngine_RunToolLoop_CancelledContext_MarksTerminated(t *testing.T) {
	db := setupTestDB(t)
	store := NewStore(db)
	storeProvider := NewStaticStoreProvider(store)

	ctx, cancel := context.WithCancel(context.Background())

	runner := &mockRunner{
		execFunc: func(execCtx context.Context, run *FSMRun, s *Store, toolsList []openai.Tool, model string) error {
			cancel() // cancel parent context during execution
			<-execCtx.Done()
			return execCtx.Err()
		},
	}

	engine := NewEngine(storeProvider, nil, nil)
	engine.RegisterRunner(FSMTypeToolLoop, runner)

	req := ToolLoopRequest{
		RunID:  "run_cancelled_test",
		ChatID: "chat_cancel",
		UserID: "user_cancel",
		Model:  "test-model",
		Messages: []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleUser, Content: "hi"},
		},
	}

	res, err := engine.RunToolLoop(ctx, req)
	require.Error(t, err)
	assert.Nil(t, res)
	assert.True(t, errors.Is(err, context.Canceled))

	persisted, err := store.GetRun(context.Background(), "run_cancelled_test")
	require.NoError(t, err)
	assert.Equal(t, RunStatusTerminated, persisted.Status)
	assert.Equal(t, StateTerminated, persisted.CurrentState)
	assert.Equal(t, "request context cancelled", persisted.ErrorText)
}

func TestEngine_Recover_StaleRun_MarksTerminated(t *testing.T) {
	db := setupTestDB(t)
	store := NewStore(db)
	storeProvider := NewStaticStoreProvider(store)
	ctx := context.Background()

	execCount := 0
	runner := &mockRunner{
		execFunc: func(execCtx context.Context, run *FSMRun, s *Store, toolsList []openai.Tool, model string) error {
			execCount++
			return nil
		},
	}

	// Set 5 minute staleness cutoff
	engine := NewEngine(storeProvider, nil, nil, WithRecoveryStalenessCutoff(5*time.Minute))
	engine.RegisterRunner(FSMTypeToolLoop, runner)

	// Create an interrupted run whose UpdatedAt is 1 hour in the past
	pastTime := time.Now().Unix() - 3600
	run := &FSMRun{
		ID:            "run_stale_test",
		ChatID:        "chat_stale",
		FSMType:       FSMTypeToolLoop,
		Status:        RunStatusRunning,
		CurrentState:  StateExecuteSteps,
		Iteration:     1,
		MaxIterations: 5,
		ContextJSON:   "[]",
		UpdatedAt:     pastTime,
		CreatedAt:     pastTime,
	}
	require.NoError(t, store.CreateRun(ctx, run))

	// Manually ensure updated_at is in the past (since CreateRun sets it to now if 0)
	_, err := db.ExecContext(ctx, "UPDATE fsm_runs SET updated_at = ? WHERE id = ?", pastTime, run.ID)
	require.NoError(t, err)

	err = engine.Recover(ctx)
	require.NoError(t, err)

	engine.Stop()

	assert.Equal(t, 0, execCount, "stale run should not be executed")
	persisted, err := store.GetRun(ctx, "run_stale_test")
	require.NoError(t, err)
	assert.Equal(t, RunStatusTerminated, persisted.Status)
	assert.Equal(t, StateTerminated, persisted.CurrentState)
	assert.Contains(t, persisted.ErrorText, "stale run expired before recovery")
}

func TestEngine_Recover_BoundedConcurrency(t *testing.T) {
	db := setupTestDB(t)
	store := NewStore(db)
	storeProvider := NewStaticStoreProvider(store)
	ctx := context.Background()

	var currentConcurrent atomic.Int32
	var maxObserved atomic.Int32
	var completedCount atomic.Int32

	runner := &mockRunner{
		execFunc: func(execCtx context.Context, run *FSMRun, s *Store, toolsList []openai.Tool, model string) error {
			cur := currentConcurrent.Add(1)
			for {
				oldMax := maxObserved.Load()
				if cur <= oldMax || maxObserved.CompareAndSwap(oldMax, cur) {
					break
				}
			}

			time.Sleep(25 * time.Millisecond)
			currentConcurrent.Add(-1)

			run.Status = RunStatusCompleted
			run.CurrentState = StateCompleted
			err := s.UpdateRun(execCtx, run)
			if err == nil {
				completedCount.Add(1)
			}
			return err
		},
	}

	const maxRecoveryConc = 2
	engine := NewEngine(storeProvider, nil, nil, WithMaxRecoveryConcurrency(maxRecoveryConc))
	engine.RegisterRunner(FSMTypeToolLoop, runner)

	const totalRuns = 6
	for i := 0; i < totalRuns; i++ {
		run := &FSMRun{
			ID:            fmt.Sprintf("run_bounded_%d", i),
			ChatID:        fmt.Sprintf("chat_%d", i),
			FSMType:       FSMTypeToolLoop,
			Status:        RunStatusRunning,
			CurrentState:  StateExecuteSteps,
			Iteration:     1,
			MaxIterations: 5,
			ContextJSON:   "[]",
		}
		require.NoError(t, store.CreateRun(ctx, run))
	}

	err := engine.Recover(ctx)
	require.NoError(t, err)

	require.Eventually(t, func() bool {
		return completedCount.Load() == int32(totalRuns)
	}, 3*time.Second, 10*time.Millisecond)

	engine.Stop()

	assert.Equal(t, int32(totalRuns), completedCount.Load(), "all runs must complete")
	assert.LessOrEqual(t, maxObserved.Load(), int32(maxRecoveryConc), "concurrency must not exceed maxRecoveryConcurrency")
}

func TestEngine_Start_RequiresDefaultModel(t *testing.T) {
	db := setupTestDB(t)
	store := NewStore(db)
	storeProvider := NewStaticStoreProvider(store)

	engine := NewEngine(storeProvider, nil, nil)
	err := engine.Start(context.Background())
	require.Error(t, err)
	assert.Contains(t, err.Error(), "fsm engine requires a default model")
}

func TestEngine_RunToolLoop_RequiresModel(t *testing.T) {
	db := setupTestDB(t)
	store := NewStore(db)
	storeProvider := NewStaticStoreProvider(store)

	engine := NewEngine(storeProvider, nil, nil)
	req := ToolLoopRequest{
		ChatID: "chat_1",
		Messages: []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleUser, Content: "hello"},
		},
	}
	res, err := engine.RunToolLoop(context.Background(), req)
	require.Error(t, err)
	assert.Nil(t, res)
	assert.Contains(t, err.Error(), "model cannot be empty")

	var runCount int
	err = db.QueryRow("SELECT COUNT(*) FROM fsm_runs").Scan(&runCount)
	require.NoError(t, err)
	assert.Equal(t, 0, runCount, "no run should be persisted when model validation fails")
}






