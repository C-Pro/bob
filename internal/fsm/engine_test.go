package fsm

import (
	"context"
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

	engine := NewEngine(storeProvider, llm, nil)

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
	engine := NewEngine(storeProvider, nil, nil, WithPollInterval(fastPoll))
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

	engine := NewEngine(storeProvider, llm, nil)

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

	engine := NewEngine(storeProvider, nil, nil, WithPollInterval(10*time.Millisecond))

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
