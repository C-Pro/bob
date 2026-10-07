package fsm

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"bob/internal/agentapi"
)

type mockFrontend struct {
	mu           sync.Mutex
	bindCalls    []agentapi.RunDescriptor
	bindErr      error
	bindTool     agentapi.Toolset
	deliverCalls []agentapi.Completion
	deliverErr   error
}

func (m *mockFrontend) Bind(ctx context.Context, desc agentapi.RunDescriptor) (agentapi.Bindings, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.bindCalls = append(m.bindCalls, desc)
	if m.bindErr != nil {
		return agentapi.Bindings{}, m.bindErr
	}
	return agentapi.Bindings{
		Tools: m.bindTool,
	}, nil
}

func (m *mockFrontend) Deliver(ctx context.Context, completion agentapi.Completion) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.deliverCalls = append(m.deliverCalls, completion)
	return m.deliverErr
}

type mockRecoveryRunner struct {
	mu        sync.Mutex
	runCalls  []*FSMRun
	runModel  string
	toolset   agentapi.Toolset
	executeFn func(ctx context.Context, run *FSMRun, store *Store, toolset agentapi.Toolset, model string) error
}

func (m *mockRecoveryRunner) Execute(ctx context.Context, run *FSMRun, store *Store, toolset agentapi.Toolset, model string) error {
	m.mu.Lock()
	m.runCalls = append(m.runCalls, run)
	m.runModel = model
	m.toolset = toolset
	fn := m.executeFn
	m.mu.Unlock()

	if fn != nil {
		return fn(ctx, run, store, toolset, model)
	}
	run.Status = RunStatusCompleted
	run.CurrentState = StateCompleted
	run.ResultJSON = "success from mock runner"
	return store.UpdateRun(ctx, run)
}

func TestEngine_Recovery_RegisteredFrontend(t *testing.T) {
	db := setupTestDB(t)
	store := NewStore(db)
	prov := NewStaticStoreProvider(store)
	ctx := context.Background()

	fe := &mockFrontend{}
	runner := &mockRecoveryRunner{}

	engine := NewEngine(prov, nil, nil,
		WithDefaultModel("default-model"),
		WithFrontend("custom_fe", fe),
	)
	engine.RegisterRunner(FSMTypeToolLoop, runner)

	now := time.Now().Unix()
	run := &FSMRun{
		ID:            "run_rec_1",
		ChatID:        "chat_custom",
		UserID:        "user_alice",
		FrontendID:    "custom_fe",
		ScopeID:       "scope_custom",
		Model:         "model-custom",
		ExecutionKind: agentapi.Interactive,
		FSMType:       FSMTypeToolLoop,
		Status:        RunStatusRunning,
		CurrentState:  StateExecuteSteps,
		Iteration:     1,
		MaxIterations: 10,
		ContextJSON:   `[]`,
		CreatedAt:     now,
		UpdatedAt:     now,
	}
	require.NoError(t, store.CreateRun(ctx, run))

	err := engine.Recover(ctx)
	require.NoError(t, err)

	// Wait for async recovery worker to finish
	require.Eventually(t, func() bool {
		fe.mu.Lock()
		defer fe.mu.Unlock()
		return len(fe.deliverCalls) > 0
	}, 3*time.Second, 20*time.Millisecond)

	fe.mu.Lock()
	require.Len(t, fe.bindCalls, 1)
	assert.Equal(t, "run_rec_1", fe.bindCalls[0].RunID)
	assert.Equal(t, "custom_fe", fe.bindCalls[0].Session.FrontendID)
	assert.Equal(t, "chat_custom", fe.bindCalls[0].Session.SessionID)
	assert.Equal(t, "scope_custom", fe.bindCalls[0].Session.ScopeID)
	assert.Equal(t, "model-custom", fe.bindCalls[0].Model)
	assert.Equal(t, agentapi.Interactive, fe.bindCalls[0].Kind)

	require.Len(t, fe.deliverCalls, 1)
	del := fe.deliverCalls[0]
	assert.Equal(t, "run_rec_1", del.Run.RunID)
	assert.Equal(t, agentapi.RunCompleted, del.Status)
	assert.Equal(t, "success from mock runner", del.Result.Content)
	fe.mu.Unlock()

	runner.mu.Lock()
	assert.Equal(t, "model-custom", runner.runModel)
	runner.mu.Unlock()

	recovered, err := store.GetRun(ctx, "run_rec_1")
	require.NoError(t, err)
	assert.Equal(t, RunStatusCompleted, recovered.Status)
}

func TestEngine_Recovery_LegacyFrontend_RegisteredBesedka(t *testing.T) {
	db := setupTestDB(t)
	store := NewStore(db)
	prov := NewStaticStoreProvider(store)
	ctx := context.Background()

	besedkaFE := &mockFrontend{}
	runner := &mockRecoveryRunner{}

	engine := NewEngine(prov, nil, nil,
		WithDefaultModel("default-model"),
		WithFrontend(LegacyFrontendID, besedkaFE),
	)
	engine.RegisterRunner(FSMTypeToolLoop, runner)

	now := time.Now().Unix()
	// Legacy run: FrontendID is empty
	run := &FSMRun{
		ID:            "run_legacy_1",
		ChatID:        "chat_legacy",
		UserID:        "user_bob",
		FrontendID:    "", // legacy
		ScopeID:       "", // legacy
		Model:         "", // legacy
		ExecutionKind: "", // legacy
		FSMType:       FSMTypeToolLoop,
		Status:        RunStatusRunning,
		CurrentState:  StateExecuteSteps,
		Iteration:     2,
		MaxIterations: 20,
		ContextJSON:   `[]`,
		CreatedAt:     now,
		UpdatedAt:     now,
	}
	require.NoError(t, store.CreateRun(ctx, run))

	err := engine.Recover(ctx)
	require.NoError(t, err)

	require.Eventually(t, func() bool {
		besedkaFE.mu.Lock()
		defer besedkaFE.mu.Unlock()
		return len(besedkaFE.deliverCalls) > 0
	}, 3*time.Second, 20*time.Millisecond)

	besedkaFE.mu.Lock()
	require.Len(t, besedkaFE.bindCalls, 1)
	assert.Equal(t, LegacyFrontendID, besedkaFE.bindCalls[0].Session.FrontendID)
	assert.Equal(t, "chat_legacy", besedkaFE.bindCalls[0].Session.SessionID)
	assert.Equal(t, "chat_legacy", besedkaFE.bindCalls[0].Session.ScopeID)
	assert.Equal(t, "default-model", besedkaFE.bindCalls[0].Model)
	assert.Equal(t, agentapi.Interactive, besedkaFE.bindCalls[0].Kind)

	require.Len(t, besedkaFE.deliverCalls, 1)
	assert.Equal(t, agentapi.RunCompleted, besedkaFE.deliverCalls[0].Status)
	besedkaFE.mu.Unlock()
}

func TestEngine_Recovery_LegacyFrontend_ResultSinkFallback(t *testing.T) {
	db := setupTestDB(t)
	store := NewStore(db)
	prov := NewStaticStoreProvider(store)
	ctx := context.Background()

	sink := &mockResultSink{}
	runner := &mockRecoveryRunner{}

	engine := NewEngine(prov, nil, nil,
		WithDefaultModel("default-model"),
		WithResultSink(sink),
	)
	engine.RegisterRunner(FSMTypeToolLoop, runner)

	now := time.Now().Unix()
	run := &FSMRun{
		ID:           "run_legacy_sink",
		ChatID:       "chat_legacy_sink",
		UserID:       "user_bob",
		FrontendID:   "",
		FSMType:      FSMTypeToolLoop,
		Status:       RunStatusRunning,
		CurrentState: StateExecuteSteps,
		Iteration:    1,
		ContextJSON:  `[]`,
		CreatedAt:    now,
		UpdatedAt:    now,
	}
	require.NoError(t, store.CreateRun(ctx, run))

	err := engine.Recover(ctx)
	require.NoError(t, err)

	require.Eventually(t, func() bool {
		sink.mu.Lock()
		defer sink.mu.Unlock()
		return len(sink.delivered) > 0
	}, 3*time.Second, 20*time.Millisecond)

	sink.mu.Lock()
	require.Len(t, sink.delivered, 1)
	assert.Equal(t, "run_legacy_sink", sink.delivered[0].ID)
	sink.mu.Unlock()
}

func TestEngine_Recovery_UnregisteredNonLegacyFrontend_FailsClosed(t *testing.T) {
	db := setupTestDB(t)
	store := NewStore(db)
	prov := NewStaticStoreProvider(store)
	ctx := context.Background()

	sink := &mockResultSink{}
	engine := NewEngine(prov, nil, nil,
		WithResultSink(sink),
	)

	now := time.Now().Unix()
	run := &FSMRun{
		ID:           "run_unknown_fe",
		ChatID:       "chat_priv",
		UserID:       "user_charlie",
		FrontendID:   "alien_frontend",
		FSMType:      FSMTypeToolLoop,
		Status:       RunStatusRunning,
		CurrentState: StateExecuteSteps,
		ContextJSON:  `[]`,
		CreatedAt:    now,
		UpdatedAt:    now,
	}
	require.NoError(t, store.CreateRun(ctx, run))

	err := engine.Recover(ctx)
	require.NoError(t, err)

	require.Eventually(t, func() bool {
		r, getErr := store.GetRun(ctx, "run_unknown_fe")
		return getErr == nil && r.Status == RunStatusFailed
	}, 3*time.Second, 20*time.Millisecond)

	r, err := store.GetRun(ctx, "run_unknown_fe")
	require.NoError(t, err)
	assert.Equal(t, RunStatusFailed, r.Status)
	assert.Contains(t, r.ErrorText, `unregistered frontend "alien_frontend" on recovery`)

	// Must NOT deliver to generic result sink (privacy isolation)
	sink.mu.Lock()
	assert.Empty(t, sink.delivered)
	sink.mu.Unlock()
}

func TestEngine_Recovery_BindFailure_FailsRunAndDelivers(t *testing.T) {
	db := setupTestDB(t)
	store := NewStore(db)
	prov := NewStaticStoreProvider(store)
	ctx := context.Background()

	fe := &mockFrontend{
		bindErr: errors.New("chat session deleted or revoked"),
	}

	engine := NewEngine(prov, nil, nil,
		WithFrontend("failing_fe", fe),
	)

	now := time.Now().Unix()
	run := &FSMRun{
		ID:           "run_bind_fail",
		ChatID:       "chat_fail",
		UserID:       "user_dave",
		FrontendID:   "failing_fe",
		FSMType:      FSMTypeToolLoop,
		Status:       RunStatusRunning,
		CurrentState: StateExecuteSteps,
		ContextJSON:  `[]`,
		CreatedAt:    now,
		UpdatedAt:    now,
	}
	require.NoError(t, store.CreateRun(ctx, run))

	err := engine.Recover(ctx)
	require.NoError(t, err)

	require.Eventually(t, func() bool {
		fe.mu.Lock()
		defer fe.mu.Unlock()
		return len(fe.deliverCalls) > 0
	}, 3*time.Second, 20*time.Millisecond)

	fe.mu.Lock()
	require.Len(t, fe.deliverCalls, 1)
	assert.Equal(t, agentapi.RunFailed, fe.deliverCalls[0].Status)
	assert.Contains(t, fe.deliverCalls[0].Error, "chat session deleted or revoked")
	fe.mu.Unlock()

	r, err := store.GetRun(ctx, "run_bind_fail")
	require.NoError(t, err)
	assert.Equal(t, RunStatusFailed, r.Status)
}

func TestEngine_Recovery_ScheduledInterruptedByRestart_FailsAndDelivers(t *testing.T) {
	db := setupTestDB(t)
	store := NewStore(db)
	prov := NewStaticStoreProvider(store)
	ctx := context.Background()

	fe := &mockFrontend{}
	engine := NewEngine(prov, nil, nil,
		WithFrontend("sched_fe", fe),
	)

	now := time.Now().Unix()
	run := &FSMRun{
		ID:            "sched_run_123",
		ChatID:        "chat_sched",
		UserID:        "user_admin",
		FrontendID:    "sched_fe",
		ExecutionKind: agentapi.Scheduled,
		FSMType:       FSMTypeToolLoop,
		Status:        RunStatusRunning,
		CurrentState:  StateExecuteSteps,
		ContextJSON:   `[]`,
		CreatedAt:     now,
		UpdatedAt:     now,
	}
	require.NoError(t, store.CreateRun(ctx, run))

	err := engine.Recover(ctx)
	require.NoError(t, err)

	require.Eventually(t, func() bool {
		fe.mu.Lock()
		defer fe.mu.Unlock()
		return len(fe.deliverCalls) > 0
	}, 3*time.Second, 20*time.Millisecond)

	fe.mu.Lock()
	require.Len(t, fe.deliverCalls, 1)
	assert.Equal(t, agentapi.RunFailed, fe.deliverCalls[0].Status)
	assert.Contains(t, fe.deliverCalls[0].Error, "scheduled run interrupted by service restart")
	fe.mu.Unlock()
}
