package scheduled_test

import (
	"context"
	"database/sql"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"bob/internal/agentapi"
	"bob/internal/agent/scheduled"
	"bob/internal/fsm"
	"bob/internal/sandbox"
	"bob/internal/scheduler"
	"bob/internal/tools"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	_ "modernc.org/sqlite"
)

func createTestGrant(schedID, envJSON, userID string, validUntil int64) *scheduler.ScheduleGrant {
	return &scheduler.ScheduleGrant{
		ID:                    "grant_" + schedID,
		ScheduleID:            schedID,
		PermissionRequestJSON: envJSON,
		ParamsHash:            scheduler.ComputeRawJSONHash(envJSON),
		GrantedBy:             userID,
		GrantedAt:             time.Now().Unix(),
		ValidUntil:            validUntil,
	}
}

func setupTestScheduleStore(t *testing.T) (*scheduler.Store, *sql.DB) {
	t.Helper()
	dbPath := filepath.Join(t.TempDir(), "scheduled_test.db")
	db, err := sql.Open("sqlite", "file:"+dbPath+"?_pragma=foreign_keys(ON)&_pragma=journal_mode(WAL)&_pragma=busy_timeout(5000)&_pragma=synchronous(NORMAL)")
	require.NoError(t, err)

	err = scheduler.EnsureScheduleSchema(context.Background(), db)
	require.NoError(t, err)

	t.Cleanup(func() { _ = db.Close() })
	return scheduler.NewStore(db), db
}

type testSandboxDriver struct {
	driverType   sandbox.DriverType
	createCount  int
	destroyCount int
}

func (d *testSandboxDriver) Type() sandbox.DriverType {
	return d.driverType
}

func (d *testSandboxDriver) Available(_ context.Context) bool {
	return true
}

func (d *testSandboxDriver) Create(_ context.Context, _ *sandbox.UserSandbox, _ string) error {
	d.createCount++
	return nil
}

func (d *testSandboxDriver) Exec(_ context.Context, _ *sandbox.UserSandbox, _ []string, _ time.Duration) (*sandbox.ExecResult, error) {
	return &sandbox.ExecResult{ExitCode: 0, Stdout: "ok"}, nil
}

func (d *testSandboxDriver) Destroy(_ context.Context, _ *sandbox.UserSandbox) error {
	d.destroyCount++
	return nil
}

func setupTestSandbox(t *testing.T) (*sandbox.Manager, *testSandboxDriver) {
	t.Helper()
	cfg := sandbox.Config{
		DataDir:            t.TempDir(),
		Enabled:            true,
		Drivers:            []string{"bwrap"},
		MaxLifetime:        30 * time.Minute,
		DefaultExecTimeout: 1 * time.Minute,
		MaxExecTimeout:     10 * time.Minute,
	}
	driver := &testSandboxDriver{driverType: sandbox.DriverBwrap}
	mgr := sandbox.NewManager(cfg, []sandbox.Driver{driver})
	t.Cleanup(func() { _ = mgr.Close() })
	return mgr, driver
}

func TestInvoker_NilSchedule(t *testing.T) {
	inv := scheduled.NewInvoker(scheduled.Config{})
	err := inv.ExecuteSchedule(context.Background(), nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "schedule cannot be nil")
}

func TestInvoker_Execute_NoGrant(t *testing.T) {
	ctx := context.Background()

	var sentMsg string
	sender := scheduled.MessageSenderFunc(func(chatID, content string) error {
		sentMsg = content
		return nil
	})

	var fsmCalled bool
	var capturedReq fsm.ToolLoopRequest
	mockFSM := scheduled.FSMRunnerFunc(func(ctx context.Context, req fsm.ToolLoopRequest) (*fsm.ToolLoopResult, error) {
		fsmCalled = true
		capturedReq = req
		return &fsm.ToolLoopResult{
			Content: "Task executed successfully.",
		}, nil
	})

	toolsRegistry := tools.NewRegistry(nil, nil, nil)

	inv := scheduled.NewInvoker(scheduled.Config{
		FSM:     mockFSM,
		Tools:   toolsRegistry,
		Sender:  sender,
		Model:   "test-model",
		BotID:   "bot_1",
		BotName: "Bob",
	})

	sched := &scheduler.Schedule{
		ID:          "sched_no_grant",
		ChatID:      "dm_user1",
		UserID:      "user1",
		Name:        "hourly_check",
		Instruction: "Check server status.",
		MaxTurns:    10,
	}

	err := inv.ExecuteSchedule(ctx, sched)
	require.NoError(t, err)
	assert.True(t, fsmCalled)
	assert.Equal(t, "dm_user1", capturedReq.ChatID)
	assert.Equal(t, "user1", capturedReq.UserID)
	assert.Contains(t, sentMsg, "Scheduled Task [hourly_check] completed:")

	// Toolset must exclude sandbox tools and schedule tools
	require.NotNil(t, capturedReq.Toolset)
	defs, err := capturedReq.Toolset.Definitions(ctx)
	require.NoError(t, err)
	for _, d := range defs {
		name := d.Schema.Function.Name
		assert.False(t, strings.HasPrefix(name, "sandbox_"), "tool %s should not be allowed without sandbox grant", name)
		assert.False(t, name == "schedule_task" || name == "cancel_schedule" || name == "list_schedules", "scheduler tool %s should not be exposed", name)
	}

	// Dispatch to forbidden tool must fail closed
	res, err := capturedReq.Toolset.Execute(ctx, agentapi.ToolCall{
		ID:   "call_1",
		Name: "schedule_task",
	})
	require.NoError(t, err)
	assert.Contains(t, res.Content, "not authorized in scheduled execution")
}

func TestInvoker_Execute_WithPreApprovedSandbox(t *testing.T) {
	ctx := context.Background()
	store, _ := setupTestScheduleStore(t)
	mgr, driver := setupTestSandbox(t)

	now := time.Now().Unix()
	sched := &scheduler.Schedule{
		ID:                 "sched_sandbox",
		Name:               "sandbox_runner",
		ChatID:             "dm_user1",
		UserID:             "user1",
		ScheduleType:       scheduler.ScheduleTypeInterval,
		IntervalSeconds:    3600,
		Instruction:        "Run tests in sandbox",
		Status:             scheduler.ScheduleStatusActive,
		RunTimeoutSeconds:  60,
		MaxTurns:           5,
		NextRunAt:          now + 3600,
		CreatedAt:          now,
		UpdatedAt:          now,
	}

	envJSON := `{"sandbox":{"driver":"bwrap","network":"none"}}`
	grant := createTestGrant("sched_sandbox", envJSON, "user1", now+86400)
	err := store.CreateSchedule(ctx, sched, grant)
	require.NoError(t, err)

	var fsmCalled bool
	var capturedReq fsm.ToolLoopRequest
	mockFSM := scheduled.FSMRunnerFunc(func(ctx context.Context, req fsm.ToolLoopRequest) (*fsm.ToolLoopResult, error) {
		fsmCalled = true
		capturedReq = req
		return &fsm.ToolLoopResult{
			Content: "Sandbox task completed.",
		}, nil
	})

	toolsRegistry := tools.NewRegistry(nil, nil, nil)
	inv := scheduled.NewInvoker(scheduled.Config{
		FSM:     mockFSM,
		Tools:   toolsRegistry,
		Sandbox: mgr,
		Model:   "test-model",
		SchedulerStoreProv: func(_ context.Context, _ string, _ bool) (*scheduler.Store, error) {
			return store, nil
		},
	})

	err = inv.ExecuteSchedule(ctx, sched)
	require.NoError(t, err)
	assert.True(t, fsmCalled)

	// Verify sandbox created and then automatically destroyed on return
	assert.Equal(t, 1, driver.createCount)
	assert.Equal(t, 1, driver.destroyCount)

	// Verify sandbox_request and sandbox_destroy are excluded, but sandbox execution tools are permitted
	require.NotNil(t, capturedReq.Toolset)
	defs, err := capturedReq.Toolset.Definitions(ctx)
	require.NoError(t, err)
	for _, d := range defs {
		name := d.Schema.Function.Name
		assert.False(t, name == "sandbox_request" || name == "sandbox_destroy", "ephemeral sandbox lifecycle tools should not be exposed")
	}
}

type mockTurnRunner struct {
	called bool
	turn   agentapi.Turn
}

func (m *mockTurnRunner) Run(_ context.Context, turn agentapi.Turn) (agentapi.Result, error) {
	m.called = true
	m.turn = turn
	return agentapi.Result{
		Content: "Runner completed scheduled turn.",
	}, nil
}

func TestInvoker_Execute_WithTurnRunner(t *testing.T) {
	ctx := context.Background()

	var sentMsg string
	sender := scheduled.MessageSenderFunc(func(chatID, content string) error {
		sentMsg = content
		return nil
	})

	runner := &mockTurnRunner{}
	inv := scheduled.NewInvoker(scheduled.Config{
		Runner: runner,
		Sender: sender,
		Model:  "gpt-model",
	})

	sched := &scheduler.Schedule{
		ID:          "sched_runner",
		ChatID:      "dm_user1",
		UserID:      "user1",
		Name:        "morning_brief",
		Instruction: "Brief the user on updates.",
		MaxTurns:    8,
	}

	err := inv.ExecuteSchedule(ctx, sched)
	require.NoError(t, err)
	assert.True(t, runner.called)
	assert.Equal(t, agentapi.Scheduled, runner.turn.Run.Kind)
	assert.Equal(t, "gpt-model", runner.turn.Run.Model)
	assert.Equal(t, 8, runner.turn.Run.MaxIterations)
	assert.Equal(t, "user1", runner.turn.Run.Actor.ID)
	assert.Contains(t, sentMsg, "Scheduled Task [morning_brief] completed:")
	assert.Contains(t, sentMsg, "Runner completed scheduled turn.")
}

func TestInvoker_SecurityChecks(t *testing.T) {
	ctx := context.Background()
	store, _ := setupTestScheduleStore(t)

	now := time.Now().Unix()

	// 1. Expired grant
	schedExpired := &scheduler.Schedule{
		ID:                 "sched_exp",
		Name:               "expired_task",
		ChatID:             "dm_user1",
		UserID:             "user1",
		ScheduleType:       scheduler.ScheduleTypeOnce,
		Instruction:        "Run once",
		Status:             scheduler.ScheduleStatusActive,
		CreatedAt:          now,
		UpdatedAt:          now,
	}
	grantExpired := createTestGrant("sched_exp", `{"sandbox":{"driver":"bwrap"}}`, "user1", now-100)
	err := store.CreateSchedule(ctx, schedExpired, grantExpired)
	require.NoError(t, err)

	inv := scheduled.NewInvoker(scheduled.Config{
		SchedulerStoreProv: func(_ context.Context, _ string, _ bool) (*scheduler.Store, error) {
			return store, nil
		},
	})

	err = inv.ExecuteSchedule(ctx, schedExpired)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "schedule grant has expired")

	// 2. Townhall sandbox forbidden
	schedTownhall := &scheduler.Schedule{
		ID:                 "sched_th",
		Name:               "townhall_sandbox",
		ChatID:             "townhall",
		UserID:             "user1",
		ScheduleType:       scheduler.ScheduleTypeOnce,
		Instruction:        "Run once",
		Status:             scheduler.ScheduleStatusActive,
		CreatedAt:          now,
		UpdatedAt:          now,
	}
	mgr, _ := setupTestSandbox(t)
	grantTH := createTestGrant("sched_th", `{"sandbox":{"driver":"bwrap"}}`, "user1", now+3600)
	err = store.CreateSchedule(ctx, schedTownhall, grantTH)
	require.NoError(t, err)

	invTH := scheduled.NewInvoker(scheduled.Config{
		Sandbox: mgr,
		SchedulerStoreProv: func(_ context.Context, _ string, _ bool) (*scheduler.Store, error) {
			return store, nil
		},
	})

	err = invTH.ExecuteSchedule(ctx, schedTownhall)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "sandbox execution is not permitted in Townhall")
}
