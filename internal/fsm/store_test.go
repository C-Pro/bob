package fsm

import (
	"context"
	"database/sql"
	"fmt"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	_ "modernc.org/sqlite"
)

func setupTestDB(t *testing.T) *sql.DB {
	t.Helper()
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "test.db")
	dsn := fmt.Sprintf("file:%s?_auto_vacuum=INCREMENTAL&_pragma=journal_mode(WAL)&_pragma=foreign_keys(ON)&_pragma=busy_timeout(5000)", dbPath)

	db, err := sql.Open("sqlite", dsn)
	require.NoError(t, err)
	db.SetMaxOpenConns(1)
	t.Cleanup(func() { _ = db.Close() })

	ctx := context.Background()
	err = EnsureDBSchema(ctx, db)
	require.NoError(t, err)

	return db
}

func TestStore_RunCRUD(t *testing.T) {
	db := setupTestDB(t)
	store := NewStore(db)
	ctx := context.Background()

	now := time.Now().Unix()
	futureResume := now + 60

	run := &FSMRun{
		ID:            "run_test_1",
		ChatID:        "chat_123",
		UserID:        "user_456",
		FSMType:       FSMTypeToolLoop,
		Status:        RunStatusRunning,
		CurrentState:  StateLLMRequest,
		Iteration:     1,
		MaxIterations: 20,
		ContextJSON:   `[{"role":"user","content":"hello"}]`,
		ResultJSON:    "",
		ErrorText:     "",
		ResumeAt:      &futureResume,
		CreatedAt:     now,
		UpdatedAt:     now,
	}

	// Create
	err := store.CreateRun(ctx, run)
	require.NoError(t, err)

	// Get
	fetched, err := store.GetRun(ctx, "run_test_1")
	require.NoError(t, err)
	assert.Equal(t, run.ID, fetched.ID)
	assert.Equal(t, run.ChatID, fetched.ChatID)
	assert.Equal(t, run.UserID, fetched.UserID)
	assert.Equal(t, run.FSMType, fetched.FSMType)
	assert.Equal(t, run.Status, fetched.Status)
	assert.Equal(t, run.CurrentState, fetched.CurrentState)
	assert.Equal(t, run.Iteration, fetched.Iteration)
	assert.Equal(t, run.MaxIterations, fetched.MaxIterations)
	assert.Equal(t, run.ContextJSON, fetched.ContextJSON)
	require.NotNil(t, fetched.ResumeAt)
	assert.Equal(t, futureResume, *fetched.ResumeAt)

	// Update
	fetched.Status = RunStatusCompleted
	fetched.CurrentState = StateCompleted
	fetched.ResultJSON = `{"response":"Hello there!"}`
	fetched.ResumeAt = nil
	err = store.UpdateRunState(ctx, fetched)
	require.NoError(t, err)

	updated, err := store.GetRun(ctx, "run_test_1")
	require.NoError(t, err)
	assert.Equal(t, RunStatusCompleted, updated.Status)
	assert.Equal(t, StateCompleted, updated.CurrentState)
	assert.Equal(t, `{"response":"Hello there!"}`, updated.ResultJSON)
	assert.Nil(t, updated.ResumeAt)

	// Not found
	_, err = store.GetRun(ctx, "non_existent")
	assert.ErrorIs(t, err, ErrRunNotFound)

	err = store.UpdateRunState(ctx, &FSMRun{ID: "non_existent"})
	assert.ErrorIs(t, err, ErrRunNotFound)
}

func TestStore_StepCRUDAndIterationQueries(t *testing.T) {
	db := setupTestDB(t)
	store := NewStore(db)
	ctx := context.Background()

	run := &FSMRun{
		ID:            "run_step_test",
		ChatID:        "chat_dm",
		UserID:        "user_dm",
		FSMType:       FSMTypeToolLoop,
		Status:        RunStatusRunning,
		CurrentState:  StateExecuteSteps,
		MaxIterations: 20,
		ContextJSON:   "[]",
	}
	require.NoError(t, store.CreateRun(ctx, run))

	steps := []FSMStep{
		{
			ID:             "step_1",
			RunID:          run.ID,
			Iteration:      1,
			StepIndex:      0,
			ToolName:       "web_search",
			ToolCallID:     "call_1",
			ArgsJSON:       `{"query":"golang"}`,
			ExecutionMode:  ExecutionModeParallel,
			Status:         StepStatusPending,
			TimeoutSeconds: 15,
		},
		{
			ID:             "step_2",
			RunID:          run.ID,
			Iteration:      1,
			StepIndex:      1,
			ToolName:       "web_fetch",
			ToolCallID:     "call_2",
			ArgsJSON:       `{"url":"https://go.dev"}`,
			ExecutionMode:  ExecutionModeParallel,
			Status:         StepStatusPending,
			TimeoutSeconds: 20,
		},
		{
			ID:             "step_3",
			RunID:          run.ID,
			Iteration:      2,
			StepIndex:      0,
			ToolName:       "sandbox_exec",
			ToolCallID:     "call_3",
			ArgsJSON:       `{"cmd":"ls"}`,
			ExecutionMode:  ExecutionModeSequential,
			Status:         StepStatusPending,
			TimeoutSeconds: 30,
		},
	}

	require.NoError(t, store.CreateSteps(ctx, steps))

	// Get individual step
	s1, err := store.GetStep(ctx, "step_1")
	require.NoError(t, err)
	assert.Equal(t, "web_search", s1.ToolName)
	assert.Equal(t, ExecutionModeParallel, s1.ExecutionMode)
	assert.Equal(t, StepStatusPending, s1.Status)
	assert.Equal(t, 15, s1.TimeoutSeconds)

	// Get steps for iteration 1
	iter1Steps, err := store.GetStepsForIteration(ctx, run.ID, 1)
	require.NoError(t, err)
	require.Len(t, iter1Steps, 2)
	assert.Equal(t, "step_1", iter1Steps[0].ID)
	assert.Equal(t, "step_2", iter1Steps[1].ID)

	// Update step 1
	now := time.Now().Unix()
	s1.Status = StepStatusCompleted
	s1.ResultJSON = `{"data":"Go is an open source language"}`
	s1.StartedAt = &now
	s1.CompletedAt = &now
	require.NoError(t, store.UpdateStep(ctx, s1))

	// Pending steps for iteration 1 should now be only step_2
	pending, err := store.GetPendingSteps(ctx, run.ID, 1)
	require.NoError(t, err)
	require.Len(t, pending, 1)
	assert.Equal(t, "step_2", pending[0].ID)

	// Not found
	_, err = store.GetStep(ctx, "unknown_step")
	assert.ErrorIs(t, err, ErrStepNotFound)

	err = store.UpdateStep(ctx, &FSMStep{ID: "unknown_step"})
	assert.ErrorIs(t, err, ErrStepNotFound)
}

func TestStore_ListActiveAndDueWaitingRuns(t *testing.T) {
	db := setupTestDB(t)
	store := NewStore(db)
	ctx := context.Background()

	now := time.Now().Unix()
	pastResume := now - 10
	futureResume := now + 100

	r1 := &FSMRun{ID: "run_active_1", ChatID: "c1", UserID: "u1", Status: RunStatusRunning, ContextJSON: "{}"}
	r2 := &FSMRun{ID: "run_waiting_due", ChatID: "c1", UserID: "u1", Status: RunStatusWaiting, ResumeAt: &pastResume, ContextJSON: "{}"}
	r3 := &FSMRun{ID: "run_waiting_future", ChatID: "c1", UserID: "u1", Status: RunStatusWaiting, ResumeAt: &futureResume, ContextJSON: "{}"}
	r4 := &FSMRun{ID: "run_completed", ChatID: "c1", UserID: "u1", Status: RunStatusCompleted, ContextJSON: "{}"}
	r5 := &FSMRun{ID: "run_failed", ChatID: "c1", UserID: "u1", Status: RunStatusFailed, ContextJSON: "{}"}

	for _, r := range []*FSMRun{r1, r2, r3, r4, r5} {
		require.NoError(t, store.CreateRun(ctx, r))
	}

	active, err := store.ListActiveRuns(ctx)
	require.NoError(t, err)
	require.Len(t, active, 3) // r1, r2, r3

	dueWaiting, err := store.ListDueWaitingRuns(ctx, now)
	require.NoError(t, err)
	require.Len(t, dueWaiting, 1)
	assert.Equal(t, "run_waiting_due", dueWaiting[0].ID)
}

func TestStore_CascadeDeleteStepsOnRunDeletion(t *testing.T) {
	db := setupTestDB(t)
	store := NewStore(db)
	ctx := context.Background()

	run := &FSMRun{ID: "run_cascade", ChatID: "c1", UserID: "u1", Status: RunStatusRunning, ContextJSON: "{}"}
	require.NoError(t, store.CreateRun(ctx, run))

	steps := []FSMStep{
		{ID: "step_c1", RunID: run.ID, Iteration: 1, StepIndex: 0, ToolName: "tool1", ArgsJSON: "{}"},
		{ID: "step_c2", RunID: run.ID, Iteration: 1, StepIndex: 1, ToolName: "tool2", ArgsJSON: "{}"},
	}
	require.NoError(t, store.CreateSteps(ctx, steps))

	// Verify steps exist
	_, err := store.GetStep(ctx, "step_c1")
	require.NoError(t, err)

	// Delete parent run
	_, err = db.ExecContext(ctx, "DELETE FROM fsm_runs WHERE id = ?", run.ID)
	require.NoError(t, err)

	// Steps must be deleted automatically by ON DELETE CASCADE
	_, err = store.GetStep(ctx, "step_c1")
	assert.ErrorIs(t, err, ErrStepNotFound)
	_, err = store.GetStep(ctx, "step_c2")
	assert.ErrorIs(t, err, ErrStepNotFound)
}

func TestStore_ChatIsolation(t *testing.T) {
	dir := t.TempDir()

	dsn1 := fmt.Sprintf("file:%s?_auto_vacuum=INCREMENTAL&_pragma=journal_mode(WAL)&_pragma=foreign_keys(ON)&_pragma=busy_timeout(5000)", filepath.Join(dir, "townhall.db"))
	db1, err := sql.Open("sqlite", dsn1)
	require.NoError(t, err)
	defer func() { _ = db1.Close() }()
	require.NoError(t, EnsureDBSchema(context.Background(), db1))

	dsn2 := fmt.Sprintf("file:%s?_auto_vacuum=INCREMENTAL&_pragma=journal_mode(WAL)&_pragma=foreign_keys(ON)&_pragma=busy_timeout(5000)", filepath.Join(dir, "dm_user1.db"))
	db2, err := sql.Open("sqlite", dsn2)
	require.NoError(t, err)
	defer func() { _ = db2.Close() }()
	require.NoError(t, EnsureDBSchema(context.Background(), db2))

	store1 := NewStore(db1)
	store2 := NewStore(db2)
	ctx := context.Background()

	r1 := &FSMRun{ID: "run_townhall", ChatID: "townhall", UserID: "user_a", Status: RunStatusRunning, ContextJSON: "{}"}
	r2 := &FSMRun{ID: "run_dm", ChatID: "dm_user1", UserID: "user_b", Status: RunStatusRunning, ContextJSON: "{}"}

	require.NoError(t, store1.CreateRun(ctx, r1))
	require.NoError(t, store2.CreateRun(ctx, r2))

	// store1 cannot see store2's run
	_, err = store1.GetRun(ctx, "run_dm")
	assert.ErrorIs(t, err, ErrRunNotFound)

	// store2 cannot see store1's run
	_, err = store2.GetRun(ctx, "run_townhall")
	assert.ErrorIs(t, err, ErrRunNotFound)
}

func TestStore_CreateSteps_Idempotent(t *testing.T) {
	db := setupTestDB(t)
	store := NewStore(db)
	ctx := context.Background()

	run := &FSMRun{
		ID:           "run_idempotent_steps",
		ChatID:       "chat_1",
		UserID:       "user_1",
		FSMType:      FSMTypeToolLoop,
		Status:       RunStatusRunning,
		CurrentState: StatePrepareSteps,
	}
	require.NoError(t, store.CreateRun(ctx, run))

	steps := []FSMStep{
		{
			ID:            "step_idem_1",
			RunID:         run.ID,
			Iteration:     1,
			StepIndex:     0,
			ToolName:      "web_search",
			ToolCallID:    "call_1",
			ArgsJSON:      `{"q":"test"}`,
			ExecutionMode: ExecutionModeParallel,
			Status:        StepStatusPending,
		},
	}

	// First insert
	require.NoError(t, store.CreateSteps(ctx, steps))

	// Second insert with same ID should not error (ON CONFLICT DO NOTHING)
	require.NoError(t, store.CreateSteps(ctx, steps))

	gotSteps, err := store.ListStepsByIteration(ctx, run.ID, 1)
	require.NoError(t, err)
	require.Len(t, gotSteps, 1)
	assert.Equal(t, "step_idem_1", gotSteps[0].ID)
}

func TestStore_OptimisticConcurrencyConflict(t *testing.T) {
	db := setupTestDB(t)
	store := NewStore(db)
	ctx := context.Background()

	run := &FSMRun{
		ID:           "run_occ_test",
		ChatID:       "chat_1",
		UserID:       "user_1",
		FSMType:      FSMTypeToolLoop,
		Status:       RunStatusRunning,
		CurrentState: StateExecuteSteps,
	}
	require.NoError(t, store.CreateRun(ctx, run))

	// Fetch two independent copies of the run
	copy1, err := store.GetRun(ctx, run.ID)
	require.NoError(t, err)
	assert.Equal(t, 0, copy1.Version)

	copy2, err := store.GetRun(ctx, run.ID)
	require.NoError(t, err)
	assert.Equal(t, 0, copy2.Version)

	// Update first copy - should succeed and increment version to 1
	copy1.Status = RunStatusWaiting
	require.NoError(t, store.UpdateRunState(ctx, copy1))
	assert.Equal(t, 1, copy1.Version)

	// Update second copy - should fail with ErrConcurrentUpdate because version 0 is stale
	copy2.Status = RunStatusCompleted
	err = store.UpdateRunState(ctx, copy2)
	require.ErrorIs(t, err, ErrConcurrentUpdate)

	// Updating a non-existent run should still return ErrRunNotFound
	err = store.UpdateRunState(ctx, &FSMRun{ID: "does_not_exist", Version: 0})
	require.ErrorIs(t, err, ErrRunNotFound)

	// Re-reading copy2 and then updating should succeed
	refreshed, err := store.GetRun(ctx, run.ID)
	require.NoError(t, err)
	assert.Equal(t, 1, refreshed.Version)
	assert.Equal(t, RunStatusWaiting, refreshed.Status)

	refreshed.Status = RunStatusCompleted
	require.NoError(t, store.UpdateRunState(ctx, refreshed))
	assert.Equal(t, 2, refreshed.Version)

	finalRun, err := store.GetRun(ctx, run.ID)
	require.NoError(t, err)
	assert.Equal(t, 2, finalRun.Version)
	assert.Equal(t, RunStatusCompleted, finalRun.Status)
}

func TestStore_ContextSelectivePersistence(t *testing.T) {
	db := setupTestDB(t)
	store := NewStore(db)
	ctx := context.Background()

	initialContext := `[{"role":"user","content":"hello"}]`
	run := &FSMRun{
		ID:           "run_ctx_sync_test",
		ChatID:       "chat_1",
		UserID:       "user_1",
		FSMType:      FSMTypeToolLoop,
		Status:       RunStatusRunning,
		CurrentState: StateInit,
		ContextJSON:  initialContext,
	}
	require.NoError(t, store.CreateRun(ctx, run))
	assert.False(t, run.IsContextDirty())

	// Updating state without mutating ContextJSON should leave it clean
	run.CurrentState = StateLLMRequest
	assert.False(t, run.IsContextDirty())
	require.NoError(t, store.UpdateRunState(ctx, run))
	assert.False(t, run.IsContextDirty())

	// Mutating ContextJSON marks it dirty
	updatedContext := `[{"role":"user","content":"hello"},{"role":"assistant","content":"hi"}]`
	run.ContextJSON = updatedContext
	assert.True(t, run.IsContextDirty())

	require.NoError(t, store.UpdateRunState(ctx, run))
	assert.False(t, run.IsContextDirty())

	fetched, err := store.GetRun(ctx, run.ID)
	require.NoError(t, err)
	assert.Equal(t, updatedContext, fetched.ContextJSON)
	assert.False(t, fetched.IsContextDirty())
}


