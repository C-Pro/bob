package fsm

import (
	"context"
	"database/sql"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRetentionManager_PruneTerminalRunsAndActiveImmunity(t *testing.T) {
	db := setupTestDB(t)
	store := NewStore(db)
	retention := NewRetentionManager(db, 7)
	ctx := context.Background()

	now := time.Now().Unix()
	tenDaysAgo := now - int64(10*24*time.Hour/time.Second)
	twoDaysAgo := now - int64(2*24*time.Hour/time.Second)

	// 1. Old completed run (should be pruned)
	oldCompleted := &FSMRun{
		ID:          "run_old_completed",
		ChatID:      "chat_1",
		UserID:      "u1",
		Status:      RunStatusCompleted,
		ContextJSON: "{}",
		CreatedAt:   tenDaysAgo,
		UpdatedAt:   tenDaysAgo,
	}
	require.NoError(t, store.CreateRun(ctx, oldCompleted))
	require.NoError(t, store.CreateSteps(ctx, []FSMStep{
		{ID: "step_old_1", RunID: oldCompleted.ID, Iteration: 1, StepIndex: 0, ToolName: "search", ArgsJSON: "{}"},
	}))

	// 2. Old failed run (should be pruned)
	oldFailed := &FSMRun{
		ID:          "run_old_failed",
		ChatID:      "chat_1",
		UserID:      "u1",
		Status:      RunStatusFailed,
		ContextJSON: "{}",
		CreatedAt:   tenDaysAgo,
		UpdatedAt:   tenDaysAgo,
	}
	require.NoError(t, store.CreateRun(ctx, oldFailed))

	// 3. Old terminated run (should be pruned)
	oldTerminated := &FSMRun{
		ID:          "run_old_terminated",
		ChatID:      "chat_1",
		UserID:      "u1",
		Status:      RunStatusTerminated,
		ContextJSON: "{}",
		CreatedAt:   tenDaysAgo,
		UpdatedAt:   tenDaysAgo,
	}
	require.NoError(t, store.CreateRun(ctx, oldTerminated))

	// 4. Old RUNNING run (IMMUNE: should NOT be pruned)
	oldRunning := &FSMRun{
		ID:          "run_old_running",
		ChatID:      "chat_1",
		UserID:      "u1",
		Status:      RunStatusRunning,
		ContextJSON: "{}",
		CreatedAt:   tenDaysAgo,
		UpdatedAt:   tenDaysAgo,
	}
	require.NoError(t, store.CreateRun(ctx, oldRunning))

	// 5. Old WAITING run (IMMUNE: should NOT be pruned)
	oldWaiting := &FSMRun{
		ID:          "run_old_waiting",
		ChatID:      "chat_1",
		UserID:      "u1",
		Status:      RunStatusWaiting,
		ContextJSON: "{}",
		CreatedAt:   tenDaysAgo,
		UpdatedAt:   tenDaysAgo,
	}
	require.NoError(t, store.CreateRun(ctx, oldWaiting))

	// 6. Recent completed run (within 7 days retention: should NOT be pruned)
	recentCompleted := &FSMRun{
		ID:          "run_recent_completed",
		ChatID:      "chat_1",
		UserID:      "u1",
		Status:      RunStatusCompleted,
		ContextJSON: "{}",
		CreatedAt:   twoDaysAgo,
		UpdatedAt:   twoDaysAgo,
	}
	require.NoError(t, store.CreateRun(ctx, recentCompleted))

	// Execute PruneAndCompact with 7-day retention
	pruned, err := retention.PruneAndCompact(ctx)
	require.NoError(t, err)
	assert.Equal(t, int64(3), pruned) // oldCompleted, oldFailed, oldTerminated

	// Verify old terminal runs are gone
	_, err = store.GetRun(ctx, "run_old_completed")
	assert.ErrorIs(t, err, ErrRunNotFound)
	_, err = store.GetRun(ctx, "run_old_failed")
	assert.ErrorIs(t, err, ErrRunNotFound)
	_, err = store.GetRun(ctx, "run_old_terminated")
	assert.ErrorIs(t, err, ErrRunNotFound)

	// Verify steps of old completed run were cascaded deleted
	_, err = store.GetStep(ctx, "step_old_1")
	assert.ErrorIs(t, err, ErrStepNotFound)

	// Verify active/waiting runs were NOT pruned despite being old
	rRunning, err := store.GetRun(ctx, "run_old_running")
	require.NoError(t, err)
	assert.Equal(t, RunStatusRunning, rRunning.Status)

	rWaiting, err := store.GetRun(ctx, "run_old_waiting")
	require.NoError(t, err)
	assert.Equal(t, RunStatusWaiting, rWaiting.Status)

	// Verify recent completed run was NOT pruned
	rRecent, err := store.GetRun(ctx, "run_recent_completed")
	require.NoError(t, err)
	assert.Equal(t, RunStatusCompleted, rRecent.Status)
}

func TestRetentionManager_IncrementalVacuum(t *testing.T) {
	// Database initialized via EnsureDBSchema has auto_vacuum = INCREMENTAL (2)
	db := setupTestDB(t)
	retention := NewRetentionManager(db, 7)
	ctx := context.Background()

	var autoVac int
	require.NoError(t, db.QueryRowContext(ctx, "PRAGMA auto_vacuum;").Scan(&autoVac))
	assert.Equal(t, 2, autoVac)

	err := retention.IncrementalVacuum(ctx, 100)
	require.NoError(t, err)

	// Default pages
	err = retention.IncrementalVacuum(ctx, 0)
	require.NoError(t, err)

	// Database with auto_vacuum = NONE (0) should log a warning and safely skip without error
	dbPathNoAV := filepath.Join(t.TempDir(), "no_av.db")
	rawDB, err := setupRawDBWithAutoVacuum0(t, dbPathNoAV)
	require.NoError(t, err)
	defer func() { _ = rawDB.Close() }()

	retentionNoAV := NewRetentionManager(rawDB, 7)
	err = retentionNoAV.IncrementalVacuum(ctx, 500)
	require.NoError(t, err)
}

func setupRawDBWithAutoVacuum0(t *testing.T, path string) (*sql.DB, error) {
	t.Helper()
	db, err := sql.Open("sqlite", "file:"+path)
	if err != nil {
		return nil, err
	}
	// Creating table first forces auto_vacuum to remain 0 (NONE)
	if _, err := db.Exec("CREATE TABLE t(a INT);"); err != nil {
		_ = db.Close()
		return nil, err
	}
	return db, nil
}

func TestRetentionManager_PruneWithoutFSMRunsTable(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "no_fsm_runs.db")
	db, err := sql.Open("sqlite", "file:"+dbPath)
	require.NoError(t, err)
	defer func() { _ = db.Close() }()

	// Database with other tables but no fsm_runs table
	_, err = db.Exec("CREATE TABLE other_table (id INT);")
	require.NoError(t, err)

	retention := NewRetentionManager(db, 7)
	pruned, err := retention.PruneTerminalRuns(context.Background(), time.Now())
	require.NoError(t, err)
	assert.Equal(t, int64(0), pruned)
}
