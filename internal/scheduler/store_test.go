package scheduler

import (
	"context"
	"database/sql"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	_ "modernc.org/sqlite"
)

func setupTestStore(t *testing.T) (*Store, *sql.DB) {
	t.Helper()
	dbPath := filepath.Join(t.TempDir(), "scheduler_test.db")
	db, err := sql.Open("sqlite", "file:"+dbPath+"?_pragma=foreign_keys(ON)&_pragma=journal_mode(WAL)")
	require.NoError(t, err)
	require.NoError(t, EnsureScheduleSchema(context.Background(), db))

	st := NewStore(db)
	t.Cleanup(func() { _ = db.Close() })
	return st, db
}

func TestStore_CreateAndGetSchedule(t *testing.T) {
	ctx := context.Background()
	st, _ := setupTestStore(t)

	now := time.Now().Unix()
	sched := &Schedule{
		ID:                "sched_1",
		Name:              "daily_brief",
		ChatID:            "chat_1",
		UserID:            "user_1",
		ScheduleType:      ScheduleTypeCron,
		CronExpr:          "0 9 * * *",
		Instruction:       "Summarize morning news",
		Status:            ScheduleStatusPendingApproval,
		NextRunAt:         now + 3600,
		RunTimeoutSeconds: 300,
		MaxTurns:          10,
	}
	grant := &ScheduleGrant{
		ID:                    "grant_1",
		ScheduleID:            "sched_1",
		PermissionRequestJSON: `{"sandbox":{"driver":"docker","network":"restricted"}}`,
		ParamsHash:            ComputeRawJSONHash(`{"sandbox":{"driver":"docker","network":"restricted"}}`),
		GrantedBy:             "",
		GrantedAt:             0,
		ValidUntil:            now + 86400,
	}

	err := st.CreateSchedule(ctx, sched, grant)
	require.NoError(t, err)

	// Fetch by ID
	fetched, err := st.GetSchedule(ctx, "sched_1")
	require.NoError(t, err)
	assert.Equal(t, "daily_brief", fetched.Name)
	assert.Equal(t, ScheduleStatusPendingApproval, fetched.Status)
	assert.Equal(t, 300, fetched.RunTimeoutSeconds)
	assert.Equal(t, 10, fetched.MaxTurns)

	// Fetch by Name
	fetchedByName, err := st.GetScheduleByName(ctx, "chat_1", "daily_brief")
	require.NoError(t, err)
	assert.Equal(t, "sched_1", fetchedByName.ID)

	// Fetch Grant
	g, err := st.GetGrant(ctx, "sched_1")
	require.NoError(t, err)
	assert.Equal(t, "grant_1", g.ID)
	assert.Equal(t, grant.ParamsHash, g.ParamsHash)

	// Invalid name rejected
	invalidSched := &Schedule{
		ID:           "sched_inv",
		Name:         "INVALID NAME",
		ChatID:       "chat_1",
		UserID:       "user_1",
		ScheduleType: ScheduleTypeInterval,
	}
	err = st.CreateSchedule(ctx, invalidSched, nil)
	assert.Error(t, err)
}

func TestStore_ApprovalAndDenialFlow(t *testing.T) {
	ctx := context.Background()
	st, _ := setupTestStore(t)

	now := time.Now().Unix()
	sched := &Schedule{
		ID:                "sched_app",
		Name:              "stock_tracker",
		ChatID:            "chat_dm1",
		UserID:            "user_alice",
		ScheduleType:      ScheduleTypeInterval,
		IntervalSeconds:   600,
		Instruction:       "Track stock tickers",
		Status:            ScheduleStatusPendingApproval,
		NextRunAt:         now + 600,
		RunTimeoutSeconds: 120,
		MaxTurns:          15,
	}
	grant := &ScheduleGrant{
		ID:                    "grant_app",
		ScheduleID:            "sched_app",
		PermissionRequestJSON: `{"sandbox":{"driver":"bwrap"}}`,
		ParamsHash:            ComputeRawJSONHash(`{"sandbox":{"driver":"bwrap"}}`),
		ValidUntil:            now + 86400,
	}
	require.NoError(t, st.CreateSchedule(ctx, sched, grant))

	// Approve schedule
	approvedSched, approvedGrant, err := st.ApproveSchedule(ctx, "chat_dm1", "stock_tracker", "user_alice", now+86400*7)
	require.NoError(t, err)
	assert.Equal(t, ScheduleStatusActive, approvedSched.Status)
	assert.Equal(t, "user_alice", approvedGrant.GrantedBy)
	assert.Equal(t, now+86400*7, approvedGrant.ValidUntil)

	// Approving an already active schedule must fail
	_, _, err = st.ApproveSchedule(ctx, "chat_dm1", "stock_tracker", "user_alice", 0)
	assert.ErrorIs(t, err, ErrScheduleNotPending)

	// Test Denial on separate pending schedule
	sched2 := &Schedule{
		ID:                "sched_deny",
		Name:              "rogue_job",
		ChatID:            "chat_dm1",
		UserID:            "user_alice",
		ScheduleType:      ScheduleTypeInterval,
		IntervalSeconds:   300,
		Instruction:       "Perform suspicious exec",
		Status:            ScheduleStatusPendingApproval,
		NextRunAt:         now + 300,
		RunTimeoutSeconds: 60,
		MaxTurns:          10,
	}
	grant2 := &ScheduleGrant{
		ID:                    "grant_deny",
		ScheduleID:            "sched_deny",
		PermissionRequestJSON: `{"sandbox":{"driver":"docker"}}`,
		ParamsHash:            ComputeRawJSONHash(`{"sandbox":{"driver":"docker"}}`),
	}
	require.NoError(t, st.CreateSchedule(ctx, sched2, grant2))

	// Tampered grant check
	schedTampered := &Schedule{
		ID:                "sched_tamp",
		Name:              "tamp_job",
		ChatID:            "chat_dm1",
		UserID:            "user_alice",
		ScheduleType:      ScheduleTypeOnce,
		Instruction:       "Tampered",
		Status:            ScheduleStatusPendingApproval,
		NextRunAt:         now + 100,
		RunTimeoutSeconds: 60,
		MaxTurns:          10,
	}
	grantTampered := &ScheduleGrant{
		ID:                    "grant_tamp",
		ScheduleID:            "sched_tamp",
		PermissionRequestJSON: `{"sandbox":{"driver":"docker"}}`,
		ParamsHash:            "invalid_hash_value",
		ValidUntil:            now + 3600,
	}
	require.NoError(t, st.CreateSchedule(ctx, schedTampered, grantTampered))
	_, _, err = st.ApproveSchedule(ctx, "chat_dm1", "tamp_job", "user_alice", 0)
	assert.Error(t, err, "tampered grant should fail approval")

	// Past valid_until check
	schedExpired := &Schedule{
		ID:                "sched_exp",
		Name:              "exp_job",
		ChatID:            "chat_dm1",
		UserID:            "user_alice",
		ScheduleType:      ScheduleTypeOnce,
		Instruction:       "Expired valid_until",
		Status:            ScheduleStatusPendingApproval,
		NextRunAt:         now + 100,
		RunTimeoutSeconds: 60,
		MaxTurns:          10,
	}
	grantExpired := &ScheduleGrant{
		ID:                    "grant_exp",
		ScheduleID:            "sched_exp",
		PermissionRequestJSON: `{"sandbox":{"driver":"docker"}}`,
		ParamsHash:            ComputeRawJSONHash(`{"sandbox":{"driver":"docker"}}`),
		ValidUntil:            now + 3600,
	}
	require.NoError(t, st.CreateSchedule(ctx, schedExpired, grantExpired))
	_, _, err = st.ApproveSchedule(ctx, "chat_dm1", "exp_job", "user_alice", now-10)
	assert.Error(t, err, "past valid_until should fail approval")

	deniedSched, err := st.DenySchedule(ctx, "chat_dm1", "rogue_job")
	require.NoError(t, err)
	assert.Equal(t, ScheduleStatusDenied, deniedSched.Status)

	// Grant must be deleted
	_, err = st.GetGrant(ctx, "sched_deny")
	assert.ErrorIs(t, err, ErrGrantNotFound)

	// Denying already denied schedule must fail
	_, err = st.DenySchedule(ctx, "chat_dm1", "rogue_job")
	assert.ErrorIs(t, err, ErrScheduleNotPending)
}

func TestStore_CancellationAndPauseResume(t *testing.T) {
	ctx := context.Background()
	st, _ := setupTestStore(t)

	now := time.Now().Unix()
	sched := &Schedule{
		ID:                "sched_pause",
		Name:              "backup_sync",
		ChatID:            "chat_1",
		UserID:            "u1",
		ScheduleType:      ScheduleTypeCron,
		CronExpr:          "@daily",
		Instruction:       "Sync backups",
		Status:            ScheduleStatusActive,
		NextRunAt:         now + 3600,
		RunTimeoutSeconds: 300,
		MaxTurns:          10,
	}
	require.NoError(t, st.CreateSchedule(ctx, sched, nil))

	// Pause
	paused, err := st.PauseSchedule(ctx, "chat_1", "backup_sync")
	require.NoError(t, err)
	assert.Equal(t, ScheduleStatusPaused, paused.Status)

	// Resume
	resumed, err := st.ResumeSchedule(ctx, "chat_1", "backup_sync")
	require.NoError(t, err)
	assert.Equal(t, ScheduleStatusActive, resumed.Status)

	// Cancel
	cancelled, err := st.CancelSchedule(ctx, "chat_1", "backup_sync")
	require.NoError(t, err)
	assert.Equal(t, ScheduleStatusCancelled, cancelled.Status)

	// Cancelling already cancelled schedule must fail
	_, err = st.CancelSchedule(ctx, "chat_1", "backup_sync")
	assert.Error(t, err)

	// Pausing cancelled schedule must fail
	_, err = st.PauseSchedule(ctx, "chat_1", "backup_sync")
	assert.Error(t, err)

	// Resuming cancelled schedule must fail
	_, err = st.ResumeSchedule(ctx, "chat_1", "backup_sync")
	assert.Error(t, err)
}

func TestStore_ClaimDueSchedules(t *testing.T) {
	ctx := context.Background()
	st, _ := setupTestStore(t)

	now := time.Now().Unix()
	s1 := &Schedule{
		ID:                "sched_due1",
		Name:              "due_task_1",
		ChatID:            "c1",
		UserID:            "u1",
		ScheduleType:      ScheduleTypeInterval,
		IntervalSeconds:   60,
		Instruction:       "run 1",
		Status:            ScheduleStatusActive,
		NextRunAt:         now - 10, // past due
		RunTimeoutSeconds: 60,
		MaxTurns:          10,
	}
	s2 := &Schedule{
		ID:                "sched_due2",
		Name:              "due_task_2",
		ChatID:            "c1",
		UserID:            "u1",
		ScheduleType:      ScheduleTypeInterval,
		IntervalSeconds:   60,
		Instruction:       "run 2",
		Status:            ScheduleStatusActive,
		NextRunAt:         now - 5, // past due
		RunTimeoutSeconds: 60,
		MaxTurns:          10,
	}
	s3 := &Schedule{
		ID:                "sched_future",
		Name:              "future_task",
		ChatID:            "c1",
		UserID:            "u1",
		ScheduleType:      ScheduleTypeInterval,
		IntervalSeconds:   60,
		Instruction:       "future",
		Status:            ScheduleStatusActive,
		NextRunAt:         now + 3600, // future
		RunTimeoutSeconds: 60,
		MaxTurns:          10,
	}
	require.NoError(t, st.CreateSchedule(ctx, s1, nil))
	require.NoError(t, st.CreateSchedule(ctx, s2, nil))
	require.NoError(t, st.CreateSchedule(ctx, s3, nil))

	lockUntil := now + 300
	claimed, err := st.ClaimDueSchedules(ctx, now, lockUntil, 10)
	require.NoError(t, err)
	require.Len(t, claimed, 2)
	assert.Equal(t, "sched_due1", claimed[0].ID)
	assert.Equal(t, "sched_due2", claimed[1].ID)

	// Second claim attempt should find nothing because rows were CAS locked to lockUntil
	claimedAgain, err := st.ClaimDueSchedules(ctx, now, lockUntil, 10)
	require.NoError(t, err)
	assert.Empty(t, claimedAgain)

	// Verify next_run_at in DB was set to lockUntil
	fresh1, err := st.GetSchedule(ctx, "sched_due1")
	require.NoError(t, err)
	assert.Equal(t, lockUntil, fresh1.NextRunAt)

	// Invalid nextRunAt on recurring schedule fails
	err = st.RecordExecutionResult(ctx, "sched_due1", "SUCCESS", -1, 0, false)
	assert.Error(t, err)

	// Record execution result for s1
	nextRun := now + 600
	err = st.RecordExecutionResult(ctx, "sched_due1", "SUCCESS", nextRun, 0, false)
	require.NoError(t, err)

	updated1, err := st.GetSchedule(ctx, "sched_due1")
	require.NoError(t, err)
	assert.Equal(t, 1, updated1.RunCount)
	assert.Equal(t, "SUCCESS", updated1.LastStatus)
	assert.Equal(t, nextRun, updated1.NextRunAt)
	assert.Equal(t, ScheduleStatusActive, updated1.Status)

	// Record execution completion for s2
	err = st.RecordExecutionResult(ctx, "sched_due2", "SUCCESS", 0, 0, true)
	require.NoError(t, err)

	updated2, err := st.GetSchedule(ctx, "sched_due2")
	require.NoError(t, err)
	assert.Equal(t, 1, updated2.RunCount)
	assert.Equal(t, ScheduleStatusCompleted, updated2.Status)

	// Recording execution result on non-active schedule fails
	err = st.RecordExecutionResult(ctx, "sched_due2", "SUCCESS", nextRun, 0, false)
	assert.Error(t, err)
}

func TestStore_PruneTerminalSchedules(t *testing.T) {
	st, _ := setupTestStore(t)

	ctx := context.Background()
	now := time.Now().Unix()

	// 1. Active schedule (should NOT be pruned)
	sActive := &Schedule{
		ID:        "s_active",
		Name:      "active_task",
		ChatID:    "chat1",
		UserID:    "u1",
		Status:    ScheduleStatusActive,
		NextRunAt: now + 3600,
		CreatedAt: now - 10000,
		UpdatedAt: now - 10000,
	}
	require.NoError(t, st.CreateSchedule(ctx, sActive, nil))

	// 2. Completed schedule updated recently (should NOT be pruned)
	sRecentCompleted := &Schedule{
		ID:        "s_recent_completed",
		Name:      "recent_task",
		ChatID:    "chat1",
		UserID:    "u1",
		Status:    ScheduleStatusCompleted,
		NextRunAt: now,
		CreatedAt: now - 500,
		UpdatedAt: now - 500,
	}
	require.NoError(t, st.CreateSchedule(ctx, sRecentCompleted, nil))

	// 3. Completed schedule updated long ago (SHOULD be pruned)
	sOldCompleted := &Schedule{
		ID:        "s_old_completed",
		Name:      "old_task",
		ChatID:    "chat1",
		UserID:    "u1",
		Status:    ScheduleStatusCompleted,
		NextRunAt: now - 20000,
		CreatedAt: now - 20000,
		UpdatedAt: now - 20000,
	}
	require.NoError(t, st.CreateSchedule(ctx, sOldCompleted, nil))
	_, err := st.DB().ExecContext(ctx, "UPDATE schedules SET updated_at = ? WHERE id = ?", now-20000, sOldCompleted.ID)
	require.NoError(t, err)

	// 4. Cancelled schedule updated long ago (SHOULD be pruned)
	sOldCancelled := &Schedule{
		ID:        "s_old_cancelled",
		Name:      "cancelled_task",
		ChatID:    "chat1",
		UserID:    "u1",
		Status:    ScheduleStatusCancelled,
		NextRunAt: now - 20000,
		CreatedAt: now - 20000,
		UpdatedAt: now - 20000,
	}
	require.NoError(t, st.CreateSchedule(ctx, sOldCancelled, nil))
	_, err = st.DB().ExecContext(ctx, "UPDATE schedules SET updated_at = ? WHERE id = ?", now-20000, sOldCancelled.ID)
	require.NoError(t, err)

	pruned, err := st.PruneTerminalSchedules(ctx, now-1000)
	require.NoError(t, err)
	assert.Equal(t, int64(2), pruned)

	// Verify surviving schedules
	_, err = st.GetSchedule(ctx, "s_active")
	assert.NoError(t, err)

	_, err = st.GetSchedule(ctx, "s_recent_completed")
	assert.NoError(t, err)

	_, err = st.GetSchedule(ctx, "s_old_completed")
	assert.ErrorIs(t, err, ErrScheduleNotFound)

	_, err = st.GetSchedule(ctx, "s_old_cancelled")
	assert.ErrorIs(t, err, ErrScheduleNotFound)
}

