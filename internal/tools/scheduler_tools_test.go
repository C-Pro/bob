package tools

import (
	"context"
	"database/sql"
	"fmt"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"bob/internal/scheduler"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	_ "modernc.org/sqlite"
)

type mockSchedulerStoreProvider struct {
	dir    string
	stores map[string]*scheduler.Store
}

func newMockSchedulerStoreProvider(t *testing.T) *mockSchedulerStoreProvider {
	t.Helper()
	return &mockSchedulerStoreProvider{
		dir:    t.TempDir(),
		stores: make(map[string]*scheduler.Store),
	}
}

func (m *mockSchedulerStoreProvider) GetSchedulerStore(ctx context.Context, chatID string, isDM bool) (*scheduler.Store, error) {
	key := chatID
	if st, ok := m.stores[key]; ok {
		return st, nil
	}
	dbPath := filepath.Join(m.dir, key+".db")
	db, err := sql.Open("sqlite", "file:"+dbPath+"?_pragma=foreign_keys(ON)&_pragma=journal_mode(WAL)")
	if err != nil {
		return nil, err
	}
	if err := scheduler.EnsureScheduleSchema(ctx, db); err != nil {
		return nil, err
	}
	st := scheduler.NewStore(db)
	m.stores[key] = st
	return st, nil
}

func setupTestRegistry(t *testing.T) (*Registry, *mockSchedulerStoreProvider) {
	t.Helper()
	prov := newMockSchedulerStoreProvider(t)
	reg := NewRegistry(nil, nil)
	reg.SetSchedulerStoreProvider(prov)
	reg.SetSchedulerLimits(1*time.Minute, 1*time.Hour, 10, 100)
	return reg, prov
}

func TestScheduleTask_NameAndBoundsValidation(t *testing.T) {
	reg, _ := setupTestRegistry(t)
	ctx := WithChatSession(context.Background(), ChatSessionContext{
		ChatID: "dm_user1",
		UserID: "user1",
		IsDM:   true,
	})

	// 1. Invalid schedule_name rejected
	invalidNames := []string{
		"Invalid Name",
		"too_many_words_in_this_name_here",
		"has-hyphen",
		"_leading_underscore",
		"",
	}
	for _, name := range invalidNames {
		args := fmt.Sprintf(`{
			"schedule_name": %q,
			"type": "interval",
			"schedule_spec": "600",
			"instruction": "Test",
			"run_timeout_seconds": 300,
			"max_turns": 20
		}`, name)
		_, err := reg.Execute(ctx, "schedule_task", args)
		assert.Error(t, err, "expected error for name: %s", name)
	}

	// 2. run_timeout_seconds out of bounds
	outOfBoundsTimeouts := []int{0, 30, 7200}
	for _, to := range outOfBoundsTimeouts {
		args := fmt.Sprintf(`{
			"schedule_name": "valid_name",
			"type": "interval",
			"schedule_spec": "600",
			"instruction": "Test",
			"run_timeout_seconds": %d,
			"max_turns": 20
		}`, to)
		_, err := reg.Execute(ctx, "schedule_task", args)
		assert.Error(t, err, "expected error for timeout: %d", to)
	}

	// 3. max_turns out of bounds
	outOfBoundsTurns := []int{0, 5, 150}
	for _, turns := range outOfBoundsTurns {
		args := fmt.Sprintf(`{
			"schedule_name": "valid_name",
			"type": "interval",
			"schedule_spec": "600",
			"instruction": "Test",
			"run_timeout_seconds": 300,
			"max_turns": %d
		}`, turns)
		_, err := reg.Execute(ctx, "schedule_task", args)
		assert.Error(t, err, "expected error for max_turns: %d", turns)
	}

	// 4. Empty instruction rejected
	argsEmptyInst := `{
		"schedule_name": "valid_name",
		"type": "interval",
		"schedule_spec": "600",
		"instruction": "   ",
		"run_timeout_seconds": 300,
		"max_turns": 20
	}`
	_, err := reg.Execute(ctx, "schedule_task", argsEmptyInst)
	assert.Error(t, err)
}

func TestScheduleTask_RecurrenceAndExpirationParsing(t *testing.T) {
	reg, _ := setupTestRegistry(t)
	ctx := WithChatSession(context.Background(), ChatSessionContext{
		ChatID: "dm_user1",
		UserID: "user1",
		IsDM:   true,
	})

	// 1. Interval minimum 300s
	argsShortInterval := `{
		"schedule_name": "short_int",
		"type": "interval",
		"schedule_spec": "60",
		"instruction": "Run every minute",
		"run_timeout_seconds": 120,
		"max_turns": 15
	}`
	_, err := reg.Execute(ctx, "schedule_task", argsShortInterval)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "at least 300 seconds")

	// Valid interval
	argsValidInterval := `{
		"schedule_name": "valid_int",
		"type": "interval",
		"schedule_spec": "600",
		"instruction": "Run every 10m",
		"run_timeout_seconds": 120,
		"max_turns": 15
	}`
	res, err := reg.Execute(ctx, "schedule_task", argsValidInterval)
	require.NoError(t, err)
	assert.Contains(t, res, "PENDING_APPROVAL")
	assert.Contains(t, res, "/schedule approve valid_int")

	// 2. Cron validation
	argsInvalidCron := `{
		"schedule_name": "bad_cron",
		"type": "cron",
		"schedule_spec": "invalid cron expr",
		"instruction": "Cron run",
		"run_timeout_seconds": 120,
		"max_turns": 15
	}`
	_, err = reg.Execute(ctx, "schedule_task", argsInvalidCron)
	assert.Error(t, err)

	argsValidCron := `{
		"schedule_name": "good_cron",
		"type": "cron",
		"schedule_spec": "0 9 * * *",
		"instruction": "Cron morning run",
		"run_timeout_seconds": 120,
		"max_turns": 15
	}`
	res, err = reg.Execute(ctx, "schedule_task", argsValidCron)
	require.NoError(t, err)
	assert.Contains(t, res, "PENDING_APPROVAL")
	assert.Contains(t, res, "/schedule approve good_cron")

	// 3. Expiration > 30 days rejected
	argsExpTooFar := `{
		"schedule_name": "far_exp",
		"type": "interval",
		"schedule_spec": "600",
		"instruction": "Test",
		"run_timeout_seconds": 120,
		"max_turns": 15,
		"expires_at": "1000h"
	}`
	_, err = reg.Execute(ctx, "schedule_task", argsExpTooFar)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "cannot exceed 30 days")
}

func TestScheduleTask_PermissionsAndSandbox(t *testing.T) {
	reg, prov := setupTestRegistry(t)

	// 1. Any schedule in Townhall strictly rejected
	thCtx := WithChatSession(context.Background(), ChatSessionContext{
		ChatID: "townhall",
		UserID: "user1",
		IsDM:   false,
	})
	argsTownhallSandbox := `{
		"schedule_name": "th_sandbox",
		"type": "interval",
		"schedule_spec": "600",
		"instruction": "Attempt sandbox in townhall",
		"run_timeout_seconds": 120,
		"max_turns": 15,
		"permissions": {
			"sandbox": {
				"driver": "docker"
			}
		}
	}`
	_, err := reg.Execute(thCtx, "schedule_task", argsTownhallSandbox)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "strictly prohibited in Townhall")

	argsTownhallNoSandbox := `{
		"schedule_name": "th_nosbx",
		"type": "interval",
		"schedule_spec": "600",
		"instruction": "Say hi in townhall",
		"run_timeout_seconds": 120,
		"max_turns": 15
	}`
	_, err = reg.Execute(thCtx, "schedule_task", argsTownhallNoSandbox)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "strictly prohibited in Townhall")

	// 2. Sandbox in DM creates PENDING_APPROVAL schedule with grant
	dmCtx := WithChatSession(context.Background(), ChatSessionContext{
		ChatID: "dm_user1",
		UserID: "user1",
		IsDM:   true,
	})
	rawPerms := `{"sandbox":{"driver":"docker","network":"restricted"}}`
	argsDMSandbox := fmt.Sprintf(`{
		"schedule_name": "dm_sandbox",
		"type": "interval",
		"schedule_spec": "600",
		"instruction": "Compile code",
		"run_timeout_seconds": 300,
		"max_turns": 20,
		"permissions": %s
	}`, rawPerms)

	res, err := reg.Execute(dmCtx, "schedule_task", argsDMSandbox)
	require.NoError(t, err)
	assert.Contains(t, res, "awaiting user approval")
	assert.Contains(t, res, "/schedule approve dm_sandbox")

	// Verify DB state
	st, err := prov.GetSchedulerStore(context.Background(), "dm_user1", true)
	require.NoError(t, err)
	sched, err := st.GetScheduleByName(context.Background(), "dm_user1", "dm_sandbox")
	require.NoError(t, err)
	assert.Equal(t, scheduler.ScheduleStatusPendingApproval, sched.Status)

	grant, err := st.GetGrant(context.Background(), sched.ID)
	require.NoError(t, err)
	assert.NotEmpty(t, grant.ParamsHash)
	assert.True(t, grant.Verify())

	// 3. Unique name conflict in same chat
	_, err = reg.Execute(dmCtx, "schedule_task", argsDMSandbox)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "already exists in this chat")
}

func TestScheduleTask_ListAndCancel(t *testing.T) {
	reg, _ := setupTestRegistry(t)
	ctx := WithChatSession(context.Background(), ChatSessionContext{
		ChatID: "dm_user1",
		UserID: "user1",
		IsDM:   true,
	})

	// Create 2 tasks
	t1 := `{
		"schedule_name": "task_alpha",
		"type": "interval",
		"schedule_spec": "600",
		"instruction": "Alpha instruction",
		"run_timeout_seconds": 120,
		"max_turns": 15
	}`
	t2 := `{
		"schedule_name": "task_beta",
		"type": "cron",
		"schedule_spec": "@daily",
		"instruction": "Beta instruction",
		"run_timeout_seconds": 120,
		"max_turns": 15
	}`
	_, err := reg.Execute(ctx, "schedule_task", t1)
	require.NoError(t, err)
	_, err = reg.Execute(ctx, "schedule_task", t2)
	require.NoError(t, err)

	// List schedules
	listRes, err := reg.Execute(ctx, "list_schedules", "{}")
	require.NoError(t, err)
	assert.Contains(t, listRes, "task_alpha")
	assert.Contains(t, listRes, "task_beta")
	assert.Contains(t, listRes, "PENDING_APPROVAL")

	// Cancel task_alpha
	cancelRes, err := reg.Execute(ctx, "cancel_schedule", `{"schedule_name": "task_alpha"}`)
	require.NoError(t, err)
	assert.Contains(t, cancelRes, "successfully cancelled")

	// Cancel non-existent schedule fails
	_, err = reg.Execute(ctx, "cancel_schedule", `{"schedule_name": "non_existent"}`)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "not found")
}

func TestScheduleTask_CronCadenceBounds(t *testing.T) {
	reg, _ := setupTestRegistry(t)
	ctx := WithChatSession(context.Background(), ChatSessionContext{
		ChatID: "dm_user1",
		UserID: "user1",
		IsDM:   true,
	})

	// 1-minute cron is rejected
	cron1m := `{
		"schedule_name": "fast_cron",
		"type": "cron",
		"schedule_spec": "* * * * *",
		"instruction": "Run every minute",
		"run_timeout_seconds": 60,
		"max_turns": 10
	}`
	_, err := reg.Execute(ctx, "schedule_task", cron1m)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "minimum cron cadence is 5 minutes")

	// 5-minute cron succeeds
	cron5m := `{
		"schedule_name": "valid_cron",
		"type": "cron",
		"schedule_spec": "*/5 * * * *",
		"instruction": "Run every 5 minutes",
		"run_timeout_seconds": 60,
		"max_turns": 10
	}`
	res, err := reg.Execute(ctx, "schedule_task", cron5m)
	require.NoError(t, err)
	assert.Contains(t, res, "PENDING_APPROVAL")
}

func TestScheduleTask_InstructionLengthCap(t *testing.T) {
	reg, _ := setupTestRegistry(t)
	ctx := WithChatSession(context.Background(), ChatSessionContext{
		ChatID: "dm_user1",
		UserID: "user1",
		IsDM:   true,
	})

	hugeInstruction := strings.Repeat("A", 4097)
	args := fmt.Sprintf(`{
		"schedule_name": "huge_inst",
		"type": "interval",
		"schedule_spec": "600",
		"instruction": %q,
		"run_timeout_seconds": 60,
		"max_turns": 10
	}`, hugeInstruction)
	_, err := reg.Execute(ctx, "schedule_task", args)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "maximum allowed length of 4096 characters")
}

func TestScheduleTask_TownhallProhibited(t *testing.T) {
	reg, _ := setupTestRegistry(t)
	thCtx := WithChatSession(context.Background(), ChatSessionContext{
		ChatID: "townhall",
		UserID: "user1",
		IsDM:   false,
	})

	args := `{
		"schedule_name": "th_recurring",
		"type": "interval",
		"schedule_spec": "600",
		"instruction": "Say hello in townhall",
		"run_timeout_seconds": 60,
		"max_turns": 10,
		"max_runs": 100
	}`
	_, err := reg.Execute(thCtx, "schedule_task", args)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "strictly prohibited in Townhall")

	_, err = reg.Execute(thCtx, "list_schedules", `{}`)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "strictly prohibited in Townhall")

	_, err = reg.Execute(thCtx, "cancel_schedule", `{"schedule_name": "th_recurring"}`)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "strictly prohibited in Townhall")
}

func TestScheduleTask_CancelOwnership(t *testing.T) {
	reg, _ := setupTestRegistry(t)
	u1Ctx := WithChatSession(context.Background(), ChatSessionContext{
		ChatID: "dm_user1",
		UserID: "user1",
		IsDM:   true,
	})
	u2Ctx := WithChatSession(context.Background(), ChatSessionContext{
		ChatID: "dm_user1",
		UserID: "user2",
		IsDM:   true,
	})

	args := `{
		"schedule_name": "user1_task",
		"type": "interval",
		"schedule_spec": "600",
		"instruction": "Say hi",
		"run_timeout_seconds": 60,
		"max_turns": 10
	}`
	_, err := reg.Execute(u1Ctx, "schedule_task", args)
	require.NoError(t, err)

	// user2 tries to cancel user1's task
	_, err = reg.Execute(u2Ctx, "cancel_schedule", `{"schedule_name": "user1_task"}`)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "permission denied")

	// user1 cancels own task
	res, err := reg.Execute(u1Ctx, "cancel_schedule", `{"schedule_name": "user1_task"}`)
	require.NoError(t, err)
	assert.Contains(t, res, "successfully cancelled")
}

func TestScheduleTask_ExpiresAtDaysParsing(t *testing.T) {
	reg, prov := setupTestRegistry(t)
	ctx := WithChatSession(context.Background(), ChatSessionContext{
		ChatID: "dm_user1",
		UserID: "user1",
		IsDM:   true,
	})

	// 7d works
	args7d := `{
		"schedule_name": "task_7d",
		"type": "interval",
		"schedule_spec": "600",
		"instruction": "Test 7d",
		"run_timeout_seconds": 60,
		"max_turns": 10,
		"expires_at": "7d"
	}`
	_, err := reg.Execute(ctx, "schedule_task", args7d)
	require.NoError(t, err)

	st, err := prov.GetSchedulerStore(ctx, "dm_user1", true)
	require.NoError(t, err)
	sched, err := st.GetScheduleByName(ctx, "dm_user1", "task_7d")
	require.NoError(t, err)
	require.NotNil(t, sched.ExpiresAt)
	assert.InDelta(t, time.Now().Add(7*24*time.Hour).Unix(), *sched.ExpiresAt, 5)

	// 31d fails (> 30 days)
	args31d := `{
		"schedule_name": "task_31d",
		"type": "interval",
		"schedule_spec": "600",
		"instruction": "Test 31d",
		"run_timeout_seconds": 60,
		"max_turns": 10,
		"expires_at": "31d"
	}`
	_, err = reg.Execute(ctx, "schedule_task", args31d)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "cannot exceed 30 days")
}

