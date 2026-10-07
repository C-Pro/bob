package commands_test

import (
	"context"
	"database/sql"
	"path/filepath"
	"testing"
	"time"

	"bob/internal/agentapi"
	"bob/internal/commands"
	"bob/internal/scheduler"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func setupTestScheduleStore(t *testing.T) (*scheduler.Store, *sql.DB) {
	t.Helper()
	dbPath := filepath.Join(t.TempDir(), "scheduler_test.db")
	db, err := sql.Open("sqlite", "file:"+dbPath+"?_pragma=foreign_keys(ON)&_pragma=journal_mode(WAL)&_pragma=busy_timeout(5000)&_pragma=synchronous(NORMAL)")
	require.NoError(t, err)

	err = scheduler.EnsureScheduleSchema(context.Background(), db)
	require.NoError(t, err)

	t.Cleanup(func() { _ = db.Close() })
	return scheduler.NewStore(db), db
}

func TestScheduleHandler_ValidationAndHelp(t *testing.T) {
	ctx := context.Background()
	store, _ := setupTestScheduleStore(t)

	resolver := commands.UserResolverFunc(func(_ context.Context, id string) string {
		if id == "user_alice" {
			return "Alice"
		}
		return id
	})
	h := commands.NewScheduleHandler(store, resolver, "bot_1")

	session := agentapi.SessionRef{
		FrontendID: "besedka",
		SessionID:  "chat_dm_1",
		ScopeID:    "chat_dm_1",
	}
	actor := agentapi.Actor{ID: "user_alice"}

	// 1. Store nil
	hNoStore := commands.NewScheduleHandler(nil, resolver, "bot_1")
	res, err := hNoStore.Handle(ctx, commands.Request{
		Session: session,
		Actor:   actor,
	})
	require.NoError(t, err)
	assert.Contains(t, res.Reply, "Scheduler storage is currently unavailable")

	// 2. Help
	res, err = h.Handle(ctx, commands.Request{
		Session: session,
		Actor:   actor,
	})
	require.NoError(t, err)
	assert.Contains(t, res.Reply, "Schedule Commands:")

	// 3. List empty
	res, err = h.Handle(ctx, commands.Request{
		Session:    session,
		Actor:      actor,
		Subcommand: "list",
	})
	require.NoError(t, err)
	assert.Contains(t, res.Reply, "No active schedules found")

	// 4. Info missing arg
	res, err = h.Handle(ctx, commands.Request{
		Session:    session,
		Actor:      actor,
		Subcommand: "info",
	})
	require.NoError(t, err)
	assert.Contains(t, res.Reply, "Please specify a schedule name")
}

func TestScheduleHandler_ApproveAndInfo(t *testing.T) {
	ctx := context.Background()
	store, _ := setupTestScheduleStore(t)

	resolver := commands.UserResolverFunc(func(_ context.Context, id string) string {
		if id == "user_alice" {
			return "Alice"
		}
		return id
	})
	h := commands.NewScheduleHandler(store, resolver, "bot_1")

	session := agentapi.SessionRef{
		FrontendID: "besedka",
		SessionID:  "chat_dm_1",
		ScopeID:    "chat_dm_1",
	}

	// Create pending schedule
	now := time.Now().Unix()
	sched := &scheduler.Schedule{
		ID:                 "sched_1",
		Name:               "daily_news",
		ChatID:             "chat_dm_1",
		UserID:             "user_alice",
		ScheduleType:       scheduler.ScheduleTypeInterval,
		IntervalSeconds:    3600,
		Instruction:        "Fetch news",
		Status:             scheduler.ScheduleStatusPendingApproval,
		RunTimeoutSeconds:  60,
		MaxTurns:           10,
		NextRunAt:          now + 3600,
		CreatedAt:          now,
		UpdatedAt:          now,
	}
	err := store.CreateSchedule(ctx, sched, nil)
	require.NoError(t, err)

	// 1. Bot cannot approve
	res, err := h.Handle(ctx, commands.Request{
		Session:    session,
		Actor:      agentapi.Actor{ID: "bot_1"},
		Subcommand: "approve",
		Args:       []string{"daily_news"},
	})
	require.NoError(t, err)
	assert.Contains(t, res.Reply, "Agents cannot self-grant or approve schedules")

	// 2. Other user cannot approve
	res, err = h.Handle(ctx, commands.Request{
		Session:    session,
		Actor:      agentapi.Actor{ID: "user_bob"},
		Subcommand: "approve",
		Args:       []string{"daily_news"},
	})
	require.NoError(t, err)
	assert.Contains(t, res.Reply, "Permission denied")

	// 3. Info before approval
	res, err = h.Handle(ctx, commands.Request{
		Session:    session,
		Actor:      agentapi.Actor{ID: "user_alice"},
		Subcommand: "info",
		Args:       []string{"daily_news"},
	})
	require.NoError(t, err)
	assert.Contains(t, res.Reply, "Schedule: `daily_news`")
	assert.Contains(t, res.Reply, "Alice")
	assert.Contains(t, res.Reply, "PENDING_APPROVAL")

	// 4. Creator approves successfully
	res, err = h.Handle(ctx, commands.Request{
		Session:    session,
		Actor:      agentapi.Actor{ID: "user_alice"},
		Subcommand: "approve",
		Args:       []string{"daily_news"},
		IsDirect:   true,
	})
	require.NoError(t, err)
	assert.True(t, res.RecordAssistantEntry)
	assert.Contains(t, res.Reply, "Schedule approved and activated")

	// Check status updated in DB
	updated, err := store.GetScheduleByName(ctx, "chat_dm_1", "daily_news")
	require.NoError(t, err)
	assert.Equal(t, scheduler.ScheduleStatusActive, updated.Status)
}

func TestScheduleHandler_DenyCancelPauseResume(t *testing.T) {
	ctx := context.Background()
	store, _ := setupTestScheduleStore(t)
	h := commands.NewScheduleHandler(store, nil, "bot_1")

	session := agentapi.SessionRef{
		FrontendID: "besedka",
		SessionID:  "chat_dm_1",
		ScopeID:    "chat_dm_1",
	}
	alice := agentapi.Actor{ID: "user_alice"}

	now := time.Now().Unix()
	sched := &scheduler.Schedule{
		ID:                 "sched_2",
		Name:               "weekly_digest",
		ChatID:             "chat_dm_1",
		UserID:             "user_alice",
		ScheduleType:       scheduler.ScheduleTypeInterval,
		IntervalSeconds:    7200,
		Instruction:        "Digest",
		Status:             scheduler.ScheduleStatusActive,
		RunTimeoutSeconds:  60,
		MaxTurns:           5,
		NextRunAt:          now + 7200,
		CreatedAt:          now,
		UpdatedAt:          now,
	}
	err := store.CreateSchedule(ctx, sched, nil)
	require.NoError(t, err)

	// Pause
	res, err := h.Handle(ctx, commands.Request{
		Session:    session,
		Actor:      alice,
		Subcommand: "pause",
		Args:       []string{"weekly_digest"},
	})
	require.NoError(t, err)
	assert.True(t, res.RecordAssistantEntry)
	assert.Contains(t, res.Reply, "Schedule paused")

	// Resume
	res, err = h.Handle(ctx, commands.Request{
		Session:    session,
		Actor:      alice,
		Subcommand: "resume",
		Args:       []string{"weekly_digest"},
	})
	require.NoError(t, err)
	assert.True(t, res.RecordAssistantEntry)
	assert.Contains(t, res.Reply, "Schedule resumed")

	// Cancel
	res, err = h.Handle(ctx, commands.Request{
		Session:    session,
		Actor:      alice,
		Subcommand: "cancel",
		Args:       []string{"weekly_digest"},
	})
	require.NoError(t, err)
	assert.True(t, res.RecordAssistantEntry)
	assert.Contains(t, res.Reply, "Schedule cancelled")

	// Create pending to test deny
	schedPending := &scheduler.Schedule{
		ID:                 "sched_3",
		Name:               "pending_task",
		ChatID:             "chat_dm_1",
		UserID:             "user_alice",
		ScheduleType:       scheduler.ScheduleTypeInterval,
		IntervalSeconds:    7200,
		Instruction:        "Pending",
		Status:             scheduler.ScheduleStatusPendingApproval,
		CreatedAt:          now,
		UpdatedAt:          now,
	}
	err = store.CreateSchedule(ctx, schedPending, nil)
	require.NoError(t, err)

	res, err = h.Handle(ctx, commands.Request{
		Session:    session,
		Actor:      alice,
		Subcommand: "deny",
		Args:       []string{"pending_task"},
	})
	require.NoError(t, err)
	assert.True(t, res.RecordAssistantEntry)
	assert.Contains(t, res.Reply, "Schedule request denied")
}
