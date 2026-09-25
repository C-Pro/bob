package scheduler

import (
	"context"
	"database/sql"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	_ "modernc.org/sqlite"
)

func setupEngineTestStore(t *testing.T) (*Store, func()) {
	t.Helper()
	dbPath := filepath.Join(t.TempDir(), "engine_test.db")
	db, err := sql.Open("sqlite", "file:"+dbPath+"?_pragma=foreign_keys(ON)&_pragma=journal_mode(WAL)")
	require.NoError(t, err)
	require.NoError(t, EnsureScheduleSchema(context.Background(), db))

	st := NewStore(db)
	return st, func() { _ = db.Close() }
}

func TestEngine_DueScheduleExecution(t *testing.T) {
	st, cleanup := setupEngineTestStore(t)
	defer cleanup()

	ctx := context.Background()
	now := time.Now().Unix()

	sched := &Schedule{
		ID:                "s_due_1",
		Name:              "due_interval",
		ChatID:            "chat_1",
		UserID:            "user_1",
		ScheduleType:      ScheduleTypeInterval,
		IntervalSeconds:   600,
		Instruction:       "Echo stats",
		Status:            ScheduleStatusActive,
		NextRunAt:         now - 10, // past due by 10s (< 15m)
		RunTimeoutSeconds: 60,
		MaxTurns:          10,
	}
	require.NoError(t, st.CreateSchedule(ctx, sched, nil))

	var executed atomic.Bool
	invoker := InvokerFunc(func(ctx context.Context, s *Schedule) error {
		assert.Equal(t, "s_due_1", s.ID)
		executed.Store(true)
		return nil
	})

	prov := NewStaticStoreProvider(st)
	engine := NewEngine(prov, invoker)
	engine.PollOnce(ctx)

	// Wait for goroutine execution to finish
	time.Sleep(50 * time.Millisecond)

	assert.True(t, executed.Load(), "due schedule must be executed by invoker")

	fresh, err := st.GetSchedule(ctx, "s_due_1")
	require.NoError(t, err)
	assert.Equal(t, 1, fresh.RunCount)
	assert.Equal(t, "SUCCESS", fresh.LastStatus)
	assert.True(t, fresh.NextRunAt > now, "next_run_at must advance to future")
}

func TestEngine_OverdueRecoveryAntiBurstSkip(t *testing.T) {
	st, cleanup := setupEngineTestStore(t)
	defer cleanup()

	ctx := context.Background()
	now := time.Now().Unix()

	// 2 hours overdue (> 15m catch-up threshold)
	sched := &Schedule{
		ID:                "s_overdue_1",
		Name:              "overdue_task",
		ChatID:            "chat_1",
		UserID:            "user_1",
		ScheduleType:      ScheduleTypeInterval,
		IntervalSeconds:   600, // 10 minutes
		Instruction:       "Heavy job",
		Status:            ScheduleStatusActive,
		NextRunAt:         now - 7200, // 2 hours ago
		RunTimeoutSeconds: 60,
		MaxTurns:          10,
	}
	require.NoError(t, st.CreateSchedule(ctx, sched, nil))

	var executed atomic.Bool
	invoker := InvokerFunc(func(ctx context.Context, s *Schedule) error {
		executed.Store(true)
		return nil
	})

	prov := NewStaticStoreProvider(st)
	engine := NewEngine(prov, invoker, WithDowntimeCutoff(15*time.Minute))
	engine.PollOnce(ctx)

	assert.False(t, executed.Load(), "overdue schedule beyond catch-up cutoff must not execute")

	fresh, err := st.GetSchedule(ctx, "s_overdue_1")
	require.NoError(t, err)
	assert.Equal(t, "SKIPPED_OVERDUE", fresh.LastStatus)
	assert.True(t, fresh.MissedCount > 0, "missed_count should be incremented")
	assert.True(t, fresh.NextRunAt > now, "next_run_at must advance to future")
}

func TestEngine_StopCancelsInFlight(t *testing.T) {
	st, cleanup := setupEngineTestStore(t)
	defer cleanup()

	now := time.Now().Unix()
	sched := &Schedule{
		ID:                "s_long_1",
		Name:              "long_task",
		ChatID:            "chat_1",
		UserID:            "user_1",
		ScheduleType:      ScheduleTypeInterval,
		IntervalSeconds:   600,
		Instruction:       "Long task",
		Status:            ScheduleStatusActive,
		NextRunAt:         now - 5,
		RunTimeoutSeconds: 60,
		MaxTurns:          10,
	}
	require.NoError(t, st.CreateSchedule(context.Background(), sched, nil))

	started := make(chan struct{})
	invoker := InvokerFunc(func(ctx context.Context, s *Schedule) error {
		close(started)
		<-ctx.Done()
		return ctx.Err()
	})

	prov := NewStaticStoreProvider(st)
	engine := NewEngine(prov, invoker, WithPollInterval(50*time.Millisecond))
	require.NoError(t, engine.Start(context.Background()))

	select {
	case <-started:
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for schedule to start")
	}

	// Stop must cancel in-flight and terminate cleanly
	engine.Stop()

	fresh, err := st.GetSchedule(context.Background(), "s_long_1")
	require.NoError(t, err)
	assert.Contains(t, fresh.LastStatus, "FAILED: context canceled")
}
