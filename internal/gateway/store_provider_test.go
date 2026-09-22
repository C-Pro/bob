package gateway

import (
	"context"
	"path/filepath"
	"testing"
	"time"

	"bob/internal/config"
	"bob/internal/fsm"
	"bob/internal/memory"
	"bob/internal/scheduler"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	_ "modernc.org/sqlite"
)

func TestMemoryStoreProvider_GetStore(t *testing.T) {
	tempDir := t.TempDir()
	cfg := &config.Config{
		DataDir: tempDir,
	}

	memMgr := memory.NewManager(cfg, nil)
	defer func() {
		_ = memMgr.Close()
	}()

	provider := NewMemoryStoreProvider(memMgr, tempDir)
	ctx := context.Background()

	// Townhall store
	thStore, err := provider.GetStore(ctx, "townhall", false)
	require.NoError(t, err)
	require.NotNil(t, thStore)

	activeRuns, err := thStore.ListActiveRuns(ctx)
	require.NoError(t, err)
	assert.Empty(t, activeRuns)

	// Repeated call returns cached store instance
	thStoreCached, err := provider.GetStore(ctx, "townhall", false)
	require.NoError(t, err)
	assert.Same(t, thStore, thStoreCached)

	// DM store
	dmStore, err := provider.GetStore(ctx, "user_123", true)
	require.NoError(t, err)
	require.NotNil(t, dmStore)

	// Repeated call returns cached DM store instance
	dmStoreCached, err := provider.GetStore(ctx, "user_123", true)
	require.NoError(t, err)
	assert.Same(t, dmStore, dmStoreCached)

	activeRuns, err = dmStore.ListActiveRuns(ctx)
	require.NoError(t, err)
	assert.Empty(t, activeRuns)
}

func TestMemoryStoreProvider_ActiveStores(t *testing.T) {
	tempDir := t.TempDir()
	cfg := &config.Config{
		DataDir: tempDir,
	}

	memMgr := memory.NewManager(cfg, nil)
	defer func() {
		_ = memMgr.Close()
	}()

	provider := NewMemoryStoreProvider(memMgr, tempDir)
	ctx := context.Background()

	// Open two stores
	_, err := provider.GetStore(ctx, "townhall", false)
	require.NoError(t, err)
	aliceStore, err := provider.GetStore(ctx, "user_alice", true)
	require.NoError(t, err)

	activeStores, err := provider.ActiveStores(ctx)
	require.NoError(t, err)
	assert.Len(t, activeStores, 2)

	// Create an active run in user_alice so it requires recovery/polling at startup.
	// townhall remains idle without active runs.
	err = aliceStore.CreateRun(ctx, &fsm.FSMRun{
		ID:           "run_alice_active",
		ChatID:       "user_alice",
		UserID:       "alice",
		IsDM:         true,
		FSMType:      fsm.FSMTypeToolLoop,
		CurrentState: fsm.StateLLMRequest,
		Status:       fsm.RunStatusRunning,
		ContextJSON:  "[]",
	})
	require.NoError(t, err)

	// Create a new provider pointing to the same dataDir without prior GetStore calls.
	// Only user_alice should be discovered because townhall has no active runs.
	memMgr2 := memory.NewManager(cfg, nil)
	defer func() {
		_ = memMgr2.Close()
	}()

	provider2 := NewMemoryStoreProvider(memMgr2, tempDir)
	discoveredStores, err := provider2.ActiveStores(ctx)
	require.NoError(t, err)
	assert.Len(t, discoveredStores, 1)
}

func TestHasActiveSchedules(t *testing.T) {
	tempDir := t.TempDir()
	cfg := &config.Config{DataDir: tempDir}
	memMgr := memory.NewManager(cfg, nil)
	defer func() { _ = memMgr.Close() }()

	provider := NewMemoryStoreProvider(memMgr, tempDir)
	ctx := context.Background()
	store, err := provider.GetSchedulerStore(ctx, "probe", true)
	require.NoError(t, err)

	dbPath := filepath.Join(tempDir, "dm_probe.db")
	active, err := hasActiveSchedules(dbPath)
	require.NoError(t, err)
	assert.False(t, active)

	now := time.Now().Unix()
	require.NoError(t, store.CreateSchedule(ctx, &scheduler.Schedule{
		ID:                "sched_active_probe",
		Name:              "active_probe",
		ChatID:            "probe",
		UserID:            "user",
		ScheduleType:      scheduler.ScheduleTypeCron,
		CronExpr:          "0 10 * * *",
		Instruction:       "probe",
		Status:            scheduler.ScheduleStatusActive,
		NextRunAt:         now + 3600,
		RunTimeoutSeconds: 300,
		MaxTurns:          15,
	}, nil))

	active, err = hasActiveSchedules(dbPath)
	require.NoError(t, err)
	assert.True(t, active)
}

func TestMemoryStoreProvider_RecoversScheduleWithoutActiveFSMRun(t *testing.T) {
	for _, schedulerFirst := range []bool{false, true} {
		name := "fsm_discovery_first"
		if schedulerFirst {
			name = "scheduler_discovery_first"
		}
		t.Run(name, func(t *testing.T) {
			tempDir := t.TempDir()
			cfg := &config.Config{DataDir: tempDir}
			ctx := context.Background()

			memMgr := memory.NewManager(cfg, nil)
			provider := NewMemoryStoreProvider(memMgr, tempDir)
			store, err := provider.GetSchedulerStore(ctx, "scheduled_chat", true)
			require.NoError(t, err)
			require.NoError(t, store.CreateSchedule(ctx, &scheduler.Schedule{
				ID:                "sched_restart",
				Name:              "restart_probe",
				ChatID:            "scheduled_chat",
				UserID:            "user",
				ScheduleType:      scheduler.ScheduleTypeCron,
				CronExpr:          "0 10 * * *",
				Instruction:       "probe restart recovery",
				Status:            scheduler.ScheduleStatusActive,
				NextRunAt:         time.Now().Add(time.Hour).Unix(),
				RunTimeoutSeconds: 300,
				MaxTurns:          15,
			}, nil))
			require.NoError(t, memMgr.Close())

			restartedMgr := memory.NewManager(cfg, nil)
			defer func() { _ = restartedMgr.Close() }()
			restartedProvider := NewMemoryStoreProvider(restartedMgr, tempDir)

			if schedulerFirst {
				schedulerStores, err := restartedProvider.ActiveSchedulerStores(ctx)
				require.NoError(t, err)
				require.Len(t, schedulerStores, 1)
			}

			fsmStores, err := restartedProvider.ActiveStores(ctx)
			require.NoError(t, err)
			assert.Empty(t, fsmStores)

			schedulerStores, err := restartedProvider.ActiveSchedulerStores(ctx)
			require.NoError(t, err)
			require.Len(t, schedulerStores, 1)
			recovered, err := schedulerStores[0].GetSchedule(ctx, "sched_restart")
			require.NoError(t, err)
			assert.Equal(t, scheduler.ScheduleStatusActive, recovered.Status)
		})
	}
}

func TestMemoryStoreProvider_ExecutesRecoveredScheduleWithoutActiveFSMRun(t *testing.T) {
	tempDir := t.TempDir()
	cfg := &config.Config{DataDir: tempDir}
	ctx := context.Background()

	memMgr := memory.NewManager(cfg, nil)
	provider := NewMemoryStoreProvider(memMgr, tempDir)
	store, err := provider.GetSchedulerStore(ctx, "scheduled_chat", true)
	require.NoError(t, err)
	require.NoError(t, store.CreateSchedule(ctx, &scheduler.Schedule{
		ID:                "sched_due_after_restart",
		Name:              "due_after_restart",
		ChatID:            "scheduled_chat",
		UserID:            "user",
		ScheduleType:      scheduler.ScheduleTypeInterval,
		IntervalSeconds:   600,
		Instruction:       "run after restart",
		Status:            scheduler.ScheduleStatusActive,
		NextRunAt:         time.Now().Add(-5 * time.Second).Unix(),
		RunTimeoutSeconds: 300,
		MaxTurns:          15,
	}, nil))
	require.NoError(t, memMgr.Close())

	restartedMgr := memory.NewManager(cfg, nil)
	defer func() { _ = restartedMgr.Close() }()
	restartedProvider := NewMemoryStoreProvider(restartedMgr, tempDir)

	fsmStores, err := restartedProvider.ActiveStores(ctx)
	require.NoError(t, err)
	assert.Empty(t, fsmStores)

	executed := make(chan string, 1)
	engine := scheduler.NewEngine(restartedProvider, scheduler.InvokerFunc(func(_ context.Context, sched *scheduler.Schedule) error {
		executed <- sched.ID
		return nil
	}))
	engine.PollOnce(ctx)

	select {
	case scheduleID := <-executed:
		assert.Equal(t, "sched_due_after_restart", scheduleID)
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for recovered schedule execution")
	}

	schedulerStores, err := restartedProvider.ActiveSchedulerStores(ctx)
	require.NoError(t, err)
	require.Len(t, schedulerStores, 1)
	require.Eventually(t, func() bool {
		recovered, getErr := schedulerStores[0].GetSchedule(ctx, "sched_due_after_restart")
		return getErr == nil && recovered.RunCount == 1 && recovered.LastStatus == "SUCCESS"
	}, 2*time.Second, 10*time.Millisecond)
}

func TestMemoryStoreProvider_NilManager(t *testing.T) {
	provider := NewMemoryStoreProvider(nil, "")
	ctx := context.Background()

	store, err := provider.GetStore(ctx, "townhall", false)
	assert.Error(t, err)
	assert.Nil(t, store)

	stores, err := provider.ActiveStores(ctx)
	assert.NoError(t, err)
	assert.Nil(t, stores)

	schedulerStores, err := provider.ActiveSchedulerStores(ctx)
	assert.NoError(t, err)
	assert.Nil(t, schedulerStores)
}
