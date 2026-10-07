package agentstore

import (
	"context"
	"database/sql"
	"fmt"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"bob/internal/config"
	"bob/internal/fsm"
	"bob/internal/knowledge"
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

func TestMemoryStoreProvider_KnowledgeSearcherMaxLimit(t *testing.T) {
	tempDir := t.TempDir()
	cfg := &config.Config{DataDir: tempDir}
	ctx := context.Background()

	memMgr := memory.NewManager(cfg, nil)
	defer func() { _ = memMgr.Close() }()

	provider := NewMemoryStoreProvider(memMgr, tempDir)
	provider.SetMaxDiscoveryLimit(25)

	kStore, err := provider.GetKnowledgeStore(ctx, "chat_dm_max", true)
	require.NoError(t, err)

	for i := 1; i <= 15; i++ {
		_, ver, err := kStore.ProposeMemory(ctx, &knowledge.KnowledgeItem{
			ChatID: "chat_dm_max",
			UserID: "user_1",
		}, &knowledge.MemoryVersion{
			Type:       knowledge.MemoryTypeFact,
			Content:    fmt.Sprintf("User profile fact item %d with query keyword for discovery testing", i),
			Confidence: 0.9,
		})
		require.NoError(t, err)
		require.NoError(t, kStore.ApproveVersion(ctx, ver.ID, "user_1"))
	}

	reconciled, err := provider.ReconcileChat(ctx, "chat_dm_max", true, 20)
	require.NoError(t, err)
	assert.Equal(t, 15, reconciled)

	searcher, err := provider.GetKnowledgeSearcher(ctx, "chat_dm_max", true, "user_1")
	require.NoError(t, err)
	require.NotNil(t, searcher)

	// Verify SearchMemories with limit > 10 returns all 15 populated items rather than being clamped to 10
	results, err := searcher.SearchMemories(ctx, "query", nil, 20)
	require.NoError(t, err)
	assert.Len(t, results, 15)
}

func TestMemoryStoreProvider_FineGrainedLocking_NoCrossChatBlocking(t *testing.T) {
	tempDir := t.TempDir()
	cfg := &config.Config{DataDir: tempDir, EmbeddingModel: "none"}
	memMgr := memory.NewManager(cfg, nil)
	defer func() { _ = memMgr.Close() }()

	provider := NewMemoryStoreProvider(memMgr, tempDir)
	ctx := context.Background()

	// Pre-initialize chat_a and chat_b
	_, err := provider.GetKnowledgeStore(ctx, "chat_a", true)
	require.NoError(t, err)
	_, err = provider.GetKnowledgeStore(ctx, "chat_b", true)
	require.NoError(t, err)

	// Hold chat_a's reconcile gate to simulate a slow indexing/reconciliation operation
	entryA := provider.getOrCreateKnowledgeEntry("chat_a", true)
	entryA.reconcileGate <- struct{}{}
	defer func() { <-entryA.reconcileGate }()

	// Concurrently reconcile chat_b. It MUST NOT be blocked by chat_a.
	done := make(chan struct{})
	go func() {
		defer close(done)
		_, reconcileErr := provider.ReconcileChat(ctx, "chat_b", true, 10)
		assert.NoError(t, reconcileErr)
	}()

	select {
	case <-done:
		// Succeeded immediately without being blocked by chat_a
	case <-time.After(1 * time.Second):
		t.Fatal("chat_b reconciliation was blocked by chat_a's reconcile gate")
	}
}

func TestMemoryStoreProvider_ConcurrentInit_SingleInstance(t *testing.T) {
	tempDir := t.TempDir()
	cfg := &config.Config{DataDir: tempDir, EmbeddingModel: "none"}
	memMgr := memory.NewManager(cfg, nil)
	defer func() { _ = memMgr.Close() }()

	provider := NewMemoryStoreProvider(memMgr, tempDir)
	ctx := context.Background()

	const concurrency = 16
	var wg sync.WaitGroup
	wg.Add(concurrency)

	stores := make([]any, concurrency)
	indexers := make([]any, concurrency)

	for i := 0; i < concurrency; i++ {
		go func(idx int) {
			defer wg.Done()
			s, err := provider.GetKnowledgeStore(ctx, "chat_concurrent_init", true)
			if err == nil {
				stores[idx] = s
			}
			idxer, err := provider.GetKnowledgeIndexer(ctx, "chat_concurrent_init", true)
			if err == nil {
				indexers[idx] = idxer
			}
		}(i)
	}

	wg.Wait()

	require.NotNil(t, stores[0])
	require.NotNil(t, indexers[0])
	for i := 1; i < concurrency; i++ {
		assert.Same(t, stores[0], stores[i], "all stores should be the exact same instance")
		assert.Same(t, indexers[0], indexers[i], "all indexers should be the exact same instance")
	}
}

func TestKnowledgeWorker_SerializesWithForegroundReconcile(t *testing.T) {
	tempDir := t.TempDir()
	cfg := &config.Config{DataDir: tempDir, EmbeddingModel: "none"}
	memMgr := memory.NewManager(cfg, nil)
	defer func() { _ = memMgr.Close() }()

	provider := NewMemoryStoreProvider(memMgr, tempDir)
	ctx := context.Background()

	_, err := provider.GetKnowledgeStore(ctx, "chat_serialized", true)
	require.NoError(t, err)

	entry := provider.getOrCreateKnowledgeEntry("chat_serialized", true)
	entry.reconcileGate <- struct{}{}

	started := make(chan struct{})
	completed := make(chan struct{})

	go func() {
		close(started)
		_, _ = provider.ReconcileChat(ctx, "chat_serialized", true, 10)
		close(completed)
	}()

	<-started

	// Verify that the second reconciliation is waiting for reconcileGate
	select {
	case <-completed:
		t.Fatal("second reconciliation should be blocked while reconcileGate is held")
	case <-time.After(50 * time.Millisecond):
		// Expected to be blocked
	}

	// Release the gate, allowing the second reconciliation to proceed
	<-entry.reconcileGate

	select {
	case <-completed:
		// Succeeded after unlock
	case <-time.After(1 * time.Second):
		t.Fatal("reconciliation failed to complete after releasing gate")
	}
}

func TestMemoryStoreProvider_ReconcileChat_ContextCancellation(t *testing.T) {
	tempDir := t.TempDir()
	cfg := &config.Config{DataDir: tempDir, EmbeddingModel: "none"}
	memMgr := memory.NewManager(cfg, nil)
	defer func() { _ = memMgr.Close() }()

	provider := NewMemoryStoreProvider(memMgr, tempDir)
	entry := provider.getOrCreateKnowledgeEntry("chat_timeout", true)
	entry.reconcileGate <- struct{}{}
	defer func() { <-entry.reconcileGate }()

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	start := time.Now()
	_, err := provider.ReconcileChat(ctx, "chat_timeout", true, 10)
	duration := time.Since(start)

	assert.ErrorIs(t, err, context.DeadlineExceeded)
	assert.Less(t, duration, 500*time.Millisecond, "ReconcileChat should return immediately when context deadline expires")
}

func TestHasPendingKnowledgeWork_ExcludesExpiredMemories(t *testing.T) {
	tempDir := t.TempDir()
	dbPath := filepath.Join(tempDir, "dm_expired_test.db")

	dsn := fmt.Sprintf("file:%s", dbPath)
	db, err := sql.Open("sqlite", dsn)
	require.NoError(t, err)
	defer func() { _ = db.Close() }()

	ctx := context.Background()
	err = knowledge.EnsureKnowledgeSchema(ctx, db)
	require.NoError(t, err)

	// Case 1: Empty database has no pending work
	hasWork, err := hasPendingKnowledgeWork(dbPath)
	require.NoError(t, err)
	assert.False(t, hasWork)

	// Case 2: Expired active memory candidate must NOT trigger pending work
	now := time.Now().Unix()
	past := now - 3600
	future := now + 3600

	_, err = db.ExecContext(ctx, `
		INSERT INTO knowledge_items (id, kind, chat_id, user_id, status, active_version_id, created_at, updated_at)
		VALUES ('item_expired', 'memory', 'expired_test', 'user1', 'active', 'mem_expired:1', ?, ?)
	`, now, now)
	require.NoError(t, err)

	_, err = db.ExecContext(ctx, `
		INSERT INTO memory_versions (id, item_id, revision, type, content, confidence, expires_at, status, chat_id, user_id, content_hash, index_status, created_at)
		VALUES ('mem_expired:1', 'item_expired', 1, 'profile', 'expired content', 1.0, ?, 'approved', 'expired_test', 'user1', 'hash_expired', 'pending', ?)
	`, past, now)
	require.NoError(t, err)

	hasWork, err = hasPendingKnowledgeWork(dbPath)
	require.NoError(t, err)
	assert.False(t, hasWork, "expired memories must not trigger pending knowledge work")

	// Case 3: Unexpired active memory candidate DOES trigger pending work
	_, err = db.ExecContext(ctx, `
		INSERT INTO knowledge_items (id, kind, chat_id, user_id, status, active_version_id, created_at, updated_at)
		VALUES ('item_valid', 'memory', 'expired_test', 'user1', 'active', 'mem_valid:1', ?, ?)
	`, now, now)
	require.NoError(t, err)

	_, err = db.ExecContext(ctx, `
		INSERT INTO memory_versions (id, item_id, revision, type, content, confidence, expires_at, status, chat_id, user_id, content_hash, index_status, created_at)
		VALUES ('mem_valid:1', 'item_valid', 1, 'profile', 'valid content', 1.0, ?, 'approved', 'expired_test', 'user1', 'hash_valid', 'pending', ?)
	`, future, now)
	require.NoError(t, err)

	hasWork, err = hasPendingKnowledgeWork(dbPath)
	require.NoError(t, err)
	assert.True(t, hasWork, "unexpired active memory should trigger pending knowledge work")
}

func TestKnowledgeWorker_DiscoversAndReconcilesUnopenedDMDatabases(t *testing.T) {
	tempDir := t.TempDir()
	cfg := &config.Config{DataDir: tempDir, EmbeddingModel: "none"}
	ctx := context.Background()

	// Phase 1: Create a DM database with pending index deletion work and pending unindexed skill
	memMgr := memory.NewManager(cfg, nil)
	provider := NewMemoryStoreProvider(memMgr, tempDir)

	store, err := provider.GetKnowledgeStore(ctx, "dm_restart_test", true)
	require.NoError(t, err)

	err = store.EnqueueIndexDeletion(ctx, "skill_pending_purge:1", knowledge.NamespaceSkills)
	require.NoError(t, err)

	pending, err := store.GetPendingIndexDeletions(ctx, 10)
	require.NoError(t, err)
	require.Len(t, pending, 1)

	// Create an active approved skill version with pending index status
	_, skillVer, err := store.ProposeSkill(ctx, &knowledge.KnowledgeItem{
		ChatID: "dm_restart_test",
		UserID: "user_test",
	}, &knowledge.SkillVersion{
		Name:                 "active_pending_skill",
		Description:          "description",
		Triggers:             []string{"test trigger"},
		Tags:                 []string{"test"},
		InstructionsMarkdown: "echo hello",
	})
	require.NoError(t, err)

	err = store.ApproveVersion(ctx, skillVer.ID, "user_test")
	require.NoError(t, err)

	// Verify it was marked pending
	skillVerAfter, err := store.GetSkillVersion(ctx, skillVer.ID)
	require.NoError(t, err)
	assert.Equal(t, knowledge.IndexStatusPending, skillVerAfter.IndexStatus)

	// Close database to simulate service shutdown
	require.NoError(t, memMgr.Close())

	// Phase 2: Start fresh provider pointing to the same dataDir without pre-opening the DM
	restartedMgr := memory.NewManager(cfg, nil)
	defer func() { _ = restartedMgr.Close() }()
	restartedProvider := NewMemoryStoreProvider(restartedMgr, tempDir)

	// Verify discovery finds unopened dm_restart_test because it has pending work
	targets := restartedProvider.DiscoverKnowledgeTargets(ctx)
	assert.Contains(t, targets, "dm_restart_test")

	// Phase 3: Run worker pass
	worker := knowledge.NewKnowledgeWorker(restartedProvider, knowledge.KnowledgeWorkerConfig{
		Interval:    10 * time.Second,
		Concurrency: 2,
		ChatTimeout: 30 * time.Second,
	})

	worker.RunPass(ctx)

	// Verify pending deletion was drained by the worker
	reopenedStore, err := restartedProvider.GetKnowledgeStore(ctx, "dm_restart_test", true)
	require.NoError(t, err)

	pendingAfter, err := reopenedStore.GetPendingIndexDeletions(ctx, 10)
	require.NoError(t, err)
	assert.Empty(t, pendingAfter, "pending index deletion should have been reconciled and drained")

	// Verify candidate skill was indexed to ready
	reconciledVer, err := reopenedStore.GetSkillVersion(ctx, skillVer.ID)
	require.NoError(t, err)
	assert.Equal(t, knowledge.IndexStatusReady, reconciledVer.IndexStatus)
}
