package gateway

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"bob/internal/config"
	"bob/internal/knowledge"
	"bob/internal/memory"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	_ "modernc.org/sqlite"
)

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

type mockKnowledgeTargetProvider struct {
	mu           sync.Mutex
	targets      []string
	reconcileFn  func(ctx context.Context, chatID string, isDM bool, limit int) (int, error)
	reconcileCnt int
}

func (m *mockKnowledgeTargetProvider) DiscoverKnowledgeTargets(ctx context.Context) []string {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.targets
}

func (m *mockKnowledgeTargetProvider) ReconcileChat(ctx context.Context, chatID string, isDM bool, limit int) (int, error) {
	m.mu.Lock()
	m.reconcileCnt++
	fn := m.reconcileFn
	m.mu.Unlock()
	if fn != nil {
		return fn(ctx, chatID, isDM, limit)
	}
	return 0, nil
}

func TestKnowledgeWorker_TransientErrorRetry(t *testing.T) {
	var attempts atomic.Int32
	mockProv := &mockKnowledgeTargetProvider{
		targets: []string{"chat_retry"},
		reconcileFn: func(ctx context.Context, chatID string, isDM bool, limit int) (int, error) {
			att := attempts.Add(1)
			if att == 1 {
				return 0, errors.New("transient cortexdb vector failure")
			}
			return 1, nil
		},
	}

	worker := NewKnowledgeWorker(mockProv, KnowledgeWorkerConfig{
		Interval:    10 * time.Millisecond,
		Concurrency: 2,
		ChatTimeout: 1 * time.Second,
	})

	ctx := context.Background()

	// First pass fails transiently
	worker.RunPass(ctx)
	assert.Equal(t, int32(1), attempts.Load())

	// Second pass succeeds
	worker.RunPass(ctx)
	assert.Equal(t, int32(2), attempts.Load())
}

func TestKnowledgeWorker_CancellationDuringConcurrencySaturation(t *testing.T) {
	gate := make(chan struct{})
	var startedCount atomic.Int32

	mockProv := &mockKnowledgeTargetProvider{
		targets: []string{"chat_1", "chat_2", "chat_3", "chat_4", "chat_5"},
		reconcileFn: func(ctx context.Context, chatID string, isDM bool, limit int) (int, error) {
			startedCount.Add(1)
			select {
			case <-gate:
				return 0, nil
			case <-ctx.Done():
				return 0, ctx.Err()
			}
		},
	}

	worker := NewKnowledgeWorker(mockProv, KnowledgeWorkerConfig{
		Interval:    1 * time.Hour,
		Concurrency: 2,
		ChatTimeout: 5 * time.Second,
	})

	ctx, cancel := context.WithCancel(context.Background())

	runDone := make(chan struct{})
	go func() {
		defer close(runDone)
		worker.RunPass(ctx)
	}()

	// Wait until concurrency limit (2) is reached
	require.Eventually(t, func() bool {
		return startedCount.Load() >= 2
	}, 1*time.Second, 10*time.Millisecond)

	// Cancel context while semaphore is saturated
	cancel()

	// Release in-flight jobs
	close(gate)

	// RunPass must exit cleanly without deadlocking on semaphore or spawning extra goroutines
	select {
	case <-runDone:
		// Succeeded
	case <-time.After(2 * time.Second):
		t.Fatal("worker.RunPass timed out or deadlocked after cancellation during concurrency saturation")
	}

	// Saturated at 2 concurrent jobs; must not have run all 5
	assert.LessOrEqual(t, startedCount.Load(), int32(3))
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
	worker := NewKnowledgeWorker(restartedProvider, KnowledgeWorkerConfig{
		Interval:    10 * time.Second,
		Concurrency: 2,
		ChatTimeout: 30 * time.Second,
	})

	worker.runPass(ctx)

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

func TestKnowledgeWorker_LifecycleAndGracefulStop(t *testing.T) {
	tempDir := t.TempDir()
	cfg := &config.Config{DataDir: tempDir, EmbeddingModel: "none"}
	memMgr := memory.NewManager(cfg, nil)
	defer func() { _ = memMgr.Close() }()

	provider := NewMemoryStoreProvider(memMgr, tempDir)
	worker := NewKnowledgeWorker(provider, KnowledgeWorkerConfig{
		Interval:    20 * time.Millisecond,
		Concurrency: 2,
		ChatTimeout: 1 * time.Second,
	})

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	worker.Start(ctx)
	assert.True(t, worker.Running())

	// Double start should be a no-op
	worker.Start(ctx)
	assert.True(t, worker.Running())

	// Stop cleanly
	worker.Stop()
	assert.False(t, worker.Running())

	// Double stop should be a safe no-op
	worker.Stop()
	assert.False(t, worker.Running())
}

func TestKnowledgeWorker_StopVersusStart_GenerationSafe(t *testing.T) {
	blockReconcile := make(chan struct{})
	reconcileStarted := make(chan struct{})
	stopTriggered := make(chan struct{})

	mockProv := &mockKnowledgeTargetProvider{
		targets: []string{"chat_gen"},
		reconcileFn: func(ctx context.Context, chatID string, isDM bool, limit int) (int, error) {
			select {
			case reconcileStarted <- struct{}{}:
			default:
			}
			go func() {
				<-ctx.Done()
				select {
				case stopTriggered <- struct{}{}:
				default:
				}
			}()
			<-blockReconcile
			return 0, nil
		},
	}

	worker := NewKnowledgeWorker(mockProv, KnowledgeWorkerConfig{
		Interval:    10 * time.Millisecond,
		Concurrency: 2,
		ChatTimeout: 5 * time.Second,
	})

	ctx := context.Background()
	worker.Start(ctx)
	require.True(t, worker.Running())

	// Wait until generation 1 job starts and blocks
	<-reconcileStarted

	// Launch Stop in a goroutine
	stopDone := make(chan struct{})
	go func() {
		defer close(stopDone)
		worker.Stop()
	}()

	// While Stop is waiting for generation 1 to drain, attempt Start.
	// Must be ignored because generation 1 is still draining.
	<-stopTriggered
	worker.Start(ctx)

	// Unblock generation 1
	close(blockReconcile)

	// Stop completes
	select {
	case <-stopDone:
	case <-time.After(2 * time.Second):
		t.Fatal("worker.Stop did not return after unblocking job")
	}

	// After stop completes, worker is not running
	assert.False(t, worker.Running())

	// Can now start generation 2 cleanly
	worker.Start(ctx)
	assert.True(t, worker.Running())
	worker.Stop()
	assert.False(t, worker.Running())
}

func TestGateway_KnowledgeWorker_ReplacementVersusStop(t *testing.T) {
	tempDir := t.TempDir()
	cfg := &config.Config{
		DataDir: tempDir,
	}

	gw := NewGateway(cfg, nil)
	lifecycleCtx, cancel := context.WithCancel(context.Background())
	defer cancel()

	gw.mu.Lock()
	gw.running = true
	gw.lifecycleCtx = lifecycleCtx
	gw.mu.Unlock()

	oldRelease := make(chan struct{})
	oldStarted := make(chan struct{})
	oldStopTriggered := make(chan struct{})
	oldProv := &mockKnowledgeTargetProvider{
		targets: []string{"c1"},
		reconcileFn: func(ctx context.Context, chatID string, isDM bool, limit int) (int, error) {
			select {
			case oldStarted <- struct{}{}:
			default:
			}
			go func() {
				<-ctx.Done()
				select {
				case oldStopTriggered <- struct{}{}:
				default:
				}
			}()
			<-oldRelease
			return 0, nil
		},
	}
	oldWorker := NewKnowledgeWorker(oldProv, KnowledgeWorkerConfig{
		Interval:    10 * time.Millisecond,
		Concurrency: 1,
	})
	oldWorker.Start(lifecycleCtx)
	<-oldStarted
	gw.SetKnowledgeWorker(oldWorker)

	newWorker := NewKnowledgeWorker(&mockKnowledgeTargetProvider{}, KnowledgeWorkerConfig{
		Interval:    10 * time.Millisecond,
		Concurrency: 1,
	})

	setDone := make(chan struct{})
	go func() {
		defer close(setDone)
		gw.SetKnowledgeWorker(newWorker)
	}()

	// While SetKnowledgeWorker is deterministically blocked stopping oldWorker, trigger Gateway.Stop()
	<-oldStopTriggered
	gw.Stop()

	// Unblock oldWorker so setter can finish
	close(oldRelease)

	select {
	case <-setDone:
	case <-time.After(2 * time.Second):
		t.Fatal("SetKnowledgeWorker timed out")
	}

	// newWorker MUST NOT be running because gateway was stopped!
	assert.False(t, newWorker.Running(), "new worker must not have been started after gateway stop")
}

func TestGateway_Start_RejectsConcurrentOrReentrant(t *testing.T) {
	tempDir := t.TempDir()
	cfg := &config.Config{DataDir: tempDir}
	gw := NewGateway(cfg, nil)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	gw.mu.Lock()
	gw.running = true
	gw.lifecycleCtx = ctx
	gw.mu.Unlock()
	defer gw.Stop()

	err := gw.Start(ctx)
	assert.ErrorContains(t, err, "gateway is already running")
}

func TestGateway_KnowledgeWorker_ConcurrentReplacement(t *testing.T) {
	tempDir := t.TempDir()
	cfg := &config.Config{DataDir: tempDir}
	gw := NewGateway(cfg, nil)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	gw.mu.Lock()
	gw.running = true
	gw.lifecycleCtx = ctx
	gw.mu.Unlock()
	defer gw.Stop()

	const count = 8
	workers := make([]*KnowledgeWorker, count)
	for i := 0; i < count; i++ {
		workers[i] = NewKnowledgeWorker(&mockKnowledgeTargetProvider{}, KnowledgeWorkerConfig{
			Interval:    50 * time.Millisecond,
			Concurrency: 1,
		})
	}

	var wg sync.WaitGroup
	wg.Add(count)
	for i := 0; i < count; i++ {
		go func(idx int) {
			defer wg.Done()
			gw.SetKnowledgeWorker(workers[idx])
		}(i)
	}
	wg.Wait()

	active := gw.KnowledgeWorker()
	require.NotNil(t, active)
	assert.True(t, active.Running(), "currently installed worker should be running")

	var runningCount int
	for _, w := range workers {
		if w.Running() {
			runningCount++
		}
	}
	assert.Equal(t, 1, runningCount, "exactly one worker should be running among concurrent replacements")
}

func TestGateway_KnowledgeWorker_ReplacementAfterContextCancellation(t *testing.T) {
	tempDir := t.TempDir()
	cfg := &config.Config{DataDir: tempDir}
	gw := NewGateway(cfg, nil)

	ctx, cancel := context.WithCancel(context.Background())
	gw.mu.Lock()
	gw.running = true
	gw.lifecycleCtx = ctx
	gw.mu.Unlock()

	// Cancel context to simulate gateway termination
	cancel()
	gw.Stop()

	newWorker := NewKnowledgeWorker(&mockKnowledgeTargetProvider{}, KnowledgeWorkerConfig{
		Interval:    10 * time.Millisecond,
		Concurrency: 1,
	})

	gw.SetKnowledgeWorker(newWorker)
	assert.False(t, newWorker.Running(), "replacement worker should not start after gateway context cancellation/stop")
}
