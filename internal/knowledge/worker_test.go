package knowledge

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

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

func TestKnowledgeWorker_LifecycleAndGracefulStop(t *testing.T) {
	mockProv := &mockKnowledgeTargetProvider{
		targets: []string{"chat_lifecycle"},
	}
	worker := NewKnowledgeWorker(mockProv, KnowledgeWorkerConfig{
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
	var startOnce, stopOnce sync.Once

	mockProv := &mockKnowledgeTargetProvider{
		targets: []string{"chat_gen"},
		reconcileFn: func(ctx context.Context, chatID string, isDM bool, limit int) (int, error) {
			startOnce.Do(func() {
				close(reconcileStarted)
			})
			go func() {
				<-ctx.Done()
				stopOnce.Do(func() {
					close(stopTriggered)
				})
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
	select {
	case <-reconcileStarted:
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for reconcileStarted")
	}

	// Launch Stop in a goroutine
	stopDone := make(chan struct{})
	go func() {
		defer close(stopDone)
		worker.Stop()
	}()

	// While Stop is waiting for generation 1 to drain, attempt Start.
	// Must be ignored because generation 1 is still draining.
	select {
	case <-stopTriggered:
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for stopTriggered")
	}
	worker.Start(ctx)

	// Unblock generation 1
	close(blockReconcile)

	// Stop completes
	select {
	case <-stopDone:
	case <-time.After(5 * time.Second):
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
