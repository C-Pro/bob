package gateway

import (
	"context"
	"sync"
	"testing"
	"time"

	"bob/internal/config"

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
	var oldStartOnce, oldStopOnce sync.Once

	oldProv := &mockKnowledgeTargetProvider{
		targets: []string{"c1"},
		reconcileFn: func(ctx context.Context, chatID string, isDM bool, limit int) (int, error) {
			oldStartOnce.Do(func() {
				close(oldStarted)
			})
			go func() {
				<-ctx.Done()
				oldStopOnce.Do(func() {
					close(oldStopTriggered)
				})
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
	select {
	case <-oldStarted:
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for oldWorker to start")
	}
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
	select {
	case <-oldStopTriggered:
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for oldStopTriggered")
	}
	gw.Stop()

	// Unblock oldWorker so setter can finish
	close(oldRelease)

	select {
	case <-setDone:
	case <-time.After(5 * time.Second):
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
