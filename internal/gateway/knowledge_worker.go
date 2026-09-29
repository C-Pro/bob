package gateway

import (
	"context"
	"errors"
	"log/slog"
	"sync"
	"time"
)

// KnowledgeWorkerConfig configures the background knowledge reconciliation engine.
type KnowledgeWorkerConfig struct {
	Interval    time.Duration
	Concurrency int
	ChatTimeout time.Duration
}

// DefaultKnowledgeWorkerConfig returns default settings for the background knowledge worker.
func DefaultKnowledgeWorkerConfig() KnowledgeWorkerConfig {
	return KnowledgeWorkerConfig{
		Interval:    1 * time.Minute,
		Concurrency: 4,
		ChatTimeout: 30 * time.Second,
	}
}

// KnowledgeTargetProvider defines the provider contract needed by KnowledgeWorker.
type KnowledgeTargetProvider interface {
	DiscoverKnowledgeTargets(ctx context.Context) []string
	ReconcileChat(ctx context.Context, chatID string, isDM bool, limit int) (int, error)
}

type workerRun struct {
	cancel context.CancelFunc
	done   chan struct{}
}

// KnowledgeWorker periodically scans and reconciles pending index deletions and unindexed candidates
// across all discovered and active DM knowledge stores.
type KnowledgeWorker struct {
	provider KnowledgeTargetProvider
	cfg      KnowledgeWorkerConfig

	mu     sync.Mutex
	curRun *workerRun
}

// NewKnowledgeWorker creates a new lifecycle-managed KnowledgeWorker.
func NewKnowledgeWorker(provider KnowledgeTargetProvider, cfg KnowledgeWorkerConfig) *KnowledgeWorker {
	if cfg.Interval <= 0 {
		cfg.Interval = 1 * time.Minute
	}
	if cfg.Concurrency <= 0 {
		cfg.Concurrency = 4
	}
	if cfg.ChatTimeout <= 0 {
		cfg.ChatTimeout = 30 * time.Second
	}
	return &KnowledgeWorker{
		provider: provider,
		cfg:      cfg,
	}
}

// Running reports whether a worker generation is currently active.
func (w *KnowledgeWorker) Running() bool {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.curRun != nil
}

// Start launches the background reconciliation loop. It immediately executes an initial pass,
// then runs periodically on cfg.Interval until Stop is called or the parent context ends.
func (w *KnowledgeWorker) Start(ctx context.Context) {
	w.mu.Lock()
	if w.curRun != nil {
		w.mu.Unlock()
		return
	}
	workerCtx, cancel := context.WithCancel(ctx)
	run := &workerRun{
		cancel: cancel,
		done:   make(chan struct{}),
	}
	w.curRun = run
	go w.runLoop(workerCtx, run)
	w.mu.Unlock()
}

// Stop gracefully shuts down the background worker and waits for all in-flight chat reconciliations to complete.
func (w *KnowledgeWorker) Stop() {
	w.mu.Lock()
	run := w.curRun
	if run == nil {
		w.mu.Unlock()
		return
	}
	run.cancel()
	w.mu.Unlock()

	<-run.done
}

func (w *KnowledgeWorker) runLoop(ctx context.Context, run *workerRun) {
	defer func() {
		w.mu.Lock()
		if w.curRun == run {
			w.curRun = nil
		}
		w.mu.Unlock()
		close(run.done)
	}()

	// 1. Immediate startup pass
	w.runPass(ctx)

	ticker := time.NewTicker(w.cfg.Interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			w.runPass(ctx)
		}
	}
}

// RunPass executes a single reconciliation pass across all target DM chats.
func (w *KnowledgeWorker) RunPass(ctx context.Context) {
	w.runPass(ctx)
}

func (w *KnowledgeWorker) runPass(ctx context.Context) {
	if w.provider == nil {
		return
	}

	targets := w.provider.DiscoverKnowledgeTargets(ctx)
	if len(targets) == 0 {
		return
	}

	sem := make(chan struct{}, w.cfg.Concurrency)
	var passWg sync.WaitGroup

passLoop:
	for _, chatID := range targets {
		select {
		case <-ctx.Done():
			break passLoop
		case sem <- struct{}{}:
		}

		passWg.Add(1)
		go func(cid string) {
			defer func() {
				<-sem
				passWg.Done()
			}()

			chatCtx, cancel := context.WithTimeout(ctx, w.cfg.ChatTimeout)
			defer cancel()

			if _, err := w.provider.ReconcileChat(chatCtx, cid, true, 100); err != nil {
				if !errors.Is(err, context.Canceled) {
					slog.Warn("background knowledge reconciliation failed for chat", "chat_id", cid, "error", err)
				}
			}
		}(chatID)
	}

	passWg.Wait()
}
