package scheduler

import (
	"context"
	"errors"
	"log/slog"
	"sync"
	"time"
)

// StoreProvider provides access to scheduler stores across active chats.
type StoreProvider interface {
	ActiveSchedulerStores(ctx context.Context) ([]*Store, error)
}

// StaticStoreProvider provides a static list of Stores (useful for unit tests).
type StaticStoreProvider struct {
	stores []*Store
}

// NewStaticStoreProvider creates a new StaticStoreProvider.
func NewStaticStoreProvider(stores ...*Store) *StaticStoreProvider {
	return &StaticStoreProvider{stores: stores}
}

// ActiveSchedulerStores returns the static list of stores.
func (p *StaticStoreProvider) ActiveSchedulerStores(ctx context.Context) ([]*Store, error) {
	return p.stores, nil
}

// Invoker executes a triggered schedule.
type Invoker interface {
	ExecuteSchedule(ctx context.Context, sched *Schedule) error
}

// InvokerFunc is an adapter allowing a function to be used as an Invoker.
type InvokerFunc func(ctx context.Context, sched *Schedule) error

// ExecuteSchedule calls f(ctx, sched).
func (f InvokerFunc) ExecuteSchedule(ctx context.Context, sched *Schedule) error {
	return f(ctx, sched)
}

// EngineOption configures the scheduler Engine.
type EngineOption func(*Engine)

// WithPollInterval sets the polling frequency of the scheduler.
func WithPollInterval(d time.Duration) EngineOption {
	return func(e *Engine) {
		if d > 0 {
			e.pollInterval = d
		}
	}
}

// WithClaimLockDuration sets how long a claimed task is locked in CAS before becoming eligible again.
func WithClaimLockDuration(d time.Duration) EngineOption {
	return func(e *Engine) {
		if d > 0 {
			e.claimLockDuration = d
		}
	}
}

// WithDowntimeCutoff sets the maximum overdue duration before anti-burst skipping triggers.
func WithDowntimeCutoff(d time.Duration) EngineOption {
	return func(e *Engine) {
		if d > 0 {
			e.downtimeCutoff = d
		}
	}
}

// WithConcurrency sets the maximum number of claimed schedules per tick.
func WithConcurrency(c int) EngineOption {
	return func(e *Engine) {
		if c > 0 {
			e.concurrency = c
		}
	}
}

// Engine orchestrates periodic polling, atomic claiming, downtime recovery, and execution dispatch for task schedules.
type Engine struct {
	provider          StoreProvider
	invoker           Invoker
	pollInterval      time.Duration
	claimLockDuration time.Duration
	downtimeCutoff    time.Duration
	concurrency       int
	running           map[string]context.CancelFunc
	mu                sync.Mutex
	stopCh            chan struct{}
	wg                sync.WaitGroup
	started           bool
	stopped           bool
}

// NewEngine initializes a new scheduler Engine.
func NewEngine(provider StoreProvider, invoker Invoker, opts ...EngineOption) *Engine {
	e := &Engine{
		provider:          provider,
		invoker:           invoker,
		pollInterval:      5 * time.Second,
		claimLockDuration: 300 * time.Second,
		downtimeCutoff:    15 * time.Minute,
		concurrency:       10,
		running:           make(map[string]context.CancelFunc),
		stopCh:            make(chan struct{}),
	}
	for _, opt := range opts {
		opt(e)
	}
	return e
}

// Start begins the periodic scheduler polling loop.
func (e *Engine) Start(ctx context.Context) error {
	e.mu.Lock()
	if e.started && !e.stopped {
		e.mu.Unlock()
		return nil
	}
	e.started = true
	e.stopped = false
	e.stopCh = make(chan struct{})
	e.wg.Add(1)
	e.mu.Unlock()

	go e.loop(ctx)
	return nil
}

// Stop stops the engine and waits for in-flight executions to complete or cancel.
func (e *Engine) Stop() {
	e.mu.Lock()
	if !e.started || e.stopped {
		e.mu.Unlock()
		return
	}
	e.stopped = true
	select {
	case <-e.stopCh:
		// already stopped
	default:
		close(e.stopCh)
	}

	for _, cancel := range e.running {
		cancel()
	}
	e.mu.Unlock()

	e.wg.Wait()
}

func (e *Engine) loop(ctx context.Context) {
	defer e.wg.Done()
	ticker := time.NewTicker(e.pollInterval)
	defer ticker.Stop()

	// Initial poll immediately upon startup
	e.poll(ctx)

	for {
		select {
		case <-ctx.Done():
			return
		case <-e.stopCh:
			return
		case <-ticker.C:
			e.poll(ctx)
		}
	}
}

// PollOnce executes a single polling cycle. Useful for deterministic unit testing.
func (e *Engine) PollOnce(ctx context.Context) {
	e.poll(ctx)
}

func (e *Engine) poll(ctx context.Context) {
	if e.provider == nil {
		return
	}

	stores, err := e.provider.ActiveSchedulerStores(ctx)
	if err != nil {
		slog.Warn("failed to fetch active scheduler stores", "error", err)
		return
	}

	now := time.Now()
	nowUnix := now.Unix()
	claimUntil := nowUnix + int64(e.claimLockDuration.Seconds())

	for _, store := range stores {
		select {
		case <-ctx.Done():
			return
		case <-e.stopCh:
			return
		default:
		}

		claimed, err := store.ClaimDueSchedules(ctx, nowUnix, claimUntil, e.concurrency)
		if err != nil {
			slog.Warn("failed to claim due schedules", "error", err)
			continue
		}

		for _, sched := range claimed {
			e.handleClaimedSchedule(ctx, store, sched, now)
		}
	}
}

func (e *Engine) handleClaimedSchedule(ctx context.Context, store *Store, sched *Schedule, now time.Time) {
	overdue := now.Unix() - sched.NextRunAt

	if overdue >= int64(e.downtimeCutoff.Seconds()) {
		// Extended downtime: skip missed runs, increment missed_count, recalculate next future occurrence
		nextTime, shouldEnd, err := CalculateNextRun(sched, now)
		if err != nil {
			slog.Error("failed to calculate next run after downtime skip", "name", sched.Name, "error", err)
			shouldEnd = true
		}

		missedPeriods := 1
		if sched.ScheduleType == ScheduleTypeInterval && sched.IntervalSeconds > 0 {
			missedPeriods = int(overdue / int64(sched.IntervalSeconds))
			if missedPeriods < 1 {
				missedPeriods = 1
			}
		} else if sched.ScheduleType == ScheduleTypeOnce {
			missedPeriods = 1
			shouldEnd = true
		}

		slog.Info("schedule overdue beyond downtime threshold; skipping missed runs",
			"name", sched.Name,
			"overdue_seconds", overdue,
			"missed_periods", missedPeriods,
		)

		writeCtx, writeCancel := context.WithTimeout(context.Background(), 5*time.Second)
		if err := store.RecordExecutionResult(writeCtx, sched.ID, "SKIPPED_DOWNTIME", nextTime.Unix(), missedPeriods, shouldEnd); err != nil {
			slog.Error("failed to record execution result after downtime skip", "schedule_id", sched.ID, "error", err)
		}
		writeCancel()
		return
	}

	// Overdue < downtimeCutoff: perform execution
	e.mu.Lock()
	if e.stopped {
		e.mu.Unlock()
		return
	}
	e.wg.Add(1)
	e.mu.Unlock()

	go func(s *Schedule) {
		defer e.wg.Done()
		e.execute(ctx, store, s)
	}(sched)
}

// ErrChatBusy is returned when a chat execution lock could not be acquired within the timeout.
var ErrChatBusy = errors.New("chat execution lock timed out")

func (e *Engine) execute(ctx context.Context, store *Store, sched *Schedule) {
	timeout := time.Duration(sched.RunTimeoutSeconds) * time.Second
	if timeout <= 0 {
		timeout = 5 * time.Minute
	}

	e.mu.Lock()
	if _, running := e.running[sched.ID]; running {
		e.mu.Unlock()
		return
	}
	execCtx, cancel := context.WithTimeout(ctx, timeout)
	e.running[sched.ID] = cancel
	e.mu.Unlock()

	defer func() {
		cancel()
		e.mu.Lock()
		delete(e.running, sched.ID)
		e.mu.Unlock()
	}()

	var execErr error
	if e.invoker != nil {
		execErr = e.invoker.ExecuteSchedule(execCtx, sched)
	}

	lastStatus := "SUCCESS"
	missedIncrement := 0
	if errors.Is(execErr, ErrChatBusy) {
		lastStatus = "SKIPPED_BUSY"
		missedIncrement = 1
	} else if execErr != nil {
		lastStatus = "FAILED: " + execErr.Error()
		runes := []rune(lastStatus)
		if len(runes) > 64 {
			lastStatus = string(runes[:64])
		}
	}

	now := time.Now()
	nextTime, shouldEnd, calcErr := CalculateNextRun(sched, now)
	if calcErr != nil || sched.ScheduleType == ScheduleTypeOnce || (sched.MaxRuns > 0 && sched.RunCount+1 >= sched.MaxRuns) {
		shouldEnd = true
	}

	writeCtx, writeCancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer writeCancel()
	if err := store.RecordExecutionResult(writeCtx, sched.ID, lastStatus, nextTime.Unix(), missedIncrement, shouldEnd); err != nil {
		slog.Error("failed to record execution result", "schedule_id", sched.ID, "error", err)
	}
}
