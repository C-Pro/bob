package fsm

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"net"
	"strings"
	"sync"
	"time"

	openai "github.com/sashabaranov/go-openai"
)

// ToolInvoker defines an interface for executing a single tool call.
type ToolInvoker interface {
	Execute(ctx context.Context, name string, argsJSON string) (string, error)
}

// ToolInvokerFunc is an adapter to allow the use of ordinary functions as ToolInvoker.
type ToolInvokerFunc func(ctx context.Context, name string, argsJSON string) (string, error)

// Execute calls f(ctx, name, argsJSON).
func (f ToolInvokerFunc) Execute(ctx context.Context, name string, argsJSON string) (string, error) {
	return f(ctx, name, argsJSON)
}

// IsReadOnlyTool reports whether a tool is pure read-only and side-effect free.
func IsReadOnlyTool(name string) bool {
	switch strings.TrimSpace(name) {
	case "web_search", "web_fetch", "recall_memory":
		return true
	default:
		return false
	}
}

// ClassifyExecutionMode determines whether a slice of OpenAI tool calls can be executed in parallel.
// If all tools are read-only, it returns ExecutionModeParallel.
// If any tool performs mutations or side-effects, it returns ExecutionModeSequential.
func ClassifyExecutionMode(tools []openai.ToolCall) ExecutionMode {
	if len(tools) <= 1 {
		return ExecutionModeSequential
	}
	for _, tc := range tools {
		if !IsReadOnlyTool(tc.Function.Name) {
			return ExecutionModeSequential
		}
	}
	return ExecutionModeParallel
}

// ClassifyStepExecutionMode inspects a slice of FSMSteps and determines the batch execution mode.
func ClassifyStepExecutionMode(steps []*FSMStep) ExecutionMode {
	if len(steps) <= 1 {
		return ExecutionModeSequential
	}
	for _, s := range steps {
		if !IsReadOnlyTool(s.ToolName) {
			return ExecutionModeSequential
		}
	}
	return ExecutionModeParallel
}

// RetryableError is an interface implemented by errors that report whether the operation can be retried.
type RetryableError interface {
	error
	IsRetryable() bool
}

// IsTransientError reports whether an error from a tool invocation is transient and eligible for retry.
func IsTransientError(err error) bool {
	if err == nil {
		return false
	}
	var retryable RetryableError
	if errors.As(err, &retryable) {
		return retryable.IsRetryable()
	}

	var netErr net.Error
	if errors.As(err, &netErr) && netErr.Timeout() {
		return true
	}

	return false
}

// RetryConfig defines the exponential backoff parameters for retrying transient step errors.
type RetryConfig struct {
	InitialBackoff time.Duration
	MaxBackoff     time.Duration
	Multiplier     float64
}

// DefaultRetryConfig provides standard retry backoff parameters.
var DefaultRetryConfig = RetryConfig{
	InitialBackoff: 100 * time.Millisecond,
	MaxBackoff:     2 * time.Second,
	Multiplier:     2.0,
}

// CalculateBackoff computes the exponential backoff duration for a given attempt number.
func CalculateBackoff(attempt int, cfg RetryConfig) time.Duration {
	if attempt <= 1 {
		return cfg.InitialBackoff
	}
	multiplier := math.Pow(cfg.Multiplier, float64(attempt-1))
	delay := time.Duration(float64(cfg.InitialBackoff) * multiplier)
	if cfg.MaxBackoff > 0 && delay > cfg.MaxBackoff {
		return cfg.MaxBackoff
	}
	return delay
}

// StepExecutorOption configures a StepExecutor.
type StepExecutorOption func(*StepExecutor)

// WithMaxWorkers sets the maximum concurrent workers for parallel step execution.
func WithMaxWorkers(n int) StepExecutorOption {
	return func(e *StepExecutor) {
		if n > 0 {
			e.maxWorkers = n
		}
	}
}

// WithRetryConfig overrides the retry backoff configuration.
func WithRetryConfig(cfg RetryConfig) StepExecutorOption {
	return func(e *StepExecutor) {
		e.retryConfig = cfg
	}
}

// StepStore defines the persistence operations required by StepExecutor.
type StepStore interface {
	UpdateStep(ctx context.Context, step *FSMStep) error
}

// StepExecutor executes FSM steps sequentially or concurrently with per-step timeouts and transient retries.
type StepExecutor struct {
	invoker     ToolInvoker
	store       StepStore
	maxWorkers  int
	retryConfig RetryConfig
}

// NewStepExecutor creates a new StepExecutor.
func NewStepExecutor(invoker ToolInvoker, store StepStore, opts ...StepExecutorOption) *StepExecutor {
	e := &StepExecutor{
		invoker:     invoker,
		store:       store,
		maxWorkers:  4,
		retryConfig: DefaultRetryConfig,
	}
	for _, opt := range opts {
		opt(e)
	}
	return e
}

// updateStep persists the given step to the store if configured, returning any storage error.
func (e *StepExecutor) updateStep(ctx context.Context, step *FSMStep) error {
	if e.store == nil {
		return nil
	}
	if err := e.store.UpdateStep(ctx, step); err != nil {
		return fmt.Errorf("fsm store update step %s: %w", step.ID, err)
	}
	return nil
}

// Execute executes a slice of FSM steps according to their execution mode (parallel or sequential).
// Status and execution metrics are updated in-place and persisted to the store if configured.
func (e *StepExecutor) Execute(ctx context.Context, steps []*FSMStep) error {
	if len(steps) == 0 {
		return nil
	}
	if e.invoker == nil {
		return fmt.Errorf("tool invoker is not configured")
	}

	mode := ClassifyStepExecutionMode(steps)
	if mode == ExecutionModeParallel {
		return e.executeParallel(ctx, steps)
	}
	return e.executeSequential(ctx, steps)
}

// ExecuteBatch is a convenience wrapper that accepts and returns a slice of value FSMSteps.
func (e *StepExecutor) ExecuteBatch(ctx context.Context, steps []FSMStep) ([]FSMStep, error) {
	if len(steps) == 0 {
		return nil, nil
	}
	ptrs := make([]*FSMStep, len(steps))
	for i := range steps {
		ptrs[i] = &steps[i]
	}
	if err := e.Execute(ctx, ptrs); err != nil {
		return nil, err
	}
	result := make([]FSMStep, len(ptrs))
	for i, p := range ptrs {
		result[i] = *p
	}
	return result, nil
}

func (e *StepExecutor) executeParallel(ctx context.Context, steps []*FSMStep) error {
	workers := e.maxWorkers
	if workers <= 0 {
		workers = 4
	}
	if len(steps) < workers {
		workers = len(steps)
	}

	sem := make(chan struct{}, workers)
	var wg sync.WaitGroup
	var errMu sync.Mutex
	var execErr error

	recordErr := func(err error) {
		if err != nil {
			errMu.Lock()
			execErr = errors.Join(execErr, err)
			errMu.Unlock()
		}
	}

	for _, step := range steps {
		step.ExecutionMode = ExecutionModeParallel
		wg.Add(1)
		go func(s *FSMStep) {
			defer wg.Done()
			select {
			case sem <- struct{}{}:
				defer func() { <-sem }()
				if err := e.executeSingleStep(ctx, s); err != nil {
					recordErr(err)
				}
			case <-ctx.Done():
				now := time.Now().Unix()
				s.Status = StepStatusFailed
				s.ErrorText = ctx.Err().Error()
				s.CompletedAt = &now
				if err := e.updateStep(ctx, s); err != nil {
					recordErr(err)
				}
				recordErr(ctx.Err())
			}
		}(step)
	}

	wg.Wait()
	if execErr != nil {
		return execErr
	}
	return ctx.Err()
}

func (e *StepExecutor) executeSequential(ctx context.Context, steps []*FSMStep) error {
	for i, step := range steps {
		step.ExecutionMode = ExecutionModeSequential

		if ctx.Err() != nil {
			now := time.Now().Unix()
			step.Status = StepStatusFailed
			step.ErrorText = ctx.Err().Error()
			step.CompletedAt = &now
			if err := e.updateStep(ctx, step); err != nil {
				return errors.Join(ctx.Err(), err)
			}
			return ctx.Err()
		}

		err := e.executeSingleStep(ctx, step)
		// If a sequential step failed or timed out, abort remaining dependent steps
		if step.Status == StepStatusFailed || step.Status == StepStatusTimedOut {
			skipErr := e.skipRemainingSteps(ctx, steps[i+1:], step.ID)
			if skipErr != nil {
				return errors.Join(err, skipErr)
			}
			return err
		}
		if err != nil {
			return err
		}
	}

	return ctx.Err()
}

func (e *StepExecutor) skipRemainingSteps(ctx context.Context, steps []*FSMStep, failedStepID string) error {
	now := time.Now().Unix()
	var errs error
	for _, s := range steps {
		if s.Status == StepStatusCompleted || s.Status == StepStatusSkipped {
			continue
		}
		s.Status = StepStatusSkipped
		s.ErrorText = fmt.Sprintf("skipped due to failure in step %s", failedStepID)
		s.CompletedAt = &now
		if err := e.updateStep(ctx, s); err != nil {
			errs = errors.Join(errs, err)
		}
	}
	return errs
}

func (e *StepExecutor) executeSingleStep(ctx context.Context, step *FSMStep) error {
	if step.Status == StepStatusCompleted || step.Status == StepStatusSkipped {
		return nil
	}

	timeoutSec := step.TimeoutSeconds
	if timeoutSec <= 0 {
		timeoutSec = 30
		step.TimeoutSeconds = timeoutSec
	}
	timeout := time.Duration(timeoutSec) * time.Second

	maxAttempts := step.MaxAttempts
	if maxAttempts <= 0 {
		maxAttempts = 3
		step.MaxAttempts = maxAttempts
	}

	now := time.Now().Unix()
	if step.StartedAt == nil {
		step.StartedAt = &now
	}
	step.Status = StepStatusRunning
	if err := e.updateStep(ctx, step); err != nil {
		return err
	}

	for {
		step.Attempt++
		stepCtx, cancel := context.WithTimeout(ctx, timeout)
		res, err := e.invoker.Execute(stepCtx, step.ToolName, step.ArgsJSON)
		cancel()

		completedTime := time.Now().Unix()
		if err == nil {
			step.Status = StepStatusCompleted
			step.ResultJSON = res
			step.ErrorText = ""
			step.CompletedAt = &completedTime
			if updateErr := e.updateStep(ctx, step); updateErr != nil {
				return updateErr
			}
			return nil
		}

		// Step timed out
		if errors.Is(stepCtx.Err(), context.DeadlineExceeded) || errors.Is(err, context.DeadlineExceeded) {
			step.Status = StepStatusTimedOut
			step.ErrorText = fmt.Sprintf("tool execution timed out after %ds", step.TimeoutSeconds)
			step.CompletedAt = &completedTime
			updateErr := e.updateStep(ctx, step)
			return errors.Join(err, updateErr)
		}

		// Parent context canceled
		if ctx.Err() != nil {
			step.Status = StepStatusFailed
			step.ErrorText = ctx.Err().Error()
			step.CompletedAt = &completedTime
			updateErr := e.updateStep(ctx, step)
			return errors.Join(ctx.Err(), updateErr)
		}

		// Transient error and retries remain
		if IsTransientError(err) && step.Attempt < maxAttempts {
			backoff := CalculateBackoff(step.Attempt, e.retryConfig)
			slog.Warn("retrying transient tool execution failure",
				"tool", step.ToolName,
				"step_id", step.ID,
				"attempt", step.Attempt,
				"max_attempts", maxAttempts,
				"backoff", backoff,
				"error", err,
			)
			select {
			case <-ctx.Done():
				step.Status = StepStatusFailed
				step.ErrorText = ctx.Err().Error()
				step.CompletedAt = &completedTime
				updateErr := e.updateStep(ctx, step)
				return errors.Join(ctx.Err(), updateErr)
			case <-time.After(backoff):
				continue
			}
		}

		// Non-transient error or attempts exhausted
		step.Status = StepStatusFailed
		step.ErrorText = err.Error()
		step.CompletedAt = &completedTime
		updateErr := e.updateStep(ctx, step)
		return errors.Join(err, updateErr)
	}
}

// HasFailedStep reports whether any step in the slice has status FAILED or TIMED_OUT.
func HasFailedStep(steps []*FSMStep) bool {
	for _, s := range steps {
		if s.Status == StepStatusFailed || s.Status == StepStatusTimedOut {
			return true
		}
	}
	return false
}
