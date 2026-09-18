package fsm

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	openai "github.com/sashabaranov/go-openai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestClassifier(t *testing.T) {
	// IsReadOnlyTool
	assert.True(t, IsReadOnlyTool("web_search"))
	assert.True(t, IsReadOnlyTool("web_fetch"))
	assert.True(t, IsReadOnlyTool("recall_memory"))
	assert.False(t, IsReadOnlyTool("sandbox_exec"))
	assert.False(t, IsReadOnlyTool("sandbox_request"))
	assert.False(t, IsReadOnlyTool("sandbox_destroy"))
	assert.False(t, IsReadOnlyTool("unknown_tool"))

	// ClassifyExecutionMode
	assert.Equal(t, ExecutionModeSequential, ClassifyExecutionMode(nil))
	assert.Equal(t, ExecutionModeSequential, ClassifyExecutionMode([]openai.ToolCall{
		{Function: openai.FunctionCall{Name: "web_search"}},
	}))

	parallelTools := []openai.ToolCall{
		{Function: openai.FunctionCall{Name: "web_search"}},
		{Function: openai.FunctionCall{Name: "web_fetch"}},
	}
	assert.Equal(t, ExecutionModeParallel, ClassifyExecutionMode(parallelTools))

	mixedTools := []openai.ToolCall{
		{Function: openai.FunctionCall{Name: "web_search"}},
		{Function: openai.FunctionCall{Name: "sandbox_exec"}},
	}
	assert.Equal(t, ExecutionModeSequential, ClassifyExecutionMode(mixedTools))

	// ClassifyStepExecutionMode
	parallelSteps := []*FSMStep{
		{ToolName: "web_search"},
		{ToolName: "recall_memory"},
	}
	assert.Equal(t, ExecutionModeParallel, ClassifyStepExecutionMode(parallelSteps))

	mixedSteps := []*FSMStep{
		{ToolName: "web_search"},
		{ToolName: "sandbox_exec"},
	}
	assert.Equal(t, ExecutionModeSequential, ClassifyStepExecutionMode(mixedSteps))

	singleStep := []*FSMStep{
		{ToolName: "web_search"},
	}
	assert.Equal(t, ExecutionModeSequential, ClassifyStepExecutionMode(singleStep))
}

type mockRetryableError struct {
	msg       string
	retryable bool
}

func (m *mockRetryableError) Error() string {
	return m.msg
}

func (m *mockRetryableError) IsRetryable() bool {
	return m.retryable
}

type mockNetTimeoutError struct{}

func (m *mockNetTimeoutError) Error() string   { return "i/o timeout" }
func (m *mockNetTimeoutError) Timeout() bool   { return true }
func (m *mockNetTimeoutError) Temporary() bool { return true }

func TestIsTransientError(t *testing.T) {
	assert.False(t, IsTransientError(nil))
	assert.False(t, IsTransientError(errors.New("invalid arguments")))
	assert.False(t, IsTransientError(errors.New("unknown tool: foo")))
	assert.False(t, IsTransientError(errors.New("command exited with code 1")))

	// RetryableError interface
	assert.True(t, IsTransientError(&mockRetryableError{msg: "rate limit", retryable: true}))
	assert.False(t, IsTransientError(&mockRetryableError{msg: "bad request", retryable: false}))

	// Wrapped RetryableError
	wrapped := fmt.Errorf("tool failed: %w", &mockRetryableError{msg: "503", retryable: true})
	assert.True(t, IsTransientError(wrapped))

	// Net error with timeout
	assert.True(t, IsTransientError(&mockNetTimeoutError{}))
}

func TestCalculateBackoff(t *testing.T) {
	cfg := RetryConfig{
		InitialBackoff: 50 * time.Millisecond,
		MaxBackoff:     300 * time.Millisecond,
		Multiplier:     2.0,
	}

	assert.Equal(t, 50*time.Millisecond, CalculateBackoff(1, cfg))
	assert.Equal(t, 100*time.Millisecond, CalculateBackoff(2, cfg))
	assert.Equal(t, 200*time.Millisecond, CalculateBackoff(3, cfg))
	assert.Equal(t, 300*time.Millisecond, CalculateBackoff(4, cfg)) // capped at MaxBackoff
}

func TestCalculateBackoff_Jitter(t *testing.T) {
	cfg := RetryConfig{
		InitialBackoff: 100 * time.Millisecond,
		MaxBackoff:     1 * time.Second,
		Multiplier:     2.0,
		Jitter:         1.0,
	}

	seen := make(map[time.Duration]bool)
	for i := 0; i < 50; i++ {
		b := CalculateBackoff(2, cfg)
		// With delay=200ms and equal jitter, backoff must be in [100ms, 200ms]
		assert.GreaterOrEqual(t, b, 100*time.Millisecond)
		assert.LessOrEqual(t, b, 200*time.Millisecond)
		seen[b] = true
	}
	// Assert randomness: across 50 calls we should see more than 1 distinct value
	assert.Greater(t, len(seen), 1)
}

func TestStepExecutor_ParallelExecutionAndConcurrencyLimit(t *testing.T) {
	var currentActive int32
	var maxObserved int32

	invoker := ToolInvokerFunc(func(ctx context.Context, name string, argsJSON string) (string, error) {
		active := atomic.AddInt32(&currentActive, 1)
		for {
			oldMax := atomic.LoadInt32(&maxObserved)
			if active <= oldMax || atomic.CompareAndSwapInt32(&maxObserved, oldMax, active) {
				break
			}
		}
		time.Sleep(30 * time.Millisecond)
		atomic.AddInt32(&currentActive, -1)
		return `{"ok": true}`, nil
	})

	fastRetry := RetryConfig{InitialBackoff: 5 * time.Millisecond, MaxBackoff: 20 * time.Millisecond, Multiplier: 2.0}
	executor := NewStepExecutor(invoker, WithMaxWorkers(2), WithRetryConfig(fastRetry))

	steps := []*FSMStep{
		{ID: "s1", ToolName: "web_search", TimeoutSeconds: 5},
		{ID: "s2", ToolName: "web_fetch", TimeoutSeconds: 5},
		{ID: "s3", ToolName: "recall_memory", TimeoutSeconds: 5},
		{ID: "s4", ToolName: "web_search", TimeoutSeconds: 5},
	}

	start := time.Now()
	err := executor.Execute(context.Background(), nil, steps)
	elapsed := time.Since(start)

	require.NoError(t, err)
	assert.False(t, HasFailedStep(steps))

	for _, s := range steps {
		assert.Equal(t, StepStatusCompleted, s.Status)
		assert.Equal(t, ExecutionModeParallel, s.ExecutionMode)
		assert.Equal(t, `{"ok": true}`, s.ResultJSON)
		assert.NotNil(t, s.StartedAt)
		assert.NotNil(t, s.CompletedAt)
	}

	// Max concurrency was restricted to 2 workers
	assert.LessOrEqual(t, maxObserved, int32(2))
	assert.GreaterOrEqual(t, maxObserved, int32(2))

	// 4 tasks taking 30ms each with concurrency 2 take ~60ms, well under sequential 120ms
	assert.Less(t, elapsed, 110*time.Millisecond)
}

func TestStepExecutor_SequentialExecutionOrderAndSkipOnFailure(t *testing.T) {
	var executedOrder []string
	var mu sync.Mutex

	invoker := ToolInvokerFunc(func(ctx context.Context, name string, argsJSON string) (string, error) {
		mu.Lock()
		executedOrder = append(executedOrder, name)
		mu.Unlock()

		if name == "step_fail" {
			return "", errors.New("command failed")
		}
		return `{"status":"ok"}`, nil
	})

	executor := NewStepExecutor(invoker)

	steps := []*FSMStep{
		{ID: "step_0", ToolName: "step_first", TimeoutSeconds: 5},
		{ID: "step_1", ToolName: "step_fail", TimeoutSeconds: 5},
		{ID: "step_2", ToolName: "step_subsequent", TimeoutSeconds: 5},
	}

	err := executor.Execute(context.Background(), nil, steps)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "command failed")
	assert.True(t, HasFailedStep(steps))

	// Verify executed order
	mu.Lock()
	assert.Equal(t, []string{"step_first", "step_fail"}, executedOrder)
	mu.Unlock()

	// Step 0 completed
	assert.Equal(t, StepStatusCompleted, steps[0].Status)
	assert.Equal(t, ExecutionModeSequential, steps[0].ExecutionMode)

	// Step 1 failed
	assert.Equal(t, StepStatusFailed, steps[1].Status)
	assert.Equal(t, "command failed", steps[1].ErrorText)

	// Step 2 was skipped due to failure in step 1
	assert.Equal(t, StepStatusSkipped, steps[2].Status)
	assert.Contains(t, steps[2].ErrorText, "skipped due to failure in step step_1")
}

func TestStepExecutor_PerStepTimeout(t *testing.T) {
	invoker := ToolInvokerFunc(func(ctx context.Context, name string, argsJSON string) (string, error) {
		select {
		case <-time.After(150 * time.Millisecond):
			return `{"slow": true}`, nil
		case <-ctx.Done():
			return "", ctx.Err()
		}
	})

	executor := NewStepExecutor(invoker)

	// Step timeout of 30ms simulated via parent deadline or step timeout
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()

	step := &FSMStep{
		ID:             "step_slow",
		ToolName:       "sandbox_exec",
		TimeoutSeconds: 1,
	}

	err := executor.Execute(ctx, nil, []*FSMStep{step})
	require.Error(t, err)
	assert.Equal(t, StepStatusTimedOut, step.Status)
	assert.Contains(t, step.ErrorText, "timed out")
}

func TestStepExecutor_ParentContextCanceled(t *testing.T) {
	invoker := ToolInvokerFunc(func(ctx context.Context, name string, argsJSON string) (string, error) {
		select {
		case <-time.After(150 * time.Millisecond):
			return `{"slow": true}`, nil
		case <-ctx.Done():
			return "", ctx.Err()
		}
	})

	executor := NewStepExecutor(invoker)

	ctx, cancel := context.WithCancel(context.Background())
	// Cancel parent context before or immediately during execution
	go func() {
		time.Sleep(10 * time.Millisecond)
		cancel()
	}()

	step := &FSMStep{
		ID:             "step_cancel",
		ToolName:       "sandbox_exec",
		TimeoutSeconds: 10,
	}

	err := executor.Execute(ctx, nil, []*FSMStep{step})
	require.Error(t, err)
	assert.Equal(t, StepStatusFailed, step.Status)
	assert.Contains(t, step.ErrorText, "context canceled")
}

func TestStepExecutor_TransientErrorRetries(t *testing.T) {
	var attempts int32

	invoker := ToolInvokerFunc(func(ctx context.Context, name string, argsJSON string) (string, error) {
		n := atomic.AddInt32(&attempts, 1)
		if n < 3 {
			return "", &mockRetryableError{msg: "rate limit exceeded", retryable: true}
		}
		return `{"recovered": true}`, nil
	})

	fastRetry := RetryConfig{InitialBackoff: 2 * time.Millisecond, MaxBackoff: 10 * time.Millisecond, Multiplier: 2.0}
	executor := NewStepExecutor(invoker, WithRetryConfig(fastRetry))

	step := &FSMStep{
		ID:          "step_retry",
		ToolName:    "web_search",
		MaxAttempts: 3,
	}

	err := executor.Execute(context.Background(), nil, []*FSMStep{step})
	require.NoError(t, err)
	assert.Equal(t, StepStatusCompleted, step.Status)
	assert.Equal(t, 3, step.Attempt)
	assert.Equal(t, `{"recovered": true}`, step.ResultJSON)
}

func TestStepExecutor_TransientErrorExhaustion(t *testing.T) {
	var attempts int32

	invoker := ToolInvokerFunc(func(ctx context.Context, name string, argsJSON string) (string, error) {
		atomic.AddInt32(&attempts, 1)
		return "", &mockRetryableError{msg: "503 Service Unavailable", retryable: true}
	})

	fastRetry := RetryConfig{InitialBackoff: 2 * time.Millisecond, MaxBackoff: 5 * time.Millisecond, Multiplier: 2.0}
	executor := NewStepExecutor(invoker, WithRetryConfig(fastRetry))

	step := &FSMStep{
		ID:          "step_exhaust",
		ToolName:    "web_search",
		MaxAttempts: 3,
	}

	err := executor.Execute(context.Background(), nil, []*FSMStep{step})
	require.Error(t, err)
	assert.Equal(t, StepStatusFailed, step.Status)
	assert.Equal(t, 3, step.Attempt)
	assert.Contains(t, step.ErrorText, "503 Service Unavailable")
}

func TestStepExecutor_StepTimeoutRetries(t *testing.T) {
	var attempts int32

	invoker := ToolInvokerFunc(func(ctx context.Context, name string, argsJSON string) (string, error) {
		n := atomic.AddInt32(&attempts, 1)
		if n < 3 {
			return "", context.DeadlineExceeded
		}
		return `{"recovered": true}`, nil
	})

	fastRetry := RetryConfig{InitialBackoff: 2 * time.Millisecond, MaxBackoff: 10 * time.Millisecond, Multiplier: 2.0}
	executor := NewStepExecutor(invoker, WithRetryConfig(fastRetry))

	step := &FSMStep{
		ID:             "step_timeout_retry",
		ToolName:       "web_search",
		TimeoutSeconds: 1,
		MaxAttempts:    3,
	}

	err := executor.Execute(context.Background(), nil, []*FSMStep{step})
	require.NoError(t, err)
	assert.Equal(t, StepStatusCompleted, step.Status)
	assert.Equal(t, 3, step.Attempt)
	assert.Equal(t, `{"recovered": true}`, step.ResultJSON)
}

func TestStepExecutor_StepTimeoutExhaustion(t *testing.T) {
	var attempts int32

	invoker := ToolInvokerFunc(func(ctx context.Context, name string, argsJSON string) (string, error) {
		atomic.AddInt32(&attempts, 1)
		return "", context.DeadlineExceeded
	})

	fastRetry := RetryConfig{InitialBackoff: 2 * time.Millisecond, MaxBackoff: 10 * time.Millisecond, Multiplier: 2.0}
	executor := NewStepExecutor(invoker, WithRetryConfig(fastRetry))

	step := &FSMStep{
		ID:             "step_timeout_exhaust",
		ToolName:       "web_search",
		TimeoutSeconds: 1,
		MaxAttempts:    3,
	}

	err := executor.Execute(context.Background(), nil, []*FSMStep{step})
	require.Error(t, err)
	assert.Equal(t, StepStatusTimedOut, step.Status)
	assert.Equal(t, 3, step.Attempt)
	assert.Contains(t, step.ErrorText, "timed out after 1s")
}

func TestStepExecutor_NonTransientErrorNoRetry(t *testing.T) {
	var attempts int32

	invoker := ToolInvokerFunc(func(ctx context.Context, name string, argsJSON string) (string, error) {
		atomic.AddInt32(&attempts, 1)
		return "", errors.New("invalid JSON arguments syntax")
	})

	executor := NewStepExecutor(invoker)

	step := &FSMStep{
		ID:          "step_nontransient",
		ToolName:    "web_search",
		MaxAttempts: 3,
	}

	err := executor.Execute(context.Background(), nil, []*FSMStep{step})
	require.Error(t, err)
	assert.Equal(t, StepStatusFailed, step.Status)
	assert.Equal(t, 1, step.Attempt) // Only 1 attempt, no useless retries
	assert.Equal(t, int32(1), atomic.LoadInt32(&attempts))
}

func TestStepExecutor_ExecuteBatchWrapper(t *testing.T) {
	invoker := ToolInvokerFunc(func(ctx context.Context, name string, argsJSON string) (string, error) {
		return fmt.Sprintf("result for %s", name), nil
	})

	executor := NewStepExecutor(invoker)

	steps := []FSMStep{
		{ID: "b1", ToolName: "web_search"},
		{ID: "b2", ToolName: "web_fetch"},
	}

	results, err := executor.ExecuteBatch(context.Background(), nil, steps)
	require.NoError(t, err)
	require.Len(t, results, 2)
	assert.Equal(t, StepStatusCompleted, results[0].Status)
	assert.Equal(t, "result for web_search", results[0].ResultJSON)
	assert.Equal(t, StepStatusCompleted, results[1].Status)
	assert.Equal(t, "result for web_fetch", results[1].ResultJSON)
}

func TestStepExecutor_WithStorePersistence(t *testing.T) {
	db := setupTestDB(t)
	store := NewStore(db)
	ctx := context.Background()

	run := &FSMRun{
		ID:          "run_exec_store",
		ChatID:      "c1",
		UserID:      "u1",
		Status:      RunStatusRunning,
		ContextJSON: "[]",
	}
	require.NoError(t, store.CreateRun(ctx, run))

	steps := []FSMStep{
		{
			ID:          "step_persisted_1",
			RunID:       run.ID,
			Iteration:   1,
			StepIndex:   0,
			ToolName:    "web_search",
			ArgsJSON:    `{"q":"go"}`,
			MaxAttempts: 3,
		},
		{
			ID:          "step_persisted_2",
			RunID:       run.ID,
			Iteration:   1,
			StepIndex:   1,
			ToolName:    "web_fetch",
			ArgsJSON:    `{"url":"https://go.dev"}`,
			MaxAttempts: 3,
		},
	}
	require.NoError(t, store.CreateSteps(ctx, steps))

	invoker := ToolInvokerFunc(func(ctx context.Context, name string, argsJSON string) (string, error) {
		return fmt.Sprintf("ok-%s", name), nil
	})

	executor := NewStepExecutor(invoker)

	stepPtrs := []*FSMStep{&steps[0], &steps[1]}
	err := executor.Execute(ctx, store, stepPtrs)
	require.NoError(t, err)

	// Fetch directly from SQLite database and verify state was persisted
	s1, err := store.GetStep(ctx, "step_persisted_1")
	require.NoError(t, err)
	assert.Equal(t, StepStatusCompleted, s1.Status)
	assert.Equal(t, "ok-web_search", s1.ResultJSON)
	assert.NotNil(t, s1.StartedAt)
	assert.NotNil(t, s1.CompletedAt)

	s2, err := store.GetStep(ctx, "step_persisted_2")
	require.NoError(t, err)
	assert.Equal(t, StepStatusCompleted, s2.Status)
	assert.Equal(t, "ok-web_fetch", s2.ResultJSON)
}

type mockFailingStore struct{}

func (m *mockFailingStore) UpdateStep(ctx context.Context, step *FSMStep) error {
	return errors.New("sqlite disk I/O error")
}

func TestStepExecutor_StoreUpdateFailurePropagated(t *testing.T) {
	invoker := ToolInvokerFunc(func(ctx context.Context, name string, argsJSON string) (string, error) {
		return `{"ok":true}`, nil
	})
	executor := NewStepExecutor(invoker)
	step := &FSMStep{
		ID:       "step_fail_store",
		ToolName: "web_search",
	}
	err := executor.Execute(context.Background(), &mockFailingStore{}, []*FSMStep{step})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "sqlite disk I/O error")
}

func TestStepExecutor_ParallelExecutionErrorPropagation(t *testing.T) {
	invoker := ToolInvokerFunc(func(ctx context.Context, name string, argsJSON string) (string, error) {
		if name == "web_search" {
			return "", errors.New("search execution failed")
		}
		return `{"ok":true}`, nil
	})
	executor := NewStepExecutor(invoker)
	steps := []*FSMStep{
		{ID: "s1", ToolName: "web_search"},
		{ID: "s2", ToolName: "web_fetch"},
	}
	err := executor.Execute(context.Background(), nil, steps)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "search execution failed")
}

func TestStepExecutor_SequentialStepFailureSkipsRemaining(t *testing.T) {
	invoker := ToolInvokerFunc(func(ctx context.Context, name string, argsJSON string) (string, error) {
		if name == "fail_tool" {
			return "", errors.New("tool execution exploded")
		}
		return `{"ok":true}`, nil
	})
	executor := NewStepExecutor(invoker)
	s1 := &FSMStep{ID: "s1", ToolName: "fail_tool", ExecutionMode: ExecutionModeSequential}
	s2 := &FSMStep{ID: "s2", ToolName: "subsequent_tool", ExecutionMode: ExecutionModeSequential}
	err := executor.Execute(context.Background(), nil, []*FSMStep{s1, s2})
	require.Error(t, err)
	assert.Equal(t, StepStatusFailed, s1.Status)
	assert.Equal(t, StepStatusSkipped, s2.Status)
	assert.Contains(t, s2.ErrorText, "skipped due to failure in step s1")
}

func TestStepExecutor_RecoveryInterruptedMutatingStepFailsClosed(t *testing.T) {
	invokerCalls := 0
	invoker := ToolInvokerFunc(func(ctx context.Context, name string, argsJSON string) (string, error) {
		invokerCalls++
		return `{"result":"done"}`, nil
	})
	executor := NewStepExecutor(invoker)

	step := &FSMStep{
		ID:            "step_mutating_running",
		ToolName:      "sandbox_exec",
		Status:        StepStatusRunning,
		ExecutionMode: ExecutionModeSequential,
	}

	err := executor.Execute(context.Background(), nil, []*FSMStep{step})
	require.NoError(t, err)
	assert.Equal(t, 0, invokerCalls, "mutating tool left running must not be re-invoked")
	assert.Equal(t, StepStatusFailed, step.Status)
	assert.Contains(t, step.ErrorText, "interrupted mid-execution; not retried (non-idempotent tool)")
	assert.NotNil(t, step.CompletedAt)
}

func TestStepExecutor_RecoveryInterruptedReadOnlyStepReexecutes(t *testing.T) {
	invokerCalls := 0
	invoker := ToolInvokerFunc(func(ctx context.Context, name string, argsJSON string) (string, error) {
		invokerCalls++
		return `{"result":"found"}`, nil
	})
	executor := NewStepExecutor(invoker)

	step := &FSMStep{
		ID:            "step_readonly_running",
		ToolName:      "web_search",
		Status:        StepStatusRunning,
		ExecutionMode: ExecutionModeSequential,
	}

	err := executor.Execute(context.Background(), nil, []*FSMStep{step})
	require.NoError(t, err)
	assert.Equal(t, 1, invokerCalls, "read-only tool left running must be re-executed")
	assert.Equal(t, StepStatusCompleted, step.Status)
	assert.Equal(t, `{"result":"found"}`, step.ResultJSON)
	assert.Empty(t, step.ErrorText)
	assert.NotNil(t, step.CompletedAt)
}

func TestStepExecutor_StepResultTruncation(t *testing.T) {
	oversized := make([]byte, 300*1024)
	for i := range oversized {
		oversized[i] = 'z'
	}

	invoker := ToolInvokerFunc(func(ctx context.Context, name string, argsJSON string) (string, error) {
		return string(oversized), nil
	})
	executor := NewStepExecutor(invoker)

	step := &FSMStep{
		ID:            "step_large_result",
		ToolName:      "web_search",
		Status:        StepStatusPending,
		ExecutionMode: ExecutionModeSequential,
	}

	err := executor.Execute(context.Background(), nil, []*FSMStep{step})
	require.NoError(t, err)
	assert.Equal(t, StepStatusCompleted, step.Status)
	assert.True(t, len(step.ResultJSON) <= MaxStepResultBytes+len(StepTruncationNotice))
	assert.True(t, strings.HasSuffix(step.ResultJSON, StepTruncationNotice))
}



