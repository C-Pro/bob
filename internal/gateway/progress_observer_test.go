package gateway

import (
	"context"
	"errors"
	"sync"
	"testing"

	"bob/internal/fsm"
	"bob/internal/models"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type mockProgressSender struct {
	mu       sync.Mutex
	messages []models.ProgressData
	nextSeq  int64
	sendErr  error
}

func (m *mockProgressSender) SendProgressMessage(ctx context.Context, chatID string, progress *models.ProgressData) (int64, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.sendErr != nil {
		return 0, m.sendErr
	}

	m.nextSeq++
	if progress != nil {
		m.messages = append(m.messages, *progress)
	}
	return m.nextSeq, nil
}

func (m *mockProgressSender) getMessages() []models.ProgressData {
	m.mu.Lock()
	defer m.mu.Unlock()
	res := make([]models.ProgressData, len(m.messages))
	copy(res, m.messages)
	return res
}

func TestGatewayProgressObserver_FullLifecycle(t *testing.T) {
	sender := &mockProgressSender{nextSeq: 41}
	obs := NewGatewayProgressObserver(sender, "dm_chat", true)

	ctx := context.Background()

	// 1. Prepare steps
	steps := []fsm.FSMStep{
		{
			ID:        "step_1",
			Iteration: 1,
			StepIndex: 0,
			ToolName:  "web_search",
			ArgsJSON:  `{"query":"golang concurrency"}`,
			Status:    fsm.StepStatusPending,
		},
	}
	obs.OnStepsPrepared(ctx, steps)

	assert.Equal(t, int64(42), obs.RootSeq())

	msgs := sender.getMessages()
	require.Len(t, msgs, 1)
	assert.Equal(t, int64(0), msgs[0].ParentSeq)
	assert.Equal(t, models.ProgressStatusRunning, msgs[0].CardStatus)
	assert.Equal(t, "Working on your request...", msgs[0].Title)

	// 2. Step transitions to RUNNING
	runningStep := steps[0]
	runningStep.Status = fsm.StepStatusRunning
	obs.OnStepUpdate(ctx, &runningStep)

	msgs = sender.getMessages()
	require.Len(t, msgs, 2)
	assert.Equal(t, int64(42), msgs[1].ParentSeq)
	require.NotNil(t, msgs[1].Step)
	assert.Equal(t, "i1_s0", msgs[1].Step.ID)
	assert.Equal(t, models.ProgressStatusRunning, msgs[1].Step.Status)
	assert.Equal(t, "Web Search", msgs[1].Step.Title)
	assert.Equal(t, "Searching for: golang concurrency", msgs[1].Step.Description)

	// 3. Step transitions to COMPLETED
	completedStep := steps[0]
	completedStep.Status = fsm.StepStatusCompleted
	obs.OnStepUpdate(ctx, &completedStep)

	msgs = sender.getMessages()
	require.Len(t, msgs, 3)
	assert.Equal(t, int64(42), msgs[2].ParentSeq)
	require.NotNil(t, msgs[2].Step)
	assert.Equal(t, "i1_s0", msgs[2].Step.ID)
	assert.Equal(t, models.ProgressStatusCompleted, msgs[2].Step.Status)

	// 4. Run finishes successfully
	obs.OnRunFinished(ctx, fsm.RunStatusCompleted, nil)

	msgs = sender.getMessages()
	require.Len(t, msgs, 4)
	assert.Equal(t, int64(42), msgs[3].ParentSeq)
	assert.Equal(t, models.ProgressStatusCompleted, msgs[3].CardStatus)
	assert.Nil(t, msgs[3].Step)
}

func TestGatewayProgressObserver_TimeoutAndSkippedMapping(t *testing.T) {
	sender := &mockProgressSender{nextSeq: 10}
	obs := NewGatewayProgressObserver(sender, "chat_timeout", false)

	ctx := context.Background()

	step1 := fsm.FSMStep{
		ID:        "s1",
		Iteration: 1,
		StepIndex: 0,
		ToolName:  "sandbox_exec",
		ArgsJSON:  `{"command":"sleep 100"}`,
		Status:    fsm.StepStatusPending,
	}
	step2 := fsm.FSMStep{
		ID:        "s2",
		Iteration: 1,
		StepIndex: 1,
		ToolName:  "sandbox_exec",
		ArgsJSON:  `{"command":"echo next"}`,
		Status:    fsm.StepStatusPending,
	}
	obs.OnStepsPrepared(ctx, []fsm.FSMStep{step1, step2})

	// Step 1 times out
	step1.Status = fsm.StepStatusTimedOut
	obs.OnStepUpdate(ctx, &step1)

	// Step 2 is skipped
	step2.Status = fsm.StepStatusSkipped
	obs.OnStepUpdate(ctx, &step2)

	// Run finishes with failure
	obs.OnRunFinished(ctx, fsm.RunStatusFailed, errors.New("timeout"))

	msgs := sender.getMessages()
	require.Len(t, msgs, 4)

	// Root card
	assert.Equal(t, models.ProgressStatusRunning, msgs[0].CardStatus)

	// Step 1: timed out -> failed with explanation
	assert.Equal(t, models.ProgressStatusFailed, msgs[1].Step.Status)
	assert.Equal(t, "Tool execution timed out.", msgs[1].Step.Description)

	// Step 2: skipped -> failed with explanation
	assert.Equal(t, models.ProgressStatusFailed, msgs[2].Step.Status)
	assert.Equal(t, "Skipped because an earlier step failed.", msgs[2].Step.Description)

	// Final card status
	assert.Equal(t, models.ProgressStatusFailed, msgs[3].CardStatus)
}

func TestGatewayProgressObserver_SendErrorDisablesSubsequent(t *testing.T) {
	sender := &mockProgressSender{sendErr: errors.New("403 Forbidden: write permission denied")}
	obs := NewGatewayProgressObserver(sender, "townhall", false)

	ctx := context.Background()

	step := fsm.FSMStep{
		ID:        "s1",
		Iteration: 1,
		StepIndex: 0,
		ToolName:  "web_search",
		ArgsJSON:  `{"query":"test"}`,
	}

	// Root card fails
	obs.OnStepsPrepared(ctx, []fsm.FSMStep{step})
	assert.Equal(t, int64(0), obs.RootSeq())

	// Step updates and finish must be graceful no-ops
	step.Status = fsm.StepStatusRunning
	obs.OnStepUpdate(ctx, &step)

	step.Status = fsm.StepStatusCompleted
	obs.OnStepUpdate(ctx, &step)

	obs.OnRunFinished(ctx, fsm.RunStatusCompleted, nil)

	// Ensure no panics occurred
}

func TestGatewayProgressObserver_ChildStepFailureDoesNotStrandRootCard(t *testing.T) {
	// Sender succeeds on root card and terminal card, but fails on child steps
	sender := &mockChildFailingSender{nextSeq: 50}
	obs := NewGatewayProgressObserver(sender, "chat_resilient", true)

	ctx := context.Background()
	step := fsm.FSMStep{
		ID:        "s1",
		Iteration: 1,
		StepIndex: 0,
		ToolName:  "web_search",
		ArgsJSON:  `{"query":"test"}`,
	}

	// 1. Root card succeeds
	obs.OnStepsPrepared(ctx, []fsm.FSMStep{step})
	assert.Equal(t, int64(51), obs.RootSeq())

	// 2. Child step fails
	step.Status = fsm.StepStatusRunning
	obs.OnStepUpdate(ctx, &step)

	// 3. Subsequent step update is skipped due to stepUpdatesDisabled
	step.Status = fsm.StepStatusCompleted
	obs.OnStepUpdate(ctx, &step)

	// 4. OnRunFinished must NOT be skipped, and must send terminal card status
	obs.OnRunFinished(ctx, fsm.RunStatusCompleted, nil)

	msgs := sender.getMessages()
	require.Len(t, msgs, 2)
	// Message 0: Root card (running)
	assert.Equal(t, int64(0), msgs[0].ParentSeq)
	assert.Equal(t, models.ProgressStatusRunning, msgs[0].CardStatus)
	// Message 1: Terminal card (completed)
	assert.Equal(t, int64(51), msgs[1].ParentSeq)
	assert.Equal(t, models.ProgressStatusCompleted, msgs[1].CardStatus)
}

type mockChildFailingSender struct {
	mu       sync.Mutex
	messages []models.ProgressData
	nextSeq  int64
}

func (m *mockChildFailingSender) SendProgressMessage(ctx context.Context, chatID string, progress *models.ProgressData) (int64, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Fail only on child step updates
	if progress != nil && progress.Step != nil {
		return 0, errors.New("simulated network error on child step")
	}

	m.nextSeq++
	if progress != nil {
		m.messages = append(m.messages, *progress)
	}
	return m.nextSeq, nil
}

func (m *mockChildFailingSender) getMessages() []models.ProgressData {
	m.mu.Lock()
	defer m.mu.Unlock()
	res := make([]models.ProgressData, len(m.messages))
	copy(res, m.messages)
	return res
}


func TestGatewayProgressObserver_DuplicateEmissionsSuppressed(t *testing.T) {
	sender := &mockProgressSender{nextSeq: 5}
	obs := NewGatewayProgressObserver(sender, "chat_dedup", true)

	ctx := context.Background()

	step := fsm.FSMStep{
		ID:        "s1",
		Iteration: 1,
		StepIndex: 0,
		ToolName:  "web_search",
		ArgsJSON:  `{"query":"retry"}`,
	}
	obs.OnStepsPrepared(ctx, []fsm.FSMStep{step})

	step.Status = fsm.StepStatusRunning
	obs.OnStepUpdate(ctx, &step)

	// Retry occurs, step is still running
	obs.OnStepUpdate(ctx, &step)
	obs.OnStepUpdate(ctx, &step)

	msgs := sender.getMessages()
	// Root card + 1 running step update (duplicates suppressed)
	require.Len(t, msgs, 2)
}

func TestGatewayProgressObserver_OnRunFinishedIdempotent(t *testing.T) {
	sender := &mockProgressSender{nextSeq: 1}
	obs := NewGatewayProgressObserver(sender, "chat_idem", true)

	ctx := context.Background()

	step := fsm.FSMStep{
		ID:        "s1",
		Iteration: 1,
		StepIndex: 0,
		ToolName:  "web_search",
	}
	obs.OnStepsPrepared(ctx, []fsm.FSMStep{step})

	obs.OnRunFinished(ctx, fsm.RunStatusCompleted, nil)
	obs.OnRunFinished(ctx, fsm.RunStatusCompleted, nil)
	obs.OnRunFinished(ctx, fsm.RunStatusFailed, errors.New("late err"))

	msgs := sender.getMessages()
	// Root card + 1 terminal card status
	require.Len(t, msgs, 2)
	assert.Equal(t, models.ProgressStatusCompleted, msgs[1].CardStatus)
}

func TestGatewayProgressObserver_ContextCancellationStillEmitsTerminal(t *testing.T) {
	sender := &mockProgressSender{nextSeq: 1}
	obs := NewGatewayProgressObserver(sender, "chat_cancel", true)

	canceledCtx, cancel := context.WithCancel(context.Background())
	cancel() // canceled immediately

	step := fsm.FSMStep{
		ID:        "s1",
		Iteration: 1,
		StepIndex: 0,
		ToolName:  "web_search",
	}
	obs.OnStepsPrepared(context.Background(), []fsm.FSMStep{step})

	// OnRunFinished uses detached background context with timeout
	obs.OnRunFinished(canceledCtx, fsm.RunStatusTerminated, canceledCtx.Err())

	msgs := sender.getMessages()
	require.Len(t, msgs, 2)
	assert.Equal(t, models.ProgressStatusFailed, msgs[1].CardStatus)
}
