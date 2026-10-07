package gateway

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"bob/internal/agentapi"
	"bob/internal/fsm"
	"bob/internal/models"
	"bob/internal/tools"
)

var _ agentapi.ProgressObserver = (*GatewayProgressObserver)(nil)
var _ fsm.ProgressObserver = (*GatewayProgressObserver)(nil)

type progressSender interface {
	SendProgressMessage(ctx context.Context, chatID string, progress *models.ProgressData) (int64, error)
}

// GatewayProgressObserver observes FSM tool loop lifecycle events and broadcasts live Besedka progress cards.
type GatewayProgressObserver struct {
	sender progressSender
	chatID string
	isDM   bool

	mu                  sync.Mutex
	rootSeq             int64
	stepUpdatesDisabled bool
	closed              bool
	lastEmitted         map[string]models.ProgressStatus
}

// NewGatewayProgressObserver creates a new GatewayProgressObserver.
func NewGatewayProgressObserver(sender progressSender, chatID string, isDM bool) *GatewayProgressObserver {
	return &GatewayProgressObserver{
		sender:      sender,
		chatID:      chatID,
		isDM:        isDM,
		lastEmitted: make(map[string]models.ProgressStatus),
	}
}

// NewGatewayProgressObserverWithRoot creates a new GatewayProgressObserver with an existing root card seq.
func NewGatewayProgressObserverWithRoot(sender progressSender, chatID string, isDM bool, rootSeq int64) *GatewayProgressObserver {
	return &GatewayProgressObserver{
		sender:      sender,
		chatID:      chatID,
		isDM:        isDM,
		rootSeq:     rootSeq,
		lastEmitted: make(map[string]models.ProgressStatus),
	}
}

// RootSeq returns the sequence number of the root progress card (0 if not created).
func (o *GatewayProgressObserver) RootSeq() int64 {
	o.mu.Lock()
	defer o.mu.Unlock()
	return o.rootSeq
}

// OnStepsPrepared is called when the FSM prepares steps for execution.
// If the root progress card has not yet been emitted, it creates the root card.
func (o *GatewayProgressObserver) OnStepsPrepared(ctx context.Context, steps []fsm.FSMStep) {
	if len(steps) == 0 {
		return
	}

	o.mu.Lock()
	defer o.mu.Unlock()

	if o.closed {
		return
	}

	if o.rootSeq == 0 {
		progress := &models.ProgressData{
			CardStatus: models.ProgressStatusRunning,
			Title:      "Working on your request...",
		}
		seq, err := o.sender.SendProgressMessage(ctx, o.chatID, progress)
		if err != nil {
			slog.Warn("failed to send root progress card, disabling progress reporting for run",
				"chat_id", o.chatID,
				"error", err,
			)
			o.closed = true
			return
		}
		o.rootSeq = seq
	}
}

// OnStepUpdate is called whenever an FSM step transitions between states.
func (o *GatewayProgressObserver) OnStepUpdate(ctx context.Context, step *fsm.FSMStep) {
	if step == nil {
		return
	}

	o.mu.Lock()
	defer o.mu.Unlock()

	if o.stepUpdatesDisabled || o.closed || o.rootSeq == 0 {
		return
	}

	var protoStatus models.ProgressStatus
	var descOverride string

	switch step.Status {
	case fsm.StepStatusRunning:
		protoStatus = models.ProgressStatusRunning
	case fsm.StepStatusCompleted:
		protoStatus = models.ProgressStatusCompleted
	case fsm.StepStatusFailed:
		protoStatus = models.ProgressStatusFailed
	case fsm.StepStatusTimedOut:
		protoStatus = models.ProgressStatusFailed
		descOverride = "Tool execution timed out."
	case fsm.StepStatusSkipped:
		protoStatus = models.ProgressStatusFailed
		descOverride = "Skipped because an earlier step failed."
	default:
		// Pending or other internal state: do not emit
		return
	}

	stepID := fmt.Sprintf("i%d_s%d", step.Iteration, step.StepIndex)

	// Avoid duplicate emissions (e.g. retry transitions maintaining running status)
	if o.lastEmitted[stepID] == protoStatus {
		return
	}

	title, desc := tools.FormatProgressStep(step.ToToolCall())
	if descOverride != "" {
		desc = descOverride
	}

	progress := &models.ProgressData{
		ParentSeq: o.rootSeq,
		Step: &models.ProgressStep{
			ID:          stepID,
			Title:       title,
			Description: desc,
			Status:      protoStatus,
		},
	}

	_, err := o.sender.SendProgressMessage(ctx, o.chatID, progress)
	if err != nil {
		slog.Warn("failed to send child progress step, disabling further step updates for run",
			"chat_id", o.chatID,
			"step_id", stepID,
			"error", err,
		)
		o.stepUpdatesDisabled = true
		return
	}

	o.lastEmitted[stepID] = protoStatus
}

// OnRunFinished marks the root progress card as completed or failed.
func (o *GatewayProgressObserver) OnRunFinished(ctx context.Context, status fsm.RunStatus, runErr error) {
	o.mu.Lock()
	defer o.mu.Unlock()

	if o.closed || o.rootSeq == 0 {
		o.closed = true
		return
	}
	o.closed = true

	cardStatus := models.ProgressStatusCompleted
	if status == fsm.RunStatusFailed || status == fsm.RunStatusTerminated || runErr != nil {
		cardStatus = models.ProgressStatusFailed
	}

	progress := &models.ProgressData{
		ParentSeq:  o.rootSeq,
		CardStatus: cardStatus,
	}

	// Use a short detached context so cancellation of the request context does not
	// prevent updating the card to its terminal status.
	sendCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if _, err := o.sender.SendProgressMessage(sendCtx, o.chatID, progress); err != nil {
		slog.Warn("failed to send terminal progress card status",
			"chat_id", o.chatID,
			"card_status", cardStatus,
			"error", err,
		)
	}
}

// Observe implements agentapi.ProgressObserver to forward generic events to GatewayProgressObserver.
func (o *GatewayProgressObserver) Observe(ctx context.Context, ev agentapi.ProgressEvent) {
	switch ev.Type {
	case agentapi.ProgressStepsPrepared:
		fsmSteps := make([]fsm.FSMStep, 0, len(ev.Steps))
		for _, s := range ev.Steps {
			fsmSteps = append(fsmSteps, fsm.FSMStep{
				ID:        s.ID,
				Iteration: s.Iteration,
				StepIndex: s.StepIndex,
				ToolName:  s.Call.Name,
				ArgsJSON:  s.Call.Arguments,
				Status:    fsm.StepStatus(s.Status),
			})
		}
		o.OnStepsPrepared(ctx, fsmSteps)
	case agentapi.ProgressStepUpdated:
		if ev.Step != nil {
			s := ev.Step
			fsmStep := fsm.FSMStep{
				ID:        s.ID,
				Iteration: s.Iteration,
				StepIndex: s.StepIndex,
				ToolName:  s.Call.Name,
				ArgsJSON:  s.Call.Arguments,
				Status:    fsm.StepStatus(s.Status),
				ErrorText: s.Error,
			}
			o.OnStepUpdate(ctx, &fsmStep)
		}
	case agentapi.ProgressRunFinished:
		var runErr error
		if ev.Error != "" {
			runErr = errors.New(ev.Error)
		}
		o.OnRunFinished(ctx, fsm.RunStatus(ev.Status), runErr)
	}
}
