package fsm

import "context"

// ProgressObserver observes FSM tool loop lifecycle events to broadcast live progress cards.
type ProgressObserver interface {
	OnStepsPrepared(ctx context.Context, steps []FSMStep)
	OnStepUpdate(ctx context.Context, step *FSMStep)
	OnRunFinished(ctx context.Context, status RunStatus, runErr error)
}

const progressObserverKey contextKey = "fsm_progress_observer"

// WithProgressObserver attaches a ProgressObserver to the context.
func WithProgressObserver(ctx context.Context, obs ProgressObserver) context.Context {
	if obs == nil {
		return ctx
	}
	return context.WithValue(ctx, progressObserverKey, obs)
}

// GetProgressObserver retrieves the ProgressObserver from context if set.
func GetProgressObserver(ctx context.Context) ProgressObserver {
	if obs, ok := ctx.Value(progressObserverKey).(ProgressObserver); ok {
		return obs
	}
	return nil
}
