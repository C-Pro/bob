package agent

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	"bob/internal/agentapi"
	"bob/internal/scheduler"
)

// ErrSessionBusy is returned when a session execution lock cannot be acquired within the timeout.
// It wraps scheduler.ErrChatBusy to preserve backward compatibility with scheduler retry handling.
var ErrSessionBusy = fmt.Errorf("%w: session execution lock busy", scheduler.ErrChatBusy)

// SessionLocker provides per-session mutex locking with timeouts and context cancellation support.
type SessionLocker struct {
	mu    sync.Mutex
	locks map[string]*sync.Mutex
}

// NewSessionLocker creates a new SessionLocker.
func NewSessionLocker() *SessionLocker {
	return &SessionLocker{
		locks: make(map[string]*sync.Mutex),
	}
}

// NormalizeSessionKey derives a canonical, collision-resistant key for a session.
// An empty FrontendID is normalized to "besedka" to maintain backward compatibility.
func NormalizeSessionKey(ref agentapi.SessionRef) string {
	fe := strings.ToLower(strings.TrimSpace(ref.FrontendID))
	if fe == "" {
		fe = "besedka"
	}
	return fmt.Sprintf("%s\x00%s", fe, ref.SessionID)
}

func (l *SessionLocker) getLock(key string) *sync.Mutex {
	l.mu.Lock()
	defer l.mu.Unlock()
	mu, ok := l.locks[key]
	if !ok {
		mu = &sync.Mutex{}
		l.locks[key] = mu
	}
	return mu
}

// TryAcquire attempts to acquire the lock for sessionKey within the given timeout.
// Returns an idempotent release function on success, or an error wrapping ErrSessionBusy on timeout.
func (l *SessionLocker) TryAcquire(ctx context.Context, sessionKey string, timeout time.Duration) (func(), error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	mu := l.getLock(sessionKey)
	if mu.TryLock() {
		var once sync.Once
		return func() {
			once.Do(func() {
				mu.Unlock()
			})
		}, nil
	}

	deadline := time.After(timeout)
	ticker := time.NewTicker(50 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-deadline:
			return nil, fmt.Errorf("%w: timed out waiting for session %s execution lock after %v", ErrSessionBusy, sessionKey, timeout)
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-ticker.C:
			if mu.TryLock() {
				var once sync.Once
				return func() {
					once.Do(func() {
						mu.Unlock()
					})
				}, nil
			}
		}
	}
}
