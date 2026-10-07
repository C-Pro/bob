package agent

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"bob/internal/agentapi"
)

func TestSessionLocker_NormalizeSessionKey(t *testing.T) {
	k1 := NormalizeSessionKey(agentapi.SessionRef{FrontendID: "", SessionID: "chat1"})
	k2 := NormalizeSessionKey(agentapi.SessionRef{FrontendID: "besedka", SessionID: "chat1"})
	k3 := NormalizeSessionKey(agentapi.SessionRef{FrontendID: "  BESEDKA  ", SessionID: "chat1"})
	assert.Equal(t, k1, k2)
	assert.Equal(t, k2, k3)

	kCLI := NormalizeSessionKey(agentapi.SessionRef{FrontendID: "cli", SessionID: "chat1"})
	assert.NotEqual(t, k1, kCLI)
}

func TestSessionLocker_TryAcquire_SuccessAndIdempotentRelease(t *testing.T) {
	locker := NewSessionLocker()
	ctx := context.Background()

	release, err := locker.TryAcquire(ctx, "sess_1", 100*time.Millisecond)
	require.NoError(t, err)
	require.NotNil(t, release)

	// Idempotent release
	release()
	release()

	// Should be acquirable again immediately
	release2, err := locker.TryAcquire(ctx, "sess_1", 100*time.Millisecond)
	require.NoError(t, err)
	release2()
}

func TestSessionLocker_TryAcquire_Timeout(t *testing.T) {
	locker := NewSessionLocker()
	ctx := context.Background()

	release, err := locker.TryAcquire(ctx, "sess_busy", 100*time.Millisecond)
	require.NoError(t, err)
	defer release()

	_, err = locker.TryAcquire(ctx, "sess_busy", 50*time.Millisecond)
	require.Error(t, err)
	assert.ErrorIs(t, err, ErrSessionBusy)
}

func TestSessionLocker_TryAcquire_ContextCancelled(t *testing.T) {
	locker := NewSessionLocker()
	ctx, cancel := context.WithCancel(context.Background())

	release, err := locker.TryAcquire(ctx, "sess_cancel", 100*time.Millisecond)
	require.NoError(t, err)
	defer release()

	cancel()
	_, err = locker.TryAcquire(ctx, "sess_cancel", 100*time.Millisecond)
	require.Error(t, err)
	assert.ErrorIs(t, err, context.Canceled)
}
