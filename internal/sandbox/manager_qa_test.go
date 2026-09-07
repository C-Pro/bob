package sandbox

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"bob/internal/config"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// recordingQADriver records calls and parameters for QA validation.
type recordingQADriver struct {
	mu             sync.Mutex
	driverType     DriverType
	available      bool
	createErr      error
	execErr        error
	destroyErr     error
	lastCmd        []string
	lastTimeout    time.Duration
	createdCount   int
	destroyedCount int
}

func (r *recordingQADriver) Type() DriverType {
	return r.driverType
}

func (r *recordingQADriver) Available(ctx context.Context) bool {
	return r.available
}

func (r *recordingQADriver) Create(ctx context.Context, sbx *UserSandbox, workspace string) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.createdCount++
	return r.createErr
}

func (r *recordingQADriver) Exec(ctx context.Context, sbx *UserSandbox, cmd []string, timeout time.Duration) (*ExecResult, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.lastCmd = cmd
	r.lastTimeout = timeout
	if r.execErr != nil {
		return nil, r.execErr
	}
	return &ExecResult{
		ExitCode: 0,
		Stdout:   "qa exec success",
		Duration: 5 * time.Millisecond,
	}, nil
}

func (r *recordingQADriver) Destroy(ctx context.Context, sbx *UserSandbox) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.destroyedCount++
	return r.destroyErr
}

// TestManager_EdgeCasesAndErrorHandling covers boundary behaviors, timeout clamping,
// rollback on driver creation failure, driver availability checks, and multi-tenant isolation.
func TestManager_EdgeCasesAndErrorHandling(t *testing.T) {
	tempDir := t.TempDir()
	cfg := &config.Config{
		DataDir:                    tempDir,
		SandboxEnabled:             true,
		SandboxDrivers:             []string{"bwrap", "docker"},
		SandboxAllowedNetworkModes: []string{"none", "restricted"},
		SandboxMaxLifetime:         30 * time.Minute,
		SandboxDefaultExecTimeout:  1 * time.Minute,
		SandboxMaxExecTimeout:      10 * time.Minute,
	}

	ctx := context.Background()

	t.Run("empty user ID is rejected", func(t *testing.T) {
		driver := &recordingQADriver{driverType: DriverBwrap, available: true}
		mgr := NewManager(cfg, []Driver{driver})
		defer func() { _ = mgr.Close() }()

		_, err := mgr.RequestSandbox(ctx, "", "chat1", RequestParams{
			Driver:      DriverBwrap,
			NetworkMode: NetworkNone,
		})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "userID cannot be empty")

		_, err = mgr.RequestSandbox(ctx, "   ", "chat1", RequestParams{
			Driver:      DriverBwrap,
			NetworkMode: NetworkNone,
		})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "userID cannot be empty")
	})

	t.Run("unregistered or unavailable driver is rejected", func(t *testing.T) {
		unavailDriver := &recordingQADriver{driverType: DriverBwrap, available: false}
		mgr := NewManager(cfg, []Driver{unavailDriver})
		defer func() { _ = mgr.Close() }()

		// Unregistered driver
		_, err := mgr.RequestSandbox(ctx, "user1", "chat1", RequestParams{
			Driver:      DriverType("unknown_driver"),
			NetworkMode: NetworkNone,
		})
		require.Error(t, err)
		assert.Contains(t, err.Error(), `requested driver "unknown_driver" is not available`)

		// Unavailable driver
		_, err = mgr.RequestSandbox(ctx, "user1", "chat1", RequestParams{
			Driver:      DriverBwrap,
			NetworkMode: NetworkNone,
		})
		require.Error(t, err)
		assert.Contains(t, err.Error(), `requested driver "bwrap" is not available`)
	})

	t.Run("driver creation error rolls back sandbox state", func(t *testing.T) {
		failDriver := &recordingQADriver{
			driverType: DriverBwrap,
			available:  true,
			createErr:  errors.New("kernel cgroup mount failed"),
		}
		mgr := NewManager(cfg, []Driver{failDriver})
		defer func() { _ = mgr.Close() }()

		_, err := mgr.RequestSandbox(ctx, "user_fail", "chat_fail", RequestParams{
			Driver:      DriverBwrap,
			NetworkMode: NetworkNone,
		})
		require.NoError(t, err)

		// Approve should attempt Create, fail, and roll back
		_, err = mgr.ApproveSandbox(ctx, "user_fail")
		require.Error(t, err)
		assert.Contains(t, err.Error(), "failed to initialize sandbox")

		// Sandbox must be purged from manager state so user is not stuck
		_, exists := mgr.GetStatus("user_fail")
		assert.False(t, exists)

		// User can request again immediately
		_, err = mgr.RequestSandbox(ctx, "user_fail", "chat_fail", RequestParams{
			Driver:      DriverBwrap,
			NetworkMode: NetworkNone,
		})
		assert.NoError(t, err)
	})

	t.Run("exec timeout clamping boundaries", func(t *testing.T) {
		driver := &recordingQADriver{driverType: DriverBwrap, available: true}
		mgr := NewManager(cfg, []Driver{driver})
		defer func() { _ = mgr.Close() }()

		_, err := mgr.RequestSandbox(ctx, "user_clamp", "chat_clamp", RequestParams{
			Driver:      DriverBwrap,
			NetworkMode: NetworkNone,
		})
		require.NoError(t, err)
		_, err = mgr.ApproveSandbox(ctx, "user_clamp")
		require.NoError(t, err)

		// 1. Requested timeout 0 -> clamped to DefaultExecTimeout (1 minute)
		_, err = mgr.Exec(ctx, "user_clamp", []string{"echo"}, 0)
		require.NoError(t, err)
		assert.Equal(t, cfg.SandboxDefaultExecTimeout, driver.lastTimeout)

		// 2. Requested timeout negative -> clamped to DefaultExecTimeout (1 minute)
		_, err = mgr.Exec(ctx, "user_clamp", []string{"echo"}, -10*time.Second)
		require.NoError(t, err)
		assert.Equal(t, cfg.SandboxDefaultExecTimeout, driver.lastTimeout)

		// 3. Requested timeout < 5s (e.g. 1s) -> clamped to minimum 5s
		_, err = mgr.Exec(ctx, "user_clamp", []string{"echo"}, 1*time.Second)
		require.NoError(t, err)
		assert.Equal(t, 5*time.Second, driver.lastTimeout)

		// 4. Requested timeout > MaxExecTimeout (e.g. 2 hours) -> clamped to MaxExecTimeout (10 minutes)
		_, err = mgr.Exec(ctx, "user_clamp", []string{"echo"}, 2*time.Hour)
		require.NoError(t, err)
		assert.Equal(t, cfg.SandboxMaxExecTimeout, driver.lastTimeout)

		// 5. In-range timeout preserved
		_, err = mgr.Exec(ctx, "user_clamp", []string{"echo"}, 3*time.Minute)
		require.NoError(t, err)
		assert.Equal(t, 3*time.Minute, driver.lastTimeout)
	})

	t.Run("destroy is idempotent", func(t *testing.T) {
		driver := &recordingQADriver{driverType: DriverBwrap, available: true}
		mgr := NewManager(cfg, []Driver{driver})
		defer func() { _ = mgr.Close() }()

		// Destroying non-existent user returns nil
		err := mgr.Destroy(ctx, "ghost_user")
		assert.NoError(t, err)

		// Destroying running sandbox
		_, err = mgr.RequestSandbox(ctx, "active_user", "chat", RequestParams{
			Driver:      DriverBwrap,
			NetworkMode: NetworkNone,
		})
		require.NoError(t, err)
		_, err = mgr.ApproveSandbox(ctx, "active_user")
		require.NoError(t, err)

		err = mgr.Destroy(ctx, "active_user")
		assert.NoError(t, err)
		assert.Equal(t, 1, driver.destroyedCount)

		// Second destroy is a safe no-op
		err = mgr.Destroy(ctx, "active_user")
		assert.NoError(t, err)
		assert.Equal(t, 1, driver.destroyedCount) // Not incremented again
	})

	t.Run("deny when no pending request returns error", func(t *testing.T) {
		mgr := NewManager(cfg, nil)
		defer func() { _ = mgr.Close() }()

		err := mgr.DenySandbox("nobody")
		require.Error(t, err)
		assert.Contains(t, err.Error(), "no pending sandbox request found to deny")
	})

	t.Run("multi-user isolation", func(t *testing.T) {
		driver := &recordingQADriver{driverType: DriverBwrap, available: true}
		mgr := NewManager(cfg, []Driver{driver})
		defer func() { _ = mgr.Close() }()

		// User A: running
		_, err := mgr.RequestSandbox(ctx, "userA", "chatA", RequestParams{
			Driver:      DriverBwrap,
			NetworkMode: NetworkNone,
		})
		require.NoError(t, err)
		_, err = mgr.ApproveSandbox(ctx, "userA")
		require.NoError(t, err)

		// User B: pending
		_, err = mgr.RequestSandbox(ctx, "userB", "chatB", RequestParams{
			Driver:      DriverBwrap,
			NetworkMode: NetworkNone,
		})
		require.NoError(t, err)

		// User C: no sandbox
		_, existsC := mgr.GetStatus("userC")
		assert.False(t, existsC)

		// Exec User A succeeds
		resA, err := mgr.Exec(ctx, "userA", []string{"ls"}, 10*time.Second)
		require.NoError(t, err)
		assert.Equal(t, "qa exec success", resA.Stdout)

		// Exec User B fails because pending approval
		_, err = mgr.Exec(ctx, "userB", []string{"ls"}, 10*time.Second)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "no active sandbox found")

		// Destroy User A does not affect User B
		err = mgr.Destroy(ctx, "userA")
		require.NoError(t, err)

		statusB, existsB := mgr.GetStatus("userB")
		require.True(t, existsB)
		assert.Equal(t, StatusPendingApproval, statusB.Status)
	})

	t.Run("pruneExpired cleans up expired sandboxes", func(t *testing.T) {
		driver := &recordingQADriver{driverType: DriverBwrap, available: true}
		mgr := NewManager(cfg, []Driver{driver})
		defer func() { _ = mgr.Close() }()

		_, err := mgr.RequestSandbox(ctx, "user_exp", "chat_exp", RequestParams{
			Driver:      DriverBwrap,
			NetworkMode: NetworkNone,
		})
		require.NoError(t, err)
		sbx, err := mgr.ApproveSandbox(ctx, "user_exp")
		require.NoError(t, err)

		// Artificially expire the sandbox
		mgr.mu.Lock()
		sbx.ExpiresAt = time.Now().Add(-5 * time.Minute)
		mgr.mu.Unlock()

		// Trigger pruneExpired
		mgr.pruneExpired()

		assert.Equal(t, 1, driver.destroyedCount)
		status, exists := mgr.GetStatus("user_exp")
		require.True(t, exists)
		assert.Equal(t, StatusExpired, status.Status)
	})
}
