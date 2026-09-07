package sandbox

import (
	"context"
	"testing"
	"time"

	"bob/internal/config"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type mockDriver struct {
	driverType   DriverType
	available    bool
	createdCount int
	execCount    int
	destroyCount int
}

func (m *mockDriver) Type() DriverType {
	return m.driverType
}

func (m *mockDriver) Available(ctx context.Context) bool {
	return m.available
}

func (m *mockDriver) Create(ctx context.Context, sbx *UserSandbox, workspace string) error {
	m.createdCount++
	return nil
}

func (m *mockDriver) Exec(ctx context.Context, sbx *UserSandbox, cmd []string, timeout time.Duration) (*ExecResult, error) {
	m.execCount++
	return &ExecResult{
		ExitCode: 0,
		Stdout:   "mock stdout",
		Duration: 10 * time.Millisecond,
	}, nil
}

func (m *mockDriver) Destroy(ctx context.Context, sbx *UserSandbox) error {
	m.destroyCount++
	return nil
}

func TestManagerLifecycle(t *testing.T) {
	tempDir := t.TempDir()
	cfg := &config.Config{
		DataDir:                   tempDir,
		SandboxEnabled:            true,
		SandboxDrivers:            []string{"bwrap", "docker"},
		SandboxAllowedImages:      []string{"alpine:latest"},
		SandboxMaxLifetime:        30 * time.Minute,
		SandboxDefaultExecTimeout: 1 * time.Minute,
		SandboxMaxExecTimeout:     10 * time.Minute,
	}

	mockBwrap := &mockDriver{driverType: DriverBwrap, available: true}
	mockDocker := &mockDriver{driverType: DriverDocker, available: true}

	mgr := NewManager(cfg, []Driver{mockBwrap, mockDocker})
	defer func() { _ = mgr.Close() }()

	ctx := context.Background()

	// 1. Available drivers
	drivers := mgr.AvailableDrivers(ctx)
	assert.ElementsMatch(t, []DriverType{DriverBwrap, DriverDocker}, drivers)

	// 2. Request sandbox for user1
	sbx, err := mgr.RequestSandbox(ctx, "user1", "chat1", RequestParams{
		Driver:      DriverBwrap,
		NetworkMode: NetworkNone,
		Reason:      "Run tests",
	})
	require.NoError(t, err)
	assert.Equal(t, StatusPendingApproval, sbx.Status)
	assert.Equal(t, "user1", sbx.UserID)
	assert.Equal(t, mgr.UserWorkspaceDir("user1"), sbx.WorkspaceDir)

	// 3. Duplicate request should fail (1-sandbox-per-user limit)
	_, err = mgr.RequestSandbox(ctx, "user1", "chat1", RequestParams{
		Driver: DriverBwrap,
	})
	assert.ErrorContains(t, err, "already have a pending sandbox")

	// 4. Exec before approval should fail
	_, err = mgr.Exec(ctx, "user1", []string{"ls"}, 10*time.Second)
	assert.ErrorContains(t, err, "no active sandbox found")

	// 5. Approve sandbox
	approved, err := mgr.ApproveSandbox(ctx, "user1")
	require.NoError(t, err)
	assert.Equal(t, StatusRunning, approved.Status)
	assert.Equal(t, mgr.UserWorkspaceDir("user1"), approved.WorkspaceDir)
	assert.Equal(t, 1, mockBwrap.createdCount)

	// 6. Request while running should fail
	_, err = mgr.RequestSandbox(ctx, "user1", "chat1", RequestParams{
		Driver: DriverBwrap,
	})
	assert.ErrorContains(t, err, "already have a running sandbox")

	// 7. Exec command
	res, err := mgr.Exec(ctx, "user1", []string{"echo", "hi"}, 10*time.Second)
	require.NoError(t, err)
	assert.Equal(t, 0, res.ExitCode)
	assert.Equal(t, "mock stdout", res.Stdout)
	assert.Equal(t, 1, mockBwrap.execCount)

	// 8. User2 can independently create a sandbox
	sbx2, err := mgr.RequestSandbox(ctx, "user2", "chat2", RequestParams{
		Driver:      DriverDocker,
		DockerImage: "alpine:latest",
		NetworkMode: NetworkNone,
	})
	require.NoError(t, err)
	assert.Equal(t, StatusPendingApproval, sbx2.Status)

	// 9. Destroy user1
	err = mgr.Destroy(ctx, "user1")
	require.NoError(t, err)
	assert.Equal(t, 1, mockBwrap.destroyCount)

	_, exists := mgr.GetStatus("user1")
	assert.False(t, exists)

	// 10. Deny user2
	err = mgr.DenySandbox("user2")
	require.NoError(t, err)
	_, exists = mgr.GetStatus("user2")
	assert.False(t, exists)
}

func TestManagerExpiration(t *testing.T) {
	tempDir := t.TempDir()
	cfg := &config.Config{
		DataDir:                   tempDir,
		SandboxEnabled:            true,
		SandboxDrivers:            []string{"bwrap"},
		SandboxMaxLifetime:        100 * time.Millisecond,
		SandboxDefaultExecTimeout: 1 * time.Minute,
		SandboxMaxExecTimeout:     10 * time.Minute,
	}

	mockBwrap := &mockDriver{driverType: DriverBwrap, available: true}
	mgr := NewManager(cfg, []Driver{mockBwrap})
	defer func() { _ = mgr.Close() }()

	ctx := context.Background()

	sbx, err := mgr.RequestSandbox(ctx, "user1", "chat1", RequestParams{
		Driver:      DriverBwrap,
		NetworkMode: NetworkNone,
	})
	require.NoError(t, err)

	_, err = mgr.ApproveSandbox(ctx, "user1")
	require.NoError(t, err)

	// Manually set expiration in the past
	mgr.mu.Lock()
	sbx.ExpiresAt = time.Now().Add(-1 * time.Second)
	mgr.mu.Unlock()

	// Exec after expiration should fail and trigger teardown
	_, err = mgr.Exec(ctx, "user1", []string{"ls"}, 10*time.Second)
	assert.ErrorContains(t, err, "expired")
	assert.Equal(t, 1, mockBwrap.destroyCount)

	status, ok := mgr.GetStatus("user1")
	require.True(t, ok)
	assert.Equal(t, StatusExpired, status.Status)
}

func TestManagerNetworkModesValidation(t *testing.T) {
	tempDir := t.TempDir()
	cfg := &config.Config{
		DataDir:                    tempDir,
		SandboxEnabled:             true,
		SandboxDrivers:             []string{"bwrap"},
		SandboxAllowedNetworkModes: []string{"none", "restricted"},
		SandboxMaxLifetime:         30 * time.Minute,
		SandboxDefaultExecTimeout:  1 * time.Minute,
		SandboxMaxExecTimeout:      10 * time.Minute,
	}

	mockBwrap := &mockDriver{driverType: DriverBwrap, available: true}
	mgr := NewManager(cfg, []Driver{mockBwrap})
	defer func() { _ = mgr.Close() }()

	ctx := context.Background()

	// Requesting "full" network mode must be rejected
	_, err := mgr.RequestSandbox(ctx, "user1", "chat1", RequestParams{
		Driver:      DriverBwrap,
		NetworkMode: NetworkFull,
		Reason:      "Need internet",
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "network mode \"full\" is not permitted by server policy")

	// Requesting "restricted" must succeed
	sbx, err := mgr.RequestSandbox(ctx, "user1", "chat1", RequestParams{
		Driver:         DriverBwrap,
		NetworkMode:    NetworkRestricted,
		AllowedDomains: []string{"pypi.org"},
		Reason:         "Need pypi",
	})
	require.NoError(t, err)
	assert.Equal(t, NetworkRestricted, sbx.Network.Mode)

	// Whole workspace mount and subdirectory mounts
	err = mgr.DenySandbox("user1")
	require.NoError(t, err)

	sbx2, err := mgr.RequestSandbox(ctx, "user1", "chat1", RequestParams{
		Driver:      DriverBwrap,
		NetworkMode: NetworkNone,
		Mounts: []UserMount{
			{RelativePath: ".", ReadOnly: true},
		},
		Reason: "Whole workspace read-only",
	})
	require.NoError(t, err)
	assert.Len(t, sbx2.Mounts, 1)
}
