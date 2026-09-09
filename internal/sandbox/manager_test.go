package sandbox

import (
	"context"
	"sync"
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
	destroyErr   error
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
	return m.destroyErr
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
	expectedWS, err := mgr.UserWorkspaceDir("user1")
	require.NoError(t, err)
	assert.Equal(t, expectedWS, sbx.WorkspaceDir)

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
	assert.Equal(t, expectedWS, approved.WorkspaceDir)
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

type slowMockDriver struct {
	mockDriver
	mu sync.Mutex
}

func (s *slowMockDriver) Create(ctx context.Context, sbx *UserSandbox, workspace string) error {
	time.Sleep(50 * time.Millisecond)
	s.mu.Lock()
	s.createdCount++
	s.mu.Unlock()
	return nil
}

func TestManager_ApproveSandbox_TOCTOURace(t *testing.T) {
	tempDir := t.TempDir()
	cfg := &config.Config{
		DataDir:                   tempDir,
		SandboxEnabled:            true,
		SandboxDrivers:            []string{"bwrap"},
		SandboxMaxLifetime:        30 * time.Minute,
		SandboxDefaultExecTimeout: 1 * time.Minute,
		SandboxMaxExecTimeout:     10 * time.Minute,
	}
	driver := &slowMockDriver{mockDriver: mockDriver{driverType: DriverBwrap, available: true}}
	mgr := NewManager(cfg, []Driver{driver})
	defer func() { _ = mgr.Close() }()

	ctx := context.Background()
	_, err := mgr.RequestSandbox(ctx, "user1", "chat1", RequestParams{
		Driver:      DriverBwrap,
		NetworkMode: NetworkNone,
		Reason:      "Race test",
	})
	require.NoError(t, err)

	var wg sync.WaitGroup
	errs := make([]error, 2)
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			_, errs[idx] = mgr.ApproveSandbox(ctx, "user1")
		}(i)
	}
	wg.Wait()

	successes := 0
	for _, err := range errs {
		if err == nil {
			successes++
		}
	}
	assert.Equal(t, 1, successes, "Exactly one ApproveSandbox call must succeed")

	driver.mu.Lock()
	assert.Equal(t, 1, driver.createdCount, "Driver.Create must only be called once")
	driver.mu.Unlock()
}

func TestManager_RequestSandbox_DestroysExpiredRunning(t *testing.T) {
	tempDir := t.TempDir()
	cfg := &config.Config{
		DataDir:                   tempDir,
		SandboxEnabled:            true,
		SandboxDrivers:            []string{"bwrap"},
		SandboxMaxLifetime:        30 * time.Minute,
		SandboxDefaultExecTimeout: 1 * time.Minute,
		SandboxMaxExecTimeout:     10 * time.Minute,
	}
	driver := &mockDriver{driverType: DriverBwrap, available: true}
	mgr := NewManager(cfg, []Driver{driver})
	defer func() { _ = mgr.Close() }()

	ctx := context.Background()

	// Inject an expired running sandbox (reaper has not collected it yet)
	mgr.mu.Lock()
	mgr.sandboxes["user_expired"] = &UserSandbox{
		UserID:    "user_expired",
		ChatID:    "chat1",
		Driver:    DriverBwrap,
		Status:    StatusRunning,
		CreatedAt: time.Now().Add(-2 * time.Hour),
		ExpiresAt: time.Now().Add(-1 * time.Hour),
	}
	mgr.mu.Unlock()

	// Requesting a new sandbox must destroy the expired running sandbox rather than leaking it
	_, err := mgr.RequestSandbox(ctx, "user_expired", "chat1", RequestParams{
		Driver:      DriverBwrap,
		NetworkMode: NetworkNone,
		Reason:      "New request after expiry",
	})
	require.NoError(t, err)

	assert.Equal(t, 1, driver.destroyCount, "expired running sandbox must be destroyed on new request")
}

func TestManager_RequestSandbox_ExpiredStatusAllowed(t *testing.T) {
	tempDir := t.TempDir()
	cfg := &config.Config{
		DataDir:                   tempDir,
		SandboxEnabled:            true,
		SandboxDrivers:            []string{"bwrap"},
		SandboxMaxLifetime:        30 * time.Minute,
		SandboxDefaultExecTimeout: 1 * time.Minute,
		SandboxMaxExecTimeout:     10 * time.Minute,
	}
	driver := &mockDriver{driverType: DriverBwrap, available: true}
	mgr := NewManager(cfg, []Driver{driver})
	defer func() { _ = mgr.Close() }()

	ctx := context.Background()

	// Inject a sandbox in StatusExpired (e.g. pruned by reaper or exec)
	mgr.mu.Lock()
	mgr.sandboxes["user_reaped"] = &UserSandbox{
		UserID:    "user_reaped",
		ChatID:    "chat1",
		Driver:    DriverBwrap,
		Status:    StatusExpired,
		CreatedAt: time.Now().Add(-2 * time.Hour),
		ExpiresAt: time.Now().Add(-1 * time.Hour),
	}
	mgr.mu.Unlock()

	// Requesting a new sandbox must succeed and replace the expired entry
	sbx, err := mgr.RequestSandbox(ctx, "user_reaped", "chat1", RequestParams{
		Driver:      DriverBwrap,
		NetworkMode: NetworkNone,
		Reason:      "New request after reaper marked expired",
	})
	require.NoError(t, err)
	assert.Equal(t, StatusPendingApproval, sbx.GetStatus())
}

func TestManager_Destroy_RetryOnError(t *testing.T) {
	tempDir := t.TempDir()
	cfg := &config.Config{
		DataDir:                   tempDir,
		SandboxEnabled:            true,
		SandboxDrivers:            []string{"bwrap"},
		SandboxMaxLifetime:        30 * time.Minute,
		SandboxDefaultExecTimeout: 1 * time.Minute,
		SandboxMaxExecTimeout:     10 * time.Minute,
	}
	driver := &mockDriver{driverType: DriverBwrap, available: true}
	mgr := NewManager(cfg, []Driver{driver})
	defer func() { _ = mgr.Close() }()

	ctx := context.Background()

	// Put a running sandbox into manager
	mgr.mu.Lock()
	mgr.sandboxes["user_err"] = &UserSandbox{
		UserID:    "user_err",
		ChatID:    "chat1",
		Driver:    DriverBwrap,
		Status:    StatusRunning,
		CreatedAt: time.Now(),
		ExpiresAt: time.Now().Add(10 * time.Minute),
	}
	mgr.mu.Unlock()

	// Simulate driver destroy failure
	driver.destroyErr = assert.AnError
	err := mgr.Destroy(ctx, "user_err")
	require.Error(t, err)

	// Verify sandbox is retained in manager map for retry
	mgr.mu.Lock()
	_, exists := mgr.sandboxes["user_err"]
	mgr.mu.Unlock()
	assert.True(t, exists, "sandbox must be retained in map when driver.Destroy fails")

	// Retry with successful driver destroy
	driver.destroyErr = nil
	err = mgr.Destroy(ctx, "user_err")
	require.NoError(t, err)

	// Verify sandbox is now removed
	mgr.mu.Lock()
	_, exists = mgr.sandboxes["user_err"]
	mgr.mu.Unlock()
	assert.False(t, exists, "sandbox must be removed from map when driver.Destroy succeeds")
}

func TestManager_ApproveAndStatus_Race(t *testing.T) {
	tempDir := t.TempDir()
	cfg := &config.Config{
		DataDir:                   tempDir,
		SandboxEnabled:            true,
		SandboxDrivers:            []string{"bwrap"},
		SandboxMaxLifetime:        30 * time.Minute,
		SandboxDefaultExecTimeout: 1 * time.Minute,
		SandboxMaxExecTimeout:     10 * time.Minute,
	}
	driver := &mockDriver{driverType: DriverBwrap, available: true}
	mgr := NewManager(cfg, []Driver{driver})
	defer func() { _ = mgr.Close() }()

	ctx := context.Background()

	_, err := mgr.RequestSandbox(ctx, "user_race", "chat1", RequestParams{
		Driver:      DriverBwrap,
		NetworkMode: NetworkNone,
		Reason:      "Race test",
	})
	require.NoError(t, err)

	var wg sync.WaitGroup
	wg.Add(2)

	go func() {
		defer wg.Done()
		for i := 0; i < 50; i++ {
			_, _ = mgr.GetStatus("user_race")
			time.Sleep(1 * time.Millisecond)
		}
	}()

	go func() {
		defer wg.Done()
		time.Sleep(5 * time.Millisecond)
		_, _ = mgr.ApproveSandbox(ctx, "user_race")
	}()

	wg.Wait()
}

func TestManager_RequestSandbox_InvalidSandboxPath(t *testing.T) {
	tempDir := t.TempDir()
	cfg := &config.Config{
		DataDir:                   tempDir,
		SandboxEnabled:            true,
		SandboxDrivers:            []string{"bwrap"},
		SandboxMaxLifetime:        30 * time.Minute,
		SandboxDefaultExecTimeout: 1 * time.Minute,
		SandboxMaxExecTimeout:     10 * time.Minute,
	}
	driver := &mockDriver{driverType: DriverBwrap, available: true}
	mgr := NewManager(cfg, []Driver{driver})
	defer func() { _ = mgr.Close() }()

	ctx := context.Background()

	// Attempting to mount onto /etc inside the sandbox must be rejected
	_, err := mgr.RequestSandbox(ctx, "user_mount", "chat1", RequestParams{
		Driver:      DriverBwrap,
		NetworkMode: NetworkNone,
		Mounts: []UserMount{
			{RelativePath: "data", SandboxPath: "/etc"},
		},
		Reason: "Attempt mount on /etc",
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "conflicts with protected system directory")

	// Attempting to mount onto / (root) must be rejected
	_, err = mgr.RequestSandbox(ctx, "user_mount", "chat1", RequestParams{
		Driver:      DriverBwrap,
		NetworkMode: NetworkNone,
		Mounts: []UserMount{
			{RelativePath: "data", SandboxPath: "/"},
		},
		Reason: "Attempt mount on root",
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "cannot be root")
}
