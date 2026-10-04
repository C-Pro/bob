package sandbox

import (
	"context"
	"errors"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type mockDriver struct {
	driverType   DriverType
	available    bool
	createdCount int
	execCount    int
	execErr      error
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
	if m.execErr != nil {
		return nil, m.execErr
	}
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
	cfg := Config{
		DataDir:            tempDir,
		Enabled:            true,
		Drivers:            []string{"bwrap", "docker"},
		AllowedImages:      []string{"alpine:latest"},
		MaxLifetime:        30 * time.Minute,
		DefaultExecTimeout: 1 * time.Minute,
		MaxExecTimeout:     10 * time.Minute,
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
	assert.ErrorContains(t, err, "already has an active sandbox")

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
	cfg := Config{
		DataDir:            tempDir,
		Enabled:            true,
		Drivers:            []string{"bwrap"},
		MaxLifetime:        100 * time.Millisecond,
		DefaultExecTimeout: 1 * time.Minute,
		MaxExecTimeout:     10 * time.Minute,
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
	cfg := Config{
		DataDir:             tempDir,
		Enabled:             true,
		Drivers:             []string{"bwrap"},
		AllowedNetworkModes: []string{"none", "restricted"},
		MaxLifetime:         30 * time.Minute,
		DefaultExecTimeout:  1 * time.Minute,
		MaxExecTimeout:      10 * time.Minute,
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
	cfg := Config{
		DataDir:            tempDir,
		Enabled:            true,
		Drivers:            []string{"bwrap"},
		MaxLifetime:        30 * time.Minute,
		DefaultExecTimeout: 1 * time.Minute,
		MaxExecTimeout:     10 * time.Minute,
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
	assert.Equal(t, 2, successes, "Both concurrent ApproveSandbox calls must succeed")

	driver.mu.Lock()
	assert.Equal(t, 1, driver.createdCount, "Driver.Create must only be called once")
	driver.mu.Unlock()
}

func TestManager_RequestSandbox_DestroysExpiredRunning(t *testing.T) {
	tempDir := t.TempDir()
	cfg := Config{
		DataDir:            tempDir,
		Enabled:            true,
		Drivers:            []string{"bwrap"},
		MaxLifetime:        30 * time.Minute,
		DefaultExecTimeout: 1 * time.Minute,
		MaxExecTimeout:     10 * time.Minute,
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
	cfg := Config{
		DataDir:            tempDir,
		Enabled:            true,
		Drivers:            []string{"bwrap"},
		MaxLifetime:        30 * time.Minute,
		DefaultExecTimeout: 1 * time.Minute,
		MaxExecTimeout:     10 * time.Minute,
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
	cfg := Config{
		DataDir:            tempDir,
		Enabled:            true,
		Drivers:            []string{"bwrap"},
		MaxLifetime:        30 * time.Minute,
		DefaultExecTimeout: 1 * time.Minute,
		MaxExecTimeout:     10 * time.Minute,
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
	cfg := Config{
		DataDir:            tempDir,
		Enabled:            true,
		Drivers:            []string{"bwrap"},
		MaxLifetime:        30 * time.Minute,
		DefaultExecTimeout: 1 * time.Minute,
		MaxExecTimeout:     10 * time.Minute,
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
			runtime.Gosched()
		}
	}()

	go func() {
		defer wg.Done()
		_, _ = mgr.ApproveSandbox(ctx, "user_race")
	}()

	wg.Wait()
}

func TestManager_RequestSandbox_InvalidSandboxPath(t *testing.T) {
	tempDir := t.TempDir()
	cfg := Config{
		DataDir:            tempDir,
		Enabled:            true,
		Drivers:            []string{"bwrap"},
		MaxLifetime:        30 * time.Minute,
		DefaultExecTimeout: 1 * time.Minute,
		MaxExecTimeout:     10 * time.Minute,
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

func TestManagerExec_ContainerNotRunning_MarksExpiredAndCleansUp(t *testing.T) {
	tempDir := t.TempDir()
	cfg := Config{
		DataDir:            tempDir,
		Enabled:            true,
		Drivers:            []string{"docker"},
		AllowedImages:      []string{"alpine:latest"},
		MaxLifetime:        30 * time.Minute,
		DefaultExecTimeout: 1 * time.Minute,
		MaxExecTimeout:     10 * time.Minute,
	}

	mockDocker := &mockDriver{
		driverType: DriverDocker,
		available:  true,
	}
	mgr := NewManager(cfg, []Driver{mockDocker})
	defer func() { _ = mgr.Close() }()

	ctx := context.Background()
	_, err := mgr.RequestSandbox(ctx, "user1", "chat1", RequestParams{
		Driver:      DriverDocker,
		DockerImage: "alpine:latest",
		NetworkMode: NetworkNone,
	})
	require.NoError(t, err)

	approved, err := mgr.ApproveSandbox(ctx, "user1")
	require.NoError(t, err)
	assert.Equal(t, StatusRunning, approved.Status)

	// Simulate container dying behind the scenes: Exec returns 409 conflict
	mockDocker.execErr = errors.New("exec create returned 409: {\"message\":\"Container mock-c is not running\"}")

	_, err = mgr.Exec(ctx, "user1", []string{"ls"}, 10*time.Second)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "sandbox container is no longer running")

	// Sandbox should be marked expired and driver Destroy called
	assert.Equal(t, StatusExpired, approved.Status)
	status, exists := mgr.GetStatus("user1")
	assert.True(t, exists)
	assert.Equal(t, StatusExpired, status.Status)
	assert.Equal(t, 1, mockDocker.destroyCount, "driver Destroy should be called on cleanup")

	// Next RequestSandbox can now succeed because existing sandbox is StatusExpired
	mockDocker.execErr = nil
	newSbx, err := mgr.RequestSandbox(ctx, "user1", "chat1", RequestParams{
		Driver:      DriverDocker,
		DockerImage: "alpine:latest",
		NetworkMode: NetworkNone,
	})
	require.NoError(t, err)
	assert.Equal(t, StatusPendingApproval, newSbx.Status)
}

func TestManager_ApproveSandbox_ConcurrentApprovalWhileCreating(t *testing.T) {
	tempDir := t.TempDir()
	cfg := Config{
		DataDir:            tempDir,
		Enabled:            true,
		Drivers:            []string{"docker"},
		AllowedImages:      []string{"golang:alpine"},
		MaxLifetime:        30 * time.Minute,
		DefaultExecTimeout: 1 * time.Minute,
		MaxExecTimeout:     10 * time.Minute,
	}

	createStarted := make(chan struct{})
	createBlock := make(chan struct{})

	blockingDriver := &blockingMockDriver{
		driverType: DriverDocker,
		available:  true,
		onCreate: func() {
			close(createStarted)
			<-createBlock
		},
	}

	mgr := NewManager(cfg, []Driver{blockingDriver})
	defer func() { _ = mgr.Close() }()

	ctx := context.Background()
	_, err := mgr.RequestSandbox(ctx, "user1", "chat1", RequestParams{
		Driver:      DriverDocker,
		DockerImage: "golang:alpine",
		NetworkMode: NetworkNone,
		Reason:      "Slow image pull reproduction",
	})
	require.NoError(t, err)

	// Goroutine 1 starts approval, which triggers driver.Create (simulating slow image pull)
	var sbx1 *UserSandbox
	var err1 error
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		sbx1, err1 = mgr.ApproveSandbox(ctx, "user1")
	}()

	// Wait until driver.Create is running
	<-createStarted

	// Goroutine 2 calls ApproveSandbox concurrently while status is StatusCreating
	var sbx2 *UserSandbox
	var err2 error
	wg.Add(1)
	go func() {
		defer wg.Done()
		sbx2, err2 = mgr.ApproveSandbox(ctx, "user1")
	}()

	// Allow a tiny moment to ensure goroutine 2 reaches ApproveSandbox
	time.Sleep(50 * time.Millisecond)

	// Unblock driver.Create (image pull finishes)
	close(createBlock)
	wg.Wait()

	require.NoError(t, err1, "First approval must succeed")
	require.NoError(t, err2, "Concurrent approval during creation must also succeed")
	assert.Same(t, sbx1, sbx2, "Both callers must receive the exact same UserSandbox instance")
	assert.Equal(t, StatusRunning, sbx1.GetStatus())

	// Only one caller may claim the continuation
	c1 := sbx1.ClaimContinuation()
	c2 := sbx2.ClaimContinuation()
	assert.True(t, (c1 && !c2) || (!c1 && c2), "Exactly one caller must successfully claim continuation")

	// Verify driver.Create was only invoked once
	blockingDriver.mu.Lock()
	assert.Equal(t, 1, blockingDriver.createdCount, "Driver.Create must be invoked exactly once")
	blockingDriver.mu.Unlock()
}

func TestManager_ApproveSandbox_AlreadyRunning(t *testing.T) {
	tempDir := t.TempDir()
	cfg := Config{
		DataDir:            tempDir,
		Enabled:            true,
		Drivers:            []string{"docker"},
		AllowedImages:      []string{"golang:alpine"},
		MaxLifetime:        30 * time.Minute,
		DefaultExecTimeout: 1 * time.Minute,
		MaxExecTimeout:     10 * time.Minute,
	}

	mockDocker := &mockDriver{
		driverType: DriverDocker,
		available:  true,
	}
	mgr := NewManager(cfg, []Driver{mockDocker})
	defer func() { _ = mgr.Close() }()

	ctx := context.Background()
	_, err := mgr.RequestSandbox(ctx, "user1", "chat1", RequestParams{
		Driver:      DriverDocker,
		DockerImage: "golang:alpine",
		NetworkMode: NetworkNone,
		Reason:      "Already running test",
	})
	require.NoError(t, err)

	// First approval succeeds
	sbx1, err := mgr.ApproveSandbox(ctx, "user1")
	require.NoError(t, err)
	assert.Equal(t, StatusRunning, sbx1.GetStatus())
	assert.True(t, sbx1.ClaimContinuation(), "First approval claims continuation")

	// Second approval on already running sandbox succeeds idempotently
	sbx2, err := mgr.ApproveSandbox(ctx, "user1")
	require.NoError(t, err)
	assert.Same(t, sbx1, sbx2)
	assert.False(t, sbx2.ClaimContinuation(), "Subsequent approval does not reclaim continuation")
}

func TestManager_Destroy_WhileCreating(t *testing.T) {
	tempDir := t.TempDir()
	cfg := Config{
		DataDir:            tempDir,
		Enabled:            true,
		Drivers:            []string{"docker"},
		AllowedImages:      []string{"golang:alpine"},
		MaxLifetime:        30 * time.Minute,
		DefaultExecTimeout: 1 * time.Minute,
		MaxExecTimeout:     10 * time.Minute,
	}

	createStarted := make(chan struct{})

	blockingDriver := &blockingMockDriver{
		driverType: DriverDocker,
		available:  true,
		onCreateCtx: func(ctx context.Context) error {
			close(createStarted)
			<-ctx.Done()
			return ctx.Err()
		},
	}

	mgr := NewManager(cfg, []Driver{blockingDriver})
	defer func() { _ = mgr.Close() }()

	ctx := context.Background()
	_, err := mgr.RequestSandbox(ctx, "user1", "chat1", RequestParams{
		Driver:      DriverDocker,
		DockerImage: "golang:alpine",
		NetworkMode: NetworkNone,
		Reason:      "Destroy while creating test",
	})
	require.NoError(t, err)

	// Goroutine starts approval
	var sbx *UserSandbox
	var approveErr error
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		sbx, approveErr = mgr.ApproveSandbox(ctx, "user1")
	}()

	<-createStarted

	// Call Destroy while creation is in-flight
	destroyErr := mgr.Destroy(ctx, "user1")
	require.NoError(t, destroyErr)

	wg.Wait()
	require.Error(t, approveErr)
	assert.Nil(t, sbx)
	assert.Contains(t, approveErr.Error(), "sandbox creation was cancelled")

	// Verify sandbox is gone from manager
	_, exists := mgr.GetStatus("user1")
	assert.False(t, exists)

	// Verify driver Destroy was invoked for rollback cleanup
	blockingDriver.mu.Lock()
	assert.Equal(t, 1, blockingDriver.destroyedCount)
	blockingDriver.mu.Unlock()
}
