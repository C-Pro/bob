package sandbox

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type provisionMockDriver struct {
	driverType   DriverType
	available    bool
	createdCount int
	lastSbx      *UserSandbox
	lastWS       string
	createDelay  time.Duration
	createErr    error
	destroyCount int
}

func (m *provisionMockDriver) Type() DriverType {
	return m.driverType
}

func (m *provisionMockDriver) Available(ctx context.Context) bool {
	return m.available
}

func (m *provisionMockDriver) Create(ctx context.Context, sbx *UserSandbox, workspace string) error {
	m.createdCount++
	m.lastSbx = sbx
	m.lastWS = workspace
	if m.createDelay > 0 {
		select {
		case <-time.After(m.createDelay):
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	return m.createErr
}

func (m *provisionMockDriver) Exec(ctx context.Context, sbx *UserSandbox, cmd []string, timeout time.Duration) (*ExecResult, error) {
	return &ExecResult{ExitCode: 0}, nil
}

func (m *provisionMockDriver) Destroy(ctx context.Context, sbx *UserSandbox) error {
	m.destroyCount++
	return nil
}

func TestManagerProvision_Success(t *testing.T) {
	tmpDir := t.TempDir()
	wsDir := filepath.Join(tmpDir, "host_workspace")
	require.NoError(t, os.Mkdir(wsDir, 0755))

	cfg := Config{
		DataDir:             tmpDir,
		Enabled:             true,
		Drivers:             []string{"bwrap"},
		AllowedNetworkModes: []string{"none", "restricted"},
		MaxLifetime:         2 * time.Hour,
	}

	driver := &provisionMockDriver{
		driverType: DriverBwrap,
		available:  true,
	}
	mgr := NewManager(cfg, []Driver{driver})
	defer func() { _ = mgr.Close() }()

	ctx := context.Background()
	params := ProvisionParams{
		Driver:            DriverBwrap,
		NetworkMode:       NetworkNone,
		WorkspaceDir:      wsDir,
		WorkspaceReadOnly: true,
		Lifetime:          30 * time.Minute,
	}

	sbx, err := mgr.Provision(ctx, "cli-user", "cli-chat", params)
	require.NoError(t, err)
	require.NotNil(t, sbx)

	assert.Equal(t, "cli-user", sbx.UserID)
	assert.Equal(t, "cli-chat", sbx.ChatID)
	assert.Equal(t, StatusRunning, sbx.GetStatus())
	assert.True(t, sbx.WorkspaceExternal)
	assert.Equal(t, wsDir, sbx.WorkspaceDir)
	assert.NotEmpty(t, sbx.ID)

	require.Len(t, sbx.Mounts, 1)
	assert.Equal(t, ".", sbx.Mounts[0].RelativePath)
	assert.Equal(t, DefaultWorkspaceMountPath, sbx.Mounts[0].SandboxPath)
	assert.True(t, sbx.Mounts[0].ReadOnly)

	assert.Equal(t, 1, driver.createdCount)
	assert.Equal(t, wsDir, driver.lastWS)

	// Close destroys running sandbox
	err = mgr.Close()
	require.NoError(t, err)
	assert.Equal(t, 1, driver.destroyCount)
}

func TestManagerProvision_Validations(t *testing.T) {
	tmpDir := t.TempDir()
	wsDir := filepath.Join(tmpDir, "workspace")
	require.NoError(t, os.Mkdir(wsDir, 0755))

	cfg := Config{
		DataDir:             tmpDir,
		Enabled:             true,
		Drivers:             []string{"bwrap", "docker"},
		AllowedNetworkModes: []string{"none"},
		AllowedImages:       []string{"alpine:latest"},
		MaxLifetime:         1 * time.Hour,
	}

	driverBwrap := &provisionMockDriver{driverType: DriverBwrap, available: true}
	driverDocker := &provisionMockDriver{driverType: DriverDocker, available: false}
	mgr := NewManager(cfg, []Driver{driverBwrap, driverDocker})
	defer func() { _ = mgr.Close() }()

	ctx := context.Background()

	t.Run("empty userID", func(t *testing.T) {
		_, err := mgr.Provision(ctx, "", "chat", ProvisionParams{
			Driver:       DriverBwrap,
			WorkspaceDir: wsDir,
			NetworkMode:  NetworkNone,
		})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "userID cannot be empty")
	})

	t.Run("empty workspace", func(t *testing.T) {
		_, err := mgr.Provision(ctx, "user1", "chat", ProvisionParams{
			Driver:       DriverBwrap,
			WorkspaceDir: "",
			NetworkMode:  NetworkNone,
		})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "workspace directory cannot be empty")
	})

	t.Run("non-existent workspace", func(t *testing.T) {
		_, err := mgr.Provision(ctx, "user1", "chat", ProvisionParams{
			Driver:       DriverBwrap,
			WorkspaceDir: filepath.Join(tmpDir, "not_real"),
			NetworkMode:  NetworkNone,
		})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "does not exist")
	})

	t.Run("workspace is file not directory", func(t *testing.T) {
		fileWS := filepath.Join(tmpDir, "file_ws")
		require.NoError(t, os.WriteFile(fileWS, []byte("data"), 0644))

		_, err := mgr.Provision(ctx, "user1", "chat", ProvisionParams{
			Driver:       DriverBwrap,
			WorkspaceDir: fileWS,
			NetworkMode:  NetworkNone,
		})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "not a directory")
	})

	t.Run("disallowed network mode", func(t *testing.T) {
		_, err := mgr.Provision(ctx, "user1", "chat", ProvisionParams{
			Driver:       DriverBwrap,
			WorkspaceDir: wsDir,
			NetworkMode:  NetworkFull,
		})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "not permitted by server policy")
	})

	t.Run("unavailable driver", func(t *testing.T) {
		_, err := mgr.Provision(ctx, "user1", "chat", ProvisionParams{
			Driver:       DriverDocker,
			WorkspaceDir: wsDir,
			NetworkMode:  NetworkNone,
		})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "not available")
	})

	t.Run("unrecognized driver", func(t *testing.T) {
		_, err := mgr.Provision(ctx, "user1", "chat", ProvisionParams{
			Driver:       DriverType("unknown"),
			WorkspaceDir: wsDir,
			NetworkMode:  NetworkNone,
		})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "not available")
	})

	t.Run("disallowed docker image", func(t *testing.T) {
		driverDocker.available = true
		_, err := mgr.Provision(ctx, "user1", "chat", ProvisionParams{
			Driver:       DriverDocker,
			DockerImage:  "malicious:latest",
			WorkspaceDir: wsDir,
			NetworkMode:  NetworkNone,
		})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "image")
	})
}

func TestManagerProvision_OnePerUser(t *testing.T) {
	tmpDir := t.TempDir()
	wsDir := filepath.Join(tmpDir, "workspace")
	require.NoError(t, os.Mkdir(wsDir, 0755))

	cfg := Config{
		DataDir:             tmpDir,
		Enabled:             true,
		Drivers:             []string{"bwrap"},
		AllowedNetworkModes: []string{"none"},
	}

	driver := &provisionMockDriver{driverType: DriverBwrap, available: true}
	mgr := NewManager(cfg, []Driver{driver})
	defer func() { _ = mgr.Close() }()

	ctx := context.Background()
	params := ProvisionParams{
		Driver:       DriverBwrap,
		NetworkMode:  NetworkNone,
		WorkspaceDir: wsDir,
		Lifetime:     10 * time.Minute,
	}

	_, err := mgr.Provision(ctx, "user1", "chat", params)
	require.NoError(t, err)

	// Second provisioning attempt for same user should fail
	_, err = mgr.Provision(ctx, "user1", "chat", params)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "already has an active sandbox")
}

func TestManagerProvision_CancellationAndRollback(t *testing.T) {
	tmpDir := t.TempDir()
	wsDir := filepath.Join(tmpDir, "workspace")
	require.NoError(t, os.Mkdir(wsDir, 0755))

	cfg := Config{
		DataDir:             tmpDir,
		Enabled:             true,
		Drivers:             []string{"bwrap"},
		AllowedNetworkModes: []string{"none"},
	}

	driver := &provisionMockDriver{
		driverType:  DriverBwrap,
		available:   true,
		createDelay: 200 * time.Millisecond,
	}
	mgr := NewManager(cfg, []Driver{driver})
	defer func() { _ = mgr.Close() }()

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	params := ProvisionParams{
		Driver:       DriverBwrap,
		NetworkMode:  NetworkNone,
		WorkspaceDir: wsDir,
	}

	_, err := mgr.Provision(ctx, "user_cancel", "chat", params)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "cancelled")
	assert.Equal(t, 1, driver.destroyCount, "driver destroy must be called on cancelled create rollback")

	// Verify user slot is clean and can be re-provisioned
	driver.createDelay = 0
	sbx, err := mgr.Provision(context.Background(), "user_cancel", "chat", params)
	require.NoError(t, err)
	assert.Equal(t, StatusRunning, sbx.GetStatus())
}

func TestManagerProvision_DriverErrorRollback(t *testing.T) {
	tmpDir := t.TempDir()
	wsDir := filepath.Join(tmpDir, "workspace")
	require.NoError(t, os.Mkdir(wsDir, 0755))

	cfg := Config{
		DataDir:             tmpDir,
		Enabled:             true,
		Drivers:             []string{"bwrap"},
		AllowedNetworkModes: []string{"none"},
	}

	driver := &provisionMockDriver{
		driverType: DriverBwrap,
		available:  true,
		createErr:  errors.New("rootfs bind mount failed"),
	}
	mgr := NewManager(cfg, []Driver{driver})
	defer func() { _ = mgr.Close() }()

	params := ProvisionParams{
		Driver:       DriverBwrap,
		NetworkMode:  NetworkNone,
		WorkspaceDir: wsDir,
	}

	_, err := mgr.Provision(context.Background(), "user_fail", "chat", params)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "rootfs bind mount failed")
	assert.Equal(t, 1, driver.destroyCount, "destroy must be called on create error")

	// Slot freed
	driver.createErr = nil
	sbx, err := mgr.Provision(context.Background(), "user_fail", "chat", params)
	require.NoError(t, err)
	assert.Equal(t, StatusRunning, sbx.GetStatus())
}
