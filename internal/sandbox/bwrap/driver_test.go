package bwrap

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"bob/internal/sandbox"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestBwrapDriver(t *testing.T) {
	driver := NewDriver()
	ctx := context.Background()

	if !driver.Available(ctx) {
		t.Skip("bwrap not available on host, skipping test")
	}

	assert.Equal(t, sandbox.DriverBwrap, driver.Type())

	tempDir := t.TempDir()
	workspace := filepath.Join(tempDir, "user_123")
	require.NoError(t, os.MkdirAll(workspace, 0o755))

	sbx := &sandbox.UserSandbox{
		UserID: "testuser1",
		Network: sandbox.NetworkPolicy{
			Mode: sandbox.NetworkNone,
		},
		Status: sandbox.StatusRunning,
	}

	err := driver.Create(ctx, sbx, workspace)
	require.NoError(t, err)
	defer func() { _ = driver.Destroy(ctx, sbx) }()

	// 1. Basic command execution
	res, err := driver.Exec(ctx, sbx, []string{"echo", "hello sandbox"}, 5*time.Second)
	require.NoError(t, err)
	assert.Equal(t, 0, res.ExitCode)
	assert.Contains(t, res.Stdout, "hello sandbox")

	// 2. Test filesystem isolation (host /home or other paths should not exist or be accessible)
	res, err = driver.Exec(ctx, sbx, []string{"sh", "-c", "cat /nonexistent_secret 2>&1"}, 5*time.Second)
	require.NoError(t, err)
	assert.NotEqual(t, 0, res.ExitCode)

	// 3. Test workspace write persistence
	res, err = driver.Exec(ctx, sbx, []string{"sh", "-c", "echo 'test content' > /workspace/output.txt && cat /workspace/output.txt"}, 5*time.Second)
	require.NoError(t, err)
	assert.Equal(t, 0, res.ExitCode)
	assert.Contains(t, res.Stdout, "test content")

	// 4. Test network isolation (mode none should fail to connect anywhere)
	res, err = driver.Exec(ctx, sbx, []string{"sh", "-c", "curl -m 1 http://1.1.1.1 2>&1"}, 5*time.Second)
	require.NoError(t, err)
	assert.NotEqual(t, 0, res.ExitCode) // curl should fail completely
}

func TestBwrap_NetworkRestrictedAirgap(t *testing.T) {
	driver := NewDriverWithConfig(Config{DataDir: t.TempDir()})
	ctx := context.Background()

	if !driver.Available(ctx) {
		t.Skip("bwrap not available on host, skipping test")
	}

	driver.SetCustomBlockedCIDRs([]string{"169.254.0.0/16"})

	// 1. Backend server on host
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("restricted-airgap-ok"))
	}))
	defer backend.Close()

	backendURL, err := url.Parse(backend.URL)
	require.NoError(t, err)

	tempDir := t.TempDir()
	workspace := filepath.Join(tempDir, "user_restricted")
	require.NoError(t, os.MkdirAll(workspace, 0o755))

	sbx := &sandbox.UserSandbox{
		UserID: "testuser_restr",
		Network: sandbox.NetworkPolicy{
			Mode:         sandbox.NetworkRestricted,
			AllowedHosts: []string{backendURL.Hostname()},
			BlockedHosts: []string{"forbidden.com"},
		},
		Status: sandbox.StatusRunning,
	}

	err = driver.Create(ctx, sbx, workspace)
	require.NoError(t, err)
	defer func() { _ = driver.Destroy(ctx, sbx) }()

	// 2. Allowed domain through proxy forwarder succeeds (explicitly bypass no_proxy for 127.0.0.1 test backend)
	res, err := driver.Exec(ctx, sbx, []string{"curl", "--noproxy", "", "-s", backend.URL + "/data"}, 10*time.Second)
	require.NoError(t, err)
	assert.Equal(t, 0, res.ExitCode, "curl output: stdout=%s stderr=%s", res.Stdout, res.Stderr)
	assert.Contains(t, res.Stdout, "restricted-airgap-ok")

	// 3. Raw socket bypass to external IP fails with kernel network unreachable error
	bypassCmd := []string{
		"python3", "-c",
		"import socket; s = socket.socket(); s.settimeout(1); s.connect(('1.1.1.1', 80))",
	}
	res, err = driver.Exec(ctx, sbx, bypassCmd, 5*time.Second)
	require.NoError(t, err)
	assert.NotEqual(t, 0, res.ExitCode)
	assert.Contains(t, res.Stderr, "Network is unreachable")

	// 4. Request to forbidden host through proxy is denied (403 Forbidden)
	res, err = driver.Exec(ctx, sbx, []string{"curl", "-s", "-w", "%{http_code}", "-o", "/dev/null", "http://forbidden.com"}, 5*time.Second)
	require.NoError(t, err)
	assert.Equal(t, "403", res.Stdout)

	// 5. Verify no_proxy / NO_PROXY environment variables are set
	envRes, err := driver.Exec(ctx, sbx, []string{"sh", "-c", "echo $no_proxy $NO_PROXY"}, 5*time.Second)
	require.NoError(t, err)
	assert.Equal(t, 0, envRes.ExitCode)
	assert.Contains(t, envRes.Stdout, "localhost,127.0.0.1 localhost,127.0.0.1")

	// 6. In-sandbox local server is accessible directly via loopback without proxy rejection (Finding F7)
	localRes, err := driver.Exec(ctx, sbx, []string{"sh", "-c", "python3 -m http.server 34567 --bind 127.0.0.1 >/dev/null 2>&1 & PID=$!; for i in $(seq 1 30); do out=$(curl -s http://127.0.0.1:34567/ 2>/dev/null); if [ -n \"$out\" ]; then echo \"$out\"; break; fi; sleep 0.1; done; kill $PID 2>/dev/null || true"}, 5*time.Second)
	require.NoError(t, err)
	assert.Equal(t, 0, localRes.ExitCode)
	assert.Contains(t, localRes.Stdout, "Directory listing")
}

func TestBwrapDriver_ResolvConf(t *testing.T) {
	driver := NewDriver()
	ctx := context.Background()

	if !driver.Available(ctx) {
		t.Skip("bwrap not available on host, skipping test")
	}

	hostResolv, err := os.ReadFile("/etc/resolv.conf")
	if os.IsNotExist(err) {
		t.Skip("/etc/resolv.conf not found on host, skipping test")
	}
	require.NoError(t, err)

	tempDir := t.TempDir()
	workspace := filepath.Join(tempDir, "user_dns")
	require.NoError(t, os.MkdirAll(workspace, 0o755))

	sbx := &sandbox.UserSandbox{
		UserID: "testuser_dns",
		Network: sandbox.NetworkPolicy{
			Mode: sandbox.NetworkFull,
		},
		Status: sandbox.StatusRunning,
	}

	err = driver.Create(ctx, sbx, workspace)
	require.NoError(t, err)
	defer func() { _ = driver.Destroy(ctx, sbx) }()

	res, err := driver.Exec(ctx, sbx, []string{"cat", "/etc/resolv.conf"}, 5*time.Second)
	require.NoError(t, err)
	t.Logf("stdout: %q, stderr: %q", res.Stdout, res.Stderr)
	assert.Equal(t, 0, res.ExitCode, "cat /etc/resolv.conf failed: %s", res.Stderr)
	assert.Equal(t, string(hostResolv), res.Stdout)
}

func TestBwrapDriver_BoundedOutput(t *testing.T) {
	driver := NewDriver()
	ctx := context.Background()

	if !driver.Available(ctx) {
		t.Skip("bwrap not available on host, skipping test")
	}

	tempDir := t.TempDir()
	workspace := filepath.Join(tempDir, "user_bounded")
	require.NoError(t, os.MkdirAll(workspace, 0o755))

	sbx := &sandbox.UserSandbox{
		UserID: "testuser_bounded",
		Network: sandbox.NetworkPolicy{
			Mode: sandbox.NetworkNone,
		},
		Status: sandbox.StatusRunning,
	}

	err := driver.Create(ctx, sbx, workspace)
	require.NoError(t, err)
	defer func() { _ = driver.Destroy(ctx, sbx) }()

	// Generate 2MB of output with sh
	res, err := driver.Exec(ctx, sbx, []string{"sh", "-c", "head -c 2097152 /dev/zero | tr '\\0' 'A'"}, 5*time.Second)
	require.NoError(t, err)
	assert.Contains(t, res.Stdout, "[output truncated")
	assert.LessOrEqual(t, len(res.Stdout), sandbox.DefaultMaxOutputBytes+200)
}

func TestBwrapDriver_ExecTimeout(t *testing.T) {
	driver := NewDriver()
	ctx := context.Background()

	if !driver.Available(ctx) {
		t.Skip("bwrap not available on host, skipping test")
	}

	tempDir := t.TempDir()
	workspace := filepath.Join(tempDir, "user_timeout")
	require.NoError(t, os.MkdirAll(workspace, 0o755))

	sbx := &sandbox.UserSandbox{
		UserID: "testuser_timeout",
		Network: sandbox.NetworkPolicy{
			Mode: sandbox.NetworkNone,
		},
		Status: sandbox.StatusRunning,
	}

	err := driver.Create(ctx, sbx, workspace)
	require.NoError(t, err)
	defer func() { _ = driver.Destroy(ctx, sbx) }()

	res, err := driver.Exec(ctx, sbx, []string{"sleep", "10"}, 50*time.Millisecond)
	require.NoError(t, err)
	require.NotNil(t, res)
	assert.Equal(t, -1, res.ExitCode)
	assert.Contains(t, res.Stderr, "command timed out")
}

func TestBuildSystemdArgs(t *testing.T) {
	// Case 1: Default mem, positive cpuLimit, default tasksMax (0 -> 1024)
	args := buildSystemdArgs(0, 1.5, 0, "/usr/bin/bwrap", []string{"--ro-bind", "/usr", "/usr"})
	assert.Contains(t, args, "MemoryMax=512M")
	assert.Contains(t, args, "TasksMax=1024")
	assert.Contains(t, args, "CPUQuota=150%")
	assert.Contains(t, args, "/usr/bin/bwrap")
	assert.Contains(t, args, "--ro-bind")

	// Case 2: Custom mem, zero cpuLimit, custom tasksMax
	args2 := buildSystemdArgs(1024, 0, 2048, "/usr/bin/bwrap", []string{"--unshare-all"})
	assert.Contains(t, args2, "MemoryMax=1024M")
	assert.Contains(t, args2, "TasksMax=2048")
	assert.NotContains(t, strings.Join(args2, " "), "CPUQuota")
}

func TestBwrapDriver_Config_CustomBlockedCIDRs(t *testing.T) {
	customCIDRs := []string{"10.0.0.0/8", "192.168.1.0/24"}
	driver := NewDriverWithConfig(Config{
		CustomBlockedCIDRs: customCIDRs,
	})
	assert.Equal(t, customCIDRs, driver.customBlockedCIDRs)
}

func TestBwrapDriver_Config_ForwarderPort(t *testing.T) {
	driver := NewDriverWithConfig(Config{
		ForwarderPort: 19090,
	})
	assert.Equal(t, 19090, driver.forwarderPort)

	driverDefault := NewDriver()
	assert.Equal(t, sandbox.DefaultForwarderPort, driverDefault.forwarderPort)
}

func TestBwrapDriver_Exec_NoProxyWhenProxyNil(t *testing.T) {
	driver := NewDriver()
	if !driver.Available(context.Background()) {
		t.Skip("bwrap not available")
	}

	tempDir := t.TempDir()
	workspace := filepath.Join(tempDir, "workspace")
	require.NoError(t, os.MkdirAll(workspace, 0o755))

	sbx := &sandbox.UserSandbox{
		UserID: "user_bwrap_no_proxy",
		Network: sandbox.NetworkPolicy{
			Mode: sandbox.NetworkRestricted,
		},
		Status: sandbox.StatusRunning,
	}
	// Note: driver.proxies[sbx.UserID] is nil (Create was not run)
	res, err := driver.Exec(context.Background(), sbx, []string{"sh", "-c", "echo http_proxy=$http_proxy"}, 5*time.Second)
	require.NoError(t, err)
	assert.Equal(t, "http_proxy=\n", res.Stdout)
}

func TestBwrapDriver_Config_DataDir_And_ForwarderBinaryCache(t *testing.T) {
	tempDataDir := t.TempDir()
	driver := NewDriverWithConfig(Config{
		DataDir: tempDataDir,
	})
	assert.Equal(t, tempDataDir, driver.dataDir)

	fwdPath, err := driver.getForwarderBinary()
	require.NoError(t, err)
	assert.NotEmpty(t, fwdPath)
	assert.Equal(t, fwdPath, driver.forwarderBinary)

	// Calling again should return cached path without re-resolving
	fwdPath2, err := driver.getForwarderBinary()
	require.NoError(t, err)
	assert.Equal(t, fwdPath, fwdPath2)
}

func TestBwrapDriver_BuildArgs_NetworkRestricted(t *testing.T) {
	tempDir := t.TempDir()
	driver := NewDriverWithConfig(Config{
		DataDir: filepath.Join(tempDir, "data"),
	})

	sbx := &sandbox.UserSandbox{
		UserID: "testuser_buildargs",
		Network: sandbox.NetworkPolicy{
			Mode: sandbox.NetworkRestricted,
		},
	}
	sbx.SetWorkspaceDir(filepath.Join(tempDir, "workspace"))

	mockProxy, err := sandbox.NewFilteringProxyWithConfig(sandbox.ProxyConfig{
		Policy:     sbx.Network,
		ListenTCP:  "none",
		SocketPath: filepath.Join(tempDir, "proxy.sock"),
	})
	require.NoError(t, err)
	defer func() { _ = mockProxy.Close() }()

	driver.mu.Lock()
	driver.proxies[sbx.UserID] = mockProxy
	driver.mu.Unlock()

	args, err := driver.buildArgs(sbx, []string{"echo", "hi"})
	require.NoError(t, err)

	// Verify isolation flags
	assert.Contains(t, args, "--unshare-all")
	assert.Contains(t, args, "--dir")
	assert.Contains(t, args, "/run/proxy")
	assert.Contains(t, args, "/run/proxy.sock")
	assert.Contains(t, args, "/run/proxy/fwd")

	// Verify environment variables
	assert.Contains(t, args, "http_proxy")
	assert.Contains(t, args, fmt.Sprintf("http://127.0.0.1:%d", driver.forwarderPort))
	assert.Contains(t, args, "no_proxy")
	assert.Contains(t, args, "localhost,127.0.0.1")

	expectedSuffix := []string{
		"/run/proxy/fwd",
		"-tcp", fmt.Sprintf("127.0.0.1:%d", driver.forwarderPort),
		"-sock", "/run/proxy.sock",
		"--",
		"echo", "hi",
	}
	require.True(t, len(args) >= len(expectedSuffix))
	assert.Equal(t, expectedSuffix, args[len(args)-len(expectedSuffix):])
}

func TestBwrapDriver_Exec_EnsureForwarderBinaryFailure(t *testing.T) {
	tempDir := t.TempDir()
	driver := NewDriverWithConfig(Config{
		DataDir:      filepath.Join(tempDir, "data"),
		ProxyFwdPath: filepath.Join(tempDir, "nonexistent", "fwd"),
	})
	driver.bwrapPath = "/bin/true"

	sbx := &sandbox.UserSandbox{
		UserID: "testuser_fail_fwd",
		Network: sandbox.NetworkPolicy{
			Mode: sandbox.NetworkRestricted,
		},
	}
	sbx.SetWorkspaceDir(filepath.Join(tempDir, "workspace"))

	mockProxy, err := sandbox.NewFilteringProxyWithConfig(sandbox.ProxyConfig{
		Policy:     sbx.Network,
		ListenTCP:  "none",
		SocketPath: filepath.Join(tempDir, "proxy.sock"),
	})
	require.NoError(t, err)
	defer func() { _ = mockProxy.Close() }()

	driver.mu.Lock()
	driver.proxies[sbx.UserID] = mockProxy
	driver.mu.Unlock()

	res, err := driver.Exec(context.Background(), sbx, []string{"echo", "hi"}, 5*time.Second)
	require.Error(t, err)
	assert.Nil(t, res)
	assert.Contains(t, err.Error(), "failed to ensure forwarder binary")
}

func TestBwrapDriver_Create_LongAndSpecialUserID(t *testing.T) {
	tempDir := t.TempDir()
	driver := NewDriverWithConfig(Config{
		DataDir: filepath.Join(tempDir, "data"),
	})
	driver.bwrapPath = "/bin/true"

	fwdPath := filepath.Join(tempDir, "data", "bin", "bob-proxy-fwd")
	require.NoError(t, os.MkdirAll(filepath.Dir(fwdPath), 0o755))
	require.NoError(t, os.WriteFile(fwdPath, []byte("#!/bin/sh\nexit 0\n"), 0o755))

	sbx := &sandbox.UserSandbox{
		UserID: "team/special/user-with-very-long-id-" + strings.Repeat("a", 100),
		Network: sandbox.NetworkPolicy{
			Mode: sandbox.NetworkRestricted,
		},
	}
	workspace := filepath.Join(tempDir, "workspace")

	ctx := context.Background()
	err := driver.Create(ctx, sbx, workspace)
	require.NoError(t, err)
	defer func() { _ = driver.Destroy(ctx, sbx) }()

	driver.mu.Lock()
	p := driver.proxies[sbx.UserID]
	driver.mu.Unlock()
	require.NotNil(t, p)
	assert.FileExists(t, p.SocketPath())
}

func TestBwrapDriver_GetForwarderBinary_DoesNotBlockMu(t *testing.T) {
	tempDir := t.TempDir()
	driver := NewDriverWithConfig(Config{
		DataDir: filepath.Join(tempDir, "data"),
	})
	driver.bwrapPath = "/bin/true"

	fwdPath := filepath.Join(tempDir, "data", "bin", "bob-proxy-fwd")
	require.NoError(t, os.MkdirAll(filepath.Dir(fwdPath), 0o755))
	require.NoError(t, os.WriteFile(fwdPath, []byte("#!/bin/sh\nexit 0\n"), 0o755))

	// Acquire fwdMu and verify d.mu operations (like Available) are unblocked
	driver.fwdMu.Lock()
	done := make(chan bool, 1)
	go func() {
		// Available acquires d.mu; it should not block on fwdMu
		_ = driver.Available(context.Background())
		done <- true
	}()

	select {
	case <-done:
		// success: Available did not block
	case <-time.After(500 * time.Millisecond):
		t.Fatal("driver.Available() blocked while fwdMu was held")
	}
	driver.fwdMu.Unlock()
}

func TestBwrapDriver_ExternalWorkspace(t *testing.T) {
	driver := NewDriver()
	ctx := context.Background()

	if !driver.Available(ctx) {
		t.Skip("bwrap not available on host, skipping test")
	}

	tempDir := t.TempDir()
	externalWS := filepath.Join(tempDir, "host_ext_ws")
	require.NoError(t, os.MkdirAll(externalWS, 0o750))
	preStat, err := os.Stat(externalWS)
	require.NoError(t, err)

	t.Run("external workspace read-only mount", func(t *testing.T) {
		sbx := &sandbox.UserSandbox{
			UserID:            "ext-user-ro",
			Status:            sandbox.StatusRunning,
			WorkspaceDir:      externalWS,
			WorkspaceExternal: true,
			Mounts: []sandbox.UserMount{
				{
					RelativePath: ".",
					SandboxPath:  sandbox.DefaultWorkspaceMountPath,
					ReadOnly:     true,
				},
			},
		}

		err := driver.Create(ctx, sbx, externalWS)
		require.NoError(t, err)
		defer func() { _ = driver.Destroy(ctx, sbx) }()

		// Verify host permissions not modified
		postStat, err := os.Stat(externalWS)
		require.NoError(t, err)
		assert.Equal(t, preStat.Mode(), postStat.Mode())

		// Write inside sandbox should fail
		res, err := driver.Exec(ctx, sbx, []string{"sh", "-c", "echo test > /workspace/fail.txt"}, 5*time.Second)
		require.NoError(t, err)
		assert.NotEqual(t, 0, res.ExitCode)
		assert.Contains(t, res.Stderr, "Read-only file system")
	})

	t.Run("external workspace read-write mount", func(t *testing.T) {
		sbx := &sandbox.UserSandbox{
			UserID:            "ext-user-rw",
			Status:            sandbox.StatusRunning,
			WorkspaceDir:      externalWS,
			WorkspaceExternal: true,
			Mounts: []sandbox.UserMount{
				{
					RelativePath: ".",
					SandboxPath:  sandbox.DefaultWorkspaceMountPath,
					ReadOnly:     false,
				},
			},
		}

		err := driver.Create(ctx, sbx, externalWS)
		require.NoError(t, err)
		defer func() { _ = driver.Destroy(ctx, sbx) }()

		// Write inside sandbox should succeed and persist on host
		res, err := driver.Exec(ctx, sbx, []string{"sh", "-c", "echo 'persisted' > /workspace/saved.txt"}, 5*time.Second)
		require.NoError(t, err)
		assert.Equal(t, 0, res.ExitCode)

		content, err := os.ReadFile(filepath.Join(externalWS, "saved.txt"))
		require.NoError(t, err)
		assert.Equal(t, "persisted\n", string(content))
	})
}

func TestBwrap_ExtraPaths(t *testing.T) {
	driver := NewDriver()
	ctx := context.Background()

	if !driver.Available(ctx) {
		t.Skip("bwrap not available on host, skipping test")
	}

	tempDir := t.TempDir()
	realTemp, err := filepath.EvalSymlinks(tempDir)
	require.NoError(t, err)

	// Create a mock custom toolchain directory
	customBinDir := filepath.Join(realTemp, "custom_toolchain", "bin")
	require.NoError(t, os.MkdirAll(customBinDir, 0o755))

	mockToolScript := filepath.Join(customBinDir, "my-custom-cli")
	scriptContent := "#!/bin/sh\necho custom-toolchain-works\n"
	require.NoError(t, os.WriteFile(mockToolScript, []byte(scriptContent), 0o755))

	workspace := filepath.Join(realTemp, "workspace")
	require.NoError(t, os.MkdirAll(workspace, 0o755))

	sbx := &sandbox.UserSandbox{
		UserID: "user-extra-path",
		Status: sandbox.StatusRunning,
		Network: sandbox.NetworkPolicy{
			Mode: sandbox.NetworkNone,
		},
		ExtraPaths: []string{filepath.Dir(customBinDir)}, // toolchain root with bin/ subdir
	}

	err = driver.Create(ctx, sbx, workspace)
	require.NoError(t, err)
	defer func() { _ = driver.Destroy(ctx, sbx) }()

	// 1. Verify custom executable is found via PATH lookup
	res, err := driver.Exec(ctx, sbx, []string{"my-custom-cli"}, 5*time.Second)
	require.NoError(t, err)
	assert.Equal(t, 0, res.ExitCode)
	assert.Contains(t, res.Stdout, "custom-toolchain-works")

	// 2. Verify cache directories and cache environment variables
	res, err = driver.Exec(ctx, sbx, []string{"sh", "-c", "echo GOCACHE=$GOCACHE; ls -d /tmp/bob-cache/go-build"}, 5*time.Second)
	require.NoError(t, err)
	assert.Equal(t, 0, res.ExitCode)
	assert.Contains(t, res.Stdout, "GOCACHE=/tmp/bob-cache/go-build")
	assert.Contains(t, res.Stdout, "/tmp/bob-cache/go-build")

	// 3. Verify invalid extra path returns error
	invalidSbx := &sandbox.UserSandbox{
		UserID: "user-invalid-path",
		Status: sandbox.StatusRunning,
		Network: sandbox.NetworkPolicy{
			Mode: sandbox.NetworkNone,
		},
		ExtraPaths: []string{"/does/not/exist/at/all"},
	}
	_, err = driver.Exec(ctx, invalidSbx, []string{"true"}, 5*time.Second)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "invalid extra path")
}

func TestBwrap_IsolatedHomeWritableOnReadOnlyWorkspace(t *testing.T) {
	driver := NewDriver()
	ctx := context.Background()

	if !driver.Available(ctx) {
		t.Skip("bwrap not available on host, skipping test")
	}

	tempDir := t.TempDir()
	workspace := filepath.Join(tempDir, "workspace")
	require.NoError(t, os.MkdirAll(workspace, 0o755))

	cacheDir := filepath.Join(tempDir, "cache")
	require.NoError(t, os.MkdirAll(cacheDir, 0o755))

	sbx := &sandbox.UserSandbox{
		UserID:            "user-ro-home",
		Status:            sandbox.StatusRunning,
		WorkspaceExternal: true,
		Mounts: []sandbox.UserMount{
			{
				RelativePath: ".",
				SandboxPath:  sandbox.DefaultWorkspaceMountPath,
				ReadOnly:     true,
			},
		},
		Network: sandbox.NetworkPolicy{
			Mode: sandbox.NetworkNone,
		},
	}
	sbx.SetWorkspaceDir(workspace)
	sbx.SetCacheDir(cacheDir)

	err := driver.Create(ctx, sbx, workspace)
	require.NoError(t, err)
	defer func() { _ = driver.Destroy(ctx, sbx) }()

	// 1. Verify workspace is indeed read-only
	res, err := driver.Exec(ctx, sbx, []string{"sh", "-c", "touch /workspace/forbidden.txt"}, 5*time.Second)
	require.NoError(t, err)
	assert.NotEqual(t, 0, res.ExitCode, "workspace should be read-only")
	assert.Contains(t, res.Stderr, "Read-only file system")

	// 2. Verify $HOME is /tmp/bob-cache/home and is writable
	res, err = driver.Exec(ctx, sbx, []string{"sh", "-c", "echo $HOME; echo 'hello from sandbox' > $HOME/persist.txt"}, 5*time.Second)
	require.NoError(t, err)
	assert.Equal(t, 0, res.ExitCode, "writing to $HOME should succeed: %s", res.Stderr)
	assert.Contains(t, res.Stdout, "/tmp/bob-cache/home")

	// 3. Verify file written to $HOME persists across commands within the active run cache
	res, err = driver.Exec(ctx, sbx, []string{"sh", "-c", "cat $HOME/persist.txt"}, 5*time.Second)
	require.NoError(t, err)
	assert.Equal(t, 0, res.ExitCode)
	assert.Contains(t, res.Stdout, "hello from sandbox")

	// 4. Verify host cache directory contains the persisted file
	hostPersistFile := filepath.Join(cacheDir, "home", "persist.txt")
	content, err := os.ReadFile(hostPersistFile)
	require.NoError(t, err)
	assert.Equal(t, "hello from sandbox\n", string(content))
}

func TestBwrap_ToolchainPrecedenceEndToEnd(t *testing.T) {
	driver := NewDriver()
	ctx := context.Background()

	if !driver.Available(ctx) {
		t.Skip("bwrap not available on host, skipping test")
	}

	tempDir := t.TempDir()
	homeDir := filepath.Join(tempDir, "home")
	t.Setenv("HOME", homeDir)

	miseDir := filepath.Join(homeDir, ".local", "share", "mise", "installs")

	// 1. Mock Go toolchain containing an older golangci-lint
	goBin := filepath.Join(miseDir, "go", "1.27.1", "bin")
	require.NoError(t, os.MkdirAll(goBin, 0o755))
	goLint := filepath.Join(goBin, "golangci-lint")
	require.NoError(t, os.WriteFile(goLint, []byte("#!/bin/sh\necho v1.64.8-shadowed\n"), 0o755))

	// 2. Mock dedicated golangci-lint toolchain
	dedicatedDir := filepath.Join(miseDir, "golangci-lint", "2.13.2", "golangci-lint-2.13.2-linux-amd64")
	require.NoError(t, os.MkdirAll(dedicatedDir, 0o755))
	dedicatedLint := filepath.Join(dedicatedDir, "golangci-lint")
	require.NoError(t, os.WriteFile(dedicatedLint, []byte("#!/bin/sh\necho v2.13.2-dedicated\n"), 0o755))

	// Host PATH deliberately lists go before dedicated linter
	pathEnv := strings.Join([]string{goBin, dedicatedDir, "/usr/bin"}, string(filepath.ListSeparator))
	discovered := sandbox.DiscoverToolchains(pathEnv, tempDir, nil)
	require.Len(t, discovered, 2)

	workspace := filepath.Join(tempDir, "workspace")
	require.NoError(t, os.MkdirAll(workspace, 0o755))

	sbx := &sandbox.UserSandbox{
		UserID:     "user-tool-precedence",
		Status:     sandbox.StatusRunning,
		Network:    sandbox.NetworkPolicy{Mode: sandbox.NetworkNone},
		ExtraPaths: discovered,
	}

	err := driver.Create(ctx, sbx, workspace)
	require.NoError(t, err)
	defer func() { _ = driver.Destroy(ctx, sbx) }()

	res, err := driver.Exec(ctx, sbx, []string{"golangci-lint"}, 5*time.Second)
	require.NoError(t, err)
	assert.Equal(t, 0, res.ExitCode)
	assert.Equal(t, "v2.13.2-dedicated\n", res.Stdout)
}
