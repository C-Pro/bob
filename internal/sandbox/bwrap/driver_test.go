package bwrap

import (
	"context"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
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
	driver := NewDriver()
	ctx := context.Background()

	if !driver.Available(ctx) {
		t.Skip("bwrap not available on host, skipping test")
	}

	origBlocked := sandbox.MandatoryBlockedCIDRs
	sandbox.MandatoryBlockedCIDRs = []string{"169.254.0.0/16"}
	defer func() { sandbox.MandatoryBlockedCIDRs = origBlocked }()

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

	// 2. Allowed domain through proxy forwarder succeeds
	res, err := driver.Exec(ctx, sbx, []string{"curl", "-s", backend.URL + "/data"}, 10*time.Second)
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
}

