package sandbox

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestValidateMountPath(t *testing.T) {
	tempDir := t.TempDir()
	workspace := filepath.Join(tempDir, "user1_workspace")
	require.NoError(t, os.MkdirAll(workspace, 0o755))

	// Valid subpath that doesn't exist yet
	path, isWhole, err := ValidateMountPath(workspace, "my_project/sub")
	require.NoError(t, err)
	assert.False(t, isWhole)
	assert.Equal(t, filepath.Join(workspace, "my_project/sub"), path)

	// Valid subpath that exists as directory
	projDir := filepath.Join(workspace, "existing_proj")
	require.NoError(t, os.MkdirAll(projDir, 0o755))
	path, isWhole, err = ValidateMountPath(workspace, "existing_proj")
	require.NoError(t, err)
	assert.False(t, isWhole)
	assert.Equal(t, projDir, path)

	// Existing path is a regular file (not directory)
	filePath := filepath.Join(workspace, "file.txt")
	require.NoError(t, os.WriteFile(filePath, []byte("data"), 0o600))
	_, _, err = ValidateMountPath(workspace, "file.txt")
	assert.ErrorContains(t, err, "must be a directory")

	// Empty workspace
	_, _, err = ValidateMountPath("", "my_project")
	assert.ErrorContains(t, err, "user workspace directory cannot be empty")

	// Root of workspace mounts
	path, isWhole, err = ValidateMountPath(workspace, "")
	require.NoError(t, err)
	assert.True(t, isWhole)
	assert.Equal(t, workspace, path)

	path, isWhole, err = ValidateMountPath(workspace, ".")
	require.NoError(t, err)
	assert.True(t, isWhole)
	assert.Equal(t, workspace, path)

	path, isWhole, err = ValidateMountPath(workspace, "/")
	require.NoError(t, err)
	assert.True(t, isWhole)
	assert.Equal(t, workspace, path)

	// Absolute path rejection
	_, _, err = ValidateMountPath(workspace, "/etc/passwd")
	assert.ErrorContains(t, err, "must be relative")

	// Traversal attacks
	_, _, err = ValidateMountPath(workspace, "../outside")
	assert.ErrorContains(t, err, "cannot traverse outside")

	_, _, err = ValidateMountPath(workspace, "foo/../../outside")
	assert.ErrorContains(t, err, "cannot traverse outside")

	// Symlink escape test
	outsideDir := filepath.Join(tempDir, "sensitive_host_dir")
	require.NoError(t, os.MkdirAll(outsideDir, 0o755))
	symlinkPath := filepath.Join(workspace, "escaped_link")
	require.NoError(t, os.Symlink(outsideDir, symlinkPath))

	_, _, err = ValidateMountPath(workspace, "escaped_link")
	assert.Error(t, err)
}

func TestValidateSandboxMountPath(t *testing.T) {
	// Empty path should be accepted (defaults to /workspace/<rel>)
	clean, err := ValidateSandboxMountPath("")
	require.NoError(t, err)
	assert.Equal(t, "", clean)

	// Valid sandbox paths
	clean, err = ValidateSandboxMountPath("/workspace/data")
	require.NoError(t, err)
	assert.Equal(t, "/workspace/data", clean)

	clean, err = ValidateSandboxMountPath("/mnt/data")
	require.NoError(t, err)
	assert.Equal(t, "/mnt/data", clean)

	// Relative paths must be rejected
	_, err = ValidateSandboxMountPath("workspace/data")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "must be an absolute path")

	// Root path must be rejected
	_, err = ValidateSandboxMountPath("/")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "cannot be root")

	// Critical system directories must be rejected
	blockedCases := []string{
		"/etc",
		"/etc/passwd",
		"/bin",
		"/bin/sh",
		"/sbin",
		"/usr",
		"/usr/bin",
		"/lib",
		"/lib64",
		"/dev",
		"/dev/null",
		"/proc",
		"/sys",
		"/tmp",
		"/tmp/foo",
		"/root",
		"/boot",
		"/run",
		"/var",
	}

	for _, tc := range blockedCases {
		_, err := ValidateSandboxMountPath(tc)
		require.Errorf(t, err, "path %q should be rejected", tc)
		assert.Contains(t, err.Error(), "conflicts with protected system directory")
	}
}

func TestValidateNetworkPolicy(t *testing.T) {
	// Mode Restricted
	p, err := ValidateNetworkPolicy(NetworkPolicy{
		Mode:         NetworkRestricted,
		AllowedHosts: []string{"PyPi.org", "pypi.org", "files.pythonhosted.org"},
		BlockedCIDRs: []string{"10.0.0.0/8"},
	})
	require.NoError(t, err)
	assert.Equal(t, NetworkRestricted, p.Mode)
	assert.ElementsMatch(t, []string{"pypi.org", "files.pythonhosted.org"}, p.AllowedHosts)

	// Ensure mandatory blocked CIDRs and hosts are present
	assert.Contains(t, p.BlockedCIDRs, "169.254.0.0/16")
	assert.Contains(t, p.BlockedCIDRs, "127.0.0.0/8")
	assert.Contains(t, p.BlockedCIDRs, "10.0.0.0/8")
	assert.Contains(t, p.BlockedCIDRs, "172.16.0.0/12")
	assert.Contains(t, p.BlockedCIDRs, "192.168.0.0/16")
	assert.Contains(t, p.BlockedHosts, "metadata.google.internal")
	assert.Contains(t, p.BlockedHosts, "169.254.169.254")
	assert.Contains(t, p.BlockedHosts, "localhost")

	// Invalid CIDR
	_, err = ValidateNetworkPolicy(NetworkPolicy{
		Mode:         NetworkRestricted,
		BlockedCIDRs: []string{"invalid-cidr"},
	})
	assert.ErrorContains(t, err, "invalid blocked CIDR")

	// Invalid mode
	_, err = ValidateNetworkPolicy(NetworkPolicy{
		Mode: "unknown_mode",
	})
	assert.ErrorContains(t, err, "invalid network mode")
}

func TestValidateImage(t *testing.T) {
	allowed := []string{"alpine:latest", "python:3.11-slim"}

	assert.NoError(t, ValidateImage("alpine:latest", allowed))
	assert.NoError(t, ValidateImage("python:3.11-slim", allowed))
	assert.NoError(t, ValidateImage(" alpine:latest ", allowed))

	assert.ErrorContains(t, ValidateImage("", allowed), "cannot be empty")
	assert.ErrorContains(t, ValidateImage("ubuntu:22.04", allowed), "not in allowed images whitelist")
}
