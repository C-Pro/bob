package sandbox

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestEnsureForwarderBinary_ExistingTarget(t *testing.T) {
	tempDataDir := t.TempDir()
	binDir := filepath.Join(tempDataDir, "bin")
	require.NoError(t, os.MkdirAll(binDir, 0o755))

	targetPath := filepath.Join(binDir, "bob-proxy-fwd")
	require.NoError(t, os.WriteFile(targetPath, []byte("#!/bin/sh\nexit 0\n"), 0o644))

	res, err := EnsureForwarderBinary(tempDataDir)
	require.NoError(t, err)
	assert.Equal(t, targetPath, res)

	info, err := os.Stat(res)
	require.NoError(t, err)
	assert.Equal(t, os.FileMode(0o755), info.Mode().Perm())
}

func TestEnsureForwarderBinary_CustomEnvOverride(t *testing.T) {
	tempDir := t.TempDir()
	customPath := filepath.Join(tempDir, "custom-fwd")
	require.NoError(t, os.WriteFile(customPath, []byte("custom-binary-payload"), 0o755))

	t.Setenv("SANDBOX_PROXY_FWD_PATH", customPath)

	tempDataDir := filepath.Join(tempDir, "data")
	res, err := EnsureForwarderBinary(tempDataDir)
	require.NoError(t, err)

	expectedTarget := filepath.Join(tempDataDir, "bin", "bob-proxy-fwd")
	assert.Equal(t, expectedTarget, res)

	content, err := os.ReadFile(res)
	require.NoError(t, err)
	assert.Equal(t, []byte("custom-binary-payload"), content)
}

func TestEnsureForwarderBinary_CompileOnDemand(t *testing.T) {
	tempDataDir := t.TempDir()

	// Clear custom env
	t.Setenv("SANDBOX_PROXY_FWD_PATH", "")

	res, err := EnsureForwarderBinary(tempDataDir)
	require.NoError(t, err)

	expectedTarget := filepath.Join(tempDataDir, "bin", "bob-proxy-fwd")
	assert.Equal(t, expectedTarget, res)

	info, err := os.Stat(res)
	require.NoError(t, err)
	assert.False(t, info.IsDir())
	assert.Equal(t, os.FileMode(0o755), info.Mode().Perm())
}
