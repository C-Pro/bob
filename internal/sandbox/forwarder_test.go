package sandbox

import (
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

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

func TestEnsureForwarderBinary_Concurrency(t *testing.T) {
	tempDataDir := t.TempDir()
	t.Setenv("SANDBOX_PROXY_FWD_PATH", "")

	const numGoroutines = 20
	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	results := make([]string, numGoroutines)
	errorsList := make([]error, numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		go func(idx int) {
			defer wg.Done()
			results[idx], errorsList[idx] = EnsureForwarderBinary(tempDataDir)
		}(i)
	}

	wg.Wait()

	expectedTarget := filepath.Join(tempDataDir, "bin", "bob-proxy-fwd")
	for i := 0; i < numGoroutines; i++ {
		require.NoError(t, errorsList[i], "goroutine %d failed", i)
		assert.Equal(t, expectedTarget, results[i])
	}
}

func TestEnsureForwarderBinary_CacheInvalidation(t *testing.T) {
	tempDir := t.TempDir()
	customPath := filepath.Join(tempDir, "custom-fwd")
	require.NoError(t, os.WriteFile(customPath, []byte("version-1"), 0o755))

	t.Setenv("SANDBOX_PROXY_FWD_PATH", customPath)
	tempDataDir := filepath.Join(tempDir, "data")

	res1, err := EnsureForwarderBinary(tempDataDir)
	require.NoError(t, err)
	content1, err := os.ReadFile(res1)
	require.NoError(t, err)
	assert.Equal(t, []byte("version-1"), content1)

	// Update custom binary with newer modtime and new content
	time.Sleep(20 * time.Millisecond)
	require.NoError(t, os.WriteFile(customPath, []byte("version-2-updated"), 0o755))

	res2, err := EnsureForwarderBinary(tempDataDir)
	require.NoError(t, err)
	content2, err := os.ReadFile(res2)
	require.NoError(t, err)
	assert.Equal(t, []byte("version-2-updated"), content2)
}

func TestEnsureForwarderBinary_BuildDisabled_NotFound(t *testing.T) {
	tempDataDir := t.TempDir()
	t.Setenv("SANDBOX_PROXY_FWD_PATH", "")
	t.Setenv("BOB_ALLOW_RUNTIME_BUILD", "0")

	_, err := EnsureForwarderBinary(tempDataDir)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "bob-proxy-fwd binary not found and cannot be built")
}

func TestEnsureForwarderBinary_InvalidCustomEnv(t *testing.T) {
	tempDataDir := t.TempDir()
	t.Setenv("SANDBOX_PROXY_FWD_PATH", "/nonexistent/path/never_exists")

	_, err := EnsureForwarderBinary(tempDataDir)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "custom forwarder binary")
}

func TestCopyFile_AtomicAndSecure(t *testing.T) {
	tempDir := t.TempDir()
	src := filepath.Join(tempDir, "src.txt")
	dst := filepath.Join(tempDir, "sub", "dst.txt")

	require.NoError(t, os.WriteFile(src, []byte("secure-payload"), 0o600))
	err := copyFile(src, dst, 0o755)
	require.NoError(t, err)

	content, err := os.ReadFile(dst)
	require.NoError(t, err)
	assert.Equal(t, []byte("secure-payload"), content)

	fi, err := os.Stat(dst)
	require.NoError(t, err)
	assert.Equal(t, os.FileMode(0o755), fi.Mode().Perm())
}

