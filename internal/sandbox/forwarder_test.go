package sandbox

import (
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"fmt"
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

	res, err := EnsureForwarderBinary(ForwarderConfig{DataDir: tempDataDir})
	require.NoError(t, err)
	assert.Equal(t, targetPath, res)

	info, err := os.Stat(res)
	require.NoError(t, err)
	assert.Equal(t, os.FileMode(0o755), info.Mode().Perm())
}

func TestEnsureForwarderBinary_CustomPath(t *testing.T) {
	tempDir := t.TempDir()
	customPath := filepath.Join(tempDir, "custom-fwd")
	require.NoError(t, os.WriteFile(customPath, []byte("custom-binary-payload"), 0o755))

	tempDataDir := filepath.Join(tempDir, "data")
	res, err := EnsureForwarderBinary(ForwarderConfig{
		DataDir:      tempDataDir,
		ProxyFwdPath: customPath,
	})
	require.NoError(t, err)

	expectedTarget := filepath.Join(tempDataDir, "bin", "bob-proxy-fwd")
	assert.Equal(t, expectedTarget, res)

	content, err := os.ReadFile(res)
	require.NoError(t, err)
	assert.Equal(t, []byte("custom-binary-payload"), content)
}

func TestEnsureForwarderBinary_Embedded(t *testing.T) {
	var buf bytes.Buffer
	gw := gzip.NewWriter(&buf)
	_, err := gw.Write([]byte("#!/bin/sh\necho embedded\n"))
	require.NoError(t, err)
	require.NoError(t, gw.Close())

	origReader := readEmbeddedForwarder
	readEmbeddedForwarder = func() ([]byte, error) {
		return buf.Bytes(), nil
	}
	t.Cleanup(func() {
		readEmbeddedForwarder = origReader
	})

	tempDataDir := t.TempDir()
	res, err := EnsureForwarderBinary(ForwarderConfig{
		DataDir:           tempDataDir,
		AllowRuntimeBuild: BoolPtr(false),
	})
	require.NoError(t, err)

	expectedTarget := filepath.Join(tempDataDir, "bin", "bob-proxy-fwd")
	assert.Equal(t, expectedTarget, res)

	content, err := os.ReadFile(res)
	require.NoError(t, err)
	assert.Equal(t, []byte("#!/bin/sh\necho embedded\n"), content)

	info, err := os.Stat(res)
	require.NoError(t, err)
	assert.Equal(t, os.FileMode(0o755), info.Mode().Perm())

	hashPath := res + ".sha256"
	hashContent, err := os.ReadFile(hashPath)
	require.NoError(t, err)
	expectedHash := fmt.Sprintf("%x\n", sha256.Sum256(buf.Bytes()))
	assert.Equal(t, expectedHash, string(hashContent))

	// Fast path cache hit
	res2, err := EnsureForwarderBinary(ForwarderConfig{
		DataDir:           tempDataDir,
		AllowRuntimeBuild: BoolPtr(false),
	})
	require.NoError(t, err)
	assert.Equal(t, expectedTarget, res2)
}

func TestEnsureForwarderBinary_CompileOnDemand(t *testing.T) {
	origReader := readEmbeddedForwarder
	readEmbeddedForwarder = func() ([]byte, error) {
		return nil, os.ErrNotExist
	}
	t.Cleanup(func() {
		readEmbeddedForwarder = origReader
	})

	tempDataDir := t.TempDir()

	res, err := EnsureForwarderBinary(ForwarderConfig{
		DataDir:           tempDataDir,
		AllowRuntimeBuild: BoolPtr(true),
	})
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

	const numGoroutines = 20
	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	results := make([]string, numGoroutines)
	errorsList := make([]error, numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		go func(idx int) {
			defer wg.Done()
			results[idx], errorsList[idx] = EnsureForwarderBinary(ForwarderConfig{
				DataDir:           tempDataDir,
				AllowRuntimeBuild: BoolPtr(true),
			})
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

	tempDataDir := filepath.Join(tempDir, "data")

	res1, err := EnsureForwarderBinary(ForwarderConfig{
		DataDir:      tempDataDir,
		ProxyFwdPath: customPath,
	})
	require.NoError(t, err)
	content1, err := os.ReadFile(res1)
	require.NoError(t, err)
	assert.Equal(t, []byte("version-1"), content1)

	// Update custom binary with newer modtime and new content
	time.Sleep(20 * time.Millisecond)
	require.NoError(t, os.WriteFile(customPath, []byte("version-2-updated"), 0o755))

	res2, err := EnsureForwarderBinary(ForwarderConfig{
		DataDir:      tempDataDir,
		ProxyFwdPath: customPath,
	})
	require.NoError(t, err)
	content2, err := os.ReadFile(res2)
	require.NoError(t, err)
	assert.Equal(t, []byte("version-2-updated"), content2)
}

func TestEnsureForwarderBinary_BuildDisabled_NotFound(t *testing.T) {
	origReader := readEmbeddedForwarder
	readEmbeddedForwarder = func() ([]byte, error) {
		return nil, os.ErrNotExist
	}
	t.Cleanup(func() {
		readEmbeddedForwarder = origReader
	})

	tempDataDir := t.TempDir()

	_, err := EnsureForwarderBinary(ForwarderConfig{
		DataDir:           tempDataDir,
		AllowRuntimeBuild: BoolPtr(false),
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "bob-proxy-fwd binary not found and cannot be built")
}

func TestEnsureForwarderBinary_InvalidCustomPath(t *testing.T) {
	tempDataDir := t.TempDir()

	_, err := EnsureForwarderBinary(ForwarderConfig{
		DataDir:      tempDataDir,
		ProxyFwdPath: "/nonexistent/path/never_exists",
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "custom forwarder binary")
}

func TestEnsureForwarderBinary_CandidateSearch(t *testing.T) {
	origReader := readEmbeddedForwarder
	readEmbeddedForwarder = func() ([]byte, error) {
		return nil, os.ErrNotExist
	}
	t.Cleanup(func() {
		readEmbeddedForwarder = origReader
	})

	execPath, err := os.Executable()
	require.NoError(t, err)
	execDir := filepath.Dir(execPath)

	testCases := []struct {
		name    string
		subDir  string
		binName string
	}{
		{
			name:    "AdjacentToExecutable",
			subDir:  "",
			binName: "bob-proxy-fwd",
		},
		{
			name:    "InBinSubdirectory",
			subDir:  "bin",
			binName: "bob-proxy-fwd",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			targetDir := execDir
			if tc.subDir != "" {
				targetDir = filepath.Join(execDir, tc.subDir)
			}

			// If testing subDir, ensure adjacent candidate does not shadow it
			if tc.subDir != "" {
				adjacentPath := filepath.Join(execDir, "bob-proxy-fwd")
				if adjData, adjErr := os.ReadFile(adjacentPath); adjErr == nil {
					require.NoError(t, os.Remove(adjacentPath))
					t.Cleanup(func() {
						assert.NoError(t, os.WriteFile(adjacentPath, adjData, 0o755))
					})
				}
			}

			_, statDirErr := os.Stat(targetDir)
			dirExisted := statDirErr == nil
			require.NoError(t, os.MkdirAll(targetDir, 0o755))

			candPath := filepath.Join(targetDir, tc.binName)
			origContent, statFileErr := os.ReadFile(candPath)
			fileExisted := statFileErr == nil

			t.Cleanup(func() {
				if fileExisted {
					assert.NoError(t, os.WriteFile(candPath, origContent, 0o755))
				} else {
					assert.NoError(t, os.Remove(candPath))
				}
				if !dirExisted && tc.subDir != "" {
					assert.NoError(t, os.Remove(targetDir))
				}
			})

			payload := []byte("candidate-fwd-" + tc.name)
			require.NoError(t, os.WriteFile(candPath, payload, 0o755))

			tempDataDir := t.TempDir()
			res, err := EnsureForwarderBinary(ForwarderConfig{
				DataDir:           tempDataDir,
				AllowRuntimeBuild: BoolPtr(false),
			})
			require.NoError(t, err)

			expected := filepath.Join(tempDataDir, "bin", "bob-proxy-fwd")
			assert.Equal(t, expected, res)

			data, err := os.ReadFile(res)
			require.NoError(t, err)
			assert.Equal(t, payload, data)

			info, err := os.Stat(res)
			require.NoError(t, err)
			assert.Equal(t, os.FileMode(0o755), info.Mode().Perm())
		})
	}
}

func TestEnsureForwarderBinaryPath_Helper(t *testing.T) {
	tempDataDir := t.TempDir()
	binDir := filepath.Join(tempDataDir, "bin")
	require.NoError(t, os.MkdirAll(binDir, 0o755))

	targetPath := filepath.Join(binDir, "bob-proxy-fwd")
	require.NoError(t, os.WriteFile(targetPath, []byte("#!/bin/sh\nexit 0\n"), 0o755))

	res, err := EnsureForwarderBinaryPath(tempDataDir)
	require.NoError(t, err)
	assert.Equal(t, targetPath, res)
}

func TestCopyFile_AtomicAndSecure(t *testing.T) {
	tempDir := t.TempDir()
	src := filepath.Join(tempDir, "src.txt")
	dst := filepath.Join(tempDir, "sub", "dst.txt")

	require.NoError(t, os.WriteFile(src, []byte("secure-payload"), 0o600))
	err := CopyFile(src, dst, 0o755)
	require.NoError(t, err)

	content, err := os.ReadFile(dst)
	require.NoError(t, err)
	assert.Equal(t, []byte("secure-payload"), content)

	fi, err := os.Stat(dst)
	require.NoError(t, err)
	assert.Equal(t, os.FileMode(0o755), fi.Mode().Perm())
}
