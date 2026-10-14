package sandbox

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRunCache_AcquireAndRemove(t *testing.T) {
	tmpDir := t.TempDir()
	SetCacheRootOverride(tmpDir)
	defer SetCacheRootOverride("")

	runID := "test_run_123"
	meta := RunCacheMetadata{
		RunID:     runID,
		ChatID:    "test-chat",
		IsDM:      true,
		CreatedAt: time.Now(),
		ExpiresAt: time.Now().Add(1 * time.Hour),
		DataDir:   "/tmp/data",
	}

	cacheDir, err := AcquireRunCacheDir(runID, meta)
	require.NoError(t, err)
	assert.True(t, filepath.IsAbs(cacheDir))

	// Verify required subdirectories exist
	expectedSubdirs := []string{"go", "go-build", "golangci-lint", "npm", "pip"}
	for _, sub := range expectedSubdirs {
		info, err := os.Stat(filepath.Join(cacheDir, sub))
		require.NoError(t, err, "subdirectory %s must exist", sub)
		assert.True(t, info.IsDir())
	}

	// Verify metadata.json was written alongside cache/
	metaFile := filepath.Join(filepath.Dir(cacheDir), "metadata.json")
	_, err = os.Stat(metaFile)
	require.NoError(t, err, "metadata.json must exist")

	// Remove run cache
	err = RemoveRunCache(runID)
	require.NoError(t, err)

	_, err = os.Stat(filepath.Dir(cacheDir))
	assert.True(t, os.IsNotExist(err), "run directory must be deleted")
}

func TestRunCache_PurgeAll(t *testing.T) {
	tmpDir := t.TempDir()
	SetCacheRootOverride(tmpDir)
	defer SetCacheRootOverride("")

	// Create 3 run caches and release leases to simulate stale/completed runs
	_, err := AcquireRunCacheDir("run_a", RunCacheMetadata{RunID: "run_a"})
	require.NoError(t, err)
	ReleaseRunCache("run_a")
	_, err = AcquireRunCacheDir("run_b", RunCacheMetadata{RunID: "run_b"})
	require.NoError(t, err)
	ReleaseRunCache("run_b")
	_, err = AcquireRunCacheDir("run_c", RunCacheMetadata{RunID: "run_c"})
	require.NoError(t, err)
	ReleaseRunCache("run_c")

	count, err := PurgeAllCaches()
	require.NoError(t, err)
	assert.Equal(t, 3, count)

	// Second purge returns 0
	count2, err := PurgeAllCaches()
	require.NoError(t, err)
	assert.Equal(t, 0, count2)
}

func TestRunCache_PurgeAll_PreservesActiveLease(t *testing.T) {
	tmpDir := t.TempDir()
	SetCacheRootOverride(tmpDir)
	defer SetCacheRootOverride("")

	// Create active run (lease held) and stale run (lease released)
	_, err := AcquireRunCacheDir("active_run", RunCacheMetadata{RunID: "active_run"})
	require.NoError(t, err)
	defer ReleaseRunCache("active_run")

	_, err = AcquireRunCacheDir("stale_run", RunCacheMetadata{RunID: "stale_run"})
	require.NoError(t, err)
	ReleaseRunCache("stale_run")

	count, err := PurgeAllCaches()
	require.NoError(t, err)
	assert.Equal(t, 1, count, "only stale run cache should be purged")

	// Verify active run directory still exists
	_, err = os.Stat(filepath.Join(tmpDir, "active_run"))
	assert.NoError(t, err, "active run cache must be preserved")

	// Verify stale run directory was removed
	_, err = os.Stat(filepath.Join(tmpDir, "stale_run"))
	assert.True(t, os.IsNotExist(err), "stale run cache must be deleted")
}

func TestRunCache_ReapStaleCaches(t *testing.T) {
	tmpDir := t.TempDir()
	SetCacheRootOverride(tmpDir)
	defer SetCacheRootOverride("")

	ctx := context.Background()
	now := time.Now()

	// 1. Expired cache (by ExpiresAt, lease released)
	_, err := AcquireRunCacheDir("expired_run", RunCacheMetadata{
		RunID:     "expired_run",
		ExpiresAt: now.Add(-10 * time.Minute),
	})
	require.NoError(t, err)
	ReleaseRunCache("expired_run")

	// 2. Active run in DB (should NOT be reaped)
	_, err = AcquireRunCacheDir("active_run", RunCacheMetadata{
		RunID:     "active_run",
		ExpiresAt: now.Add(1 * time.Hour),
	})
	require.NoError(t, err)
	defer ReleaseRunCache("active_run")

	// 3. Terminal run in DB (lease released, should be reaped)
	_, err = AcquireRunCacheDir("terminal_run", RunCacheMetadata{
		RunID:     "terminal_run",
		ExpiresAt: now.Add(1 * time.Hour),
	})
	require.NoError(t, err)
	ReleaseRunCache("terminal_run")

	checker := func(_ context.Context, meta RunCacheMetadata) (bool, error) {
		switch meta.RunID {
		case "active_run":
			return false, nil
		case "terminal_run":
			return true, nil
		default:
			return false, nil
		}
	}

	reaped, err := ReapStaleCaches(ctx, checker, 24*time.Hour)
	require.NoError(t, err)
	assert.Equal(t, 2, reaped, "should reap expired_run and terminal_run")

	// Verify active_run still exists
	_, err = os.Stat(filepath.Join(tmpDir, "active_run"))
	assert.NoError(t, err, "active_run must still exist")

	// Verify others were removed
	_, err = os.Stat(filepath.Join(tmpDir, "expired_run"))
	assert.True(t, os.IsNotExist(err))
	_, err = os.Stat(filepath.Join(tmpDir, "terminal_run"))
	assert.True(t, os.IsNotExist(err))
}

func TestSanitizeRunKey_NoCollision(t *testing.T) {
	k1 := SanitizeRunKey("a/run")
	k2 := SanitizeRunKey("b/run")
	assert.NotEqual(t, k1, k2)
	assert.Equal(t, "a_run", k1)
	assert.Equal(t, "b_run", k2)
}
