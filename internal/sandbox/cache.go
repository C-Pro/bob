package sandbox

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// DefaultRunCacheExpiry defines the fallback expiration for unmanaged run caches.
const DefaultRunCacheExpiry = 2 * time.Hour

// RunCacheMetadata describes an active or stale sandbox run cache directory.
type RunCacheMetadata struct {
	RunID     string    `json:"run_id"`
	ChatID    string    `json:"chat_id,omitempty"`
	IsDM      bool      `json:"is_dm,omitempty"`
	CreatedAt time.Time `json:"created_at"`
	ExpiresAt time.Time `json:"expires_at"`
	DataDir   string    `json:"data_dir,omitempty"`
}

var (
	cacheRootOverride string
	cacheRootMu       sync.RWMutex
)

// SetCacheRootOverride sets an override directory for sandbox run caches (used in tests).
func SetCacheRootOverride(dir string) {
	cacheRootMu.Lock()
	defer cacheRootMu.Unlock()
	cacheRootOverride = dir
}

// CacheRootDir returns the base root directory where per-run sandbox caches are stored.
func CacheRootDir() string {
	cacheRootMu.RLock()
	override := cacheRootOverride
	cacheRootMu.RUnlock()
	if override != "" {
		return override
	}

	cacheDir, err := os.UserCacheDir()
	if err != nil || strings.TrimSpace(cacheDir) == "" {
		home := os.Getenv("HOME")
		if home != "" {
			cacheDir = filepath.Join(home, ".cache")
		} else {
			cacheDir = "./data/cache"
		}
	}
	return filepath.Join(cacheDir, "bob", "sandboxes")
}

// SanitizeRunKey prevents path traversal when using runID in filenames.
func SanitizeRunKey(runID string) string {
	clean := strings.TrimSpace(runID)
	clean = strings.ReplaceAll(clean, "..", "_")
	clean = strings.ReplaceAll(clean, "/", "_")
	clean = strings.ReplaceAll(clean, "\\", "_")
	clean = strings.ReplaceAll(clean, ":", "_")
	clean = strings.ReplaceAll(clean, " ", "_")
	clean = filepath.Clean(clean)
	return clean
}

// AcquireRunCacheDir initializes or returns the private cache directory for a run.
// It creates <rootDir>/<runID>/cache and subdirectories for toolchain caches,
// writing metadata.json alongside the cache directory.
func AcquireRunCacheDir(runID string, meta RunCacheMetadata) (string, error) {
	cleanID := SanitizeRunKey(runID)
	if cleanID == "" || cleanID == "." {
		return "", fmt.Errorf("invalid run ID %q", runID)
	}

	baseDir := filepath.Join(CacheRootDir(), cleanID)
	cacheDir := filepath.Join(baseDir, "cache")

	// Ensure cache subdirectories exist
	subdirs := []string{
		cacheDir,
		filepath.Join(cacheDir, "go"),
		filepath.Join(cacheDir, "go-build"),
		filepath.Join(cacheDir, "golangci-lint"),
		filepath.Join(cacheDir, "npm"),
		filepath.Join(cacheDir, "pip"),
	}
	for _, dir := range subdirs {
		if err := os.MkdirAll(dir, 0o700); err != nil {
			return "", fmt.Errorf("failed to create cache directory %s: %w", dir, err)
		}
	}

	metaPath := filepath.Join(baseDir, "metadata.json")
	if _, err := os.Stat(metaPath); os.IsNotExist(err) {
		if meta.RunID == "" {
			meta.RunID = runID
		}
		if meta.CreatedAt.IsZero() {
			meta.CreatedAt = time.Now()
		}
		if meta.ExpiresAt.IsZero() {
			meta.ExpiresAt = meta.CreatedAt.Add(DefaultRunCacheExpiry)
		}
		data, err := json.MarshalIndent(meta, "", "  ")
		if err == nil {
			_ = os.WriteFile(metaPath, data, 0o600)
		}
	}

	return cacheDir, nil
}

// RemoveRunCache deletes the entire run cache directory for the specified runID.
func RemoveRunCache(runID string) error {
	cleanID := SanitizeRunKey(runID)
	if cleanID == "" || cleanID == "." {
		return nil
	}
	baseDir := filepath.Join(CacheRootDir(), cleanID)
	return os.RemoveAll(baseDir)
}

// PurgeAllCaches removes all run caches under the sandbox cache root directory.
func PurgeAllCaches() (int, error) {
	rootDir := CacheRootDir()
	entries, err := os.ReadDir(rootDir)
	if err != nil {
		if os.IsNotExist(err) {
			return 0, nil
		}
		return 0, fmt.Errorf("failed to read sandbox cache directory %s: %w", rootDir, err)
	}

	count := 0
	var firstErr error
	for _, entry := range entries {
		entryPath := filepath.Join(rootDir, entry.Name())
		if err := os.RemoveAll(entryPath); err != nil {
			if firstErr == nil {
				firstErr = err
			}
			slog.Warn("failed to remove sandbox cache entry", "path", entryPath, "error", err)
		} else {
			count++
		}
	}
	return count, firstErr
}

// RunStatusChecker inspects whether a run has reached a terminal status.
type RunStatusChecker func(ctx context.Context, meta RunCacheMetadata) (isTerminal bool, err error)

// ReapStaleCaches scans the cache root directory and deletes caches that have expired
// or whose associated FSM runs have completed/failed.
func ReapStaleCaches(ctx context.Context, checker RunStatusChecker, maxAge time.Duration) (int, error) {
	rootDir := CacheRootDir()
	entries, err := os.ReadDir(rootDir)
	if err != nil {
		if os.IsNotExist(err) {
			return 0, nil
		}
		return 0, fmt.Errorf("failed to read sandbox cache directory: %w", err)
	}

	now := time.Now()
	reaped := 0

	for _, entry := range entries {
		if ctx.Err() != nil {
			return reaped, ctx.Err()
		}
		entryPath := filepath.Join(rootDir, entry.Name())
		metaPath := filepath.Join(entryPath, "metadata.json")

		data, err := os.ReadFile(metaPath)
		if err != nil {
			// No metadata. Check folder mtime against maxAge fallback
			if fi, sErr := entry.Info(); sErr == nil && maxAge > 0 {
				if now.Sub(fi.ModTime()) > maxAge {
					if rmErr := os.RemoveAll(entryPath); rmErr == nil {
						reaped++
					}
				}
			}
			continue
		}

		var meta RunCacheMetadata
		if err := json.Unmarshal(data, &meta); err != nil {
			// Corrupt metadata; fallback to mtime
			if fi, sErr := entry.Info(); sErr == nil && maxAge > 0 {
				if now.Sub(fi.ModTime()) > maxAge {
					if rmErr := os.RemoveAll(entryPath); rmErr == nil {
						reaped++
					}
				}
			}
			continue
		}

		// Check if expired
		if !meta.ExpiresAt.IsZero() && now.After(meta.ExpiresAt) {
			if rmErr := os.RemoveAll(entryPath); rmErr == nil {
				reaped++
			}
			continue
		}

		// Check if FSM run is terminal
		if checker != nil {
			terminal, chkErr := checker(ctx, meta)
			if chkErr != nil {
				slog.Warn("status check failed during sandbox cache reaping; retaining entry", "run_id", meta.RunID, "error", chkErr)
				continue
			}
			if terminal {
				if rmErr := os.RemoveAll(entryPath); rmErr == nil {
					reaped++
				}
				continue
			}
		}
	}

	return reaped, nil
}
