package sandbox

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
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
	cacheRootDataDir  string
	resolvedCacheRoot string
	cacheRootMu       sync.RWMutex

	activeLocksMu sync.Mutex
	activeLocks   = make(map[string]*os.File)
)

var errCacheActive = errors.New("sandbox cache is actively locked")

func lockRunCache(baseDir, cleanID string) error {
	activeLocksMu.Lock()
	defer activeLocksMu.Unlock()

	if _, ok := activeLocks[cleanID]; ok {
		return nil
	}

	lockPath := filepath.Join(baseDir, "run.lock")
	f, err := os.OpenFile(lockPath, os.O_RDWR|os.O_CREATE, 0o600)
	if err != nil {
		return err
	}

	if err := syscall.Flock(int(f.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
		_ = f.Close()
		return err
	}

	activeLocks[cleanID] = f
	return nil
}

// ReleaseRunCache releases the process advisory lock on a run cache without deleting the directory.
func ReleaseRunCache(runID string) {
	cleanID := SanitizeRunKey(runID)
	activeLocksMu.Lock()
	defer activeLocksMu.Unlock()

	if f, ok := activeLocks[cleanID]; ok {
		_ = syscall.Flock(int(f.Fd()), syscall.LOCK_UN)
		_ = f.Close()
		delete(activeLocks, cleanID)
	}
}

// tryLockRunCache attempts to acquire an exclusive lock on the run cache directory.
// Returns the open lock file and true on success, or nil and false if the cache is actively locked.
func tryLockRunCache(entryPath string) (*os.File, bool) {
	cleanID := filepath.Base(entryPath)
	activeLocksMu.Lock()
	if _, ok := activeLocks[cleanID]; ok {
		activeLocksMu.Unlock()
		return nil, false
	}
	activeLocksMu.Unlock()

	lockPath := filepath.Join(entryPath, "run.lock")
	f, err := os.OpenFile(lockPath, os.O_RDWR|os.O_CREATE, 0o600)
	if err != nil {
		return nil, false
	}

	if err := syscall.Flock(int(f.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
		_ = f.Close()
		return nil, false
	}

	return f, true
}

func tryLockAndRemove(entryPath string) error {
	lockFile, locked := tryLockRunCache(entryPath)
	if !locked {
		return errCacheActive
	}
	defer func() {
		_ = lockFile.Close()
	}()
	return os.RemoveAll(entryPath)
}

// SetCacheRootOverride sets an override directory for sandbox run caches (used in tests).
func SetCacheRootOverride(dir string) {
	cacheRootMu.Lock()
	defer cacheRootMu.Unlock()
	cacheRootOverride = dir
	resolvedCacheRoot = ""
	if dir == "" {
		cacheRootDataDir = ""
	}
}

// SetCacheRootDataDir configures a data directory candidate to use for sandbox run caches.
func SetCacheRootDataDir(dir string) {
	clean := strings.TrimSpace(dir)
	if clean == "" {
		return
	}
	cacheRootMu.Lock()
	defer cacheRootMu.Unlock()
	if cacheRootDataDir != clean {
		cacheRootDataDir = clean
		resolvedCacheRoot = ""
	}
}

func testWritableDir(dir string) bool {
	clean := strings.TrimSpace(dir)
	if clean == "" {
		return false
	}
	if err := os.MkdirAll(clean, 0o700); err != nil {
		return false
	}
	f, err := os.CreateTemp(clean, ".probe-*")
	if err != nil {
		return false
	}
	_ = f.Close()
	_ = os.Remove(f.Name())
	return true
}

// CacheRootDir returns the base root directory where per-run sandbox caches are stored.
func CacheRootDir() string {
	cacheRootMu.RLock()
	if cacheRootOverride != "" {
		res := cacheRootOverride
		cacheRootMu.RUnlock()
		return res
	}
	if resolvedCacheRoot != "" {
		res := resolvedCacheRoot
		cacheRootMu.RUnlock()
		return res
	}
	cacheRootMu.RUnlock()

	cacheRootMu.Lock()
	defer cacheRootMu.Unlock()

	if cacheRootOverride != "" {
		return cacheRootOverride
	}
	if resolvedCacheRoot != "" {
		return resolvedCacheRoot
	}

	candidates := make([]string, 0, 8)

	// 1. Standard user cache directory
	if userCache, err := os.UserCacheDir(); err == nil && strings.TrimSpace(userCache) != "" {
		candidates = append(candidates, filepath.Join(userCache, "bob", "sandboxes"))
	}
	if xdg := os.Getenv("XDG_CACHE_HOME"); strings.TrimSpace(xdg) != "" {
		candidates = append(candidates, filepath.Join(xdg, "bob", "sandboxes"))
	}
	if home := os.Getenv("HOME"); strings.TrimSpace(home) != "" {
		candidates = append(candidates, filepath.Join(home, ".cache", "bob", "sandboxes"))
	}

	// 2. Data directory candidates
	if cacheRootDataDir != "" {
		candidates = append(candidates, filepath.Join(cacheRootDataDir, "cache", "sandboxes"))
	}
	candidates = append(candidates,
		filepath.Join(".", "data", "cache", "sandboxes"),
		filepath.Join("/data", "cache", "sandboxes"),
	)

	// 3. System temp directory
	candidates = append(candidates,
		filepath.Join(os.TempDir(), ".cache", "bob", "sandboxes"),
		filepath.Join(os.TempDir(), "bob", "sandboxes"),
	)

	// 4. Working directory fallback
	candidates = append(candidates, filepath.Join(".", ".cache", "bob", "sandboxes"))

	seen := make(map[string]struct{}, len(candidates))
	for _, cand := range candidates {
		cand = filepath.Clean(cand)
		if _, exists := seen[cand]; exists {
			continue
		}
		seen[cand] = struct{}{}
		if testWritableDir(cand) {
			resolvedCacheRoot = cand
			return cand
		}
	}

	fallback := filepath.Join(os.TempDir(), "bob", "sandboxes")
	resolvedCacheRoot = fallback
	return fallback
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

	if meta.DataDir != "" {
		SetCacheRootDataDir(meta.DataDir)
	}

	rootDir := CacheRootDir()
	baseDir := filepath.Join(rootDir, cleanID)
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
			slog.Warn("failed to create cache directory in root, falling back to temp dir", "root", rootDir, "error", err)
			cacheRootMu.Lock()
			resolvedCacheRoot = ""
			cacheRootMu.Unlock()

			fallbackRoot := filepath.Join(os.TempDir(), ".cache", "bob", "sandboxes")
			if testWritableDir(fallbackRoot) {
				cacheRootMu.Lock()
				resolvedCacheRoot = fallbackRoot
				cacheRootMu.Unlock()
				baseDir = filepath.Join(fallbackRoot, cleanID)
				cacheDir = filepath.Join(baseDir, "cache")
				fallbackSubdirs := []string{
					cacheDir,
					filepath.Join(cacheDir, "go"),
					filepath.Join(cacheDir, "go-build"),
					filepath.Join(cacheDir, "golangci-lint"),
					filepath.Join(cacheDir, "npm"),
					filepath.Join(cacheDir, "pip"),
				}
				for _, fbDir := range fallbackSubdirs {
					if fbErr := os.MkdirAll(fbDir, 0o700); fbErr != nil {
						return "", fmt.Errorf("failed to create cache directory %s after fallback: %w", fbDir, fbErr)
					}
				}
				break
			}
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

	// Acquire process lease on the run cache directory
	if err := lockRunCache(baseDir, cleanID); err != nil {
		slog.Warn("failed to acquire advisory lock on run cache", "run_id", runID, "error", err)
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
	ReleaseRunCache(runID)
	return os.RemoveAll(baseDir)
}

// PurgeAllCaches removes all inactive run caches under the sandbox cache root directory.
// Entries currently held by an active sandbox or runner process are skipped.
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
		if !entry.IsDir() {
			continue
		}
		entryPath := filepath.Join(rootDir, entry.Name())
		err := tryLockAndRemove(entryPath)
		if errors.Is(err, errCacheActive) {
			slog.Info("skipping active sandbox cache during purge", "path", entryPath)
			continue
		}
		if err != nil {
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
// Entries currently locked by an active process are preserved.
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

	removeIfInactive := func(entryPath string) bool {
		err := tryLockAndRemove(entryPath)
		if errors.Is(err, errCacheActive) {
			slog.Info("skipping active sandbox cache during reaping", "path", entryPath)
			return false
		}
		return err == nil
	}

	for _, entry := range entries {
		if ctx.Err() != nil {
			return reaped, ctx.Err()
		}
		if !entry.IsDir() {
			continue
		}
		entryPath := filepath.Join(rootDir, entry.Name())
		metaPath := filepath.Join(entryPath, "metadata.json")

		data, err := os.ReadFile(metaPath)
		if err != nil {
			// No metadata. Check folder mtime against maxAge fallback
			if fi, sErr := entry.Info(); sErr == nil && maxAge > 0 {
				if now.Sub(fi.ModTime()) > maxAge {
					if removeIfInactive(entryPath) {
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
					if removeIfInactive(entryPath) {
						reaped++
					}
				}
			}
			continue
		}

		// Check if expired
		if !meta.ExpiresAt.IsZero() && now.After(meta.ExpiresAt) {
			if removeIfInactive(entryPath) {
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
				if removeIfInactive(entryPath) {
					reaped++
				}
				continue
			}
		}
	}

	return reaped, nil
}
