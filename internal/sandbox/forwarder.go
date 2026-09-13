package sandbox

import (
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"embed"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
)

var forwarderMu sync.RWMutex

//go:embed embedded
var embeddedFS embed.FS

var readEmbeddedForwarder = func() ([]byte, error) {
	return embeddedFS.ReadFile("embedded/bob-proxy-fwd.gz")
}

// CopyFile copies a file from src to dst atomically with the specified mode.
func CopyFile(src, dst string, mode os.FileMode) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer func() { _ = in.Close() }()

	dstDir := filepath.Dir(dst)
	if err := os.MkdirAll(dstDir, 0o755); err != nil {
		return err
	}

	tmpFile, err := os.CreateTemp(dstDir, ".bob-proxy-fwd-*.tmp")
	if err != nil {
		return err
	}
	tmpDst := tmpFile.Name()
	defer func() {
		_ = tmpFile.Close()
		_ = os.Remove(tmpDst)
	}()

	if _, err := io.Copy(tmpFile, in); err != nil {
		return err
	}

	if err := tmpFile.Chmod(mode); err != nil {
		return err
	}

	if err := tmpFile.Close(); err != nil {
		return err
	}

	return os.Rename(tmpDst, dst)
}

func extractEmbeddedForwarder(targetPath string) (bool, error) {
	gzData, err := readEmbeddedForwarder()
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return false, nil
		}
		return false, fmt.Errorf("failed to read embedded forwarder: %w", err)
	}
	if len(gzData) == 0 {
		return false, nil
	}

	hash := fmt.Sprintf("%x", sha256.Sum256(gzData))
	hashPath := targetPath + ".sha256"

	// If already extracted, verify hash and executable permission
	if fi, err := os.Stat(targetPath); err == nil && !fi.IsDir() && fi.Size() > 0 && fi.Mode()&0o111 != 0 {
		if cachedHash, err := os.ReadFile(hashPath); err == nil && strings.TrimSpace(string(cachedHash)) == hash {
			return true, nil
		}
	}

	dstDir := filepath.Dir(targetPath)
	if err := os.MkdirAll(dstDir, 0o755); err != nil {
		return false, fmt.Errorf("failed to create directory for forwarder: %w", err)
	}

	gzReader, err := gzip.NewReader(bytes.NewReader(gzData))
	if err != nil {
		return false, fmt.Errorf("failed to create gzip reader for embedded forwarder: %w", err)
	}
	defer func() { _ = gzReader.Close() }()

	tmpFile, err := os.CreateTemp(dstDir, ".bob-proxy-fwd-*.tmp")
	if err != nil {
		return false, fmt.Errorf("failed to create temp file for forwarder extraction: %w", err)
	}
	tmpDst := tmpFile.Name()
	defer func() {
		_ = tmpFile.Close()
		_ = os.Remove(tmpDst)
	}()

	const maxDecompressedForwarderSize = 64 * 1024 * 1024 // 64 MB limit
	// nosemgrep: go.lang.security.decompression_bomb.potential-dos-via-decompression-bomb
	if _, err := io.Copy(tmpFile, io.LimitReader(gzReader, maxDecompressedForwarderSize)); err != nil {
		return false, fmt.Errorf("failed to decompress embedded forwarder: %w", err)
	}

	if err := tmpFile.Chmod(0o755); err != nil {
		return false, fmt.Errorf("failed to chmod extracted forwarder: %w", err)
	}

	if err := tmpFile.Close(); err != nil {
		return false, fmt.Errorf("failed to close extracted forwarder temp file: %w", err)
	}

	if err := os.Rename(tmpDst, targetPath); err != nil {
		return false, fmt.Errorf("failed to replace forwarder binary: %w", err)
	}

	_ = os.WriteFile(hashPath, []byte(hash+"\n"), 0o644)

	return true, nil
}

func isBuildAllowed(cfg ForwarderConfig) bool {
	if cfg.AllowRuntimeBuild != nil {
		return *cfg.AllowRuntimeBuild
	}
	for _, arg := range os.Args {
		if strings.HasPrefix(arg, "-test.") {
			return true
		}
	}
	return false
}

func findRepoRoot() string {
	dir, err := os.Getwd()
	if err != nil {
		return ""
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	return ""
}

func resolveSourceBinary(customPath string) (string, os.FileInfo, error) {
	if custom := strings.TrimSpace(customPath); custom != "" {
		absCustom, err := filepath.Abs(custom)
		if err != nil {
			return "", nil, fmt.Errorf("failed to resolve custom forwarder path %q: %w", custom, err)
		}
		fi, err := os.Stat(absCustom)
		if err != nil {
			return "", nil, fmt.Errorf("custom forwarder binary %q not found: %w", absCustom, err)
		}
		if fi.IsDir() {
			return "", nil, fmt.Errorf("custom forwarder path %q is a directory", absCustom)
		}
		if !fi.Mode().IsRegular() {
			return "", nil, fmt.Errorf("custom forwarder binary %q is not a regular file", absCustom)
		}
		return absCustom, fi, nil
	}

	if execPath, err := os.Executable(); err == nil {
		execDir := filepath.Dir(execPath)
		candidates := []string{
			filepath.Join(execDir, "bob-proxy-fwd"),
			filepath.Join(execDir, "bin", "bob-proxy-fwd"),
		}
		for _, cand := range candidates {
			if fi, err := os.Stat(cand); err == nil && !fi.IsDir() && fi.Mode().IsRegular() {
				if absCand, err := filepath.Abs(cand); err == nil {
					return absCand, fi, nil
				}
			}
		}
	}

	return "", nil, nil
}

// EnsureForwarderBinary locates or prepares the bob-proxy-fwd binary inside dataDir/bin.
// It accepts a ForwarderConfig specifying dataDir, optional custom binary path, and runtime build flag.
func EnsureForwarderBinary(cfg ForwarderConfig) (string, error) {
	cleanDataDir := strings.TrimSpace(cfg.DataDir)
	if cleanDataDir == "" {
		cleanDataDir = "./data"
	}
	absDataDir, err := filepath.Abs(cleanDataDir)
	if err != nil {
		return "", fmt.Errorf("failed to resolve data dir %q: %w", cleanDataDir, err)
	}

	targetPath := filepath.Join(absDataDir, "bin", "bob-proxy-fwd")

	// Custom path override takes precedence
	if custom := strings.TrimSpace(cfg.ProxyFwdPath); custom != "" {
		sourcePath, sourceFi, err := resolveSourceBinary(custom)
		if err != nil {
			return "", err
		}
		forwarderMu.Lock()
		defer forwarderMu.Unlock()
		if fi, err := os.Stat(targetPath); err == nil && !fi.IsDir() && fi.Size() == sourceFi.Size() && !sourceFi.ModTime().After(fi.ModTime()) && fi.Mode()&0o111 != 0 {
			return targetPath, nil
		}
		if err := CopyFile(sourcePath, targetPath, 0o755); err != nil {
			return "", fmt.Errorf("failed to copy forwarder binary from %q to %q: %w", sourcePath, targetPath, err)
		}
		return targetPath, nil
	}

	// Fast-path: check if target is already cached and valid under read lock
	forwarderMu.RLock()
	if fi, err := os.Stat(targetPath); err == nil && !fi.IsDir() && fi.Size() > 0 && fi.Mode()&0o111 != 0 {
		hashPath := targetPath + ".sha256"
		if gzData, err := readEmbeddedForwarder(); err == nil && len(gzData) > 0 {
			expectedHash := fmt.Sprintf("%x", sha256.Sum256(gzData))
			if cachedHash, err := os.ReadFile(hashPath); err == nil && strings.TrimSpace(string(cachedHash)) == expectedHash {
				forwarderMu.RUnlock()
				return targetPath, nil
			}
		} else {
			forwarderMu.RUnlock()
			return targetPath, nil
		}
	}
	forwarderMu.RUnlock()

	forwarderMu.Lock()
	defer forwarderMu.Unlock()

	// 1. Try extracting from embedded forwarder binary
	extracted, err := extractEmbeddedForwarder(targetPath)
	if err != nil {
		return "", fmt.Errorf("failed to extract embedded forwarder: %w", err)
	}
	if extracted {
		return targetPath, nil
	}

	// 2. Check if an existing valid binary already exists at targetPath (e.g. pre-populated in docker)
	if fi, err := os.Stat(targetPath); err == nil && !fi.IsDir() && fi.Size() > 0 {
		if fi.Mode()&0o111 == 0 {
			if chmodErr := os.Chmod(targetPath, 0o755); chmodErr != nil {
				slog.Warn("failed to chmod cached forwarder binary", "path", targetPath, "error", chmodErr)
			}
		}
		return targetPath, nil
	}

	// 3. Fallback to source binary candidates on host/filesystem
	sourcePath, _, err := resolveSourceBinary("")
	if err != nil {
		return "", err
	}
	if sourcePath != "" {
		if err := CopyFile(sourcePath, targetPath, 0o755); err != nil {
			return "", fmt.Errorf("failed to copy forwarder binary from %q to %q: %w", sourcePath, targetPath, err)
		}
		return targetPath, nil
	}

	// 4. Fallback: Compile on-demand if Go toolchain is available in dev or test environment
	if isBuildAllowed(cfg) {
		goPath, err := exec.LookPath("go")
		if err == nil && goPath != "" {
			if err := os.MkdirAll(filepath.Dir(targetPath), 0o755); err != nil {
				return "", fmt.Errorf("failed to create forwarder bin directory: %w", err)
			}

			repoRoot := findRepoRoot()
			buildPkg := "./cmd/bob-proxy-fwd"
			cmd := exec.Command(goPath, "build", "-mod=vendor", "-ldflags=-w -s", "-o", targetPath, buildPkg)
			if repoRoot != "" {
				cmd.Dir = repoRoot
			}
			cmd.Env = append(os.Environ(),
				"CGO_ENABLED=0",
				"GOOS=linux",
				"GOARCH="+runtime.GOARCH,
			)
			var stderr strings.Builder
			cmd.Stderr = &stderr

			if buildErr := cmd.Run(); buildErr == nil {
				if err := os.Chmod(targetPath, 0o755); err != nil {
					return "", fmt.Errorf("failed to chmod compiled forwarder %q: %w", targetPath, err)
				}
				return targetPath, nil
			} else {
				slog.Warn("failed to compile bob-proxy-fwd on demand", "error", buildErr, "details", stderr.String())
			}
		}
	}

	return "", errors.New("bob-proxy-fwd binary not found and cannot be built; ensure 'make build-fwd' was run or set SANDBOX_PROXY_FWD_PATH")
}

// EnsureForwarderBinaryPath prepares the forwarder using only a data directory with default options.
func EnsureForwarderBinaryPath(dataDir string) (string, error) {
	return EnsureForwarderBinary(ForwarderConfig{DataDir: dataDir})
}

