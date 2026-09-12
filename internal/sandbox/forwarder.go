package sandbox

import (
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
)

var forwarderMu sync.RWMutex

func copyFile(src, dst string, mode os.FileMode) error {
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

	if err := tmpFile.Chmod(mode); err != nil {
		return err
	}

	if _, err := io.Copy(tmpFile, in); err != nil {
		return err
	}

	if err := tmpFile.Close(); err != nil {
		return err
	}

	return os.Rename(tmpDst, dst)
}

func isBuildAllowed() bool {
	if os.Getenv("BOB_ALLOW_RUNTIME_BUILD") == "0" {
		return false
	}
	if os.Getenv("BOB_ALLOW_RUNTIME_BUILD") == "1" || os.Getenv("ENV") == "development" {
		return true
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

// EnsureForwarderBinary locates or prepares the bob-proxy-fwd binary inside dataDir/bin.
// It ensures the returned binary exists under dataDir (for sibling Docker toHostPath compatibility),
// is statically built for Linux, and has executable permissions.
func EnsureForwarderBinary(dataDir string) (string, error) {
	cleanDataDir := strings.TrimSpace(dataDir)
	if cleanDataDir == "" {
		cleanDataDir = "./data"
	}
	absDataDir, err := filepath.Abs(cleanDataDir)
	if err != nil {
		return "", fmt.Errorf("failed to resolve data dir %q: %w", cleanDataDir, err)
	}

	targetPath := filepath.Join(absDataDir, "bin", "bob-proxy-fwd")

	// Fast-path: check cache under read lock
	forwarderMu.RLock()
	if fi, err := os.Stat(targetPath); err == nil && !fi.IsDir() && fi.Size() > 0 {
		custom := strings.TrimSpace(os.Getenv("SANDBOX_PROXY_FWD_PATH"))
		if custom == "" {
			if fi.Mode()&0o111 != 0 {
				forwarderMu.RUnlock()
				return targetPath, nil
			}
		} else if customFi, err := os.Stat(custom); err == nil && !customFi.IsDir() {
			if !customFi.ModTime().After(fi.ModTime()) && customFi.Size() == fi.Size() && fi.Mode()&0o111 != 0 {
				forwarderMu.RUnlock()
				return targetPath, nil
			}
		}
	}
	forwarderMu.RUnlock()

	forwarderMu.Lock()
	defer forwarderMu.Unlock()

	// Double-checked locking after acquiring write lock
	if fi, err := os.Stat(targetPath); err == nil && !fi.IsDir() && fi.Size() > 0 {
		custom := strings.TrimSpace(os.Getenv("SANDBOX_PROXY_FWD_PATH"))
		if custom == "" {
			if chmodErr := os.Chmod(targetPath, 0o755); chmodErr != nil {
				slog.Warn("failed to chmod cached forwarder binary", "path", targetPath, "error", chmodErr)
			}
			return targetPath, nil
		}
		if customFi, err := os.Stat(custom); err == nil && !customFi.IsDir() {
			if !customFi.ModTime().After(fi.ModTime()) && customFi.Size() == fi.Size() {
				if chmodErr := os.Chmod(targetPath, 0o755); chmodErr != nil {
					slog.Warn("failed to chmod cached forwarder binary", "path", targetPath, "error", chmodErr)
				}
				return targetPath, nil
			}
		}
	}

	// 1. Check custom environment override
	if custom := strings.TrimSpace(os.Getenv("SANDBOX_PROXY_FWD_PATH")); custom != "" {
		if fi, err := os.Stat(custom); err == nil && !fi.IsDir() {
			absCustom, err := filepath.Abs(custom)
			if err != nil {
				return "", fmt.Errorf("failed to resolve custom forwarder path %q: %w", custom, err)
			}
			if err := copyFile(absCustom, targetPath, 0o755); err != nil {
				return "", fmt.Errorf("failed to copy custom forwarder to dataDir: %w", err)
			}
			return targetPath, nil
		}
	}

	// 2. Check candidate search paths for pre-built binary (trusted paths only)
	var candidates []string
	if execPath, err := os.Executable(); err == nil {
		execDir := filepath.Dir(execPath)
		candidates = append(candidates,
			filepath.Join(execDir, "bob-proxy-fwd"),
			filepath.Join(execDir, "bin", "bob-proxy-fwd"),
		)
	}

	for _, cand := range candidates {
		if fi, err := os.Stat(cand); err == nil && !fi.IsDir() {
			absCand, err := filepath.Abs(cand)
			if err != nil {
				continue
			}
			if err := copyFile(absCand, targetPath, 0o755); err == nil {
				return targetPath, nil
			}
		}
	}

	// 3. Fallback: Compile on-demand if Go toolchain is available in dev or test environment
	if isBuildAllowed() {
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
