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

var forwarderMu sync.Mutex

func copyFile(src, dst string, mode os.FileMode) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer func() { _ = in.Close() }()

	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return err
	}

	tmpDst := dst + ".tmp"
	out, err := os.OpenFile(tmpDst, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, mode)
	if err != nil {
		return err
	}

	if _, err := io.Copy(out, in); err != nil {
		_ = out.Close()
		_ = os.Remove(tmpDst)
		return err
	}

	if err := out.Close(); err != nil {
		_ = os.Remove(tmpDst)
		return err
	}

	return os.Rename(tmpDst, dst)
}

// EnsureForwarderBinary locates or prepares the bob-proxy-fwd binary inside dataDir/bin.
// It ensures the returned binary exists under dataDir (for sibling Docker toHostPath compatibility),
// is statically built for Linux, and has executable permissions.
func EnsureForwarderBinary(dataDir string) (string, error) {
	forwarderMu.Lock()
	defer forwarderMu.Unlock()

	cleanDataDir := strings.TrimSpace(dataDir)
	if cleanDataDir == "" {
		cleanDataDir = "./data"
	}
	absDataDir, err := filepath.Abs(cleanDataDir)
	if err != nil {
		return "", fmt.Errorf("failed to resolve data dir %q: %w", cleanDataDir, err)
	}

	targetPath := filepath.Join(absDataDir, "bin", "bob-proxy-fwd")

	if fi, err := os.Stat(targetPath); err == nil && !fi.IsDir() {
		_ = os.Chmod(targetPath, 0o755)
		return targetPath, nil
	}

	// 1. Check custom environment override
	if custom := strings.TrimSpace(os.Getenv("SANDBOX_PROXY_FWD_PATH")); custom != "" {
		if fi, err := os.Stat(custom); err == nil && !fi.IsDir() {
			absCustom, err := filepath.Abs(custom)
			if err != nil {
				return "", fmt.Errorf("failed to resolve custom forwarder path %q: %w", custom, err)
			}
			if err := copyFile(absCustom, targetPath, 0o755); err != nil {
				slog.Warn("failed to copy custom forwarder to dataDir, using original", "source", absCustom, "error", err)
				return absCustom, nil
			}
			return targetPath, nil
		}
	}

	// 2. Check candidate search paths for pre-built binary
	var candidates []string

	// Current working directory bin/
	candidates = append(candidates, filepath.Join("bin", "bob-proxy-fwd"))

	// Beside current running executable
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

	// 3. Fallback: Compile on-demand if Go toolchain is available (e.g. dev or test environment)
	goPath, err := exec.LookPath("go")
	if err == nil && goPath != "" {
		if err := os.MkdirAll(filepath.Dir(targetPath), 0o755); err != nil {
			return "", fmt.Errorf("failed to create forwarder bin directory: %w", err)
		}

		cmd := exec.Command(goPath, "build", "-ldflags=-w -s", "-o", targetPath, "bob/cmd/bob-proxy-fwd")
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

	return "", errors.New("bob-proxy-fwd binary not found and cannot be built; ensure 'make build-fwd' was run or set SANDBOX_PROXY_FWD_PATH")
}
