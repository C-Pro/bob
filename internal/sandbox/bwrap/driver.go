package bwrap

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"syscall"
	"time"

	"bob/internal/sandbox"
)

// Driver implements sandbox.Driver using bubblewrap (/usr/bin/bwrap).
type Driver struct {
	bwrapPath          string
	cpuLimit           float64
	memoryLimitMB      int
	mu                 sync.Mutex
	proxies            map[string]*sandbox.FilteringProxy // keyed by userID
	customBlockedCIDRs []string
}

// Config provides configuration parameters for the Bubblewrap driver.
type Config struct {
	CPULimit      float64
	MemoryLimitMB int
}

// SetCustomBlockedCIDRs overrides default mandatory blocked CIDRs for testing.
func (d *Driver) SetCustomBlockedCIDRs(cidrs []string) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.customBlockedCIDRs = cidrs
}

// NewDriver creates a new Bubblewrap driver with default limits.
func NewDriver() *Driver {
	return NewDriverWithConfig(Config{})
}

// NewDriverWithConfig creates a new Bubblewrap driver with custom resource limits.
func NewDriverWithConfig(cfg Config) *Driver {
	path, _ := exec.LookPath("bwrap")
	mem := cfg.MemoryLimitMB
	if mem <= 0 {
		mem = 512
	}
	cpu := cfg.CPULimit
	if cpu <= 0 {
		cpu = 1.0
	}
	return &Driver{
		bwrapPath:     path,
		cpuLimit:      cpu,
		memoryLimitMB: mem,
		proxies:       make(map[string]*sandbox.FilteringProxy),
	}
}

// Type returns the driver type.
func (d *Driver) Type() sandbox.DriverType {
	return sandbox.DriverBwrap
}

// Available checks if the bwrap binary exists on PATH.
func (d *Driver) Available(ctx context.Context) bool {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.bwrapPath != "" {
		return true
	}
	path, err := exec.LookPath("bwrap")
	if err == nil && path != "" {
		d.bwrapPath = path
		return true
	}
	return false
}

// Create initializes workspace resources and sets up a network filtering proxy if needed.
func (d *Driver) Create(ctx context.Context, sbx *sandbox.UserSandbox, userWorkspaceDir string) error {
	if !d.Available(ctx) {
		return errors.New("bubblewrap (bwrap) is not available on host")
	}

	absWorkspace, err := filepath.Abs(userWorkspaceDir)
	if err != nil {
		return fmt.Errorf("failed to resolve user workspace directory: %w", err)
	}
	if err := os.MkdirAll(absWorkspace, 0o755); err != nil {
		return fmt.Errorf("failed to create user workspace directory: %w", err)
	}
	sbx.SetWorkspaceDir(absWorkspace)

	if sbx.Network.Mode == sandbox.NetworkRestricted {
		d.mu.Lock()
		defer d.mu.Unlock()
		if existing, ok := d.proxies[sbx.UserID]; ok {
			if err := existing.Close(); err != nil {
				slog.Warn("failed to close existing bwrap proxy", "user", sbx.UserID, "error", err)
			}
		}
		sockDir, err := os.MkdirTemp("", fmt.Sprintf("bob-proxy-%s-", sbx.UserID))
		if err != nil {
			return fmt.Errorf("failed to create proxy socket directory: %w", err)
		}
		sockPath := filepath.Join(sockDir, "proxy.sock")
		proxy, err := sandbox.NewFilteringProxyWithConfig(sandbox.ProxyConfig{
			Policy:        sbx.Network,
			ListenTCP:     "127.0.0.1:0",
			SocketPath:    sockPath,
			CustomBlocked: d.customBlockedCIDRs,
		})
		if err != nil {
			return fmt.Errorf("failed to start filtering proxy for user %s: %w", sbx.UserID, err)
		}
		d.proxies[sbx.UserID] = proxy
	}

	return nil
}

// Exec executes a command inside the bwrap sandbox.
func (d *Driver) Exec(ctx context.Context, sbx *sandbox.UserSandbox, cmd []string, timeout time.Duration) (*sandbox.ExecResult, error) {
	if !d.Available(ctx) {
		return nil, errors.New("bubblewrap (bwrap) is not available on host")
	}
	if len(cmd) == 0 {
		return nil, errors.New("command cannot be empty")
	}

	execCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	args := []string{
		"--die-with-parent",
	}

	// Network configuration
	switch sbx.Network.Mode {
	case sandbox.NetworkFull:
		// Unshare namespaces except network
		args = append(args, "--unshare-user", "--unshare-ipc", "--unshare-pid", "--unshare-uts")
		d.mountNetFiles(&args)

	case sandbox.NetworkRestricted:
		// Enforce kernel network namespace airgap: unshare all namespaces including network.
		// Loopback (lo) is the ONLY network interface inside the sandbox.
		// Any raw TCP/UDP socket attempt to external or LAN IPs fails immediately with ENETUNREACH.
		args = append(args, "--unshare-all")
		d.mountNetFiles(&args)

	case sandbox.NetworkNone:
		fallthrough
	default:
		// Isolate everything including network
		args = append(args, "--unshare-all")
	}

	// Mount essential host system directories (read-only)
	systemDirs := []string{"/usr", "/lib", "/lib64", "/bin", "/etc/alternatives"}
	for _, sysDir := range systemDirs {
		if fi, err := os.Stat(sysDir); err == nil && fi.IsDir() {
			args = append(args, "--ro-bind", sysDir, sysDir)
		}
	}

	// Mount standard pseudo filesystems
	args = append(args,
		"--proc", "/proc",
		"--dev", "/dev",
		"--tmpfs", "/tmp",
	)

	// Mount user workspace directory or subdirectories
	userWorkspaceDir := sbx.GetWorkspaceDir()
	if userWorkspaceDir == "" {
		userWorkspaceDir = filepath.Clean(filepath.Join("./data/sandboxes", sbx.UserID))
	}
	absWorkspace, err := filepath.Abs(userWorkspaceDir)
	if err != nil {
		return nil, fmt.Errorf("failed to resolve workspace directory: %w", err)
	}
	if err := os.MkdirAll(absWorkspace, 0o755); err != nil {
		return nil, fmt.Errorf("failed to ensure workspace directory: %w", err)
	}

	var wholeWorkspaceMount *sandbox.UserMount
	var subMounts []sandbox.UserMount

	for i := range sbx.Mounts {
		m := &sbx.Mounts[i]
		_, isWhole, err := sandbox.ValidateMountPath(absWorkspace, m.RelativePath)
		if err != nil {
			return nil, fmt.Errorf("invalid mount %q: %w", m.RelativePath, err)
		}
		if isWhole {
			if wholeWorkspaceMount == nil {
				wholeWorkspaceMount = m
			}
		} else {
			subMounts = append(subMounts, *m)
		}
	}

	if wholeWorkspaceMount != nil {
		if wholeWorkspaceMount.ReadOnly {
			args = append(args, "--ro-bind", absWorkspace, sandbox.DefaultWorkspaceMountPath)
		} else {
			args = append(args, "--bind", absWorkspace, sandbox.DefaultWorkspaceMountPath)
		}
	} else {
		// Isolate scratch workspace on tmpfs
		args = append(args, "--tmpfs", sandbox.DefaultWorkspaceMountPath)
		for _, m := range subMounts {
			hostSubpath, _, err := sandbox.ValidateMountPath(absWorkspace, m.RelativePath)
			if err != nil {
				return nil, fmt.Errorf("invalid mount %q: %w", m.RelativePath, err)
			}
			if err := os.MkdirAll(hostSubpath, 0o755); err != nil {
				return nil, fmt.Errorf("failed to prepare mount path %q: %w", hostSubpath, err)
			}
			cleanDest, err := sandbox.ValidateSandboxMountPath(m.SandboxPath)
			if err != nil {
				return nil, fmt.Errorf("invalid sandbox mount destination %q: %w", m.SandboxPath, err)
			}
			sandboxDest := cleanDest
			if sandboxDest == "" {
				sandboxDest = filepath.Join(sandbox.DefaultWorkspaceMountPath, m.RelativePath)
			}
			if m.ReadOnly {
				args = append(args, "--ro-bind", hostSubpath, sandboxDest)
			} else {
				args = append(args, "--bind", hostSubpath, sandboxDest)
			}
		}
	}

	args = append(args, "--chdir", sandbox.DefaultWorkspaceMountPath)

	var proxySockPath string
	var fwdBinary string
	if sbx.Network.Mode == sandbox.NetworkRestricted {
		d.mu.Lock()
		proxy := d.proxies[sbx.UserID]
		d.mu.Unlock()
		if proxy != nil && proxy.SocketPath() != "" {
			proxySockPath = proxy.SocketPath()
			args = append(args, "--dir", "/run", "--bind", proxySockPath, "/run/proxy.sock")

			fwd, err := sandbox.EnsureForwarderBinary("")
			if err != nil {
				return nil, fmt.Errorf("failed to ensure forwarder binary: %w", err)
			}
			fwdBinary = fwd
			args = append(args, "--dir", "/run/proxy", "--ro-bind", fwdBinary, "/run/proxy/fwd")
		}
	}

	// Configure environment inside the sandbox
	args = append(args,
		"--clearenv",
		"--setenv", "PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
		"--setenv", "HOME", sandbox.DefaultWorkspaceMountPath,
		"--setenv", "PWD", sandbox.DefaultWorkspaceMountPath,
		"--setenv", "TERM", "dumb",
	)

	if sbx.Network.Mode == sandbox.NetworkRestricted {
		proxyAddr := "http://127.0.0.1:18080"
		if proxySockPath == "" {
			d.mu.Lock()
			proxy := d.proxies[sbx.UserID]
			d.mu.Unlock()
			if proxy != nil {
				proxyAddr = proxy.Addr()
			}
		}
		args = append(args,
			"--setenv", "http_proxy", proxyAddr,
			"--setenv", "https_proxy", proxyAddr,
			"--setenv", "HTTP_PROXY", proxyAddr,
			"--setenv", "HTTPS_PROXY", proxyAddr,
			"--setenv", "all_proxy", proxyAddr,
			"--setenv", "ALL_PROXY", proxyAddr,
		)
	}

	// Append command
	if fwdBinary != "" {
		args = append(args, "/run/proxy/fwd", "-tcp", "127.0.0.1:18080", "-sock", "/run/proxy.sock", "--")
		args = append(args, cmd...)
	} else {
		args = append(args, cmd...)
	}

	var bwrapCmd *exec.Cmd
	if hasSystemdUserScope() {
		sysArgs := buildSystemdArgs(d.memoryLimitMB, d.cpuLimit, d.bwrapPath, args)
		// nosemgrep: go.lang.security.audit.dangerous-exec-command.dangerous-exec-command
		bwrapCmd = exec.CommandContext(execCtx, "systemd-run", sysArgs...)
	} else {
		slog.Warn("systemd-run --user not available; bwrap running without cgroup resource limits (MemoryMax/TasksMax/CPUQuota)", "user", sbx.UserID)
		// nosemgrep: go.lang.security.audit.dangerous-exec-command.dangerous-exec-command
		bwrapCmd = exec.CommandContext(execCtx, d.bwrapPath, args...)
	}

	stdout := sandbox.NewBoundedBuffer(sandbox.DefaultMaxOutputBytes)
	stderr := sandbox.NewBoundedBuffer(sandbox.DefaultMaxOutputBytes)
	bwrapCmd.Stdout = stdout
	bwrapCmd.Stderr = stderr

	startTime := time.Now()
	err = bwrapCmd.Run()
	duration := time.Since(startTime)

	if execCtx.Err() != nil {
		errStr := stderr.String()
		if errStr != "" {
			errStr += "\ncommand timed out"
		} else {
			errStr = "command timed out"
		}
		return &sandbox.ExecResult{
			ExitCode: -1,
			Stdout:   stdout.String(),
			Stderr:   errStr,
			Duration: duration,
		}, nil
	}

	exitCode := 0
	if err != nil {
		var exitErr *exec.ExitError
		if errors.As(err, &exitErr) {
			if status, ok := exitErr.Sys().(syscall.WaitStatus); ok {
				exitCode = status.ExitStatus()
			} else {
				exitCode = 1
			}
		} else {
			return nil, fmt.Errorf("bwrap execution failed: %w", err)
		}
	}

	return &sandbox.ExecResult{
		ExitCode: exitCode,
		Stdout:   stdout.String(),
		Stderr:   stderr.String(),
		Duration: duration,
	}, nil
}

// Destroy cleans up resources and shuts down filtering proxies associated with the user.
func (d *Driver) Destroy(ctx context.Context, sbx *sandbox.UserSandbox) error {
	d.mu.Lock()
	defer d.mu.Unlock()
	if proxy, ok := d.proxies[sbx.UserID]; ok {
		if err := proxy.Close(); err != nil {
			slog.Warn("failed to close bwrap filtering proxy on destroy", "user", sbx.UserID, "error", err)
		}
		delete(d.proxies, sbx.UserID)
	}
	return nil
}

func (d *Driver) mountNetFiles(args *[]string) {
	netFiles := []string{"/etc/resolv.conf", "/etc/ssl", "/etc/ca-certificates"}
	for _, f := range netFiles {
		realPath, err := filepath.EvalSymlinks(f)
		if err != nil {
			realPath = f
		}
		if _, err := os.Stat(realPath); err == nil {
			*args = append(*args, "--ro-bind", realPath, f)
		}
	}
}

func hasSystemdUserScope() bool {
	cmd := exec.Command("systemd-run", "--user", "--scope", "-q", "true")
	return cmd.Run() == nil
}

func buildSystemdArgs(memMB int, cpuLimit float64, bwrapPath string, bwrapArgs []string) []string {
	if memMB <= 0 {
		memMB = 512
	}
	sysArgs := []string{
		"--user", "--scope", "-q",
		"-p", fmt.Sprintf("MemoryMax=%dM", memMB),
		"-p", "TasksMax=64",
	}
	if cpuLimit > 0 {
		sysArgs = append(sysArgs, "-p", fmt.Sprintf("CPUQuota=%d%%", int(cpuLimit*100)))
	}
	sysArgs = append(sysArgs, "--", bwrapPath)
	sysArgs = append(sysArgs, bwrapArgs...)
	return sysArgs
}
