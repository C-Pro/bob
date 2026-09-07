package bwrap

import (
	"bytes"
	"context"
	"errors"
	"fmt"
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
	bwrapPath string
	mu        sync.Mutex
	proxies   map[string]*sandbox.FilteringProxy // keyed by userID
}

// NewDriver creates a new Bubblewrap driver.
func NewDriver() *Driver {
	path, _ := exec.LookPath("bwrap")
	return &Driver{
		bwrapPath: path,
		proxies:   make(map[string]*sandbox.FilteringProxy),
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

	if err := os.MkdirAll(userWorkspaceDir, 0o755); err != nil {
		return fmt.Errorf("failed to create user workspace directory: %w", err)
	}
	sbx.WorkspaceDir = userWorkspaceDir

	if sbx.Network.Mode == sandbox.NetworkRestricted {
		d.mu.Lock()
		defer d.mu.Unlock()
		if existing, ok := d.proxies[sbx.UserID]; ok {
			_ = existing.Close()
		}
		sockPath := filepath.Join(os.TempDir(), fmt.Sprintf("bob-proxy-%s.sock", sbx.UserID))
		proxy, err := sandbox.NewFilteringProxyWithConfig(sandbox.ProxyConfig{
			Policy:     sbx.Network,
			ListenTCP:  "127.0.0.1:0",
			SocketPath: sockPath,
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
	userWorkspaceDir := sbx.WorkspaceDir
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
			args = append(args, "--ro-bind", absWorkspace, "/workspace")
		} else {
			args = append(args, "--bind", absWorkspace, "/workspace")
		}
	} else {
		// Isolate scratch workspace on tmpfs
		args = append(args, "--tmpfs", "/workspace")
		for _, m := range subMounts {
			hostSubpath, _, err := sandbox.ValidateMountPath(absWorkspace, m.RelativePath)
			if err != nil {
				return nil, fmt.Errorf("invalid mount %q: %w", m.RelativePath, err)
			}
			if err := os.MkdirAll(hostSubpath, 0o755); err != nil {
				return nil, fmt.Errorf("failed to prepare mount path %q: %w", hostSubpath, err)
			}
			sandboxDest := m.SandboxPath
			if sandboxDest == "" {
				sandboxDest = filepath.Join("/workspace", m.RelativePath)
			}
			if m.ReadOnly {
				args = append(args, "--ro-bind", hostSubpath, sandboxDest)
			} else {
				args = append(args, "--bind", hostSubpath, sandboxDest)
			}
		}
	}

	args = append(args, "--chdir", "/workspace")

	var proxySockPath string
	if sbx.Network.Mode == sandbox.NetworkRestricted {
		d.mu.Lock()
		proxy := d.proxies[sbx.UserID]
		d.mu.Unlock()
		if proxy != nil && proxy.SocketPath() != "" {
			proxySockPath = proxy.SocketPath()
			args = append(args, "--dir", "/run", "--bind", proxySockPath, "/run/proxy.sock")
		}
	}

	// Append command
	if proxySockPath != "" {
		forwarderScript := `
if command -v socat >/dev/null 2>&1; then
    socat TCP-LISTEN:18080,bind=127.0.0.1,reuseaddr,fork UNIX-CONNECT:/run/proxy.sock </dev/null >/dev/null 2>&1 &
    FWD_PID=$!
elif command -v python3 >/dev/null 2>&1; then
    python3 -c 'import socket,threading
def p(a,b):
 try:
  while 1:
   d=a.recv(4096)
   if not d:break
   b.sendall(d)
 except:pass
 finally:
  try:a.close();b.close()
  except:pass
s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
s.bind(("127.0.0.1",18080))
s.listen(16)
while 1:
 c,_=s.accept()
 u=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
 u.connect("/run/proxy.sock")
 threading.Thread(target=p,args=(c,u),daemon=True).start()
 threading.Thread(target=p,args=(u,c),daemon=True).start()
' </dev/null >/dev/null 2>&1 &
    FWD_PID=$!
fi
sleep 0.05
"$@"
EXIT_CODE=$?
if [ -n "$FWD_PID" ]; then
    kill -9 $FWD_PID 2>/dev/null || true
fi
exit $EXIT_CODE
`
		args = append(args, "/bin/sh", "-c", forwarderScript, "--")
		args = append(args, cmd...)
	} else {
		args = append(args, cmd...)
	}

	// nosemgrep: go.lang.security.audit.dangerous-exec-command.dangerous-exec-command
	bwrapCmd := exec.CommandContext(execCtx, d.bwrapPath, args...)

	// Configure environment variables
	env := []string{
		"PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
		"HOME=/workspace",
		"TERM=dumb",
	}

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
		env = append(env,
			"http_proxy="+proxyAddr,
			"https_proxy="+proxyAddr,
			"HTTP_PROXY="+proxyAddr,
			"HTTPS_PROXY="+proxyAddr,
			"all_proxy="+proxyAddr,
			"ALL_PROXY="+proxyAddr,
		)
	}
	bwrapCmd.Env = env

	var stdout, stderr bytes.Buffer
	bwrapCmd.Stdout = &stdout
	bwrapCmd.Stderr = &stderr

	startTime := time.Now()
	err = bwrapCmd.Run()
	duration := time.Since(startTime)

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
		_ = proxy.Close()
		delete(d.proxies, sbx.UserID)
	}
	return nil
}

func (d *Driver) mountNetFiles(args *[]string) {
	netFiles := []string{"/etc/resolv.conf", "/etc/ssl", "/etc/ca-certificates"}
	for _, f := range netFiles {
		if fi, err := os.Stat(f); err == nil {
			if fi.IsDir() {
				*args = append(*args, "--ro-bind", f, f)
			} else {
				*args = append(*args, "--ro-bind", f, f)
			}
		}
	}
}
