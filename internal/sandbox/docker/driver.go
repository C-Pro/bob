package docker

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"bob/internal/sandbox"
)

// Driver implements sandbox.Driver using the Docker Engine API over Unix socket.
// It is 100% pure Go and CGO-free, requiring no 3rd-party Docker SDK dependencies.
type Driver struct {
	socketPath    string
	allowedImages []string
	cpuLimit      float64
	memoryLimitMB int
	dataDir       string
	hostDataDir   string
	client        *http.Client
	mu            sync.Mutex
	proxies            map[string]*sandbox.FilteringProxy // keyed by userID
	customBlockedCIDRs []string
}

// Config provides configuration parameters for the Docker driver.
type Config struct {
	SocketPath    string
	AllowedImages []string
	CPULimit      float64
	MemoryLimitMB int
	DataDir       string
	HostDataDir   string
}

// NewDriver creates a new Docker Sibling driver.
func NewDriver(cfg Config) *Driver {
	socket := cfg.SocketPath
	if socket == "" {
		socket = "/var/run/docker.sock"
	}
	cpu := cfg.CPULimit
	if cpu <= 0 {
		cpu = 1.0
	}
	mem := cfg.MemoryLimitMB
	if mem <= 0 {
		mem = 512
	}

	transport := &http.Transport{
		DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
			return (&net.Dialer{}).DialContext(ctx, "unix", socket)
		},
		DisableKeepAlives: false,
	}

	hostData := strings.TrimSpace(cfg.HostDataDir)
	if hostData != "" {
		hostData = filepath.Clean(hostData)
	}

	return &Driver{
		socketPath:    socket,
		allowedImages: cfg.AllowedImages,
		cpuLimit:      cpu,
		memoryLimitMB: mem,
		dataDir:       strings.TrimSpace(cfg.DataDir),
		hostDataDir:   hostData,
		client: &http.Client{
			Transport: transport,
			Timeout:   10 * time.Minute,
		},
		proxies: make(map[string]*sandbox.FilteringProxy),
	}
}

// toHostPath translates a container-local path under dataDir to the corresponding hostDataDir path.
// If hostDataDir is empty, containerPath is returned unchanged as host and container paths match.
// If hostDataDir is set, containerPath must be within dataDir, otherwise an error is returned.
func (d *Driver) toHostPath(containerPath string) (string, error) {
	if d.hostDataDir == "" {
		return containerPath, nil
	}
	if d.dataDir == "" {
		return "", errors.New("host data directory is configured but data directory is empty")
	}
	absData, err := filepath.Abs(d.dataDir)
	if err != nil {
		return "", fmt.Errorf("failed to resolve data directory %q: %w", d.dataDir, err)
	}
	absPath, err := filepath.Abs(containerPath)
	if err != nil {
		return "", fmt.Errorf("failed to resolve container path %q: %w", containerPath, err)
	}
	rel, err := filepath.Rel(absData, absPath)
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("path %q is outside data directory %q and cannot be mapped to host data directory %q", containerPath, d.dataDir, d.hostDataDir)
	}
	return filepath.Join(d.hostDataDir, rel), nil
}

// Type returns the driver type.
func (d *Driver) Type() sandbox.DriverType {
	return sandbox.DriverDocker
}

// SetCustomBlockedCIDRs sets custom blocked CIDRs for testing.
func (d *Driver) SetCustomBlockedCIDRs(cidrs []string) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.customBlockedCIDRs = cidrs
}

// Available pings the Docker daemon to check if the Unix socket is functional.
func (d *Driver) Available(ctx context.Context) bool {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, "http://localhost/_ping", nil)
	if err != nil {
		return false
	}
	resp, err := d.client.Do(req)
	if err != nil {
		return false
	}
	defer func() { _ = resp.Body.Close() }()
	return resp.StatusCode == http.StatusOK
}

// Create spawns an idle sandbox container and initializes user mounts.
func (d *Driver) Create(ctx context.Context, sbx *sandbox.UserSandbox, userWorkspaceDir string) (err error) {
	image := strings.TrimSpace(sbx.DockerImage)
	if image == "" {
		if len(d.allowedImages) > 0 {
			image = d.allowedImages[0]
		} else {
			image = "alpine:latest"
		}
		sbx.DockerImage = image
	}

	if err := sandbox.ValidateImage(image, d.allowedImages); err != nil {
		return err
	}

	if err := os.MkdirAll(userWorkspaceDir, 0o755); err != nil {
		return fmt.Errorf("failed to create user workspace directory: %w", err)
	}

	absWorkspace, err := filepath.Abs(userWorkspaceDir)
	if err != nil {
		return fmt.Errorf("failed to resolve workspace directory: %w", err)
	}
	sbx.SetWorkspaceDir(absWorkspace)

	createdProxy := false
	var restrictedBinds []string
	if sbx.Network.Mode == sandbox.NetworkRestricted {
		fwdBinary, err := sandbox.EnsureForwarderBinary(d.dataDir)
		if err != nil {
			return fmt.Errorf("failed to ensure proxy forwarder binary: %w", err)
		}
		hostFwdPath, err := d.toHostPath(fwdBinary)
		if err != nil {
			return fmt.Errorf("failed to resolve host path for forwarder binary: %w", err)
		}

		proxyDir := filepath.Join(absWorkspace, ".proxy")
		if err := os.MkdirAll(proxyDir, 0o700); err != nil {
			return fmt.Errorf("failed to create proxy directory: %w", err)
		}
		hostProxyDir, err := d.toHostPath(proxyDir)
		if err != nil {
			return fmt.Errorf("failed to resolve host path for proxy directory: %w", err)
		}
		sockPath := filepath.Join(proxyDir, "proxy.sock")

		d.mu.Lock()
		if existing, ok := d.proxies[sbx.UserID]; ok {
			if err := existing.Close(); err != nil {
				slog.Warn("failed to close existing proxy", "user", sbx.UserID, "error", err)
			}
		}
		proxy, err := sandbox.NewFilteringProxyWithConfig(sandbox.ProxyConfig{
			Policy:        sbx.Network,
			SocketPath:    sockPath,
			CustomBlocked: d.customBlockedCIDRs,
		})
		if err != nil {
			d.mu.Unlock()
			return fmt.Errorf("failed to start filtering proxy for user %s: %w", sbx.UserID, err)
		}
		_ = os.Chmod(sockPath, 0o666)
		d.proxies[sbx.UserID] = proxy
		createdProxy = true
		d.mu.Unlock()

		restrictedBinds = append(restrictedBinds,
			fmt.Sprintf("%s:/run/proxy/fwd:ro", hostFwdPath),
			fmt.Sprintf("%s:/run/proxy:rw", hostProxyDir),
		)
	}
	defer func() {
		if err != nil && createdProxy {
			d.mu.Lock()
			if p, ok := d.proxies[sbx.UserID]; ok {
				if cErr := p.Close(); cErr != nil {
					slog.Warn("error closing proxy during create rollback", "user", sbx.UserID, "error", cErr)
				}
				delete(d.proxies, sbx.UserID)
			}
			d.mu.Unlock()
		}
	}()

	var binds []string
	var wholeWorkspaceMount *sandbox.UserMount
	var subMounts []sandbox.UserMount

	for i := range sbx.Mounts {
		m := &sbx.Mounts[i]
		_, isWhole, err := sandbox.ValidateMountPath(absWorkspace, m.RelativePath)
		if err != nil {
			return fmt.Errorf("invalid mount %q: %w", m.RelativePath, err)
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
		mode := "rw"
		if wholeWorkspaceMount.ReadOnly {
			mode = "ro"
		}
		hostMountSource, err := d.toHostPath(absWorkspace)
		if err != nil {
			return fmt.Errorf("failed to resolve host path for workspace: %w", err)
		}
		binds = append(binds, fmt.Sprintf("%s:%s:%s", hostMountSource, sandbox.DefaultWorkspaceMountPath, mode))
	} else {
		for _, m := range subMounts {
			hostSubpath, _, err := sandbox.ValidateMountPath(absWorkspace, m.RelativePath)
			if err != nil {
				return fmt.Errorf("invalid mount %q: %w", m.RelativePath, err)
			}
			if err := os.MkdirAll(hostSubpath, 0o755); err != nil {
				return fmt.Errorf("failed to prepare mount subpath %q: %w", hostSubpath, err)
			}
			cleanDest, err := sandbox.ValidateSandboxMountPath(m.SandboxPath)
			if err != nil {
				return fmt.Errorf("invalid sandbox mount destination %q: %w", m.SandboxPath, err)
			}
			dest := cleanDest
			if dest == "" {
				dest = filepath.Join(sandbox.DefaultWorkspaceMountPath, m.RelativePath)
			}
			mode := "rw"
			if m.ReadOnly {
				mode = "ro"
			}
			hostMountSource, err := d.toHostPath(hostSubpath)
			if err != nil {
				return fmt.Errorf("failed to resolve host path for mount %q: %w", m.RelativePath, err)
			}
			binds = append(binds, fmt.Sprintf("%s:%s:%s", hostMountSource, dest, mode))
		}
	}

	binds = append(binds, restrictedBinds...)

	networkMode := "none"
	if sbx.Network.Mode == sandbox.NetworkFull {
		networkMode = "bridge"
	}

	memBytes := int64(d.memoryLimitMB) * 1024 * 1024
	nanoCPUs := int64(d.cpuLimit * 1e9)

	pidsLimit := int64(64)
	hostConfig := map[string]interface{}{
		"NetworkMode": networkMode,
		"Binds":       binds,
		"Memory":      memBytes,
		"NanoCPUs":    nanoCPUs,
		"PidsLimit":   pidsLimit,
		"SecurityOpt": []string{"no-new-privileges"},
		"CapDrop":     []string{"ALL"},
	}

	containerCmd := []string{"sleep", "infinity"}
	if sbx.Network.Mode == sandbox.NetworkRestricted {
		containerCmd = []string{
			"/run/proxy/fwd",
			"-daemon",
			"-tcp", "127.0.0.1:18080",
			"-sock", "/run/proxy/proxy.sock",
		}
	}

	createPayload := map[string]interface{}{
		"Image":      image,
		"Cmd":        containerCmd,
		"User":       fmt.Sprintf("%d:%d", os.Getuid(), os.Getgid()),
		"WorkingDir": sandbox.DefaultWorkspaceMountPath,
		"HostConfig": hostConfig,
	}

	bodyJSON, err := json.Marshal(createPayload)
	if err != nil {
		return fmt.Errorf("failed to encode container payload: %w", err)
	}

	createReq, err := http.NewRequestWithContext(ctx, http.MethodPost, "http://localhost/containers/create", bytes.NewReader(bodyJSON))
	if err != nil {
		return fmt.Errorf("failed to build container create request: %w", err)
	}
	createReq.Header.Set("Content-Type", "application/json")

	createResp, err := d.client.Do(createReq)
	if err != nil {
		return fmt.Errorf("failed to connect to docker daemon: %w", err)
	}

	if createResp.StatusCode == http.StatusNotFound {
		_ = createResp.Body.Close()
		// Image is missing locally; pull it and retry container create
		if pullErr := d.pullImage(ctx, image); pullErr != nil {
			return fmt.Errorf("image %s missing locally and pull failed: %w", image, pullErr)
		}

		retryReq, rErr := http.NewRequestWithContext(ctx, http.MethodPost, "http://localhost/containers/create", bytes.NewReader(bodyJSON))
		if rErr != nil {
			return fmt.Errorf("failed to build retry container create request: %w", rErr)
		}
		retryReq.Header.Set("Content-Type", "application/json")

		createResp, err = d.client.Do(retryReq)
		if err != nil {
			return fmt.Errorf("failed to retry container create after image pull: %w", err)
		}
	}
	defer func() { _ = createResp.Body.Close() }()

	if createResp.StatusCode != http.StatusCreated {
		respBody, _ := io.ReadAll(createResp.Body)
		return fmt.Errorf("docker container create returned status %d: %s", createResp.StatusCode, string(respBody))
	}

	var createResult struct {
		ID string `json:"Id"`
	}
	if err := json.NewDecoder(createResp.Body).Decode(&createResult); err != nil {
		return fmt.Errorf("failed to decode container create response: %w", err)
	}
	sbx.SetInternalID(createResult.ID)

	// Start container
	startURL := fmt.Sprintf("http://localhost/containers/%s/start", sbx.GetInternalID())
	startReq, err := http.NewRequestWithContext(ctx, http.MethodPost, startURL, nil)
	if err != nil {
		if dErr := d.Destroy(ctx, sbx); dErr != nil {
			slog.Warn("failed to cleanup container on start request error", "user", sbx.UserID, "error", dErr)
		}
		return fmt.Errorf("failed to build container start request: %w", err)
	}

	startResp, err := d.client.Do(startReq)
	if err != nil {
		if dErr := d.Destroy(ctx, sbx); dErr != nil {
			slog.Warn("failed to cleanup container on start error", "user", sbx.UserID, "error", dErr)
		}
		return fmt.Errorf("failed to start container: %w", err)
	}
	defer func() { _ = startResp.Body.Close() }()

	if startResp.StatusCode != http.StatusNoContent && startResp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(startResp.Body)
		if dErr := d.Destroy(ctx, sbx); dErr != nil {
			slog.Warn("failed to cleanup container on unexpected status", "user", sbx.UserID, "error", dErr)
		}
		return fmt.Errorf("docker container start returned status %d: %s", startResp.StatusCode, string(respBody))
	}

	return nil
}

// Exec executes a command inside the running sandbox container.
func (d *Driver) Exec(ctx context.Context, sbx *sandbox.UserSandbox, cmd []string, timeout time.Duration) (*sandbox.ExecResult, error) {
	internalID := sbx.GetInternalID()
	if internalID == "" {
		return nil, errors.New("container has not been created or is not running")
	}

	execCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	var env []string
	env = append(env,
		"HOME="+sandbox.DefaultWorkspaceMountPath,
		"PWD="+sandbox.DefaultWorkspaceMountPath,
	)
	if sbx.Network.Mode == sandbox.NetworkRestricted {
		proxyAddr := "http://127.0.0.1:18080"
		env = append(env,
			"http_proxy="+proxyAddr,
			"https_proxy="+proxyAddr,
			"HTTP_PROXY="+proxyAddr,
			"HTTPS_PROXY="+proxyAddr,
			"all_proxy="+proxyAddr,
			"ALL_PROXY="+proxyAddr,
		)
	}

	// 1. Create exec instance
	execCreatePayload := map[string]interface{}{
		"AttachStdout": true,
		"AttachStderr": true,
		"Cmd":          cmd,
		"Env":          env,
		"WorkingDir":   sandbox.DefaultWorkspaceMountPath,
	}
	bodyJSON, _ := json.Marshal(execCreatePayload)

	createURL := fmt.Sprintf("http://localhost/containers/%s/exec", internalID)
	req, err := http.NewRequestWithContext(execCtx, http.MethodPost, createURL, bytes.NewReader(bodyJSON))
	if err != nil {
		return nil, fmt.Errorf("failed to build exec create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := d.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to create exec instance: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusCreated {
		b, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("exec create returned %d: %s", resp.StatusCode, string(b))
	}

	var execCreateResult struct {
		ID string `json:"Id"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&execCreateResult); err != nil {
		return nil, fmt.Errorf("failed to decode exec create response: %w", err)
	}
	execID := execCreateResult.ID

	// 2. Start exec instance and stream multiplexed stdout/stderr
	execStartPayload := map[string]interface{}{
		"Detach": false,
		"Tty":    false,
	}
	startBody, _ := json.Marshal(execStartPayload)

	startURL := fmt.Sprintf("http://localhost/exec/%s/start", execID)
	startReq, err := http.NewRequestWithContext(execCtx, http.MethodPost, startURL, bytes.NewReader(startBody))
	if err != nil {
		return nil, fmt.Errorf("failed to build exec start request: %w", err)
	}
	startReq.Header.Set("Content-Type", "application/json")

	startTime := time.Now()
	startResp, err := d.client.Do(startReq)
	if err != nil {
		if execCtx.Err() != nil {
			d.killExecProcess(internalID, execID)
			return &sandbox.ExecResult{
				ExitCode: -1,
				Stdout:   "",
				Stderr:   "command timed out",
				Duration: time.Since(startTime),
			}, nil
		}
		return nil, fmt.Errorf("failed to start exec: %w", err)
	}
	defer func() { _ = startResp.Body.Close() }()

	stdout := sandbox.NewBoundedBuffer(sandbox.DefaultMaxOutputBytes)
	stderr := sandbox.NewBoundedBuffer(sandbox.DefaultMaxOutputBytes)
	err = demuxDockerStream(startResp.Body, stdout, stderr)
	duration := time.Since(startTime)
	if execCtx.Err() != nil {
		d.killExecProcess(internalID, execID)
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
	if err != nil && !errors.Is(err, io.EOF) {
		return nil, fmt.Errorf("error reading exec stream: %w", err)
	}

	// 3. Inspect exec instance to retrieve exit code
	inspectURL := fmt.Sprintf("http://localhost/exec/%s/json", execID)
	inspectReq, err := http.NewRequestWithContext(ctx, http.MethodGet, inspectURL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to build exec inspect request: %w", err)
	}

	inspectResp, err := d.client.Do(inspectReq)
	if err != nil {
		return nil, fmt.Errorf("failed to inspect exec: %w", err)
	}
	defer func() { _ = inspectResp.Body.Close() }()

	var inspectResult struct {
		ExitCode int  `json:"ExitCode"`
		Running  bool `json:"Running"`
	}
	if err := json.NewDecoder(inspectResp.Body).Decode(&inspectResult); err != nil {
		return nil, fmt.Errorf("failed to decode exec inspect response: %w", err)
	}

	return &sandbox.ExecResult{
		ExitCode: inspectResult.ExitCode,
		Stdout:   stdout.String(),
		Stderr:   stderr.String(),
		Duration: duration,
	}, nil
}

// Destroy terminates and removes the Docker container and stops any filtering proxy.
func (d *Driver) Destroy(ctx context.Context, sbx *sandbox.UserSandbox) error {
	d.mu.Lock()
	if proxy, ok := d.proxies[sbx.UserID]; ok {
		if err := proxy.Close(); err != nil {
			slog.Warn("failed to close filtering proxy on destroy", "user", sbx.UserID, "error", err)
		}
		delete(d.proxies, sbx.UserID)
	}
	d.mu.Unlock()

	internalID := sbx.GetInternalID()
	if internalID == "" {
		return nil
	}

	deleteURL := fmt.Sprintf("http://localhost/containers/%s?v=1&force=true", internalID)
	req, err := http.NewRequestWithContext(ctx, http.MethodDelete, deleteURL, nil)
	if err != nil {
		return fmt.Errorf("failed to build container delete request: %w", err)
	}

	resp, err := d.client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to delete container %s: %w", internalID, err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusNoContent && resp.StatusCode != http.StatusNotFound {
		return fmt.Errorf("failed to delete container %s: returned status %d", internalID, resp.StatusCode)
	}

	sbx.SetInternalID("")
	return nil
}

// demuxDockerStream demultiplexes Docker's 8-byte framed stdout/stderr stream.
// Frame Header:
// byte 0: stream type (1 = stdout, 2 = stderr)
// bytes 1-3: unused
// bytes 4-7: big-endian uint32 payload length
func demuxDockerStream(r io.Reader, stdout, stderr io.Writer) error {
	header := make([]byte, 8)
	for {
		_, err := io.ReadFull(r, header)
		if err != nil {
			if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
				return nil
			}
			return err
		}

		streamType := header[0]
		frameLen := binary.BigEndian.Uint32(header[4:8])

		var dest io.Writer
		switch streamType {
		case 1:
			dest = stdout
		case 2:
			dest = stderr
		default:
			dest = io.Discard
		}

		_, err = io.CopyN(dest, r, int64(frameLen))
		if err != nil {
			return err
		}
	}
}

func (d *Driver) killExecProcess(internalID, _ string) {
	if internalID == "" {
		return
	}
	// Restart the container with t=0 (instant SIGKILL).
	// This synchronously kills all processes in the container cgroup, waits for exit,
	// and restarts the container with 'sleep infinity', eliminating race conditions.
	restartCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	restartURL := fmt.Sprintf("http://localhost/containers/%s/restart?t=0", internalID)
	req, err := http.NewRequestWithContext(restartCtx, http.MethodPost, restartURL, nil)
	if err != nil {
		slog.Error("failed to create container restart request on exec timeout", "container", internalID, "error", err)
		return
	}

	resp, err := d.client.Do(req)
	if err != nil {
		slog.Error("failed to restart container on exec timeout", "container", internalID, "error", err)
		return
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusNoContent && resp.StatusCode != http.StatusOK {
		slog.Warn("unexpected status restarting container on exec timeout", "container", internalID, "status", resp.StatusCode)
	}
}

// pullImage streams and downloads a docker image from registry using Docker Engine API.
func (d *Driver) pullImage(ctx context.Context, image string) error {
	slog.Info("pulling missing docker image for sandbox", "image", image)

	pullCtx := ctx
	if _, hasDeadline := ctx.Deadline(); !hasDeadline {
		var cancel context.CancelFunc
		pullCtx, cancel = context.WithTimeout(ctx, 5*time.Minute)
		defer cancel()
	}

	pullURL := fmt.Sprintf("http://localhost/images/create?fromImage=%s", url.QueryEscape(image))
	req, err := http.NewRequestWithContext(pullCtx, http.MethodPost, pullURL, nil)
	if err != nil {
		return fmt.Errorf("failed to build image pull request: %w", err)
	}

	resp, err := d.client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to connect to docker daemon for image pull: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("docker image pull returned status %d: %s", resp.StatusCode, string(respBody))
	}

	// Drain streaming progress messages and verify no error occurred mid-pull
	dec := json.NewDecoder(resp.Body)
	for {
		var msg struct {
			Status      string `json:"status"`
			Error       string `json:"error"`
			ErrorDetail struct {
				Message string `json:"message"`
			} `json:"errorDetail"`
		}
		if err := dec.Decode(&msg); err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return fmt.Errorf("error reading image pull stream: %w", err)
		}
		if msg.Error != "" {
			return fmt.Errorf("docker image pull failed: %s", msg.Error)
		}
	}

	slog.Info("successfully pulled docker image for sandbox", "image", image)
	return nil
}
