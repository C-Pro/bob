package docker

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
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
	client        *http.Client
	mu            sync.Mutex
	proxies       map[string]*sandbox.FilteringProxy // keyed by userID
}

// Config provides configuration parameters for the Docker driver.
type Config struct {
	SocketPath    string
	AllowedImages []string
	CPULimit      float64
	MemoryLimitMB int
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

	return &Driver{
		socketPath:    socket,
		allowedImages: cfg.AllowedImages,
		cpuLimit:      cpu,
		memoryLimitMB: mem,
		client: &http.Client{
			Transport: transport,
			Timeout:   10 * time.Minute,
		},
		proxies: make(map[string]*sandbox.FilteringProxy),
	}
}

// Type returns the driver type.
func (d *Driver) Type() sandbox.DriverType {
	return sandbox.DriverDocker
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
func (d *Driver) Create(ctx context.Context, sbx *sandbox.UserSandbox, userWorkspaceDir string) error {
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
	sbx.WorkspaceDir = absWorkspace

	if sbx.Network.Mode == sandbox.NetworkRestricted {
		d.mu.Lock()
		if existing, ok := d.proxies[sbx.UserID]; ok {
			_ = existing.Close()
		}
		proxy, err := sandbox.NewFilteringProxyWithConfig(sandbox.ProxyConfig{
			Policy:    sbx.Network,
			ListenTCP: "0.0.0.0:0",
		})
		if err != nil {
			d.mu.Unlock()
			return fmt.Errorf("failed to start filtering proxy for user %s: %w", sbx.UserID, err)
		}
		d.proxies[sbx.UserID] = proxy
		d.mu.Unlock()
	}

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
		binds = append(binds, fmt.Sprintf("%s:/workspace:%s", absWorkspace, mode))
	} else {
		for _, m := range subMounts {
			hostSubpath, _, err := sandbox.ValidateMountPath(absWorkspace, m.RelativePath)
			if err != nil {
				return fmt.Errorf("invalid mount %q: %w", m.RelativePath, err)
			}
			if err := os.MkdirAll(hostSubpath, 0o755); err != nil {
				return fmt.Errorf("failed to prepare mount subpath %q: %w", hostSubpath, err)
			}
			dest := m.SandboxPath
			if dest == "" {
				dest = filepath.Join("/workspace", m.RelativePath)
			}
			mode := "rw"
			if m.ReadOnly {
				mode = "ro"
			}
			binds = append(binds, fmt.Sprintf("%s:%s:%s", hostSubpath, dest, mode))
		}
	}

	networkMode := "none"
	if sbx.Network.Mode == sandbox.NetworkFull || sbx.Network.Mode == sandbox.NetworkRestricted {
		networkMode = "bridge"
	}

	memBytes := int64(d.memoryLimitMB) * 1024 * 1024
	nanoCPUs := int64(d.cpuLimit * 1e9)

	hostConfig := map[string]interface{}{
		"NetworkMode": networkMode,
		"Binds":       binds,
		"Memory":      memBytes,
		"NanoCPUs":    nanoCPUs,
	}
	if sbx.Network.Mode == sandbox.NetworkRestricted {
		hostConfig["ExtraHosts"] = []string{"host.docker.internal:host-gateway"}
	}

	createPayload := map[string]interface{}{
		"Image":      image,
		"Cmd":        []string{"sleep", "infinity"},
		"WorkingDir": "/workspace",
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
	sbx.InternalID = createResult.ID

	// Start container
	startURL := fmt.Sprintf("http://localhost/containers/%s/start", sbx.InternalID)
	startReq, err := http.NewRequestWithContext(ctx, http.MethodPost, startURL, nil)
	if err != nil {
		_ = d.Destroy(ctx, sbx)
		return fmt.Errorf("failed to build container start request: %w", err)
	}

	startResp, err := d.client.Do(startReq)
	if err != nil {
		_ = d.Destroy(ctx, sbx)
		return fmt.Errorf("failed to start container: %w", err)
	}
	defer func() { _ = startResp.Body.Close() }()

	if startResp.StatusCode != http.StatusNoContent && startResp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(startResp.Body)
		_ = d.Destroy(ctx, sbx)
		return fmt.Errorf("docker container start returned status %d: %s", startResp.StatusCode, string(respBody))
	}

	return nil
}

// Exec executes a command inside the running sandbox container.
func (d *Driver) Exec(ctx context.Context, sbx *sandbox.UserSandbox, cmd []string, timeout time.Duration) (*sandbox.ExecResult, error) {
	if sbx.InternalID == "" {
		return nil, errors.New("container has not been created or is not running")
	}

	execCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	var env []string
	if sbx.Network.Mode == sandbox.NetworkRestricted {
		d.mu.Lock()
		proxy := d.proxies[sbx.UserID]
		d.mu.Unlock()
		if proxy != nil {
			proxyAddr := fmt.Sprintf("http://host.docker.internal:%d", proxy.Port())
			env = append(env,
				"http_proxy="+proxyAddr,
				"https_proxy="+proxyAddr,
				"HTTP_PROXY="+proxyAddr,
				"HTTPS_PROXY="+proxyAddr,
				"all_proxy="+proxyAddr,
				"ALL_PROXY="+proxyAddr,
			)
		}
	}

	// 1. Create exec instance
	execCreatePayload := map[string]interface{}{
		"AttachStdout": true,
		"AttachStderr": true,
		"Cmd":          cmd,
		"Env":          env,
	}
	bodyJSON, _ := json.Marshal(execCreatePayload)

	createURL := fmt.Sprintf("http://localhost/containers/%s/exec", sbx.InternalID)
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
		return nil, fmt.Errorf("failed to start exec: %w", err)
	}
	defer func() { _ = startResp.Body.Close() }()

	var stdout, stderr bytes.Buffer
	err = demuxDockerStream(startResp.Body, &stdout, &stderr)
	duration := time.Since(startTime)
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
		_ = proxy.Close()
		delete(d.proxies, sbx.UserID)
	}
	d.mu.Unlock()

	if sbx.InternalID == "" {
		return nil
	}

	deleteURL := fmt.Sprintf("http://localhost/containers/%s?v=1&force=true", sbx.InternalID)
	req, err := http.NewRequestWithContext(ctx, http.MethodDelete, deleteURL, nil)
	if err != nil {
		return fmt.Errorf("failed to build container delete request: %w", err)
	}

	resp, err := d.client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to delete container %s: %w", sbx.InternalID, err)
	}
	defer func() { _ = resp.Body.Close() }()

	sbx.InternalID = ""
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
