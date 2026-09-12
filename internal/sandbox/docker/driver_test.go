package docker

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"bob/internal/sandbox"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDockerDriverWithMockServer(t *testing.T) {
	tempDir := t.TempDir()
	sockPath := filepath.Join(tempDir, "mock_docker.sock")

	listener, err := net.Listen("unix", sockPath)
	require.NoError(t, err)
	defer func() { _ = listener.Close() }()

	mux := http.NewServeMux()

	mux.HandleFunc("/_ping", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("OK"))
	})

	mux.HandleFunc("/containers/create", func(w http.ResponseWriter, r *http.Request) {
		var payload map[string]interface{}
		_ = json.NewDecoder(r.Body).Decode(&payload)
		assert.Equal(t, "alpine:latest", payload["Image"])

		hostCfg, ok := payload["HostConfig"].(map[string]interface{})
		assert.True(t, ok)
		assert.Equal(t, "none", hostCfg["NetworkMode"])
		assert.Equal(t, float64(64), hostCfg["PidsLimit"])
		assert.Equal(t, []interface{}{"no-new-privileges"}, hostCfg["SecurityOpt"])
		assert.Equal(t, []interface{}{"ALL"}, hostCfg["CapDrop"])
		assert.Equal(t, fmt.Sprintf("%d:%d", os.Getuid(), os.Getgid()), payload["User"])

		w.WriteHeader(http.StatusCreated)
		_, _ = w.Write([]byte(`{"Id":"mock-container-abc"}`))
	})

	mux.HandleFunc("/containers/mock-container-abc/start", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	})

	mux.HandleFunc("/containers/mock-container-abc/exec", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusCreated)
		_, _ = w.Write([]byte(`{"Id":"mock-exec-xyz"}`))
	})

	mux.HandleFunc("/exec/mock-exec-xyz/start", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		// Write stdout frame: type=1, len=11, payload="mock output"
		payload := []byte("mock output")
		header := make([]byte, 8)
		header[0] = 1
		binary.BigEndian.PutUint32(header[4:8], uint32(len(payload)))
		_, _ = w.Write(header)
		_, _ = w.Write(payload)
	})

	mux.HandleFunc("/exec/mock-exec-xyz/json", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"ExitCode":0,"Running":false}`))
	})

	mux.HandleFunc("/containers/mock-container-abc", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodDelete {
			w.WriteHeader(http.StatusNoContent)
		}
	})

	server := &http.Server{Handler: mux}
	go func() {
		_ = server.Serve(listener)
	}()
	defer func() { _ = server.Close() }()

	driver := NewDriver(Config{
		SocketPath:    sockPath,
		AllowedImages: []string{"alpine:latest"},
		CPULimit:      1.0,
		MemoryLimitMB: 512,
	})

	ctx := context.Background()
	assert.Equal(t, sandbox.DriverDocker, driver.Type())
	assert.True(t, driver.Available(ctx))

	userWorkspace := filepath.Join(tempDir, "user_workspace")
	require.NoError(t, os.MkdirAll(userWorkspace, 0o755))

	sbx := &sandbox.UserSandbox{
		UserID:      "testuser",
		DockerImage: "alpine:latest",
		Network: sandbox.NetworkPolicy{
			Mode: sandbox.NetworkNone,
		},
		Status: sandbox.StatusRunning,
	}

	err = driver.Create(ctx, sbx, userWorkspace)
	require.NoError(t, err)
	assert.Equal(t, "mock-container-abc", sbx.GetInternalID())

	execRes, err := driver.Exec(ctx, sbx, []string{"echo", "hi"}, 5*time.Second)
	require.NoError(t, err)
	assert.Equal(t, 0, execRes.ExitCode)
	assert.Equal(t, "mock output", execRes.Stdout)

	err = driver.Destroy(ctx, sbx)
	require.NoError(t, err)
	assert.Empty(t, sbx.GetInternalID())
}

func TestDockerDriver_ConcurrentExecDestroyRace(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if strings.Contains(r.URL.Path, "/exec") {
			_ = json.NewEncoder(w).Encode(map[string]interface{}{"Id": "exec-abc"})
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer ts.Close()

	driver := NewDriver(Config{SocketPath: ts.URL})
	driver.client = ts.Client()

	sbx := &sandbox.UserSandbox{
		UserID: "testuser",
	}
	sbx.SetInternalID("container-race-test")

	ctx := context.Background()
	var wg sync.WaitGroup

	// Goroutine 1: Rapid Exec calls reading InternalID
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < 50; i++ {
			_, _ = driver.Exec(ctx, sbx, []string{"echo", "1"}, time.Second)
		}
	}()

	// Goroutine 2: Rapid SetInternalID / Destroy clearing InternalID
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < 50; i++ {
			_ = driver.Destroy(ctx, sbx)
			sbx.SetInternalID("container-race-test")
		}
	}()

	wg.Wait()
}

func TestDemuxDockerStream(t *testing.T) {
	var input bytes.Buffer

	// Frame 1: stdout, "stdout message\n"
	msg1 := []byte("stdout message\n")
	hdr1 := make([]byte, 8)
	hdr1[0] = 1
	binary.BigEndian.PutUint32(hdr1[4:8], uint32(len(msg1)))
	input.Write(hdr1)
	input.Write(msg1)

	// Frame 2: stderr, "stderr warning\n"
	msg2 := []byte("stderr warning\n")
	hdr2 := make([]byte, 8)
	hdr2[0] = 2
	binary.BigEndian.PutUint32(hdr2[4:8], uint32(len(msg2)))
	input.Write(hdr2)
	input.Write(msg2)

	var stdout, stderr bytes.Buffer
	err := demuxDockerStream(&input, &stdout, &stderr)
	require.NoError(t, err)

	assert.Equal(t, "stdout message\n", stdout.String())
	assert.Equal(t, "stderr warning\n", stderr.String())
}

func TestDockerDriver_NetworkRestrictedProxyReachability(t *testing.T) {
	tempDir := t.TempDir()
	sockPath := filepath.Join(tempDir, "mock_docker_restr.sock")

	listener, err := net.Listen("unix", sockPath)
	require.NoError(t, err)
	defer func() { _ = listener.Close() }()

	var capturedCreateHostConfig map[string]interface{}
	var capturedCreateCmd []interface{}
	var capturedExecEnv []string
	var capturedExecWorkingDir string

	mux := http.NewServeMux()
	mux.HandleFunc("/_ping", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("OK"))
	})

	mux.HandleFunc("/containers/create", func(w http.ResponseWriter, r *http.Request) {
		var payload map[string]interface{}
		_ = json.NewDecoder(r.Body).Decode(&payload)
		capturedCreateHostConfig, _ = payload["HostConfig"].(map[string]interface{})
		capturedCreateCmd, _ = payload["Cmd"].([]interface{})

		w.WriteHeader(http.StatusCreated)
		_, _ = w.Write([]byte(`{"Id":"mock-restr-container"}`))
	})

	mux.HandleFunc("/containers/mock-restr-container/start", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	})

	mux.HandleFunc("/containers/mock-restr-container/exec", func(w http.ResponseWriter, r *http.Request) {
		var payload map[string]interface{}
		_ = json.NewDecoder(r.Body).Decode(&payload)
		if wd, ok := payload["WorkingDir"].(string); ok {
			capturedExecWorkingDir = wd
		}
		if envList, ok := payload["Env"].([]interface{}); ok {
			for _, e := range envList {
				capturedExecEnv = append(capturedExecEnv, e.(string))
			}
		}

		w.WriteHeader(http.StatusCreated)
		_, _ = w.Write([]byte(`{"Id":"mock-restr-exec"}`))
	})

	mux.HandleFunc("/exec/mock-restr-exec/start", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		payload := []byte("ok")
		header := make([]byte, 8)
		header[0] = 1
		binary.BigEndian.PutUint32(header[4:8], uint32(len(payload)))
		_, _ = w.Write(header)
		_, _ = w.Write(payload)
	})

	mux.HandleFunc("/exec/mock-restr-exec/json", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"ExitCode":0,"Running":false}`))
	})

	mux.HandleFunc("/containers/mock-restr-container", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodDelete {
			w.WriteHeader(http.StatusNoContent)
		}
	})

	server := &http.Server{Handler: mux}
	go func() {
		_ = server.Serve(listener)
	}()
	defer func() { _ = server.Close() }()

	dataDir := filepath.Join(tempDir, "data")
	driver := NewDriver(Config{
		SocketPath:    sockPath,
		AllowedImages: []string{"alpine:latest"},
		CPULimit:      1.0,
		MemoryLimitMB: 512,
		DataDir:       dataDir,
	})

	ctx := context.Background()
	userWorkspace := filepath.Join(tempDir, "user_workspace")
	require.NoError(t, os.MkdirAll(userWorkspace, 0o755))

	sbx := &sandbox.UserSandbox{
		UserID:      "testuser_docker_restr",
		DockerImage: "alpine:latest",
		Network: sandbox.NetworkPolicy{
			Mode:         sandbox.NetworkRestricted,
			AllowedHosts: []string{"api.github.com"},
		},
		Status: sandbox.StatusRunning,
	}

	err = driver.Create(ctx, sbx, userWorkspace)
	require.NoError(t, err)
	defer func() { _ = driver.Destroy(ctx, sbx) }()

	// Verify HostConfig
	require.NotNil(t, capturedCreateHostConfig)
	assert.Equal(t, "none", capturedCreateHostConfig["NetworkMode"])
	assert.Nil(t, capturedCreateHostConfig["ExtraHosts"])
	assert.Equal(t, true, capturedCreateHostConfig["Init"])

	// Verify Cmd (Defect 1 & Scenario 2)
	expectedCmd := []interface{}{
		"/run/proxy/fwd",
		"-daemon",
		"-tcp",
		"127.0.0.1:18080",
		"-sock",
		"/run/proxy/proxy.sock",
	}
	assert.Equal(t, expectedCmd, capturedCreateCmd)

	binds, ok := capturedCreateHostConfig["Binds"].([]interface{})
	require.True(t, ok)
	var hasProxyBind bool
	expectedProxyPrefix := driver.getProxyDir(sbx.UserID)
	for _, b := range binds {
		str, _ := b.(string)
		assert.False(t, strings.HasSuffix(str, ":/run/proxy/fwd:ro"), "overlapping /run/proxy/fwd bind mount must not be used")
		if strings.HasSuffix(str, ":/run/proxy:ro") && strings.HasPrefix(str, expectedProxyPrefix) {
			hasProxyBind = true
		}
	}
	assert.True(t, hasProxyBind, "expected single /run/proxy:ro bind mount under dataDir")

	// Verify that the forwarder binary exists and is executable in the proxy directory on host
	fwdFi, err := os.Stat(filepath.Join(expectedProxyPrefix, "fwd"))
	require.NoError(t, err)
	assert.True(t, fwdFi.Mode()&0o111 != 0, "forwarder binary must be executable")

	// Ensure workspace is not contaminated with .proxy
	_, statErr := os.Stat(filepath.Join(userWorkspace, ".proxy"))
	assert.True(t, os.IsNotExist(statErr), "user workspace should not contain .proxy directory")

	// Execute command
	res, err := driver.Exec(ctx, sbx, []string{"echo", "test"}, 5*time.Second)
	require.NoError(t, err)
	assert.Equal(t, 0, res.ExitCode)

	// Verify working directory and environment variables in exec instance
	assert.Equal(t, sandbox.DefaultWorkspaceMountPath, capturedExecWorkingDir)
	assert.Contains(t, capturedExecEnv, "HOME="+sandbox.DefaultWorkspaceMountPath)
	assert.Contains(t, capturedExecEnv, "PWD="+sandbox.DefaultWorkspaceMountPath)

	var hasHttpProxy, hasLoopbackProxy, hasNoProxy, hasNoProxyUpper bool
	for _, envVar := range capturedExecEnv {
		if strings.HasPrefix(envVar, "HTTP_PROXY=") {
			hasHttpProxy = true
			if strings.Contains(envVar, "127.0.0.1:18080") {
				hasLoopbackProxy = true
			}
		}
		if envVar == "no_proxy=localhost,127.0.0.1" {
			hasNoProxy = true
		}
		if envVar == "NO_PROXY=localhost,127.0.0.1" {
			hasNoProxyUpper = true
		}
	}
	assert.True(t, hasHttpProxy, "expected HTTP_PROXY in exec env")
	assert.True(t, hasLoopbackProxy, "expected HTTP_PROXY to point to loopback forwarder 127.0.0.1:18080")
	assert.True(t, hasNoProxy, "expected no_proxy=localhost,127.0.0.1 in exec env")
	assert.True(t, hasNoProxyUpper, "expected NO_PROXY=localhost,127.0.0.1 in exec env")
}

func TestDockerDriver_Create_ForwarderBinaryMissing(t *testing.T) {
	tempDir := t.TempDir()
	sockPath := filepath.Join(tempDir, "mock_docker_fwd_missing.sock")

	listener, err := net.Listen("unix", sockPath)
	require.NoError(t, err)
	defer func() { _ = listener.Close() }()

	driver := NewDriver(Config{
		SocketPath:    sockPath,
		AllowedImages: []string{"alpine:latest"},
		DataDir:       filepath.Join(tempDir, "data"),
		ProxyFwdPath:  filepath.Join(tempDir, "nonexistent", "bob-proxy-fwd"),
	})

	ctx := context.Background()
	userWorkspace := filepath.Join(tempDir, "workspace")
	require.NoError(t, os.MkdirAll(userWorkspace, 0o755))

	sbx := &sandbox.UserSandbox{
		UserID:      "testuser_fwd_missing",
		DockerImage: "alpine:latest",
		Network: sandbox.NetworkPolicy{
			Mode: sandbox.NetworkRestricted,
		},
		Status: sandbox.StatusRunning,
	}

	err = driver.Create(ctx, sbx, userWorkspace)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "failed to ensure proxy forwarder binary")

	driver.mu.Lock()
	defer driver.mu.Unlock()
	assert.Nil(t, driver.proxies[sbx.UserID], "no proxy should be stored on failure")
}


func TestDockerDriver_AutoPullMissingImage_Success(t *testing.T) {
	tempDir := t.TempDir()
	sockPath := filepath.Join(tempDir, "mock_docker_pull.sock")

	listener, err := net.Listen("unix", sockPath)
	require.NoError(t, err)
	defer func() { _ = listener.Close() }()

	var createCalls int
	var pullCalled bool

	mux := http.NewServeMux()
	mux.HandleFunc("/_ping", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("OK"))
	})

	mux.HandleFunc("/images/create", func(w http.ResponseWriter, r *http.Request) {
		pullCalled = true
		assert.Equal(t, "test-image:latest", r.URL.Query().Get("fromImage"))
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"status":"Pulling from test-image"}` + "\n" + `{"status":"Digest: sha256:abc"}` + "\n"))
	})

	mux.HandleFunc("/containers/create", func(w http.ResponseWriter, r *http.Request) {
		createCalls++
		if createCalls == 1 {
			// First attempt fails with 404 No such image
			w.WriteHeader(http.StatusNotFound)
			_, _ = w.Write([]byte(`{"message":"No such image: test-image:latest"}`))
			return
		}
		// Second attempt after pull succeeds with 201 Created
		w.WriteHeader(http.StatusCreated)
		_, _ = w.Write([]byte(`{"Id":"container-after-pull-123"}`))
	})

	mux.HandleFunc("/containers/container-after-pull-123/start", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	})

	mux.HandleFunc("/containers/container-after-pull-123", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodDelete {
			w.WriteHeader(http.StatusNoContent)
		}
	})

	server := &http.Server{Handler: mux}
	go func() {
		_ = server.Serve(listener)
	}()
	defer func() { _ = server.Close() }()

	driver := NewDriver(Config{
		SocketPath:    sockPath,
		AllowedImages: []string{"test-image:latest"},
	})

	ctx := context.Background()
	userWorkspace := filepath.Join(tempDir, "user_workspace")
	require.NoError(t, os.MkdirAll(userWorkspace, 0o755))

	sbx := &sandbox.UserSandbox{
		UserID:      "testuser_pull",
		DockerImage: "test-image:latest",
		Network: sandbox.NetworkPolicy{
			Mode: sandbox.NetworkNone,
		},
		Status: sandbox.StatusRunning,
	}

	err = driver.Create(ctx, sbx, userWorkspace)
	require.NoError(t, err)
	assert.True(t, pullCalled, "expected image pull to be called on missing image")
	assert.Equal(t, 2, createCalls, "expected container create to be retried after pull")
	assert.Equal(t, "container-after-pull-123", sbx.GetInternalID())

	err = driver.Destroy(ctx, sbx)
	require.NoError(t, err)
}

func TestDockerDriver_AutoPullMissingImage_PullError(t *testing.T) {
	tempDir := t.TempDir()
	sockPath := filepath.Join(tempDir, "mock_docker_pull_err.sock")

	listener, err := net.Listen("unix", sockPath)
	require.NoError(t, err)
	defer func() { _ = listener.Close() }()

	mux := http.NewServeMux()
	mux.HandleFunc("/_ping", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("OK"))
	})

	mux.HandleFunc("/containers/create", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		_, _ = w.Write([]byte(`{"message":"No such image: forbidden-image:latest"}`))
	})

	mux.HandleFunc("/images/create", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		// Stream reports mid-pull error
		_, _ = w.Write([]byte(`{"errorDetail":{"message":"pull access denied"},"error":"pull access denied"}` + "\n"))
	})

	server := &http.Server{Handler: mux}
	go func() {
		_ = server.Serve(listener)
	}()
	defer func() { _ = server.Close() }()

	driver := NewDriver(Config{
		SocketPath:    sockPath,
		AllowedImages: []string{"forbidden-image:latest"},
	})

	ctx := context.Background()
	userWorkspace := filepath.Join(tempDir, "user_workspace")
	require.NoError(t, os.MkdirAll(userWorkspace, 0o755))

	sbx := &sandbox.UserSandbox{
		UserID:      "testuser_pull_fail",
		DockerImage: "forbidden-image:latest",
		Network: sandbox.NetworkPolicy{
			Mode: sandbox.NetworkNone,
		},
		Status: sandbox.StatusRunning,
	}

	err = driver.Create(ctx, sbx, userWorkspace)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "pull access denied")
}

func TestDockerDriver_BoundedOutput(t *testing.T) {
	tempDir := t.TempDir()
	sockPath := filepath.Join(tempDir, "mock_docker.sock")

	listener, err := net.Listen("unix", sockPath)
	require.NoError(t, err)
	defer func() { _ = listener.Close() }()

	mux := http.NewServeMux()
	mux.HandleFunc("/_ping", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("OK"))
	})
	mux.HandleFunc("/containers/mock-c/exec", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusCreated)
		_, _ = w.Write([]byte(`{"Id":"exec-big"}`))
	})
	mux.HandleFunc("/exec/exec-big/start", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		// Send 2MB of stdout data in chunks
		chunk := bytes.Repeat([]byte("A"), 64*1024)
		header := make([]byte, 8)
		header[0] = 1 // stdout
		binary.BigEndian.PutUint32(header[4:8], uint32(len(chunk)))
		for i := 0; i < 32; i++ { // 32 * 64KB = 2MB
			_, _ = w.Write(header)
			_, _ = w.Write(chunk)
		}
	})
	mux.HandleFunc("/exec/exec-big/json", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"ExitCode":0,"Running":false}`))
	})

	server := &http.Server{Handler: mux}
	go func() {
		_ = server.Serve(listener)
	}()
	defer func() { _ = server.Close() }()

	driver := NewDriver(Config{
		SocketPath:    sockPath,
		AllowedImages: []string{"alpine:latest"},
	})

	ctx := context.Background()
	sbx := &sandbox.UserSandbox{
		UserID:     "user1",
		Status:     sandbox.StatusRunning,
		InternalID: "mock-c",
	}

	res, err := driver.Exec(ctx, sbx, []string{"head", "-c", "2M", "/dev/zero"}, 5*time.Second)
	require.NoError(t, err)
	assert.Contains(t, res.Stdout, "[output truncated")
	assert.LessOrEqual(t, len(res.Stdout), sandbox.DefaultMaxOutputBytes+200)
}

func TestDockerDriver_ExecTimeoutKill(t *testing.T) {
	tempDir := t.TempDir()
	sockPath := filepath.Join(tempDir, "mock_docker.sock")

	listener, err := net.Listen("unix", sockPath)
	require.NoError(t, err)
	defer func() { _ = listener.Close() }()

	var mu sync.Mutex
	restarted := false

	mux := http.NewServeMux()
	mux.HandleFunc("/_ping", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("OK"))
	})
	mux.HandleFunc("/containers/mock-c/exec", func(w http.ResponseWriter, r *http.Request) {
		var payload map[string]interface{}
		_ = json.NewDecoder(r.Body).Decode(&payload)
		cmdList, _ := payload["Cmd"].([]interface{})
		execID := "exec-subsequent"
		if len(cmdList) > 0 && cmdList[0] == "sleep" {
			execID = "exec-hang"
		}
		w.WriteHeader(http.StatusCreated)
		_, _ = fmt.Fprintf(w, `{"Id":%q}`, execID)
	})
	mux.HandleFunc("/exec/exec-hang/start", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		payload := []byte("starting task...")
		header := make([]byte, 8)
		header[0] = 1 // stdout
		binary.BigEndian.PutUint32(header[4:8], uint32(len(payload)))
		_, _ = w.Write(header)
		_, _ = w.Write(payload)
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		<-r.Context().Done()
	})
	mux.HandleFunc("/exec/exec-subsequent/start", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		payload := []byte("subsequent output")
		header := make([]byte, 8)
		header[0] = 1
		binary.BigEndian.PutUint32(header[4:8], uint32(len(payload)))
		_, _ = w.Write(header)
		_, _ = w.Write(payload)
	})
	mux.HandleFunc("/exec/exec-subsequent/json", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"ExitCode":0,"Running":false}`))
	})
	mux.HandleFunc("/containers/mock-c/restart", func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		if r.URL.Query().Get("t") == "0" {
			restarted = true
		}
		mu.Unlock()
		w.WriteHeader(http.StatusNoContent)
	})

	server := &http.Server{Handler: mux}
	go func() {
		_ = server.Serve(listener)
	}()
	defer func() { _ = server.Close() }()

	driver := NewDriver(Config{
		SocketPath:    sockPath,
		AllowedImages: []string{"alpine:latest"},
	})

	ctx := context.Background()
	sbx := &sandbox.UserSandbox{
		UserID:     "user_timeout",
		Status:     sandbox.StatusRunning,
		InternalID: "mock-c",
	}

	res, err := driver.Exec(ctx, sbx, []string{"sleep", "100"}, 50*time.Millisecond)
	require.NoError(t, err)
	require.NotNil(t, res)
	assert.Equal(t, -1, res.ExitCode)
	assert.Contains(t, res.Stdout, "starting task...")
	assert.Contains(t, res.Stderr, "command timed out")

	mu.Lock()
	didRestart := restarted
	mu.Unlock()
	assert.False(t, didRestart, "container must not be restarted when targeted kill succeeds")

	// Verify that a subsequent command succeeds after the timeout container recovery
	res2, err := driver.Exec(ctx, sbx, []string{"echo", "subsequent"}, 5*time.Second)
	require.NoError(t, err)
	require.NotNil(t, res2)
	assert.Equal(t, 0, res2.ExitCode)
	assert.Equal(t, "subsequent output", res2.Stdout)
}

func TestDockerDriver_Destroy_StatusValidation(t *testing.T) {
	tempDir := t.TempDir()
	sockPath := filepath.Join(tempDir, "mock_docker.sock")

	listener, err := net.Listen("unix", sockPath)
	require.NoError(t, err)
	defer func() { _ = listener.Close() }()

	deleteStatus := http.StatusInternalServerError

	mux := http.NewServeMux()
	mux.HandleFunc("/_ping", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("OK"))
	})
	mux.HandleFunc("/containers/mock-c", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodDelete {
			w.WriteHeader(deleteStatus)
		}
	})

	server := &http.Server{Handler: mux}
	go func() {
		_ = server.Serve(listener)
	}()
	defer func() { _ = server.Close() }()

	driver := NewDriver(Config{
		SocketPath:    sockPath,
		AllowedImages: []string{"alpine:latest"},
	})

	ctx := context.Background()
	sbx := &sandbox.UserSandbox{
		UserID:     "user_del",
		Status:     sandbox.StatusRunning,
		InternalID: "mock-c",
	}

	// 1. Delete returns 500 Internal Server Error
	deleteStatus = http.StatusInternalServerError
	err = driver.Destroy(ctx, sbx)
	require.Error(t, err, "destroy must fail if DELETE returns 500")
	assert.Equal(t, "mock-c", sbx.GetInternalID(), "internal ID must be preserved when destroy fails")

	// 2. Delete returns 204 No Content
	deleteStatus = http.StatusNoContent
	err = driver.Destroy(ctx, sbx)
	require.NoError(t, err, "destroy must succeed if DELETE returns 204")
	assert.Equal(t, "", sbx.GetInternalID(), "internal ID must be cleared after successful destroy")

	// 3. Delete returns 404 Not Found (idempotent success)
	sbx.SetInternalID("mock-c-404")
	deleteStatus = http.StatusNotFound
	err = driver.Destroy(ctx, sbx)
	require.NoError(t, err, "destroy must succeed idempotently if DELETE returns 404")
	assert.Equal(t, "", sbx.GetInternalID(), "internal ID must be cleared after 404 destroy")
}

func TestDockerDriver_Create_ProxyCleanupOnError(t *testing.T) {
	tempDir := t.TempDir()
	sockPath := filepath.Join(tempDir, "mock_docker.sock")

	listener, err := net.Listen("unix", sockPath)
	require.NoError(t, err)
	defer func() { _ = listener.Close() }()

	mux := http.NewServeMux()
	mux.HandleFunc("/_ping", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("OK"))
	})
	// Container create fails with 500
	mux.HandleFunc("/containers/create", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`{"message":"daemon failure"}`))
	})

	server := &http.Server{Handler: mux}
	go func() {
		_ = server.Serve(listener)
	}()
	defer func() { _ = server.Close() }()

	driver := NewDriver(Config{
		SocketPath:    sockPath,
		AllowedImages: []string{"alpine:latest"},
	})

	ctx := context.Background()
	userWorkspace := filepath.Join(tempDir, "user_workspace")
	require.NoError(t, os.MkdirAll(userWorkspace, 0o755))

	sbx := &sandbox.UserSandbox{
		UserID:      "user_leak_test",
		DockerImage: "alpine:latest",
		Network: sandbox.NetworkPolicy{
			Mode:         sandbox.NetworkRestricted,
			AllowedHosts: []string{"example.com"},
		},
		Status: sandbox.StatusRunning,
	}

	err = driver.Create(ctx, sbx, userWorkspace)
	require.Error(t, err, "Create must fail when daemon returns 500")

	driver.mu.Lock()
	_, proxyExists := driver.proxies["user_leak_test"]
	driver.mu.Unlock()
	assert.False(t, proxyExists, "proxy must not be leaked in driver.proxies on create failure")
}

func TestToHostPath(t *testing.T) {
	tempDir := t.TempDir()
	containerDataDir := filepath.Join(tempDir, "container", "data")
	hostDataDir := filepath.Join(tempDir, "host", "data")
	require.NoError(t, os.MkdirAll(containerDataDir, 0o755))

	d := NewDriver(Config{
		DataDir:     containerDataDir,
		HostDataDir: hostDataDir,
	})

	// Subpath under dataDir translates to hostDataDir
	containerSub := filepath.Join(containerDataDir, "sandboxes", "user123")
	expectedHostSub := filepath.Join(hostDataDir, "sandboxes", "user123")
	res, err := d.toHostPath(containerSub)
	require.NoError(t, err)
	assert.Equal(t, expectedHostSub, res)

	// Nested subpath under dataDir translates properly
	nestedSub := filepath.Join(containerDataDir, "sandboxes", "user123", "code", "file.txt")
	expectedNestedHost := filepath.Join(hostDataDir, "sandboxes", "user123", "code", "file.txt")
	res, err = d.toHostPath(nestedSub)
	require.NoError(t, err)
	assert.Equal(t, expectedNestedHost, res)

	// Root of dataDir translates to root of hostDataDir
	res, err = d.toHostPath(containerDataDir)
	require.NoError(t, err)
	assert.Equal(t, hostDataDir, res)

	// Path outside dataDir returns an error when HostDataDir is set
	outsidePath := filepath.Join(tempDir, "other", "sandboxes")
	_, err = d.toHostPath(outsidePath)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "outside data directory")

	// When HostDataDir is empty, paths are resolved to absolute host paths without error
	dNoHost := NewDriver(Config{
		DataDir:     containerDataDir,
		HostDataDir: "",
	})
	res, err = dNoHost.toHostPath(containerSub)
	require.NoError(t, err)
	assert.Equal(t, containerSub, res)

	// Relative path is resolved to absolute
	relPath := filepath.Join("data", "proxies", "user1")
	res, err = dNoHost.toHostPath(relPath)
	require.NoError(t, err)
	expectedAbs, _ := filepath.Abs(relPath)
	assert.Equal(t, expectedAbs, res)
	assert.True(t, filepath.IsAbs(res))

	// getProxyDir always returns absolute path
	proxyDir := dNoHost.getProxyDir("user1")
	assert.True(t, filepath.IsAbs(proxyDir))

	// When DataDir is empty but HostDataDir is set, an error is returned
	dNoData := NewDriver(Config{
		DataDir:     "",
		HostDataDir: hostDataDir,
	})
	_, err = dNoData.toHostPath(containerSub)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "data directory is empty")
}

func TestDockerDriver_HostDataDirBinds(t *testing.T) {
	tempDir := t.TempDir()
	sockPath := filepath.Join(tempDir, "mock_docker.sock")

	containerDataDir := filepath.Join(tempDir, "container", "data")
	hostDataDir := filepath.Join(tempDir, "host", "data")
	require.NoError(t, os.MkdirAll(containerDataDir, 0o755))

	listener, err := net.Listen("unix", sockPath)
	require.NoError(t, err)
	defer func() { _ = listener.Close() }()

	var lastBinds []interface{}
	var bindsMu sync.Mutex

	mux := http.NewServeMux()
	mux.HandleFunc("/_ping", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("OK"))
	})
	mux.HandleFunc("/containers/create", func(w http.ResponseWriter, r *http.Request) {
		var payload map[string]interface{}
		_ = json.NewDecoder(r.Body).Decode(&payload)
		hostCfg := payload["HostConfig"].(map[string]interface{})
		bindsMu.Lock()
		if bindsRaw, ok := hostCfg["Binds"].([]interface{}); ok {
			lastBinds = bindsRaw
		} else {
			lastBinds = nil
		}
		bindsMu.Unlock()
		w.WriteHeader(http.StatusCreated)
		_, _ = w.Write([]byte(`{"Id":"mock-container-binds"}`))
	})
	mux.HandleFunc("/containers/mock-container-binds/start", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	})
	mux.HandleFunc("/containers/mock-container-binds", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodDelete {
			w.WriteHeader(http.StatusNoContent)
		}
	})

	server := &http.Server{Handler: mux}
	go func() { _ = server.Serve(listener) }()
	defer func() { _ = server.Close() }()

	driver := NewDriver(Config{
		SocketPath:    sockPath,
		AllowedImages: []string{"alpine:latest"},
		DataDir:       containerDataDir,
		HostDataDir:   hostDataDir,
	})

	ctx := context.Background()
	userWorkspace := filepath.Join(containerDataDir, "sandboxes", "user_binds")
	require.NoError(t, os.MkdirAll(userWorkspace, 0o755))

	// 1. Whole workspace mount ('.')
	sbxWhole := &sandbox.UserSandbox{
		UserID:      "user_binds",
		DockerImage: "alpine:latest",
		Network:     sandbox.NetworkPolicy{Mode: sandbox.NetworkNone},
		Mounts: []sandbox.UserMount{
			{RelativePath: ".", ReadOnly: false},
		},
		Status: sandbox.StatusRunning,
	}
	err = driver.Create(ctx, sbxWhole, userWorkspace)
	require.NoError(t, err)

	bindsMu.Lock()
	require.Len(t, lastBinds, 1)
	expectedWholeHostBind := fmt.Sprintf("%s:%s:rw", filepath.Join(hostDataDir, "sandboxes", "user_binds"), sandbox.DefaultWorkspaceMountPath)
	assert.Equal(t, expectedWholeHostBind, lastBinds[0])
	bindsMu.Unlock()
	_ = driver.Destroy(ctx, sbxWhole)

	// 2. Subpath mount ('project')
	subDir := filepath.Join(userWorkspace, "project")
	require.NoError(t, os.MkdirAll(subDir, 0o755))

	sbxSub := &sandbox.UserSandbox{
		UserID:      "user_binds",
		DockerImage: "alpine:latest",
		Network:     sandbox.NetworkPolicy{Mode: sandbox.NetworkNone},
		Mounts: []sandbox.UserMount{
			{RelativePath: "project", ReadOnly: true},
		},
		Status: sandbox.StatusRunning,
	}
	err = driver.Create(ctx, sbxSub, userWorkspace)
	require.NoError(t, err)

	bindsMu.Lock()
	require.Len(t, lastBinds, 1)
	expectedSubHostBind := fmt.Sprintf("%s:%s:ro", filepath.Join(hostDataDir, "sandboxes", "user_binds", "project"), filepath.Join(sandbox.DefaultWorkspaceMountPath, "project"))
	assert.Equal(t, expectedSubHostBind, lastBinds[0])
	bindsMu.Unlock()
	_ = driver.Destroy(ctx, sbxSub)

	// 3. Workspace outside dataDir returns error during Create
	outsideWorkspace := filepath.Join(tempDir, "outside", "workspace")
	sbxOutside := &sandbox.UserSandbox{
		UserID:      "user_outside",
		DockerImage: "alpine:latest",
		Network:     sandbox.NetworkPolicy{Mode: sandbox.NetworkNone},
		Mounts: []sandbox.UserMount{
			{RelativePath: ".", ReadOnly: false},
		},
		Status: sandbox.StatusRunning,
	}
	err = driver.Create(ctx, sbxOutside, outsideWorkspace)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "failed to resolve host path for workspace")
}

func TestDockerDriver_RealDocker_NetworkRestrictedAirgap(t *testing.T) {
	ctx := context.Background()
	driver := NewDriver(Config{
		SocketPath:    "/var/run/docker.sock",
		AllowedImages: []string{"alpine:latest"},
		CPULimit:      1.0,
		MemoryLimitMB: 256,
		DataDir:       t.TempDir(),
	})

	if !driver.Available(ctx) {
		t.Skip("Docker daemon is not available on host, skipping test")
	}

	driver.SetCustomBlockedCIDRs([]string{"169.254.0.0/16"})

	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("docker-restricted-airgap-ok"))
	}))
	defer backend.Close()

	backendURL, err := url.Parse(backend.URL)
	require.NoError(t, err)

	tempDir := t.TempDir()
	workspace := filepath.Join(tempDir, "user_restr_real")
	require.NoError(t, os.MkdirAll(workspace, 0o755))

	sbx := &sandbox.UserSandbox{
		UserID:      "testuser_real_restr",
		DockerImage: "alpine:latest",
		Network: sandbox.NetworkPolicy{
			Mode:         sandbox.NetworkRestricted,
			AllowedHosts: []string{backendURL.Hostname()},
			BlockedHosts: []string{"forbidden.com"},
		},
		Status: sandbox.StatusRunning,
	}

	err = driver.Create(ctx, sbx, workspace)
	require.NoError(t, err)
	defer func() { _ = driver.Destroy(ctx, sbx) }()

	// 1. Raw socket attempt to external IP fails with network unreachable error
	res, err := driver.Exec(ctx, sbx, []string{"nc", "-w", "1", "1.1.1.1", "80"}, 5*time.Second)
	require.NoError(t, err)
	assert.NotEqual(t, 0, res.ExitCode, "direct connection to external IP should fail in airgap: %s", res.Stderr)

	// 2. Allowed domain through loopback proxy forwarder succeeds
	res, err = driver.Exec(ctx, sbx, []string{"wget", "-q", "-O", "-", backend.URL + "/data"}, 10*time.Second)
	require.NoError(t, err)
	assert.Equal(t, 0, res.ExitCode, "wget output: stdout=%s stderr=%s", res.Stdout, res.Stderr)
	assert.Contains(t, res.Stdout, "docker-restricted-airgap-ok")

	// 3. Blocked domain is rejected by proxy
	res, err = driver.Exec(ctx, sbx, []string{"wget", "-q", "-O", "-", "http://forbidden.com/data"}, 5*time.Second)
	require.NoError(t, err)
	assert.NotEqual(t, 0, res.ExitCode, "forbidden host should be blocked")

	// 4. In-container no_proxy / NO_PROXY environment variables are set (Finding F7)
	res, err = driver.Exec(ctx, sbx, []string{"sh", "-c", "echo $no_proxy $NO_PROXY"}, 5*time.Second)
	require.NoError(t, err)
	assert.Equal(t, 0, res.ExitCode)
	assert.Contains(t, res.Stdout, "localhost,127.0.0.1 localhost,127.0.0.1")
}

func TestDockerDriver_RealDocker_NetworkRestricted_HTTPSConnect(t *testing.T) {
	ctx := context.Background()
	driver := NewDriver(Config{
		SocketPath:    "/var/run/docker.sock",
		AllowedImages: []string{"alpine:latest"},
		CPULimit:      1.0,
		MemoryLimitMB: 256,
		DataDir:       t.TempDir(),
	})

	if !driver.Available(ctx) {
		t.Skip("Docker daemon is not available on host, skipping test")
	}

	driver.SetCustomBlockedCIDRs([]string{"169.254.0.0/16"})

	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		if _, err := w.Write([]byte("https-connect-ok")); err != nil {
			t.Logf("Failed to write response: %v", err)
		}
	}))
	defer backend.Close()

	backendURL, err := url.Parse(backend.URL)
	require.NoError(t, err)

	workspace := filepath.Join(t.TempDir(), "user_https_real")
	require.NoError(t, os.MkdirAll(workspace, 0o755))

	sbx := &sandbox.UserSandbox{
		UserID:      "testuser_real_https",
		DockerImage: "alpine:latest",
		Network: sandbox.NetworkPolicy{
			Mode:         sandbox.NetworkRestricted,
			AllowedHosts: []string{backendURL.Hostname()},
		},
		Status: sandbox.StatusRunning,
	}

	err = driver.Create(ctx, sbx, workspace)
	require.NoError(t, err)
	defer func() {
		if destroyErr := driver.Destroy(ctx, sbx); destroyErr != nil {
			t.Logf("Failed to destroy sandbox: %v", destroyErr)
		}
	}()

	connectScript := fmt.Sprintf(
		"(printf 'CONNECT %s:%s HTTP/1.1\\r\\nHost: %s:%s\\r\\n\\r\\nGET /secure-data HTTP/1.1\\r\\nHost: %s\\r\\nConnection: close\\r\\n\\r\\n'; sleep 1) | nc -w 5 127.0.0.1 18080",
		backendURL.Hostname(), backendURL.Port(),
		backendURL.Hostname(), backendURL.Port(),
		backendURL.Hostname(),
	)

	res, err := driver.Exec(ctx, sbx, []string{"sh", "-c", connectScript}, 10*time.Second)
	require.NoError(t, err)
	assert.Equal(t, 0, res.ExitCode, "exec output: stdout=%s stderr=%s", res.Stdout, res.Stderr)
	assert.Contains(t, res.Stdout, "200 Connection Established")
	assert.Contains(t, res.Stdout, "https-connect-ok")
}

func TestDockerDriver_RealDocker_NetworkRestricted_DomainResolution(t *testing.T) {
	ctx := context.Background()

	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		if _, err := w.Write([]byte("mock-domain-airgap-ok")); err != nil {
			t.Logf("Failed to write mock response: %v", err)
		}
	}))
	defer backend.Close()

	backendURL, err := url.Parse(backend.URL)
	require.NoError(t, err)
	backendPort := backendURL.Port()

	dnsPC, err := net.ListenPacket("udp", "127.0.0.1:0")
	require.NoError(t, err)
	defer func() {
		if closeErr := dnsPC.Close(); closeErr != nil {
			t.Logf("Failed to close dnsPC: %v", closeErr)
		}
	}()

	go func() {
		buf := make([]byte, 512)
		for {
			n, clientAddr, err := dnsPC.ReadFrom(buf)
			if err != nil {
				return
			}
			if n < 12 {
				continue
			}
			qnameEnd := 12
			for qnameEnd < n && buf[qnameEnd] != 0 {
				qnameEnd += int(buf[qnameEnd]) + 1
			}
			qnameEnd++
			if qnameEnd+4 > n {
				continue
			}
			qtype := binary.BigEndian.Uint16(buf[qnameEnd : qnameEnd+2])

			qLen := qnameEnd + 4
			if qtype == 1 { // A record
				resp := make([]byte, qLen+16)
				copy(resp[:qLen], buf[:qLen])
				resp[2] = 0x81 // QR=1, RD=1
				resp[3] = 0x80 // RA=1, RCODE=0
				resp[4] = 0x00
				resp[5] = 0x01 // QDCOUNT = 1
				resp[6] = 0x00
				resp[7] = 0x01 // ANCOUNT = 1
				resp[8] = 0x00
				resp[9] = 0x00 // NSCOUNT = 0
				resp[10] = 0x00
				resp[11] = 0x00 // ARCOUNT = 0
				resp[qLen] = 0xc0
				resp[qLen+1] = 0x0c
				resp[qLen+2] = 0x00
				resp[qLen+3] = 0x01 // Type A
				resp[qLen+4] = 0x00
				resp[qLen+5] = 0x01 // Class IN
				resp[qLen+6] = 0x00
				resp[qLen+7] = 0x00
				resp[qLen+8] = 0x00
				resp[qLen+9] = 0x3c // TTL 60
				resp[qLen+10] = 0x00
				resp[qLen+11] = 0x04
				copy(resp[qLen+12:qLen+16], net.IPv4(127, 0, 0, 1).To4())
				if _, writeErr := dnsPC.WriteTo(resp[:qLen+16], clientAddr); writeErr != nil {
					return
				}
			} else { // Other types (AAAA): empty NOERROR
				resp := make([]byte, qLen)
				copy(resp[:qLen], buf[:qLen])
				resp[2] = 0x81
				resp[3] = 0x80
				resp[4] = 0x00
				resp[5] = 0x01 // QDCOUNT = 1
				resp[6] = 0x00
				resp[7] = 0x00 // ANCOUNT = 0
				resp[8] = 0x00
				resp[9] = 0x00
				resp[10] = 0x00
				resp[11] = 0x00
				if _, writeErr := dnsPC.WriteTo(resp[:qLen], clientAddr); writeErr != nil {
					return
				}
			}
		}
	}()

	mockResolver := &net.Resolver{
		PreferGo: true,
		Dial: func(ctx context.Context, network, address string) (net.Conn, error) {
			var d net.Dialer
			return d.DialContext(ctx, "udp", dnsPC.LocalAddr().String())
		},
	}

	driver := NewDriver(Config{
		SocketPath:    "/var/run/docker.sock",
		AllowedImages: []string{"alpine:latest"},
		CPULimit:      1.0,
		MemoryLimitMB: 256,
		DataDir:       t.TempDir(),
		Resolver:      mockResolver,
	})

	if !driver.Available(ctx) {
		t.Skip("Docker daemon is not available on host, skipping test")
	}

	driver.SetCustomBlockedCIDRs([]string{"169.254.0.0/16"})

	workspace := filepath.Join(t.TempDir(), "user_domain_real")
	require.NoError(t, os.MkdirAll(workspace, 0o755))

	const domain = "mock.sandbox.local"
	sbx := &sandbox.UserSandbox{
		UserID:      "testuser_real_domain",
		DockerImage: "alpine:latest",
		Network: sandbox.NetworkPolicy{
			Mode:         sandbox.NetworkRestricted,
			AllowedHosts: []string{domain},
		},
		Status: sandbox.StatusRunning,
	}

	err = driver.Create(ctx, sbx, workspace)
	require.NoError(t, err)
	defer func() {
		if destroyErr := driver.Destroy(ctx, sbx); destroyErr != nil {
			t.Logf("Failed to destroy sandbox: %v", destroyErr)
		}
	}()

	targetURL := fmt.Sprintf("http://%s:%s/data", domain, backendPort)
	res, err := driver.Exec(ctx, sbx, []string{"wget", "-q", "-O", "-", targetURL}, 10*time.Second)
	require.NoError(t, err)
	assert.Equal(t, 0, res.ExitCode, "wget output: stdout=%s stderr=%s", res.Stdout, res.Stderr)
	assert.Contains(t, res.Stdout, "mock-domain-airgap-ok")
}

func TestDockerDriver_RealDocker_NetworkRestricted_ConcurrentExec(t *testing.T) {
	ctx := context.Background()
	driver := NewDriver(Config{
		SocketPath:    "/var/run/docker.sock",
		AllowedImages: []string{"alpine:latest"},
		CPULimit:      1.0,
		MemoryLimitMB: 256,
		DataDir:       t.TempDir(),
	})

	if !driver.Available(ctx) {
		t.Skip("Docker daemon is not available on host, skipping test")
	}

	driver.SetCustomBlockedCIDRs([]string{"169.254.0.0/16"})

	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		if _, err := w.Write([]byte("concurrent-exec-ok")); err != nil {
			t.Logf("Failed to write response: %v", err)
		}
	}))
	defer backend.Close()

	backendURL, err := url.Parse(backend.URL)
	require.NoError(t, err)

	workspace := filepath.Join(t.TempDir(), "user_concurrent_real")
	require.NoError(t, os.MkdirAll(workspace, 0o755))

	sbx := &sandbox.UserSandbox{
		UserID:      "testuser_real_concurrent",
		DockerImage: "alpine:latest",
		Network: sandbox.NetworkPolicy{
			Mode:         sandbox.NetworkRestricted,
			AllowedHosts: []string{backendURL.Hostname()},
		},
		Status: sandbox.StatusRunning,
	}

	err = driver.Create(ctx, sbx, workspace)
	require.NoError(t, err)
	defer func() {
		if destroyErr := driver.Destroy(ctx, sbx); destroyErr != nil {
			t.Logf("Failed to destroy sandbox: %v", destroyErr)
		}
	}()

	const concurrency = 10
	errCh := make(chan error, concurrency)
	var wg sync.WaitGroup

	for i := 0; i < concurrency; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			res, err := driver.Exec(ctx, sbx, []string{"wget", "-q", "-O", "-", backend.URL + "/data"}, 10*time.Second)
			if err != nil {
				errCh <- err
				return
			}
			if res.ExitCode != 0 {
				errCh <- fmt.Errorf("unexpected exit code %d: %s", res.ExitCode, res.Stderr)
				return
			}
			if !strings.Contains(res.Stdout, "concurrent-exec-ok") {
				errCh <- fmt.Errorf("unexpected stdout: %s", res.Stdout)
				return
			}
			errCh <- nil
		}()
	}

	wg.Wait()
	close(errCh)

	for err := range errCh {
		assert.NoError(t, err)
	}
}


func TestDockerDriver_Destroy_ProxySocketCleanup(t *testing.T) {
	tempDir := t.TempDir()
	sockPath := filepath.Join(tempDir, "docker.sock")
	listener, err := net.Listen("unix", sockPath)
	require.NoError(t, err)

	mux := http.NewServeMux()
	mux.HandleFunc("/_ping", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("OK"))
	})
	mux.HandleFunc("/containers/create", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusCreated)
		_, _ = w.Write([]byte(`{"Id":"mock-cleanup-container"}`))
	})
	mux.HandleFunc("/containers/mock-cleanup-container/start", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	})
	mux.HandleFunc("/containers/mock-cleanup-container", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodDelete {
			w.WriteHeader(http.StatusNoContent)
		}
	})

	server := &http.Server{Handler: mux}
	go func() { _ = server.Serve(listener) }()
	defer func() { _ = server.Close() }()

	dataDir := filepath.Join(tempDir, "data")
	driver := NewDriver(Config{
		SocketPath:    sockPath,
		AllowedImages: []string{"alpine:latest"},
		DataDir:       dataDir,
	})

	ctx := context.Background()
	userWorkspace := filepath.Join(tempDir, "user_workspace")
	require.NoError(t, os.MkdirAll(userWorkspace, 0o755))

	sbx := &sandbox.UserSandbox{
		UserID:      "user_cleanup_test",
		DockerImage: "alpine:latest",
		Network: sandbox.NetworkPolicy{
			Mode:         sandbox.NetworkRestricted,
			AllowedHosts: []string{"api.github.com"},
		},
		Status: sandbox.StatusRunning,
	}

	err = driver.Create(ctx, sbx, userWorkspace)
	require.NoError(t, err)

	expectedProxyDir := driver.getProxyDir(sbx.UserID)
	expectedSocket := filepath.Join(expectedProxyDir, "proxy.sock")

	_, err = os.Stat(expectedSocket)
	require.NoError(t, err, "proxy socket should exist after Create()")

	err = driver.Destroy(ctx, sbx)
	require.NoError(t, err)

	_, err = os.Stat(expectedSocket)
	assert.True(t, os.IsNotExist(err), "proxy.sock should be removed after Destroy()")

	_, err = os.Stat(expectedProxyDir)
	assert.True(t, os.IsNotExist(err), "proxy directory should be removed after Destroy()")
}

func TestDockerDriver_Create_ExistingProxyDirectory(t *testing.T) {
	tempDir := t.TempDir()
	sockPath := filepath.Join(tempDir, "docker.sock")
	listener, err := net.Listen("unix", sockPath)
	require.NoError(t, err)

	mux := http.NewServeMux()
	mux.HandleFunc("/_ping", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("OK"))
	})
	mux.HandleFunc("/containers/create", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusCreated)
		_, _ = w.Write([]byte(`{"Id":"mock-exist-container"}`))
	})
	mux.HandleFunc("/containers/mock-exist-container/start", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	})
	mux.HandleFunc("/containers/mock-exist-container", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodDelete {
			w.WriteHeader(http.StatusNoContent)
		}
	})

	server := &http.Server{Handler: mux}
	go func() { _ = server.Serve(listener) }()
	defer func() { _ = server.Close() }()

	dataDir := filepath.Join(tempDir, "data")
	driver := NewDriver(Config{
		SocketPath:    sockPath,
		AllowedImages: []string{"alpine:latest"},
		DataDir:       dataDir,
	})

	ctx := context.Background()
	userWorkspace := filepath.Join(tempDir, "user_workspace")
	require.NoError(t, os.MkdirAll(userWorkspace, 0o755))

	sbx := &sandbox.UserSandbox{
		UserID:      "user_stale_proxy_test",
		DockerImage: "alpine:latest",
		Network: sandbox.NetworkPolicy{
			Mode:         sandbox.NetworkRestricted,
			AllowedHosts: []string{"api.github.com"},
		},
		Status: sandbox.StatusRunning,
	}

	// Pre-create dirty/stale proxy socket simulating a dirty crash
	staleProxyDir := driver.getProxyDir(sbx.UserID)
	require.NoError(t, os.MkdirAll(staleProxyDir, 0o755))
	staleSock := filepath.Join(staleProxyDir, "proxy.sock")
	require.NoError(t, os.WriteFile(staleSock, []byte("stale dead socket"), 0o660))

	err = driver.Create(ctx, sbx, userWorkspace)
	require.NoError(t, err, "Create() should succeed and overwrite stale proxy socket cleanly")
	defer func() { _ = driver.Destroy(ctx, sbx) }()

	fi, err := os.Stat(staleSock)
	require.NoError(t, err)
	assert.Equal(t, os.ModeSocket, fi.Mode()&os.ModeSocket)
}

func TestDockerDriver_Config_CustomBlockedCIDRs(t *testing.T) {
	customCIDRs := []string{"10.0.0.0/8", "192.168.1.0/24"}
	driver := NewDriver(Config{
		CustomBlockedCIDRs: customCIDRs,
	})
	assert.Equal(t, customCIDRs, driver.customBlockedCIDRs)
}

func TestDockerDriver_Config_ForwarderPort(t *testing.T) {
	driver := NewDriver(Config{
		ForwarderPort: 19090,
	})
	assert.Equal(t, 19090, driver.forwarderPort)

	driverDefault := NewDriver(Config{})
	assert.Equal(t, sandbox.DefaultForwarderPort, driverDefault.forwarderPort)
}

func TestDockerDriver_Exec_NoProxyWhenProxyNil(t *testing.T) {
	tempDir := t.TempDir()
	sockPath := filepath.Join(tempDir, "mock_docker.sock")
	listener, err := net.Listen("unix", sockPath)
	require.NoError(t, err)

	var capturedExecEnv []string
	mux := http.NewServeMux()
	mux.HandleFunc("/_ping", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("OK"))
	})
	mux.HandleFunc("/containers/test-container-id/exec", func(w http.ResponseWriter, r *http.Request) {
		var payload map[string]interface{}
		_ = json.NewDecoder(r.Body).Decode(&payload)
		if envList, ok := payload["Env"].([]interface{}); ok {
			for _, e := range envList {
				if s, ok := e.(string); ok {
					capturedExecEnv = append(capturedExecEnv, s)
				}
			}
		}
		w.WriteHeader(http.StatusCreated)
		_ = json.NewEncoder(w).Encode(map[string]string{"Id": "exec-id-123"})
	})
	mux.HandleFunc("/exec/exec-id-123/start", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	mux.HandleFunc("/exec/exec-id-123/json", func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]interface{}{"Running": false, "ExitCode": 0})
	})

	server := &http.Server{Handler: mux}
	go func() { _ = server.Serve(listener) }()
	defer func() { _ = server.Close() }()

	driver := NewDriver(Config{
		SocketPath: sockPath,
	})

	sbx := &sandbox.UserSandbox{
		UserID: "user_no_proxy",
		Network: sandbox.NetworkPolicy{
			Mode: sandbox.NetworkRestricted,
		},
		Status: sandbox.StatusRunning,
	}
	sbx.SetInternalID("test-container-id")

	_, err = driver.Exec(context.Background(), sbx, []string{"echo", "hi"}, 5*time.Second)
	require.NoError(t, err)

	for _, envVar := range capturedExecEnv {
		assert.False(t, strings.HasPrefix(envVar, "http_proxy="), "http_proxy should not be set when proxy is nil")
		assert.False(t, strings.HasPrefix(envVar, "HTTP_PROXY="), "HTTP_PROXY should not be set when proxy is nil")
	}
}

func TestDockerDriver_GetProxyDir_LongUserID_HostDataDir(t *testing.T) {
	tempDir := t.TempDir()
	dataDir := filepath.Join(tempDir, "container_data")
	hostDataDir := filepath.Join(tempDir, "host_data")
	require.NoError(t, os.MkdirAll(dataDir, 0o755))
	require.NoError(t, os.MkdirAll(hostDataDir, 0o755))

	driver := NewDriver(Config{
		DataDir:     dataDir,
		HostDataDir: hostDataDir,
	})

	longSpecialUserID := "org/team/subgroup/user-with-very-long-id-" + strings.Repeat("x", 120)
	proxyDir := driver.getProxyDir(longSpecialUserID)
	require.NotEmpty(t, proxyDir)

	// Ensure proxyDir is within dataDir so toHostPath succeeds
	hostPath, err := driver.toHostPath(proxyDir)
	require.NoError(t, err, "toHostPath must succeed for long userID with HostDataDir configured")
	assert.True(t, strings.HasPrefix(hostPath, hostDataDir), "mapped host path must be inside hostDataDir")

	// Verify socket path length
	sockPath := filepath.Join(proxyDir, "proxy.sock")
	assert.True(t, len(sockPath) < 108, "proxy.sock path must be < 108 characters")
}

func TestDockerDriver_RealDocker_ConcurrentExec_OneTimeoutOneSucceeds(t *testing.T) {
	ctx := context.Background()
	driver := NewDriver(Config{
		SocketPath:    "/var/run/docker.sock",
		AllowedImages: []string{"alpine:latest"},
		CPULimit:      1.0,
		MemoryLimitMB: 256,
		DataDir:       t.TempDir(),
	})

	if !driver.Available(ctx) {
		t.Skip("Docker daemon is not available on host, skipping test")
	}

	tempDir := t.TempDir()
	workspace := filepath.Join(tempDir, "user_concurrent_timeout")
	require.NoError(t, os.MkdirAll(workspace, 0o755))

	sbx := &sandbox.UserSandbox{
		UserID:      "user_conc_timeout",
		DockerImage: "alpine:latest",
		Network: sandbox.NetworkPolicy{
			Mode: sandbox.NetworkNone,
		},
		Status: sandbox.StatusRunning,
	}

	err := driver.Create(ctx, sbx, workspace)
	require.NoError(t, err)
	defer func() { _ = driver.Destroy(ctx, sbx) }()

	var wg sync.WaitGroup
	var resTimeout, resSuccess *sandbox.ExecResult
	var errTimeout, errSuccess error

	wg.Add(2)

	// Exec 1: sleep 10 with 500ms timeout -> should time out and be killed
	go func() {
		defer wg.Done()
		resTimeout, errTimeout = driver.Exec(ctx, sbx, []string{"sleep", "10"}, 500*time.Millisecond)
	}()

	// Exec 2: sleep 1 with 5s timeout -> should succeed concurrently without being killed by Exec 1
	go func() {
		defer wg.Done()
		time.Sleep(100 * time.Millisecond) // ensure Exec 1 starts first
		resSuccess, errSuccess = driver.Exec(ctx, sbx, []string{"sleep", "1"}, 5*time.Second)
	}()

	wg.Wait()

	require.NoError(t, errTimeout)
	require.NotNil(t, resTimeout)
	assert.Equal(t, -1, resTimeout.ExitCode)
	assert.Contains(t, resTimeout.Stderr, "command timed out")

	require.NoError(t, errSuccess)
	require.NotNil(t, resSuccess)
	assert.Equal(t, 0, resSuccess.ExitCode, "concurrent exec must succeed and not be killed by timeout of another exec")
}

