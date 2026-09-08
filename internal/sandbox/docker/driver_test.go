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

	driver := NewDriver(Config{
		SocketPath:    sockPath,
		AllowedImages: []string{"alpine:latest"},
		CPULimit:      1.0,
		MemoryLimitMB: 512,
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
	assert.Equal(t, "bridge", capturedCreateHostConfig["NetworkMode"])
	extraHosts, ok := capturedCreateHostConfig["ExtraHosts"].([]interface{})
	require.True(t, ok)
	assert.Contains(t, extraHosts, "host.docker.internal:host-gateway")

	// Execute command
	res, err := driver.Exec(ctx, sbx, []string{"echo", "test"}, 5*time.Second)
	require.NoError(t, err)
	assert.Equal(t, 0, res.ExitCode)

	// Verify working directory and environment variables in exec instance
	assert.Equal(t, sandbox.DefaultWorkspaceMountPath, capturedExecWorkingDir)
	assert.Contains(t, capturedExecEnv, "HOME="+sandbox.DefaultWorkspaceMountPath)
	assert.Contains(t, capturedExecEnv, "PWD="+sandbox.DefaultWorkspaceMountPath)

	var hasHttpProxy, hasHostDockerInternal bool
	for _, envVar := range capturedExecEnv {
		if strings.HasPrefix(envVar, "HTTP_PROXY=") {
			hasHttpProxy = true
			if strings.Contains(envVar, "host.docker.internal:") {
				hasHostDockerInternal = true
			}
		}
	}
	assert.True(t, hasHttpProxy, "expected HTTP_PROXY in exec env")
	assert.True(t, hasHostDockerInternal, "expected HTTP_PROXY to point to host.docker.internal")
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
	assert.True(t, didRestart, "container must be restarted atomically on exec timeout")

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
