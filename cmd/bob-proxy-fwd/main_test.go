package main

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"syscall"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func startMockUnixEchoServer(t *testing.T, sockPath string) (net.Listener, *sync.WaitGroup) {
	ln, err := net.Listen("unix", sockPath)
	require.NoError(t, err)

	var wg sync.WaitGroup
	go func() {
		for {
			conn, err := ln.Accept()
			if err != nil {
				return
			}
			wg.Add(1)
			go func(c net.Conn) {
				defer wg.Done()
				defer func() { _ = c.Close() }()
				_, _ = io.Copy(c, c)
			}(conn)
		}
	}()

	return ln, &wg
}

func getFreeTCPPort(t *testing.T) int {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	defer func() { _ = ln.Close() }()
	return ln.Addr().(*net.TCPAddr).Port
}

func TestServeForwarder_BidirectionalEcho(t *testing.T) {
	tempDir := t.TempDir()
	sockPath := filepath.Join(tempDir, "echo.sock")

	unixLn, wg := startMockUnixEchoServer(t, sockPath)
	defer func() {
		_ = unixLn.Close()
		wg.Wait()
	}()

	tcpLn, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	go func() {
		_ = serveForwarder(ctx, tcpLn, sockPath)
	}()

	tcpAddr := tcpLn.Addr().String()

	// Connect to forwarder
	conn, err := net.Dial("tcp", tcpAddr)
	require.NoError(t, err)
	defer func() { _ = conn.Close() }()

	msg := []byte("hello-bob-forwarder")
	_, err = conn.Write(msg)
	require.NoError(t, err)

	buf := make([]byte, len(msg))
	_, err = io.ReadFull(conn, buf)
	require.NoError(t, err)
	assert.Equal(t, msg, buf)
}

func TestServeForwarder_ConcurrentTraffic(t *testing.T) {
	tempDir := t.TempDir()
	sockPath := filepath.Join(tempDir, "echo_concurrent.sock")

	unixLn, wg := startMockUnixEchoServer(t, sockPath)
	defer func() {
		_ = unixLn.Close()
		wg.Wait()
	}()

	tcpLn, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	go func() {
		_ = serveForwarder(ctx, tcpLn, sockPath)
	}()

	tcpAddr := tcpLn.Addr().String()
	numClients := 10
	var clientWg sync.WaitGroup
	clientWg.Add(numClients)

	for i := 0; i < numClients; i++ {
		go func(id int) {
			defer clientWg.Done()
			conn, err := net.Dial("tcp", tcpAddr)
			if !assert.NoError(t, err) {
				return
			}
			defer func() { _ = conn.Close() }()

			payload := []byte(fmt.Sprintf("client-%d-message-data", id))
			_, err = conn.Write(payload)
			assert.NoError(t, err)

			buf := make([]byte, len(payload))
			_, err = io.ReadFull(conn, buf)
			assert.NoError(t, err)
			assert.Equal(t, payload, buf)
		}(i)
	}

	clientWg.Wait()
}

func TestRun_DaemonMode(t *testing.T) {
	tempDir := t.TempDir()
	sockPath := filepath.Join(tempDir, "daemon_echo.sock")

	unixLn, wg := startMockUnixEchoServer(t, sockPath)
	defer func() {
		_ = unixLn.Close()
		wg.Wait()
	}()

	port := getFreeTCPPort(t)
	tcpAddr := fmt.Sprintf("127.0.0.1:%d", port)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	doneCh := make(chan int, 1)
	var stdout, stderr bytes.Buffer

	go func() {
		code := run(ctx, []string{
			"-daemon",
			"-tcp", tcpAddr,
			"-sock", sockPath,
		}, os.Stdin, &stdout, &stderr)
		doneCh <- code
	}()

	// Wait for listener to be up
	var conn net.Conn
	for i := 0; i < 50; i++ {
		var dialErr error
		conn, dialErr = net.DialTimeout("tcp", tcpAddr, 100*time.Millisecond)
		if dialErr == nil {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	require.NotNil(t, conn, "failed to connect to daemon forwarder")

	msg := []byte("daemon-test-string")
	_, err := conn.Write(msg)
	require.NoError(t, err)

	buf := make([]byte, len(msg))
	_, err = io.ReadFull(conn, buf)
	require.NoError(t, err)
	assert.Equal(t, msg, buf)
	_ = conn.Close()

	cancel()
	select {
	case code := <-doneCh:
		assert.Equal(t, 0, code)
	case <-time.After(3 * time.Second):
		t.Fatal("daemon did not exit cleanly on context cancellation")
	}
}

func TestRun_WrapperMode(t *testing.T) {
	tempDir := t.TempDir()
	sockPath := filepath.Join(tempDir, "wrap_echo.sock")

	unixLn, wg := startMockUnixEchoServer(t, sockPath)
	defer func() {
		_ = unixLn.Close()
		wg.Wait()
	}()

	port := getFreeTCPPort(t)
	tcpAddr := fmt.Sprintf("127.0.0.1:%d", port)

	ctx := context.Background()
	var stdout, stderr bytes.Buffer

	// Success exit code
	code := run(ctx, []string{
		"-tcp", tcpAddr,
		"-sock", sockPath,
		"--",
		"echo", "wrapper-success-output",
	}, os.Stdin, &stdout, &stderr)

	assert.Equal(t, 0, code)
	assert.Contains(t, stdout.String(), "wrapper-success-output")

	// Custom non-zero exit code
	stdout.Reset()
	stderr.Reset()
	code = run(ctx, []string{
		"-tcp", tcpAddr,
		"-sock", sockPath,
		"--",
		"sh", "-c", "exit 42",
	}, os.Stdin, &stdout, &stderr)

	assert.Equal(t, 42, code)
}

func TestStartReaper(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	startReaper(ctx)

	cmd := exec.Command("true")
	err := cmd.Start()
	require.NoError(t, err)

	pid := cmd.Process.Pid
	require.Eventually(t, func() bool {
		return syscall.Kill(pid, 0) != nil
	}, 2*time.Second, 10*time.Millisecond)
}

type mockTemporaryListener struct {
	net.Listener
	mu         sync.Mutex
	errorCount int
	maxErrors  int
	closed     bool
}

func (m *mockTemporaryListener) Accept() (net.Conn, error) {
	m.mu.Lock()
	if m.closed {
		m.mu.Unlock()
		return nil, net.ErrClosed
	}
	if m.errorCount < m.maxErrors {
		m.errorCount++
		m.mu.Unlock()
		return nil, syscall.EMFILE
	}
	m.mu.Unlock()
	return m.Listener.Accept()
}

func (m *mockTemporaryListener) Close() error {
	m.mu.Lock()
	m.closed = true
	m.mu.Unlock()
	return m.Listener.Close()
}

func TestServeForwarder_TemporaryAcceptError(t *testing.T) {
	tempDir := t.TempDir()
	sockPath := filepath.Join(tempDir, "echo.sock")

	unixLn, wg := startMockUnixEchoServer(t, sockPath)
	defer func() {
		_ = unixLn.Close()
		wg.Wait()
	}()

	realTCP, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)

	mockLn := &mockTemporaryListener{
		Listener:  realTCP,
		maxErrors: 2,
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	fwdErrCh := make(chan error, 1)
	go func() {
		fwdErrCh <- serveForwarder(ctx, mockLn, sockPath)
	}()

	// Connect client - should succeed after temporary error retries
	client, err := net.Dial("tcp", realTCP.Addr().String())
	require.NoError(t, err)
	defer func() { _ = client.Close() }()

	_, err = client.Write([]byte("ping"))
	require.NoError(t, err)

	buf := make([]byte, 4)
	_, err = io.ReadFull(client, buf)
	require.NoError(t, err)
	assert.Equal(t, "ping", string(buf))

	cancel()
	_ = mockLn.Close()
	err = <-fwdErrCh
	assert.NoError(t, err)
}

type mockFatalListener struct {
	net.Listener
}

func (m *mockFatalListener) Accept() (net.Conn, error) {
	return nil, fmt.Errorf("fatal non-temporary hardware error")
}

func TestServeForwarder_FatalAcceptError(t *testing.T) {
	tempDir := t.TempDir()
	sockPath := filepath.Join(tempDir, "echo.sock")

	realTCP, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	defer func() { _ = realTCP.Close() }()

	mockLn := &mockFatalListener{Listener: realTCP}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	err = serveForwarder(ctx, mockLn, sockPath)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "accept error")
}

func TestServeForwarder_UnixSocketNotFound(t *testing.T) {
	tempDir := t.TempDir()
	nonexistentSock := filepath.Join(tempDir, "nonexistent.sock")

	tcpLn, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	fwdDone := make(chan error, 1)
	go func() {
		fwdDone <- serveForwarder(ctx, tcpLn, nonexistentSock)
	}()

	client, err := net.Dial("tcp", tcpLn.Addr().String())
	require.NoError(t, err)
	defer func() { _ = client.Close() }()

	// Client read should immediately return EOF because unix socket dial failed
	buf := make([]byte, 16)
	n, err := client.Read(buf)
	assert.Equal(t, 0, n)
	assert.ErrorIs(t, err, io.EOF)

	cancel()
	err = <-fwdDone
	assert.NoError(t, err)
}

func TestBridge_UpstreamAbruptClose(t *testing.T) {
	tempDir := t.TempDir()
	sockPath := filepath.Join(tempDir, "abrupt.sock")

	unixLn, err := net.Listen("unix", sockPath)
	require.NoError(t, err)
	defer func() { _ = unixLn.Close() }()

	serverAccepted := make(chan net.Conn, 1)
	go func() {
		conn, err := unixLn.Accept()
		if err == nil {
			serverAccepted <- conn
		}
	}()

	tcpLn, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	fwdDone := make(chan error, 1)
	go func() {
		fwdDone <- serveForwarder(ctx, tcpLn, sockPath)
	}()

	client, err := net.Dial("tcp", tcpLn.Addr().String())
	require.NoError(t, err)
	defer func() { _ = client.Close() }()

	serverConn := <-serverAccepted
	_, err = client.Write([]byte("hello"))
	require.NoError(t, err)

	buf := make([]byte, 5)
	_, err = io.ReadFull(serverConn, buf)
	require.NoError(t, err)

	// Server abruptly closes connection
	_ = serverConn.Close()

	// Client reading from forwarder should receive EOF
	n, err := client.Read(buf)
	assert.Equal(t, 0, n)
	assert.ErrorIs(t, err, io.EOF)

	cancel()
	err = <-fwdDone
	assert.NoError(t, err)
}

func TestBridge_HalfClosedConnection(t *testing.T) {
	tempDir := t.TempDir()
	sockPath := filepath.Join(tempDir, "halfclosed.sock")

	unixLn, err := net.Listen("unix", sockPath)
	require.NoError(t, err)
	defer func() { _ = unixLn.Close() }()

	go func() {
		conn, err := unixLn.Accept()
		if err != nil {
			return
		}
		defer func() { _ = conn.Close() }()

		req, err := io.ReadAll(conn)
		if err != nil {
			return
		}
		if string(req) == "PING_REQUEST" {
			_, _ = conn.Write([]byte("PONG_RESPONSE"))
		}
	}()

	tcpLn, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	fwdDone := make(chan error, 1)
	go func() {
		fwdDone <- serveForwarder(ctx, tcpLn, sockPath)
	}()

	clientConn, err := net.Dial("tcp", tcpLn.Addr().String())
	require.NoError(t, err)
	defer func() { _ = clientConn.Close() }()

	tcpClient, ok := clientConn.(*net.TCPConn)
	require.True(t, ok)

	_, err = tcpClient.Write([]byte("PING_REQUEST"))
	require.NoError(t, err)
	err = tcpClient.CloseWrite()
	require.NoError(t, err)

	resp, err := io.ReadAll(tcpClient)
	require.NoError(t, err)
	assert.Equal(t, "PONG_RESPONSE", string(resp))

	cancel()
	err = <-fwdDone
	assert.NoError(t, err)
}

func TestServeForwarder_DialTimeout(t *testing.T) {
	origTimeout := defaultDialTimeout
	defaultDialTimeout = 50 * time.Millisecond
	defer func() { defaultDialTimeout = origTimeout }()

	tempDir := t.TempDir()
	sockPath := filepath.Join(tempDir, "stall.sock")

	// Create raw unix socket with backlog 1
	fd, err := syscall.Socket(syscall.AF_UNIX, syscall.SOCK_STREAM, 0)
	require.NoError(t, err)
	defer func() { _ = syscall.Close(fd) }()

	sa := &syscall.SockaddrUnix{Name: sockPath}
	err = syscall.Bind(fd, sa)
	require.NoError(t, err)
	err = syscall.Listen(fd, 1)
	require.NoError(t, err)

	// Fill backlog with non-blocking connects so next connect blocks
	for i := 0; i < 5; i++ {
		connFd, err := syscall.Socket(syscall.AF_UNIX, syscall.SOCK_STREAM, 0)
		if err == nil {
			defer func(f int) { _ = syscall.Close(f) }(connFd)
			_ = syscall.SetNonblock(connFd, true)
			_ = syscall.Connect(connFd, sa)
		}
	}

	tcpLn, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	fwdDone := make(chan error, 1)
	go func() {
		fwdDone <- serveForwarder(ctx, tcpLn, sockPath)
	}()

	client, err := net.Dial("tcp", tcpLn.Addr().String())
	require.NoError(t, err)
	defer func() { _ = client.Close() }()

	// Client should receive EOF when dialer times out and closes client connection
	buf := make([]byte, 16)
	_ = client.SetReadDeadline(time.Now().Add(1 * time.Second))
	n, err := client.Read(buf)
	assert.Equal(t, 0, n)
	assert.ErrorIs(t, err, io.EOF)

	cancel()
	err = <-fwdDone
	assert.NoError(t, err)
}

func TestBridge_IdleTimeout(t *testing.T) {
	c1, c2 := net.Pipe()
	defer func() { _ = c1.Close(); _ = c2.Close() }()

	done := make(chan struct{})
	go func() {
		bridge(c1, c2, 50*time.Millisecond)
		close(done)
	}()

	select {
	case <-done:
		// Succeeded: bridge returned after idle timeout
	case <-time.After(2 * time.Second):
		t.Fatal("bridge did not time out on idle connection")
	}
}




