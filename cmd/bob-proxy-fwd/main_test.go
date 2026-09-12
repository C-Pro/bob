package main

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net"
	"os"
	"path/filepath"
	"sync"
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
