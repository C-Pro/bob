package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"net"
	"os"
	"os/exec"
	"os/signal"
	"sync"
	"syscall"
	"time"
)

const (
	maxConcurrentConns = 128
)

var (
	defaultDialTimeout = 10 * time.Second
	defaultIdleTimeout = 2 * time.Minute
)

type closeWriter interface {
	CloseWrite() error
}

type idleTimeoutConn struct {
	net.Conn
	peer        net.Conn
	idleTimeout time.Duration
}

func (c *idleTimeoutConn) Read(b []byte) (int, error) {
	if c.idleTimeout > 0 {
		deadline := time.Now().Add(c.idleTimeout)
		_ = c.SetReadDeadline(deadline)
		if c.peer != nil {
			_ = c.peer.SetReadDeadline(deadline)
		}
	}
	return c.Conn.Read(b)
}

func (c *idleTimeoutConn) Write(b []byte) (int, error) {
	if c.idleTimeout > 0 {
		deadline := time.Now().Add(c.idleTimeout)
		_ = c.SetWriteDeadline(deadline)
		if c.peer != nil {
			_ = c.peer.SetReadDeadline(deadline)
		}
	}
	return c.Conn.Write(b)
}

func (c *idleTimeoutConn) CloseWrite() error {
	if cw, ok := c.Conn.(closeWriter); ok {
		return cw.CloseWrite()
	}
	return nil
}

func bridge(client net.Conn, upstream net.Conn, idleTimeout time.Duration) {
	if idleTimeout > 0 {
		initDeadline := time.Now().Add(idleTimeout)
		_ = client.SetDeadline(initDeadline)
		_ = upstream.SetDeadline(initDeadline)
	}

	wrappedClient := &idleTimeoutConn{Conn: client, peer: upstream, idleTimeout: idleTimeout}
	wrappedUpstream := &idleTimeoutConn{Conn: upstream, peer: client, idleTimeout: idleTimeout}

	var wg sync.WaitGroup
	wg.Add(2)

	pipe := func(dst io.Writer, src io.Reader, dstConn net.Conn) {
		defer wg.Done()
		if _, err := io.Copy(dst, src); err != nil && !errors.Is(err, net.ErrClosed) && !errors.Is(err, io.EOF) {
			slog.Debug("bridge copy error", "error", err)
		}
		if cw, ok := dstConn.(closeWriter); ok {
			_ = cw.CloseWrite()
		} else {
			_ = dstConn.Close()
		}
	}

	go pipe(wrappedUpstream, wrappedClient, upstream)
	go pipe(wrappedClient, wrappedUpstream, client)

	wg.Wait()
	_ = client.Close()
	_ = upstream.Close()
}

func isTemporaryAcceptError(err error) bool {
	if errors.Is(err, syscall.EMFILE) || errors.Is(err, syscall.ENFILE) ||
		errors.Is(err, syscall.ECONNABORTED) || errors.Is(err, syscall.ECONNRESET) ||
		errors.Is(err, syscall.ENOBUFS) {
		return true
	}
	var netErr net.Error
	if errors.As(err, &netErr) {
		//nolint:staticcheck // Temporary is used for legacy net.Error compatibility
		return netErr.Timeout() || netErr.Temporary()
	}
	return false
}

func serveForwarder(ctx context.Context, ln net.Listener, sockPath string) error {
	var connsMu sync.Mutex
	// nosemgrep: trailofbits.go.iterate-over-empty-map.iterate-over-empty-map
	activeConns := make(map[net.Conn]struct{})
	sem := make(chan struct{}, maxConcurrentConns)
	var fwdWg sync.WaitGroup

	go func() {
		<-ctx.Done()
		_ = ln.Close()

		connsMu.Lock()
		for c := range activeConns {
			_ = c.Close()
		}
		connsMu.Unlock()
	}()

	var tempDelay time.Duration
acceptLoop:
	for {
		client, err := ln.Accept()
		if err != nil {
			if errors.Is(err, net.ErrClosed) || ctx.Err() != nil {
				break acceptLoop
			}
			if isTemporaryAcceptError(err) {
				if tempDelay == 0 {
					tempDelay = 5 * time.Millisecond
				} else {
					tempDelay *= 2
				}
				if max := 1 * time.Second; tempDelay > max {
					tempDelay = max
				}
				slog.Warn("temporary forwarder accept error, backing off", "error", err, "delay", tempDelay)
				select {
				case <-ctx.Done():
					break acceptLoop
				case <-time.After(tempDelay):
				}
				continue
			}
			fwdWg.Wait()
			return fmt.Errorf("accept error: %w", err)
		}
		tempDelay = 0

		select {
		case sem <- struct{}{}:
		case <-ctx.Done():
			_ = client.Close()
			break acceptLoop
		}

		connsMu.Lock()
		activeConns[client] = struct{}{}
		connsMu.Unlock()

		fwdWg.Add(1)
		go func(c net.Conn) {
			defer func() {
				<-sem
				fwdWg.Done()
				connsMu.Lock()
				delete(activeConns, c)
				connsMu.Unlock()
			}()

			var dialer net.Dialer
			dialCtx, cancel := context.WithTimeout(ctx, defaultDialTimeout)
			defer cancel()

			upstream, err := dialer.DialContext(dialCtx, "unix", sockPath)
			if err != nil {
				slog.Error("failed to connect to unix socket", "socket", sockPath, "error", err)
				_ = c.Close()
				return
			}

			bridge(c, upstream, defaultIdleTimeout)
		}(client)
	}

	fwdWg.Wait()
	return nil
}

func startReaper(ctx context.Context) {
	sigCh := make(chan os.Signal, 32)
	signal.Notify(sigCh, syscall.SIGCHLD)
	go func() {
		defer signal.Stop(sigCh)
		for {
			select {
			case <-ctx.Done():
				return
			case <-sigCh:
				for {
					var status syscall.WaitStatus
					pid, err := syscall.Wait4(-1, &status, syscall.WNOHANG, nil)
					if pid <= 0 || err != nil {
						break
					}
				}
			}
		}
	}()
}

func run(ctx context.Context, args []string, stdin io.Reader, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("bob-proxy-fwd", flag.ContinueOnError)
	fs.SetOutput(stderr)

	tcpAddr := fs.String("tcp", "127.0.0.1:18080", "TCP address to listen on")
	sockPath := fs.String("sock", "/run/proxy/proxy.sock", "Unix domain socket path")
	daemonMode := fs.Bool("daemon", false, "Run as background daemon")

	if err := fs.Parse(args); err != nil {
		return 2
	}

	cmdArgs := fs.Args()
	if *daemonMode && len(cmdArgs) > 0 {
		_, _ = fmt.Fprintln(stderr, "cannot specify command arguments in daemon mode")
		return 2
	}
	isDaemon := *daemonMode || len(cmdArgs) == 0

	lc := net.ListenConfig{}
	ln, err := lc.Listen(ctx, "tcp", *tcpAddr)
	if err != nil {
		_, _ = fmt.Fprintf(stderr, "failed to bind TCP listener on %s: %v\n", *tcpAddr, err)
		return 1
	}
	defer func() { _ = ln.Close() }()

	if isDaemon {
		slog.Info("starting proxy forwarder daemon", "tcp", ln.Addr().String(), "socket", *sockPath)
		startReaper(ctx)
		if err := serveForwarder(ctx, ln, *sockPath); err != nil {
			_, _ = fmt.Fprintf(stderr, "forwarder error: %v\n", err)
			return 1
		}
		return 0
	}

	fwdCtx, cancelFwd := context.WithCancel(ctx)
	defer cancelFwd()

	errCh := make(chan error, 1)
	go func() {
		errCh <- serveForwarder(fwdCtx, ln, *sockPath)
	}()

	// nosemgrep: go.lang.security.audit.dangerous-exec-command.dangerous-exec-command
	cmd := exec.CommandContext(ctx, cmdArgs[0], cmdArgs[1:]...)
	cmd.Stdin = stdin
	cmd.Stdout = stdout
	cmd.Stderr = stderr

	runErr := cmd.Run()
	cancelFwd()
	fwdErr := <-errCh
	if fwdErr != nil && !errors.Is(fwdErr, net.ErrClosed) && !errors.Is(fwdErr, context.Canceled) {
		slog.Error("forwarder error in wrapper mode", "error", fwdErr)
		_, _ = fmt.Fprintf(stderr, "forwarder error: %v\n", fwdErr)
	}

	if runErr != nil {
		var exitErr *exec.ExitError
		if errors.As(runErr, &exitErr) {
			return exitErr.ExitCode()
		}
		_, _ = fmt.Fprintf(stderr, "failed to execute %s: %v\n", cmdArgs[0], runErr)
		return 1
	}

	if fwdErr != nil && !errors.Is(fwdErr, net.ErrClosed) && !errors.Is(fwdErr, context.Canceled) {
		return 1
	}

	return 0
}

func main() {
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	code := run(ctx, os.Args[1:], os.Stdin, os.Stdout, os.Stderr)
	os.Exit(code)
}
