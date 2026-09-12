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

type closeWriter interface {
	CloseWrite() error
}

func bridge(client net.Conn, upstream net.Conn) {
	var wg sync.WaitGroup
	wg.Add(2)

	pipe := func(dst, src net.Conn) {
		defer wg.Done()
		if _, err := io.Copy(dst, src); err != nil && !errors.Is(err, net.ErrClosed) && !errors.Is(err, io.EOF) {
			slog.Debug("bridge copy error", "error", err)
		}
		if cw, ok := dst.(closeWriter); ok {
			_ = cw.CloseWrite()
		} else {
			_ = dst.Close()
		}
	}

	go pipe(upstream, client)
	go pipe(client, upstream)

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
	activeConns := make(map[net.Conn]struct{})

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
	for {
		client, err := ln.Accept()
		if err != nil {
			if errors.Is(err, net.ErrClosed) || ctx.Err() != nil {
				return nil
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
					return nil
				case <-time.After(tempDelay):
				}
				continue
			}
			return fmt.Errorf("accept error: %w", err)
		}
		tempDelay = 0

		connsMu.Lock()
		activeConns[client] = struct{}{}
		connsMu.Unlock()

		go func(c net.Conn) {
			defer func() {
				connsMu.Lock()
				delete(activeConns, c)
				connsMu.Unlock()
			}()

			var dialer net.Dialer
			dialCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
			defer cancel()

			upstream, err := dialer.DialContext(dialCtx, "unix", sockPath)
			if err != nil {
				slog.Error("failed to connect to unix socket", "socket", sockPath, "error", err)
				_ = c.Close()
				return
			}

			bridge(c, upstream)
		}(client)
	}
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

	cmd := exec.CommandContext(ctx, cmdArgs[0], cmdArgs[1:]...)
	cmd.Stdin = stdin
	cmd.Stdout = stdout
	cmd.Stderr = stderr

	runErr := cmd.Run()
	cancelFwd()
	<-errCh

	if runErr != nil {
		var exitErr *exec.ExitError
		if errors.As(runErr, &exitErr) {
			return exitErr.ExitCode()
		}
		_, _ = fmt.Fprintf(stderr, "failed to execute %s: %v\n", cmdArgs[0], runErr)
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
