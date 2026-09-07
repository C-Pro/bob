package sandbox

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// ProxyConfig specifies configuration options for FilteringProxy.
type ProxyConfig struct {
	Policy             NetworkPolicy
	ListenTCP          string   // TCP bind address (e.g. "127.0.0.1:0" or "0.0.0.0:0"). If empty, defaults to "127.0.0.1:0".
	SocketPath         string   // Optional path for a Unix domain socket listener.
	CustomBlocked      []string // Optional test override to bypass defaultMandatoryBlockedCIDRs.
	AllowedClientCIDRs []string // Optional allowed client CIDRs in addition to loopback.
}

// FilteringProxy is an in-process HTTP/CONNECT proxy that enforces domain and CIDR whitelisting/blacklisting.
type FilteringProxy struct {
	policy             NetworkPolicy
	listener           net.Listener
	unixListener       net.Listener
	server             *http.Server
	addr               string
	socketPath         string
	port               int
	closed             bool
	mu                 sync.Mutex
	connsMu            sync.Mutex
	activeConns        map[net.Conn]struct{}
	allowedClientCIDRs []*net.IPNet
	httpClient         *http.Client
}

type closeWriter interface {
	CloseWrite() error
}

func closeWrite(conn net.Conn) {
	if cw, ok := conn.(closeWriter); ok {
		_ = cw.CloseWrite()
	}
}

// NewFilteringProxy starts an in-process filtering proxy bound to 127.0.0.1 on a free port.
func NewFilteringProxy(policy NetworkPolicy) (*FilteringProxy, error) {
	return NewFilteringProxyWithConfig(ProxyConfig{
		Policy:    policy,
		ListenTCP: "127.0.0.1:0",
	})
}

// NewFilteringProxyWithConfig starts a filtering proxy with customized TCP and Unix socket listeners.
func NewFilteringProxyWithConfig(cfg ProxyConfig) (*FilteringProxy, error) {
	mandatory := defaultMandatoryBlockedCIDRs
	if cfg.CustomBlocked != nil {
		mandatory = cfg.CustomBlocked
	}
	sanitizedPolicy, err := ValidateNetworkPolicyWithOverrides(cfg.Policy, mandatory)
	if err != nil {
		return nil, fmt.Errorf("invalid proxy network policy: %w", err)
	}

	bindTCP := cfg.ListenTCP
	if bindTCP == "" {
		bindTCP = "127.0.0.1:0"
	}

	ln, err := net.Listen("tcp", bindTCP)
	if err != nil {
		return nil, fmt.Errorf("failed to bind filtering proxy listener: %w", err)
	}

	var port int
	if tcpAddr, ok := ln.Addr().(*net.TCPAddr); ok {
		port = tcpAddr.Port
	}

	var unixLn net.Listener
	if cfg.SocketPath != "" {
		_ = os.Remove(cfg.SocketPath)
		if err := os.MkdirAll(filepath.Dir(cfg.SocketPath), 0o755); err != nil {
			_ = ln.Close()
			return nil, fmt.Errorf("failed to create proxy socket directory: %w", err)
		}
		unixLn, err = net.Listen("unix", cfg.SocketPath)
		if err != nil {
			_ = ln.Close()
			return nil, fmt.Errorf("failed to bind proxy unix socket %s: %w", cfg.SocketPath, err)
		}
		if err := os.Chmod(cfg.SocketPath, 0o666); err != nil {
			_ = unixLn.Close()
			_ = ln.Close()
			return nil, fmt.Errorf("failed to chmod proxy unix socket %s: %w", cfg.SocketPath, err)
		}
	}

	var allowedClientNets []*net.IPNet
	for _, rawCIDR := range cfg.AllowedClientCIDRs {
		_, ipNet, err := net.ParseCIDR(strings.TrimSpace(rawCIDR))
		if err == nil && ipNet != nil {
			allowedClientNets = append(allowedClientNets, ipNet)
		}
	}

	fp := &FilteringProxy{
		policy:             sanitizedPolicy,
		listener:           ln,
		unixListener:       unixLn,
		addr:               ln.Addr().String(),
		socketPath:         cfg.SocketPath,
		port:               port,
		activeConns:        make(map[net.Conn]struct{}),
		allowedClientCIDRs: allowedClientNets,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
			CheckRedirect: func(req *http.Request, via []*http.Request) error {
				return http.ErrUseLastResponse
			},
		},
	}

	fp.server = &http.Server{
		Handler:      http.HandlerFunc(fp.handleRequest),
		ReadTimeout:  60 * time.Second,
		WriteTimeout: 60 * time.Second,
	}

	go func() {
		if err := fp.server.Serve(ln); err != nil && !errors.Is(err, http.ErrServerClosed) {
			slog.Error("filtering proxy TCP server error", "error", err)
		}
	}()

	if unixLn != nil {
		go func() {
			if err := fp.server.Serve(unixLn); err != nil && !errors.Is(err, http.ErrServerClosed) {
				slog.Error("filtering proxy Unix socket server error", "error", err)
			}
		}()
	}

	return fp, nil
}

// Addr returns the proxy address, e.g. "http://127.0.0.1:45678".
func (p *FilteringProxy) Addr() string {
	return "http://" + p.addr
}

// HostPort returns the host:port string of the proxy.
func (p *FilteringProxy) HostPort() string {
	return p.addr
}

// Port returns the TCP port the proxy is listening on.
func (p *FilteringProxy) Port() int {
	return p.port
}

// SocketPath returns the Unix domain socket path if configured.
func (p *FilteringProxy) SocketPath() string {
	return p.socketPath
}

func (p *FilteringProxy) trackConn(c net.Conn) {
	p.connsMu.Lock()
	defer p.connsMu.Unlock()
	if p.activeConns != nil {
		p.activeConns[c] = struct{}{}
	}
}

func (p *FilteringProxy) untrackConn(c net.Conn) {
	p.connsMu.Lock()
	defer p.connsMu.Unlock()
	if p.activeConns != nil {
		delete(p.activeConns, c)
	}
}

// Close stops the proxy listener and terminates active connections.
func (p *FilteringProxy) Close() error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.closed {
		return nil
	}
	p.closed = true

	// Terminate active hijacked connections
	p.connsMu.Lock()
	for c := range p.activeConns {
		_ = c.Close()
	}
	p.activeConns = make(map[net.Conn]struct{})
	p.connsMu.Unlock()

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	err := p.server.Shutdown(ctx)
	if p.unixListener != nil {
		if uErr := p.unixListener.Close(); uErr != nil && !errors.Is(uErr, net.ErrClosed) && err == nil {
			err = uErr
		}
	}
	if p.socketPath != "" {
		if rmErr := os.Remove(p.socketPath); rmErr != nil && !os.IsNotExist(rmErr) && err == nil {
			err = rmErr
		}
	}
	return err
}

func (p *FilteringProxy) logAccess(clientAddr, method, target, proto string, statusCode int, bytesSent int64, action, reason string) {
	clientIP, _, err := net.SplitHostPort(clientAddr)
	if err != nil {
		clientIP = clientAddr
	}
	nowStr := time.Now().Format("02/Jan/2006:15:04:05 -0700")
	reasonSuffix := ""
	if reason != "" {
		reasonSuffix = ": " + reason
	}
	logLine := fmt.Sprintf("%s - - [%s] \"%s %s %s\" %d %d \"-\" \"-\" [%s%s]",
		clientIP, nowStr, method, target, proto, statusCode, bytesSent, action, reasonSuffix)
	slog.Info(logLine,
		"client", clientIP,
		"method", method,
		"target", target,
		"status", statusCode,
		"action", action,
	)
}

var hopByHopHeaders = []string{
	"Connection",
	"Keep-Alive",
	"Proxy-Authenticate",
	"Proxy-Authorization",
	"Proxy-Connection",
	"TE",
	"Trailer",
	"Trailers",
	"Transfer-Encoding",
	"Upgrade",
}

func removeHopByHopHeaders(header http.Header) {
	for _, connHeader := range header.Values("Connection") {
		for _, h := range strings.Split(connHeader, ",") {
			header.Del(strings.TrimSpace(h))
		}
	}
	for _, h := range hopByHopHeaders {
		header.Del(h)
	}
}

func (p *FilteringProxy) handleRequest(w http.ResponseWriter, req *http.Request) {
	// Verify that client is loopback, local private subnet (e.g. docker bridge), or unix socket
	if req.RemoteAddr != "" && req.RemoteAddr != "@" {
		clientHost, _, err := net.SplitHostPort(req.RemoteAddr)
		if err == nil {
			ip := net.ParseIP(clientHost)
			if ip != nil {
				authorized := ip.IsLoopback()
				if !authorized {
					for _, cidr := range p.allowedClientCIDRs {
						if cidr.Contains(ip) {
							authorized = true
							break
						}
					}
				}
				if !authorized {
					p.logAccess(req.RemoteAddr, req.Method, req.URL.Host, req.Proto, http.StatusForbidden, 0, "DENIED", "client IP not authorized")
					http.Error(w, "Access Denied: unauthorized client IP", http.StatusForbidden)
					return
				}
			}
		}
	}

	targetHost := req.URL.Host
	if targetHost == "" {
		targetHost = req.Host
	}

	host, port, err := net.SplitHostPort(targetHost)
	if err != nil {
		host = targetHost
		if req.Method == http.MethodConnect {
			port = "443"
		} else {
			port = "80"
		}
	}

	verifiedIP, err := p.resolveAndValidate(req.Context(), host)
	if err != nil {
		p.logAccess(req.RemoteAddr, req.Method, targetHost, req.Proto, http.StatusForbidden, 0, "DENIED", err.Error())
		http.Error(w, fmt.Sprintf("Access Denied by Sandbox Policy: %v", err), http.StatusForbidden)
		return
	}

	if req.Method == http.MethodConnect {
		p.handleConnect(w, req, verifiedIP, port, targetHost)
	} else {
		p.handleHTTP(w, req, verifiedIP, port, targetHost)
	}
}

func (p *FilteringProxy) resolveAndValidate(ctx context.Context, host string) (net.IP, error) {
	cleanHost := strings.ToLower(strings.TrimSpace(host))

	// 1. Check blocked hostnames
	for _, blocked := range p.policy.BlockedHosts {
		if cleanHost == strings.ToLower(blocked) || strings.HasSuffix(cleanHost, "."+strings.ToLower(blocked)) {
			return nil, fmt.Errorf("host %q is blacklisted", cleanHost)
		}
	}

	// 2. Resolve IP and check CIDR blocks
	var ips []net.IP
	if parsedIP := net.ParseIP(cleanHost); parsedIP != nil {
		ips = append(ips, parsedIP)
	} else {
		lookupCtx, cancel := context.WithTimeout(ctx, 3*time.Second)
		defer cancel()
		resolved, err := net.DefaultResolver.LookupIP(lookupCtx, "ip", cleanHost)
		if err != nil {
			return nil, fmt.Errorf("failed to resolve host %q: %w", cleanHost, err)
		}
		if len(resolved) == 0 {
			return nil, fmt.Errorf("host %q resolved to no IP addresses", cleanHost)
		}
		ips = resolved
	}

	for _, ip := range ips {
		for _, cidrStr := range p.policy.BlockedCIDRs {
			_, cidr, err := net.ParseCIDR(cidrStr)
			if err == nil && cidr.Contains(ip) {
				return nil, fmt.Errorf("resolved IP %s matches blocked CIDR %s", ip.String(), cidrStr)
			}
		}
	}

	// 3. If in Restricted mode, verify host or IP is explicitly permitted in AllowedHosts whitelist (default-deny)
	if p.policy.Mode == NetworkRestricted {
		if len(p.policy.AllowedHosts) == 0 {
			return nil, fmt.Errorf("restricted network policy requires at least one allowed domain; host %q blocked by default-deny", cleanHost)
		}
		allowed := false
		for _, allowedHost := range p.policy.AllowedHosts {
			cleanAllowed := strings.ToLower(strings.TrimSpace(allowedHost))
			if cleanHost == cleanAllowed || strings.HasSuffix(cleanHost, "."+cleanAllowed) {
				allowed = true
				break
			}
			// Also check if allowed is a CIDR/IP
			if parsedAllowedIP := net.ParseIP(cleanAllowed); parsedAllowedIP != nil {
				for _, ip := range ips {
					if ip.Equal(parsedAllowedIP) {
						allowed = true
						break
					}
				}
			}
		}
		if !allowed {
			return nil, fmt.Errorf("host %q is not in allowed domains whitelist", cleanHost)
		}
	}

	return ips[0], nil
}

func (p *FilteringProxy) handleConnect(w http.ResponseWriter, req *http.Request, targetIP net.IP, port, targetHost string) {
	targetAddr := net.JoinHostPort(targetIP.String(), port)
	destConn, err := net.DialTimeout("tcp", targetAddr, 10*time.Second)
	if err != nil {
		p.logAccess(req.RemoteAddr, req.Method, targetHost, req.Proto, http.StatusBadGateway, 0, "ERROR", err.Error())
		http.Error(w, fmt.Sprintf("Failed to connect to target: %v", err), http.StatusBadGateway)
		return
	}
	defer func() { _ = destConn.Close() }()

	hijacker, ok := w.(http.Hijacker)
	if !ok {
		p.logAccess(req.RemoteAddr, req.Method, targetHost, req.Proto, http.StatusInternalServerError, 0, "ERROR", "hijacking not supported")
		http.Error(w, "Hijacking not supported", http.StatusInternalServerError)
		return
	}

	clientConn, _, err := hijacker.Hijack()
	if err != nil {
		p.logAccess(req.RemoteAddr, req.Method, targetHost, req.Proto, http.StatusServiceUnavailable, 0, "ERROR", err.Error())
		http.Error(w, fmt.Sprintf("Hijacking failed: %v", err), http.StatusServiceUnavailable)
		return
	}
	defer func() { _ = clientConn.Close() }()

	p.trackConn(clientConn)
	defer p.untrackConn(clientConn)
	p.trackConn(destConn)
	defer p.untrackConn(destConn)

	_, err = clientConn.Write([]byte("HTTP/1.1 200 Connection Established\r\n\r\n"))
	if err != nil {
		p.logAccess(req.RemoteAddr, req.Method, targetHost, req.Proto, http.StatusBadGateway, 0, "ERROR", err.Error())
		return
	}

	ctx := req.Context()
	ctxDone := make(chan struct{})
	go func() {
		select {
		case <-ctx.Done():
			_ = clientConn.Close()
			_ = destConn.Close()
		case <-ctxDone:
		}
	}()
	defer close(ctxDone)

	// Bidirectional tunnel
	var wg sync.WaitGroup
	wg.Add(2)
	var bytesToDest, bytesToClient int64

	go func() {
		defer wg.Done()
		n, _ := io.Copy(destConn, clientConn)
		atomic.AddInt64(&bytesToDest, n)
		closeWrite(destConn)
	}()

	go func() {
		defer wg.Done()
		n, _ := io.Copy(clientConn, destConn)
		atomic.AddInt64(&bytesToClient, n)
		closeWrite(clientConn)
	}()

	wg.Wait()
	p.logAccess(req.RemoteAddr, req.Method, targetHost, req.Proto, http.StatusOK, bytesToDest+bytesToClient, "ALLOWED", "")
}

func (p *FilteringProxy) handleHTTP(w http.ResponseWriter, req *http.Request, targetIP net.IP, port, targetHost string) {
	outReq := req.Clone(req.Context())
	outReq.RequestURI = "" // RequestURI must not be set on client requests
	if outReq.URL.Scheme == "" {
		outReq.URL.Scheme = "http"
	}
	if outReq.URL.Host == "" {
		outReq.URL.Host = targetHost
	}

	// Strip hop-by-hop headers from outgoing request
	removeHopByHopHeaders(outReq.Header)

	// Dial verified destination IP directly to eliminate DNS rebinding / TOCTOU
	targetAddr := net.JoinHostPort(targetIP.String(), port)
	transport := &http.Transport{
		DialContext: func(ctx context.Context, network, _ string) (net.Conn, error) {
			var d net.Dialer
			return d.DialContext(ctx, network, targetAddr)
		},
		ResponseHeaderTimeout: 30 * time.Second,
		IdleConnTimeout:       30 * time.Second,
	}

	client := &http.Client{
		Transport: transport,
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			return http.ErrUseLastResponse
		},
		Timeout: 60 * time.Second,
	}

	resp, err := client.Do(outReq)
	if err != nil {
		p.logAccess(req.RemoteAddr, req.Method, req.URL.String(), req.Proto, http.StatusBadGateway, 0, "ERROR", err.Error())
		http.Error(w, fmt.Sprintf("Proxy error: %v", err), http.StatusBadGateway)
		return
	}
	defer func() { _ = resp.Body.Close() }()

	// Strip hop-by-hop headers from upstream response
	removeHopByHopHeaders(resp.Header)

	for k, vv := range resp.Header {
		for _, v := range vv {
			w.Header().Add(k, v)
		}
	}
	w.WriteHeader(resp.StatusCode)
	n, err := io.Copy(w, resp.Body)
	if err != nil && !errors.Is(err, context.Canceled) {
		slog.Debug("error copying proxy response body", "target", targetHost, "error", err)
	}
	p.logAccess(req.RemoteAddr, req.Method, req.URL.String(), req.Proto, resp.StatusCode, n, "ALLOWED", "")
}
