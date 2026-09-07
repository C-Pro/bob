package sandbox

import (
	"bufio"
	"context"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestFilteringProxyHTTP(t *testing.T) {
	// 1. Create a dummy backend HTTP server
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("backend response"))
	}))
	defer backend.Close()

	backendURL, err := url.Parse(backend.URL)
	require.NoError(t, err)

	// 2. Start proxy with restricted access allowing only the backend host
	proxy, err := NewFilteringProxyWithConfig(ProxyConfig{
		Policy: NetworkPolicy{
			Mode:         NetworkRestricted,
			AllowedHosts: []string{backendURL.Hostname()},
			BlockedHosts: []string{"forbidden.com"},
		},
		CustomBlocked: []string{"169.254.0.0/16"},
	})
	require.NoError(t, err)
	defer func() { _ = proxy.Close() }()

	proxyURL, err := url.Parse(proxy.Addr())
	require.NoError(t, err)

	client := &http.Client{
		Transport: &http.Transport{
			Proxy: http.ProxyURL(proxyURL),
		},
	}

	// 3. Test allowed request -> 200 OK
	resp, err := client.Get(backend.URL + "/test")
	require.NoError(t, err)
	defer func() { _ = resp.Body.Close() }()
	body, _ := io.ReadAll(resp.Body)
	assert.Equal(t, http.StatusOK, resp.StatusCode)
	assert.Equal(t, "backend response", string(body))

	// 4. Test blocked host (explicitly blacklisted) -> 403 Forbidden
	resp, err = client.Get("http://forbidden.com/secret")
	require.NoError(t, err)
	defer func() { _ = resp.Body.Close() }()
	assert.Equal(t, http.StatusForbidden, resp.StatusCode)

	// 5. Test mandatory metadata IP -> 403 Forbidden
	resp, err = client.Get("http://169.254.169.254/computeMetadata/v1/")
	require.NoError(t, err)
	defer func() { _ = resp.Body.Close() }()
	assert.Equal(t, http.StatusForbidden, resp.StatusCode)

	// 6. Test domain not in whitelist -> 403 Forbidden
	resp, err = client.Get("http://unauthorized.org/page")
	require.NoError(t, err)
	defer func() { _ = resp.Body.Close() }()
	assert.Equal(t, http.StatusForbidden, resp.StatusCode)
}

func TestFilteringProxyConnect(t *testing.T) {
	// Create a dummy backend TCP echo server
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	defer func() { _ = ln.Close() }()

	go func() {
		for {
			conn, err := ln.Accept()
			if err != nil {
				return
			}
			go func(c net.Conn) {
				defer func() { _ = c.Close() }()
				buf := make([]byte, 1024)
				n, _ := c.Read(buf)
				_, _ = c.Write([]byte("ECHO:" + string(buf[:n])))
			}(conn)
		}
	}()

	backendHost, backendPort, err := net.SplitHostPort(ln.Addr().String())
	require.NoError(t, err)

	proxy, err := NewFilteringProxyWithConfig(ProxyConfig{
		Policy: NetworkPolicy{
			Mode:         NetworkRestricted,
			AllowedHosts: []string{backendHost},
		},
		CustomBlocked: []string{"169.254.0.0/16"},
	})
	require.NoError(t, err)
	defer func() { _ = proxy.Close() }()

	// Connect to proxy and issue CONNECT request to allowed backend
	proxyConn, err := net.Dial("tcp", proxy.HostPort())
	require.NoError(t, err)
	defer func() { _ = proxyConn.Close() }()

	req := "CONNECT " + ln.Addr().String() + " HTTP/1.1\r\nHost: " + ln.Addr().String() + "\r\n\r\n"
	_, err = proxyConn.Write([]byte(req))
	require.NoError(t, err)

	reader := bufio.NewReader(proxyConn)
	statusLine, err := reader.ReadString('\n')
	require.NoError(t, err)
	assert.Contains(t, statusLine, "200 Connection Established")

	// Read remaining headers
	for {
		line, err := reader.ReadString('\n')
		require.NoError(t, err)
		if strings.TrimSpace(line) == "" {
			break
		}
	}

	// Send payload through tunnel
	_, err = proxyConn.Write([]byte("hello"))
	require.NoError(t, err)

	buf := make([]byte, 100)
	n, err := reader.Read(buf)
	require.NoError(t, err)
	assert.Equal(t, "ECHO:hello", string(buf[:n]))

	// Now try CONNECT to cloud metadata IP -> must fail with 403
	proxyConn2, err := net.Dial("tcp", proxy.HostPort())
	require.NoError(t, err)
	defer func() { _ = proxyConn2.Close() }()

	req2 := "CONNECT 169.254.169.254:80 HTTP/1.1\r\nHost: 169.254.169.254:80\r\n\r\n"
	_, err = proxyConn2.Write([]byte(req2))
	require.NoError(t, err)

	reader2 := bufio.NewReader(proxyConn2)
	statusLine2, err := reader2.ReadString('\n')
	require.NoError(t, err)
	assert.Contains(t, statusLine2, "403 Forbidden")
	_ = backendPort
}

func TestFilteringProxyBlocksLoopbackAndPrivateNetworks(t *testing.T) {
	proxy, err := NewFilteringProxy(NetworkPolicy{
		Mode:         NetworkRestricted,
		AllowedHosts: []string{"127.0.0.1", "10.1.2.3", "192.168.1.1"},
	})
	require.NoError(t, err)
	defer func() { _ = proxy.Close() }()

	proxyURL, err := url.Parse(proxy.Addr())
	require.NoError(t, err)

	client := &http.Client{
		Transport: &http.Transport{
			Proxy: http.ProxyURL(proxyURL),
		},
	}

	// 127.0.0.1 must be blocked
	resp, err := client.Get("http://127.0.0.1:1234/test")
	require.NoError(t, err)
	defer func() { _ = resp.Body.Close() }()
	assert.Equal(t, http.StatusForbidden, resp.StatusCode)

	// 10.1.2.3 must be blocked
	resp2, err := client.Get("http://10.1.2.3:1234/test")
	require.NoError(t, err)
	defer func() { _ = resp2.Body.Close() }()
	assert.Equal(t, http.StatusForbidden, resp2.StatusCode)

	// 192.168.1.1 must be blocked
	resp3, err := client.Get("http://192.168.1.1:1234/test")
	require.NoError(t, err)
	defer func() { _ = resp3.Body.Close() }()
	assert.Equal(t, http.StatusForbidden, resp3.StatusCode)
}

func TestFilteringProxyHopByHopHeaders(t *testing.T) {
	var receivedHeaders http.Header
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedHeaders = r.Header.Clone()
		w.Header().Set("Proxy-Authenticate", "Basic")
		w.Header().Set("Upgrade", "websocket")
		w.Header().Set("Keep-Alive", "timeout=5")
		w.Header().Set("X-Regular-Header", "allowed")
		w.WriteHeader(http.StatusOK)
	}))
	defer backend.Close()

	backendURL, err := url.Parse(backend.URL)
	require.NoError(t, err)

	proxy, err := NewFilteringProxyWithConfig(ProxyConfig{
		Policy: NetworkPolicy{
			Mode:         NetworkRestricted,
			AllowedHosts: []string{backendURL.Hostname()},
		},
		CustomBlocked: []string{"169.254.0.0/16"},
	})
	require.NoError(t, err)
	defer func() { _ = proxy.Close() }()

	proxyURL, err := url.Parse(proxy.Addr())
	require.NoError(t, err)

	client := &http.Client{
		Transport: &http.Transport{
			Proxy: http.ProxyURL(proxyURL),
		},
	}

	req, err := http.NewRequest(http.MethodGet, backend.URL+"/test", nil)
	require.NoError(t, err)
	req.Header.Set("Connection", "close, X-Client-Hop")
	req.Header.Set("X-Client-Hop", "strip-me")
	req.Header.Set("Proxy-Connection", "keep-alive")
	req.Header.Set("Upgrade", "websocket")
	req.Header.Set("X-Normal-Header", "keep-me")

	resp, err := client.Do(req)
	require.NoError(t, err)
	defer func() { _ = resp.Body.Close() }()

	// Verify upstream did NOT receive hop-by-hop headers
	assert.Empty(t, receivedHeaders.Get("Upgrade"))
	assert.Empty(t, receivedHeaders.Get("Proxy-Connection"))
	assert.Empty(t, receivedHeaders.Get("X-Client-Hop"))
	assert.Equal(t, "keep-me", receivedHeaders.Get("X-Normal-Header"))

	// Verify client did NOT receive upstream hop-by-hop headers
	assert.Empty(t, resp.Header.Get("Proxy-Authenticate"))
	assert.Empty(t, resp.Header.Get("Upgrade"))
	assert.Empty(t, resp.Header.Get("Keep-Alive"))
	assert.Equal(t, "allowed", resp.Header.Get("X-Regular-Header"))
}

func TestFilteringProxyDNSRebindingPrevention(t *testing.T) {
	ips, err := net.LookupIP("example.com")
	require.NoError(t, err)
	require.NotEmpty(t, ips)

	// Block the resolved IP of example.com
	blockedCIDR := ips[0].String() + "/32"

	proxy, err := NewFilteringProxy(NetworkPolicy{
		Mode:         NetworkRestricted,
		AllowedHosts: []string{"example.com"},
		BlockedCIDRs: []string{blockedCIDR},
	})
	require.NoError(t, err)
	defer func() { _ = proxy.Close() }()

	proxyURL, err := url.Parse(proxy.Addr())
	require.NoError(t, err)

	client := &http.Client{
		Transport: &http.Transport{
			Proxy: http.ProxyURL(proxyURL),
		},
	}

	// Request should be blocked because the resolved IP matches BlockedCIDRs
	resp, err := client.Get("http://example.com/test")
	require.NoError(t, err)
	defer func() { _ = resp.Body.Close() }()
	assert.Equal(t, http.StatusForbidden, resp.StatusCode)
}

func TestFilteringProxyUnixSocket(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("unix-proxy-ok"))
	}))
	defer backend.Close()

	backendURL, err := url.Parse(backend.URL)
	require.NoError(t, err)

	tempDir := t.TempDir()
	sockPath := filepath.Join(tempDir, "proxy.sock")

	proxy, err := NewFilteringProxyWithConfig(ProxyConfig{
		Policy: NetworkPolicy{
			Mode:         NetworkRestricted,
			AllowedHosts: []string{backendURL.Hostname()},
		},
		ListenTCP:     "127.0.0.1:0",
		SocketPath:    sockPath,
		CustomBlocked: []string{"169.254.0.0/16"},
	})
	require.NoError(t, err)
	defer func() { _ = proxy.Close() }()

	assert.Equal(t, sockPath, proxy.SocketPath())
	assert.Greater(t, proxy.Port(), 0)
	assert.FileExists(t, sockPath)

	// Test proxying over the Unix domain socket
	client := &http.Client{
		Transport: &http.Transport{
			DialContext: func(ctx context.Context, network, addr string) (net.Conn, error) {
				var d net.Dialer
				return d.DialContext(ctx, "unix", sockPath)
			},
		},
	}

	req, err := http.NewRequest(http.MethodGet, backend.URL+"/test", nil)
	require.NoError(t, err)

	resp, err := client.Do(req)
	require.NoError(t, err)
	defer func() { _ = resp.Body.Close() }()

	body, err := io.ReadAll(resp.Body)
	require.NoError(t, err)
	assert.Equal(t, http.StatusOK, resp.StatusCode)
	assert.Equal(t, "unix-proxy-ok", string(body))

	// Close proxy and verify socket is removed
	require.NoError(t, proxy.Close())
	assert.NoFileExists(t, sockPath)
}

func TestMandatoryBlockedCIDRs_Immutability(t *testing.T) {
	cidrs1 := MandatoryBlockedCIDRs()
	assert.NotEmpty(t, cidrs1)

	// Mutate the returned slice
	cidrs1[0] = "0.0.0.0/0"

	// Subsequent call must return pristine original list
	cidrs2 := MandatoryBlockedCIDRs()
	assert.NotEqual(t, "0.0.0.0/0", cidrs2[0])
	assert.Equal(t, "127.0.0.0/8", cidrs2[0])

	// ValidateNetworkPolicy must still include 127.0.0.0/8
	pol, err := ValidateNetworkPolicy(NetworkPolicy{
		Mode: NetworkRestricted,
	})
	require.NoError(t, err)
	assert.Contains(t, pol.BlockedCIDRs, "127.0.0.0/8")
}

func TestFilteringProxy_UnixSocket_ConnectHalfCloseAndClose(t *testing.T) {
	// 1. Backend TCP server: reads a request, writes a response, and half-closes (CloseWrite)
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	defer func() { _ = ln.Close() }()

	go func() {
		for {
			conn, err := ln.Accept()
			if err != nil {
				return
			}
			go func(c net.Conn) {
				defer func() { _ = c.Close() }()
				buf := make([]byte, 4)
				_, err := io.ReadFull(c, buf)
				if err != nil {
					return
				}
				_, _ = c.Write([]byte("ECHO:" + string(buf)))
				if tc, ok := c.(*net.TCPConn); ok {
					_ = tc.CloseWrite()
				}
				// Server keeps read side open for a moment
				time.Sleep(1 * time.Second)
			}(conn)
		}
	}()

	backendHost, _, err := net.SplitHostPort(ln.Addr().String())
	require.NoError(t, err)

	tmpDir := t.TempDir()
	sockPath := filepath.Join(tmpDir, "proxy.sock")

	proxy, err := NewFilteringProxyWithConfig(ProxyConfig{
		Policy: NetworkPolicy{
			Mode:         NetworkRestricted,
			AllowedHosts: []string{backendHost},
		},
		ListenTCP:     "127.0.0.1:0",
		SocketPath:    sockPath,
		CustomBlocked: []string{"169.254.0.0/16"},
	})
	require.NoError(t, err)
	defer func() { _ = proxy.Close() }()

	// 2. Connect via Unix domain socket to proxy
	clientConn, err := net.Dial("unix", sockPath)
	require.NoError(t, err)
	defer func() { _ = clientConn.Close() }()

	unixConn, ok := clientConn.(*net.UnixConn)
	require.True(t, ok)

	// Send CONNECT
	req := "CONNECT " + ln.Addr().String() + " HTTP/1.1\r\nHost: " + ln.Addr().String() + "\r\n\r\n"
	_, err = unixConn.Write([]byte(req))
	require.NoError(t, err)

	reader := bufio.NewReader(unixConn)
	statusLine, err := reader.ReadString('\n')
	require.NoError(t, err)
	assert.Contains(t, statusLine, "200 Connection Established")

	for {
		line, err := reader.ReadString('\n')
		require.NoError(t, err)
		if strings.TrimSpace(line) == "" {
			break
		}
	}

	// 3. Write 4 bytes to server. Server will reply and call CloseWrite().
	// Client should receive response and EOF without waiting for the server to call Close() after 1 second.
	_, err = unixConn.Write([]byte("ping"))
	require.NoError(t, err)

	start := time.Now()
	// Read response until EOF
	resp, err := io.ReadAll(reader)
	duration := time.Since(start)
	require.NoError(t, err)
	assert.Equal(t, "ECHO:ping", string(resp))
	// If half-close was propagated to unixConn, duration must be < 500ms (not waiting 1s for server Close)
	assert.Less(t, duration, 500*time.Millisecond, "CloseWrite on destConn must immediately propagate EOF to clientConn")

	// 4. Test that proxy.Close() actively closes hijacked connections immediately
	conn2, err := net.Dial("unix", sockPath)
	require.NoError(t, err)
	defer func() { _ = conn2.Close() }()

	_, err = conn2.Write([]byte(req))
	require.NoError(t, err)
	reader2 := bufio.NewReader(conn2)
	statusLine2, err := reader2.ReadString('\n')
	require.NoError(t, err)
	assert.Contains(t, statusLine2, "200 Connection Established")
	for {
		line, err := reader2.ReadString('\n')
		require.NoError(t, err)
		if strings.TrimSpace(line) == "" {
			break
		}
	}

	// Now close the proxy; conn2 should be closed immediately (< 100ms)
	closeStart := time.Now()
	require.NoError(t, proxy.Close())
	buf := make([]byte, 10)
	_, err = conn2.Read(buf)
	closeDuration := time.Since(closeStart)
	assert.Error(t, err, "hijacked connection should be closed when proxy is closed")
	assert.Less(t, closeDuration, 200*time.Millisecond, "proxy.Close must immediately terminate hijacked connections")
}

func TestFilteringProxy_ClientIPAuthorization(t *testing.T) {
	// 1. Default proxy: only loopback is authorized; LAN private IP (192.168.1.50) must be rejected
	proxy, err := NewFilteringProxyWithConfig(ProxyConfig{
		Policy: NetworkPolicy{
			Mode:         NetworkRestricted,
			AllowedHosts: []string{"example.com"},
		},
		CustomBlocked: []string{"169.254.0.0/16"},
	})
	require.NoError(t, err)
	defer func() { _ = proxy.Close() }()

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "http://example.com/", nil)
	req.RemoteAddr = "192.168.1.50:54321"

	proxy.handleRequest(rec, req)
	assert.Equal(t, http.StatusForbidden, rec.Code, "unauthorized LAN private IP must be rejected with 403")
	assert.Contains(t, rec.Body.String(), "unauthorized client IP")

	// 2. Proxy with Docker subnet allowed: 172.17.0.2 is authorized, 192.168.1.50 is rejected
	dockerProxy, err := NewFilteringProxyWithConfig(ProxyConfig{
		Policy: NetworkPolicy{
			Mode:         NetworkRestricted,
			AllowedHosts: []string{"example.com"},
		},
		CustomBlocked:      []string{"169.254.0.0/16"},
		AllowedClientCIDRs: []string{"172.16.0.0/12"},
	})
	require.NoError(t, err)
	defer func() { _ = dockerProxy.Close() }()

	rec2 := httptest.NewRecorder()
	req2 := httptest.NewRequest(http.MethodGet, "http://example.com/", nil)
	req2.RemoteAddr = "192.168.1.50:54321"
	dockerProxy.handleRequest(rec2, req2)
	assert.Equal(t, http.StatusForbidden, rec2.Code)

	rec3 := httptest.NewRecorder()
	req3 := httptest.NewRequest(http.MethodGet, "http://example.com/", nil)
	req3.RemoteAddr = "172.17.0.2:54321"
	dockerProxy.handleRequest(rec3, req3)
	// 172.17.0.2 should pass the client IP check (not fail with unauthorized client IP)
	assert.NotContains(t, rec3.Body.String(), "unauthorized client IP")
}

