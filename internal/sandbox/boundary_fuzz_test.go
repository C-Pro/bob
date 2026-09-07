package sandbox

import (
	"context"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"bob/internal/config"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// FuzzUserWorkspaceDir validates path normalization and directory containment
// in UserWorkspaceDir across arbitrary inputs, testing for traversal sequences.
func FuzzUserWorkspaceDir(f *testing.F) {
	seeds := []string{
		"user1",
		"alice-123",
		"",
		"../traversal",
		"../../etc/passwd",
		"..\\..\\windows\\system32",
		"user/sub",
		"/root",
		"user\x00evil",
		"a/b/c/d/e/f",
		".hidden",
		"...",
		"....//....//",
		"user@domain.com",
		"user:123",
		"user*name",
		"user~name",
		"user$name",
	}
	for _, s := range seeds {
		f.Add(s)
	}

	tempDir := f.TempDir()
	cfg := &config.Config{
		DataDir: tempDir,
	}
	mgr := &Manager{cfg: cfg}
	sandboxBase := filepath.Join(tempDir, "sandboxes")

	f.Fuzz(func(t *testing.T, userID string) {
		// Invariant: UserWorkspaceDir must never panic on arbitrary input
		workspace, err := mgr.UserWorkspaceDir(userID)
		if err != nil {
			return
		}
		if workspace == "" {
			t.Errorf("workspace path should not be empty")
		}

		// Check directory containment: path must reside strictly inside sandboxBase
		rel, err := filepath.Rel(sandboxBase, workspace)
		isContained := err == nil && rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator))
		if !isContained {
			t.Errorf("Path traversal escape: input=%q -> workspace=%q (rel=%q)", userID, workspace, rel)
		}
	})
}

// FuzzResolveAndValidate tests FilteringProxy domain filtering and IP validation logic
// against diverse hostname inputs, IP notations, CIDRs, and boundary edge cases.
func FuzzResolveAndValidate(f *testing.F) {
	seeds := []string{
		"127.0.0.1",
		"169.254.169.254",
		"metadata.google.internal",
		"localhost",
		"10.0.0.1",
		"172.16.0.1",
		"192.168.1.1",
		"::1",
		"fe80::1",
		"fc00::1",
		"::ffff:127.0.0.1",
		"example.com",
		"sub.example.com",
		"allowed.org",
		"evil.com",
		"",
		"   ",
		"trailing.dot.",
		"http://malformed",
		"user@host:80",
		"0.0.0.0",
		"255.255.255.255",
	}
	for _, s := range seeds {
		f.Add(s)
	}

	proxy := &FilteringProxy{
		policy: NetworkPolicy{
			Mode:         NetworkRestricted,
			AllowedHosts: []string{"allowed.org", "sub.allowed.org"},
			BlockedHosts: MandatoryBlockedHosts,
			BlockedCIDRs: MandatoryBlockedCIDRs,
		},
	}

	ctx := context.Background()

	f.Fuzz(func(t *testing.T, host string) {
		// Invariant: resolveAndValidate must never panic on arbitrary strings
		ip, err := proxy.resolveAndValidate(ctx, host)

		cleanHost := strings.ToLower(strings.TrimSpace(host))

		// Invariant: Mandatory blocked hosts must always be rejected
		for _, blocked := range MandatoryBlockedHosts {
			if cleanHost == strings.ToLower(blocked) || strings.HasSuffix(cleanHost, "."+strings.ToLower(blocked)) {
				if err == nil {
					t.Errorf("expected error for blocked host %q, got nil (ip=%v)", host, ip)
				}
			}
		}

		// Invariant: If parsed as IP and matches blocked CIDR, must always be rejected
		if parsed := net.ParseIP(cleanHost); parsed != nil {
			for _, cidrStr := range MandatoryBlockedCIDRs {
				_, cidr, parseErr := net.ParseCIDR(cidrStr)
				if parseErr == nil && cidr.Contains(parsed) {
					if err == nil {
						t.Errorf("expected error for blocked CIDR IP %s (%s), got nil", parsed, cidrStr)
					}
				}
			}
		}
	})
}

// FuzzValidateMountPath exercises mount path validation against directory traversal,
// symlink escapes, and boundary conditions relative to user workspace.
func FuzzValidateMountPath(f *testing.F) {
	seeds := []string{
		"",
		".",
		"/",
		"my_dir",
		"my_dir/subdir",
		"../outside",
		"foo/../../bar",
		"/abs/path",
		"\\windows\\path",
		".git",
		"a/b/c/d",
		"spaces in path",
		"....//....//",
	}
	for _, s := range seeds {
		f.Add(s)
	}

	tempDir := f.TempDir()
	workspace := filepath.Join(tempDir, "workspace")
	if err := os.MkdirAll(workspace, 0o755); err != nil {
		f.Fatalf("failed to create temp workspace: %v", err)
	}

	f.Fuzz(func(t *testing.T, relPath string) {
		// Invariant: ValidateMountPath must never panic
		resolved, isWhole, err := ValidateMountPath(workspace, relPath)
		if err != nil {
			return // Expected rejection for invalid paths
		}

		// Invariant: If successful, the path must be contained within workspace
		absWorkspace, _ := filepath.Abs(workspace)
		rel, relErr := filepath.Rel(absWorkspace, resolved)
		escaped := relErr != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator))
		if escaped {
			t.Errorf("path traversal escaped workspace: relPath=%q -> resolved=%q (rel=%s)", relPath, resolved, rel)
		}

		// Invariant: Root mount invariants
		clean := filepath.Clean(strings.TrimSpace(relPath))
		if clean == "." || clean == "" || strings.TrimSpace(relPath) == "/" {
			if !isWhole {
				t.Errorf("expected isWhole=true for %q", relPath)
			}
			if resolved != absWorkspace {
				t.Errorf("expected resolved == absWorkspace for %q, got %q", relPath, resolved)
			}
		}
	})
}

// TestUserWorkspaceDir_PathTraversalDefect documents and confirms the path traversal
// vulnerability in UserWorkspaceDir where filepath.Clean preserves "../" prefixes.
func TestUserWorkspaceDir_PathTraversalDefect(t *testing.T) {
	tempDir := t.TempDir()
	cfg := &config.Config{
		DataDir: filepath.Join(tempDir, "data"),
	}
	mgr := &Manager{cfg: cfg}
	sandboxBase := filepath.Join(cfg.DataDir, "sandboxes")

	testCases := []struct {
		name              string
		userID            string
		expectError       bool
		expectedResultDir string
	}{
		{
			name:              "Normal alphanumeric UserID is contained",
			userID:            "user-12345",
			expectError:       false,
			expectedResultDir: filepath.Join(sandboxBase, "user-12345"),
		},
		{
			name:        "Path traversal one level up escapes sandboxes to data",
			userID:      "../victim",
			expectError: true,
		},
		{
			name:        "Path traversal two levels up escapes data to tempDir",
			userID:      "../../sensitive_host_file",
			expectError: true,
		},
		{
			name:        "Deep path traversal resolves to host root",
			userID:      "../../../../../../../../../../etc",
			expectError: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			actualPath, err := mgr.UserWorkspaceDir(tc.userID)
			if tc.expectError {
				assert.Error(t, err, "Path traversal must be rejected with error")
			} else {
				require.NoError(t, err)
				assert.Equal(t, tc.expectedResultDir, actualPath)

				rel, err := filepath.Rel(sandboxBase, actualPath)
				require.NoError(t, err)
				isContained := rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator))
				assert.True(t, isContained, "Path should remain inside sandboxBase")
			}
		})
	}
}

// TestFilteringProxy_RestrictedEmptyDomains_DefaultOpen verifies that in NetworkRestricted mode,
// an empty AllowedHosts slice causes the whitelist check to be bypassed completely (default-open defect).
func TestFilteringProxy_RestrictedEmptyDomains_DefaultOpen(t *testing.T) {
	// Setup proxy with Restricted mode but EMPTY AllowedHosts
	proxy := &FilteringProxy{
		policy: NetworkPolicy{
			Mode:         NetworkRestricted,
			AllowedHosts: []string{}, // Empty domain whitelist
			BlockedHosts: MandatoryBlockedHosts,
			BlockedCIDRs: MandatoryBlockedCIDRs,
		},
	}

	// 1. Mandatory blocked host should still be blocked by step 1
	_, err := proxy.resolveAndValidate(context.Background(), "localhost")
	assert.Error(t, err, "localhost must be blocked")

	// 2. Public internet IP (e.g. 93.184.216.34 for example.com)
	// DEFECT: Because len(p.policy.AllowedHosts) == 0, the whitelist loop is skipped entirely!
	ip, err := proxy.resolveAndValidate(context.Background(), "93.184.216.34")
	assert.NoError(t, err, "DEFECT CONFIRMED: resolveAndValidate returned nil error because whitelist loop was skipped when AllowedHosts is empty")
	assert.Equal(t, "93.184.216.34", ip.String(), "Public IP was permitted through restricted proxy with empty whitelist")

	// 3. Contrast with populated AllowedHosts: unlisted IP is correctly rejected
	proxyStrict := &FilteringProxy{
		policy: NetworkPolicy{
			Mode:         NetworkRestricted,
			AllowedHosts: []string{"allowed.example.com"},
			BlockedHosts: MandatoryBlockedHosts,
			BlockedCIDRs: MandatoryBlockedCIDRs,
		},
	}
	_, errStrict := proxyStrict.resolveAndValidate(context.Background(), "93.184.216.34")
	assert.ErrorContains(t, errStrict, "is not in allowed domains whitelist")
}

// TestFilteringProxy_LANClientIPAuthorization documents the open LAN proxy risk:
// handleRequest allows any RFC 1918 private client IP without authentication.
func TestFilteringProxy_LANClientIPAuthorization(t *testing.T) {
	proxy := &FilteringProxy{
		policy: NetworkPolicy{
			Mode:         NetworkNone,
			BlockedHosts: MandatoryBlockedHosts,
			BlockedCIDRs: MandatoryBlockedCIDRs,
		},
	}

	testCases := []struct {
		name          string
		clientAddr    string
		expectBlocked bool
	}{
		{
			name:          "Loopback client is permitted",
			clientAddr:    "127.0.0.1:54321",
			expectBlocked: false,
		},
		{
			name:          "LAN client 192.168.1.50 is permitted due to ip.IsPrivate()",
			clientAddr:    "192.168.1.50:54321",
			expectBlocked: false, // DEFECT: LAN clients are allowed without auth
		},
		{
			name:          "LAN client 10.0.0.200 is permitted due to ip.IsPrivate()",
			clientAddr:    "10.0.0.200:54321",
			expectBlocked: false, // DEFECT: LAN clients are allowed without auth
		},
		{
			name:          "Docker bridge client 172.17.0.2 is permitted",
			clientAddr:    "172.17.0.2:54321",
			expectBlocked: false,
		},
		{
			name:          "Public internet client IP 203.0.113.50 is rejected",
			clientAddr:    "203.0.113.50:54321",
			expectBlocked: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			clientHost, _, err := net.SplitHostPort(tc.clientAddr)
			require.NoError(t, err)

			ip := net.ParseIP(clientHost)
			require.NotNil(t, ip)

			isUnauthorized := !ip.IsLoopback() && !ip.IsPrivate() && !ip.IsLinkLocalUnicast()
			assert.Equal(t, tc.expectBlocked, isUnauthorized)

			// Verify via synthetic HTTP request through proxy handler
			req := httptest.NewRequest(http.MethodGet, "http://localhost/", nil)
			req.RemoteAddr = tc.clientAddr
			w := httptest.NewRecorder()

			proxy.handleRequest(w, req)
			if tc.expectBlocked {
				assert.Equal(t, http.StatusForbidden, w.Code)
				assert.Contains(t, w.Body.String(), "unauthorized client IP")
			} else {
				// Failed at policy resolution or connection, NOT at client IP check
				assert.NotContains(t, w.Body.String(), "unauthorized client IP")
			}
		})
	}
}

// TestApproveSandbox_StateRaceCondition demonstrates the TOCTOU window where m.mu is released
// before driver.Create() completes, leaving Status == StatusPendingApproval.
func TestApproveSandbox_StateRaceCondition(t *testing.T) {
	tempDir := t.TempDir()
	cfg := &config.Config{
		DataDir:                   tempDir,
		SandboxEnabled:            true,
		SandboxDrivers:            []string{"bwrap"},
		SandboxMaxLifetime:        30 * time.Minute,
		SandboxDefaultExecTimeout: 1 * time.Minute,
		SandboxMaxExecTimeout:     10 * time.Minute,
	}

	createStarted := make(chan struct{})
	createBlock := make(chan struct{})

	blockingDriver := &blockingMockDriver{
		driverType: DriverBwrap,
		available:  true,
		onCreate: func() {
			close(createStarted)
			<-createBlock
		},
	}

	mgr := NewManager(cfg, []Driver{blockingDriver})
	defer func() { _ = mgr.Close() }()

	ctx := context.Background()
	_, err := mgr.RequestSandbox(ctx, "race_user", "chat1", RequestParams{
		Driver:      DriverBwrap,
		NetworkMode: NetworkNone,
	})
	require.NoError(t, err)

	// Goroutine 1 calls ApproveSandbox
	var approveErr error
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		_, approveErr = mgr.ApproveSandbox(ctx, "race_user")
	}()

	// Wait until driver.Create() has begun
	<-createStarted

	// In this window, status is STILL StatusPendingApproval because m.mu was unlocked before driver.Create()
	mgr.mu.Lock()
	sbx := mgr.sandboxes["race_user"]
	statusDuringCreate := sbx.Status
	mgr.mu.Unlock()

	assert.Equal(t, StatusPendingApproval, statusDuringCreate, "DEFECT: status remains StatusPendingApproval while driver.Create() is running")

	// Unblock driver.Create()
	close(createBlock)
	wg.Wait()
	assert.NoError(t, approveErr)
}

type blockingMockDriver struct {
	driverType DriverType
	available  bool
	onCreate   func()
	mu         sync.Mutex
}

func (m *blockingMockDriver) Type() DriverType {
	return m.driverType
}

func (m *blockingMockDriver) Available(ctx context.Context) bool {
	return m.available
}

func (m *blockingMockDriver) Create(ctx context.Context, sbx *UserSandbox, workspace string) error {
	if m.onCreate != nil {
		m.onCreate()
	}
	return nil
}

func (m *blockingMockDriver) Exec(ctx context.Context, sbx *UserSandbox, cmd []string, timeout time.Duration) (*ExecResult, error) {
	return &ExecResult{ExitCode: 0}, nil
}

func (m *blockingMockDriver) Destroy(ctx context.Context, sbx *UserSandbox) error {
	return nil
}
