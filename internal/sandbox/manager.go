package sandbox

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"

	"bob/internal/config"
)

// RequestParams contains the parameters requested by an agent for creating a sandbox.
type RequestParams struct {
	Driver          DriverType
	DockerImage     string
	NetworkMode     NetworkMode
	AllowedDomains  []string
	Mounts          []UserMount
	LifetimeMinutes int
	Reason          string
}

// Manager orchestrates user sandboxes, enforces the 1-sandbox-per-user limit,
// handles user approval transitions, and cleans up expired sessions.
type Manager struct {
	cfg       *config.Config
	drivers   map[DriverType]Driver
	sandboxes map[string]*UserSandbox // Keyed strictly by UserID
	mu        sync.Mutex
	stopCh    chan struct{}
	wg        sync.WaitGroup
}

// NewManager creates a new Sandbox Manager and registers configured drivers.
func NewManager(cfg *config.Config, drivers []Driver) *Manager {
	driverMap := make(map[DriverType]Driver)
	for _, d := range drivers {
		driverMap[d.Type()] = d
	}

	m := &Manager{
		cfg:       cfg,
		drivers:   driverMap,
		sandboxes: make(map[string]*UserSandbox),
		stopCh:    make(chan struct{}),
	}

	m.wg.Add(1)
	go m.reaperLoop()

	return m
}

// Close stops the background reaper and cleans up all active sandboxes.
func (m *Manager) Close() error {
	m.mu.Lock()
	select {
	case <-m.stopCh:
		m.mu.Unlock()
		return nil
	default:
		close(m.stopCh)
	}

	var toDestroy []*UserSandbox
	for _, sbx := range m.sandboxes {
		if sbx.GetStatus() == StatusRunning || sbx.GetStatus() == StatusExpired {
			toDestroy = append(toDestroy, sbx)
		}
	}
	m.sandboxes = make(map[string]*UserSandbox)
	m.mu.Unlock()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	for _, sbx := range toDestroy {
		if d, ok := m.drivers[sbx.Driver]; ok {
			if err := d.Destroy(ctx, sbx); err != nil {
				slog.Warn("failed to destroy sandbox on manager shutdown", "user", sbx.UserID, "error", err)
			}
		}
	}

	m.wg.Wait()
	return nil
}

// AvailableDrivers returns a list of driver types that are currently available and operational.
func (m *Manager) AvailableDrivers(ctx context.Context) []DriverType {
	var available []DriverType
	for _, driverName := range m.cfg.SandboxDrivers {
		dt := DriverType(strings.TrimSpace(driverName))
		if d, ok := m.drivers[dt]; ok && d.Available(ctx) {
			available = append(available, dt)
		}
	}
	return available
}

// AllowedNetworkModes returns the configured permitted network modes.
func (m *Manager) AllowedNetworkModes() []string {
	if m.cfg != nil && len(m.cfg.SandboxAllowedNetworkModes) > 0 {
		return m.cfg.SandboxAllowedNetworkModes
	}
	return []string{string(NetworkNone), string(NetworkRestricted)}
}

var validUserIDRegex = regexp.MustCompile(`^[a-zA-Z0-9_-]{1,64}$`)

// UserWorkspaceDir returns the absolute path to the user's private workspace.
// It strictly validates userID against traversal attacks and containment under DataDir/sandboxes.
func (m *Manager) UserWorkspaceDir(userID string) (string, error) {
	if !validUserIDRegex.MatchString(userID) {
		return "", fmt.Errorf("invalid userID %q: must match ^[a-zA-Z0-9_-]{1,64}$", userID)
	}
	cleanUID := filepath.Clean(userID)
	sandboxBase := filepath.Join(m.cfg.DataDir, "sandboxes")
	target := filepath.Join(sandboxBase, cleanUID)
	rel, err := filepath.Rel(sandboxBase, target)
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("path traversal detected for userID %q", userID)
	}
	return target, nil
}

// RequestSandbox creates a pending sandbox request awaiting explicit human user approval.
func (m *Manager) RequestSandbox(ctx context.Context, userID, chatID string, params RequestParams) (*UserSandbox, error) {
	if !m.cfg.SandboxEnabled {
		return nil, errors.New("sandbox execution is disabled by configuration")
	}
	if strings.TrimSpace(userID) == "" {
		return nil, errors.New("userID cannot be empty")
	}

	var toDestroy *UserSandbox
	var destroyDriver Driver

	m.mu.Lock()
	// Check 1-live-sandbox-per-user limit
	if existing, ok := m.sandboxes[userID]; ok {
		if existing.GetStatus() == StatusRunning {
			if time.Now().Before(existing.ExpiresAt) {
				m.mu.Unlock()
				return nil, errors.New("you already have a running sandbox; destroy it first with /sandbox destroy")
			}
			toDestroy = existing
			destroyDriver = m.drivers[existing.Driver]
			delete(m.sandboxes, userID)
		} else if existing.GetStatus() == StatusPendingApproval {
			m.mu.Unlock()
			return nil, errors.New("you already have a pending sandbox request awaiting approval; use /sandbox approve or /sandbox deny")
		} else if existing.GetStatus() == StatusCreating {
			m.mu.Unlock()
			return nil, errors.New("a sandbox is currently being created; please wait")
		} else if existing.GetStatus() == StatusExpired {
			toDestroy = existing
			destroyDriver = m.drivers[existing.Driver]
			delete(m.sandboxes, userID)
		}
	}
	m.mu.Unlock()

	if toDestroy != nil && destroyDriver != nil {
		if err := destroyDriver.Destroy(ctx, toDestroy); err != nil {
			slog.Warn("failed to destroy expired sandbox on new request", "user", userID, "error", err)
		}
	}

	// Validate network mode is permitted by server policy
	allowedModes := m.AllowedNetworkModes()
	modeAllowed := false
	for _, mode := range allowedModes {
		if strings.EqualFold(mode, string(params.NetworkMode)) {
			modeAllowed = true
			break
		}
	}
	if !modeAllowed {
		return nil, fmt.Errorf("network mode %q is not permitted by server policy (allowed: %s)", params.NetworkMode, strings.Join(allowedModes, ", "))
	}

	// Validate driver
	available := false
	for _, driverName := range m.cfg.SandboxDrivers {
		if DriverType(strings.TrimSpace(driverName)) == params.Driver {
			if d, ok := m.drivers[params.Driver]; ok && d.Available(ctx) {
				available = true
				break
			}
		}
	}
	if !available {
		return nil, fmt.Errorf("requested driver %q is not available", params.Driver)
	}

	// Validate image if docker driver
	if params.Driver == DriverDocker {
		if err := ValidateImage(params.DockerImage, m.cfg.SandboxAllowedImages); err != nil {
			return nil, err
		}
	}

	// Validate network policy
	netPolicy, err := ValidateNetworkPolicy(NetworkPolicy{
		Mode:         params.NetworkMode,
		AllowedHosts: params.AllowedDomains,
	})
	if err != nil {
		return nil, fmt.Errorf("invalid network configuration: %w", err)
	}

	// Validate mounts relative to user workspace
	userWorkspace, err := m.UserWorkspaceDir(userID)
	if err != nil {
		return nil, fmt.Errorf("invalid user workspace: %w", err)
	}
	for _, mount := range params.Mounts {
		if _, _, err := ValidateMountPath(userWorkspace, mount.RelativePath); err != nil {
			return nil, fmt.Errorf("invalid mount: %w", err)
		}
		if _, err := ValidateSandboxMountPath(mount.SandboxPath); err != nil {
			return nil, fmt.Errorf("invalid sandbox mount path: %w", err)
		}
	}

	// Calculate lifetime
	lifetime := time.Duration(params.LifetimeMinutes) * time.Minute
	if lifetime <= 0 || lifetime > m.cfg.SandboxMaxLifetime {
		lifetime = m.cfg.SandboxMaxLifetime
	}

	now := time.Now()
	sbx := &UserSandbox{
		UserID:       userID,
		ChatID:       chatID,
		Driver:       params.Driver,
		DockerImage:  params.DockerImage,
		Network:      netPolicy,
		Mounts:       params.Mounts,
		Reason:       params.Reason,
		CreatedAt:    now,
		ExpiresAt:    now.Add(lifetime),
		Status:       StatusPendingApproval,
		WorkspaceDir: userWorkspace,
	}

	m.mu.Lock()
	if existing, ok := m.sandboxes[userID]; ok && existing.GetStatus() != StatusNone {
		m.mu.Unlock()
		return nil, errors.New("a sandbox request or instance was started concurrently; please retry")
	}
	m.sandboxes[userID] = sbx
	m.mu.Unlock()
	return sbx, nil
}

// ApproveSandbox transitions a pending sandbox request to running and invokes driver creation.
func (m *Manager) ApproveSandbox(ctx context.Context, userID string) (*UserSandbox, error) {
	m.mu.Lock()
	sbx, ok := m.sandboxes[userID]
	if !ok || sbx.GetStatus() != StatusPendingApproval {
		m.mu.Unlock()
		return nil, errors.New("no pending sandbox request found to approve")
	}

	driver, ok := m.drivers[sbx.Driver]
	if !ok {
		m.mu.Unlock()
		return nil, fmt.Errorf("driver %s not found", sbx.Driver)
	}
	sbx.SetStatus(StatusCreating)
	m.mu.Unlock()

	userWorkspace, err := m.UserWorkspaceDir(userID)
	if err != nil {
		m.mu.Lock()
		delete(m.sandboxes, userID)
		m.mu.Unlock()
		return nil, fmt.Errorf("invalid user workspace: %w", err)
	}
	if err := os.MkdirAll(userWorkspace, 0o755); err != nil {
		m.mu.Lock()
		delete(m.sandboxes, userID)
		m.mu.Unlock()
		return nil, fmt.Errorf("failed to create user workspace: %w", err)
	}

	sbx.SetWorkspaceDir(userWorkspace)
	if err := driver.Create(ctx, sbx, userWorkspace); err != nil {
		if dErr := driver.Destroy(ctx, sbx); dErr != nil {
			slog.Warn("failed to destroy sandbox during create rollback", "user", userID, "error", dErr)
		}
		m.mu.Lock()
		delete(m.sandboxes, userID)
		m.mu.Unlock()
		return nil, fmt.Errorf("failed to initialize sandbox: %w", err)
	}

	m.mu.Lock()
	current, stillExists := m.sandboxes[userID]
	if !stillExists || current != sbx {
		m.mu.Unlock()
		if err := driver.Destroy(ctx, sbx); err != nil {
			slog.Warn("failed to destroy cancelled sandbox", "user", userID, "error", err)
		}
		return nil, errors.New("sandbox creation was cancelled")
	}
	sbx.SetStatus(StatusRunning)
	m.mu.Unlock()

	return sbx, nil
}

// DenySandbox rejects and cancels a user's pending sandbox request.
func (m *Manager) DenySandbox(userID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	sbx, ok := m.sandboxes[userID]
	if !ok || sbx.Status != StatusPendingApproval {
		if ok && sbx.Status == StatusCreating {
			return errors.New("sandbox is currently being created and cannot be denied")
		}
		return errors.New("no pending sandbox request found to deny")
	}

	delete(m.sandboxes, userID)
	return nil
}

// Exec executes a command inside the user's active sandbox.
func (m *Manager) Exec(ctx context.Context, userID string, cmd []string, requestedTimeout time.Duration) (*ExecResult, error) {
	m.mu.Lock()
	sbx, ok := m.sandboxes[userID]
	if !ok || sbx.Status != StatusRunning {
		m.mu.Unlock()
		return nil, errors.New("no active sandbox found for this user; request one first with sandbox_request and approve it")
	}

	// Check expiration
	if time.Now().After(sbx.ExpiresAt) {
		driver := m.drivers[sbx.Driver]
		sbx.Status = StatusExpired
		m.mu.Unlock()
		if driver != nil {
			if err := driver.Destroy(ctx, sbx); err != nil {
				slog.Error("failed to destroy expired sandbox during exec", "user_id", userID, "driver", sbx.Driver, "err", err)
			}
		}
		return nil, errors.New("sandbox lifetime has expired; please request a new sandbox")
	}

	driver, ok := m.drivers[sbx.Driver]
	if !ok {
		m.mu.Unlock()
		return nil, fmt.Errorf("driver %s not available", sbx.Driver)
	}
	m.mu.Unlock()

	// Clamp execution timeout
	execTimeout := requestedTimeout
	if execTimeout <= 0 {
		execTimeout = m.cfg.SandboxDefaultExecTimeout
	}
	if execTimeout < 5*time.Second {
		execTimeout = 5 * time.Second
	}
	if execTimeout > m.cfg.SandboxMaxExecTimeout {
		execTimeout = m.cfg.SandboxMaxExecTimeout
	}

	return driver.Exec(ctx, sbx, cmd, execTimeout)
}

// Destroy terminates and removes the user's active sandbox.
func (m *Manager) Destroy(ctx context.Context, userID string) error {
	m.mu.Lock()
	sbx, ok := m.sandboxes[userID]
	if !ok {
		m.mu.Unlock()
		return nil // Already destroyed
	}
	driver := m.drivers[sbx.Driver]
	m.mu.Unlock()

	if driver != nil && (sbx.GetStatus() == StatusRunning || sbx.GetStatus() == StatusExpired) {
		if err := driver.Destroy(ctx, sbx); err != nil {
			return err
		}
	}

	m.mu.Lock()
	if cur, ok := m.sandboxes[userID]; ok && cur == sbx {
		delete(m.sandboxes, userID)
	}
	m.mu.Unlock()
	return nil
}

// GetStatus returns the sandbox status for the user.
func (m *Manager) GetStatus(userID string) (*UserSandbox, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	sbx, ok := m.sandboxes[userID]
	if !ok {
		return nil, false
	}
	// Copy to prevent data races
	cpy := sbx.Clone()
	if cpy.Status == StatusRunning && time.Now().After(cpy.ExpiresAt) {
		cpy.Status = StatusExpired
	}
	return cpy, true
}

func (m *Manager) reaperLoop() {
	defer m.wg.Done()
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-m.stopCh:
			return
		case <-ticker.C:
			m.pruneExpired()
		}
	}
}

func (m *Manager) pruneExpired() {
	m.mu.Lock()
	now := time.Now()
	var toDestroy []*UserSandbox

	for userID, sbx := range m.sandboxes {
		if sbx.Status == StatusRunning && now.After(sbx.ExpiresAt) {
			toDestroy = append(toDestroy, sbx)
			sbx.Status = StatusExpired
		} else if sbx.Status == StatusExpired && now.After(sbx.ExpiresAt.Add(1*time.Hour)) {
			delete(m.sandboxes, userID)
		}
	}
	m.mu.Unlock()

	if len(toDestroy) == 0 {
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	for _, sbx := range toDestroy {
		if d, ok := m.drivers[sbx.Driver]; ok {
			if err := d.Destroy(ctx, sbx); err != nil {
				slog.Warn("failed to destroy expired sandbox", "user", sbx.UserID, "error", err)
			}
		}
	}
}
