package sandbox

import (
	"context"
	"sync"
	"time"
)

// DriverType identifies the underlying isolation technology.
type DriverType string

const (
	DriverBwrap  DriverType = "bwrap"
	DriverDocker DriverType = "docker"
)

// NetworkMode defines the network isolation level.
type NetworkMode string

const (
	NetworkNone       NetworkMode = "none"
	NetworkRestricted NetworkMode = "restricted"
	NetworkFull       NetworkMode = "full"
)

// NetworkPolicy configures network access for a sandbox.
type NetworkPolicy struct {
	Mode         NetworkMode
	AllowedHosts []string // e.g. ["pypi.org", "files.pythonhosted.org"]
	BlockedHosts []string // e.g. ["metadata.google.internal"]
	BlockedCIDRs []string // e.g. ["169.254.169.254/32"]
}

// UserMount defines a mount point restricted to the user's private workspace.
type UserMount struct {
	RelativePath string // Path relative to user workspace (e.g. "project/data")
	SandboxPath  string // Path inside sandbox (e.g. "/workspace/data")
	ReadOnly     bool
}

// DevicePolicy defines hardware/device access (reserved for future GPU support).
type DevicePolicy struct {
	AllowGPU    bool
	DevicePaths []string
}

// SandboxStatus represents the lifecycle state of a user's sandbox.
type SandboxStatus string

const (
	StatusNone            SandboxStatus = "none"
	StatusPendingApproval SandboxStatus = "pending_approval"
	StatusCreating        SandboxStatus = "creating"
	StatusRunning         SandboxStatus = "running"
	StatusExpired         SandboxStatus = "expired"
)

// UserSandbox holds the state and configuration of a user's single sandbox.
type UserSandbox struct {
	mu           sync.RWMutex
	UserID       string
	ChatID       string
	Driver       DriverType
	DockerImage  string
	Network      NetworkPolicy
	Mounts       []UserMount
	Devices      DevicePolicy
	Reason       string
	CreatedAt    time.Time
	ExpiresAt    time.Time
	Status       SandboxStatus
	InternalID   string // Driver-specific handle (e.g. docker container ID)
	WorkspaceDir string // Absolute path to user sandbox workspace on host
}

// GetInternalID returns the internal driver handle safely under read lock.
func (s *UserSandbox) GetInternalID() string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.InternalID
}

// SetInternalID sets the internal driver handle safely under write lock.
func (s *UserSandbox) SetInternalID(id string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.InternalID = id
}

// Clone returns a shallow copy of UserSandbox with its own mutex.
func (s *UserSandbox) Clone() *UserSandbox {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return &UserSandbox{
		UserID:       s.UserID,
		ChatID:       s.ChatID,
		Driver:       s.Driver,
		DockerImage:  s.DockerImage,
		Network:      s.Network,
		Mounts:       s.Mounts,
		Devices:      s.Devices,
		Reason:       s.Reason,
		CreatedAt:    s.CreatedAt,
		ExpiresAt:    s.ExpiresAt,
		Status:       s.Status,
		InternalID:   s.InternalID,
		WorkspaceDir: s.WorkspaceDir,
	}
}

// ExecResult contains the output and status of a command executed in a sandbox.
type ExecResult struct {
	ExitCode int
	Stdout   string
	Stderr   string
	Duration time.Duration
}

// Driver defines the interface for sandbox execution engines.
type Driver interface {
	Type() DriverType
	Available(ctx context.Context) bool
	Create(ctx context.Context, sbx *UserSandbox, userWorkspaceDir string) error
	Exec(ctx context.Context, sbx *UserSandbox, cmd []string, timeout time.Duration) (*ExecResult, error)
	Destroy(ctx context.Context, sbx *UserSandbox) error
}
