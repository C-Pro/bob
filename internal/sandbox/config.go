package sandbox

import (
	"errors"
	"fmt"
	"path/filepath"
	"strings"
	"time"
)

// DefaultSandboxAllowedImages defines standard safe container images.
var DefaultSandboxAllowedImages = []string{"alpine:latest", "golang:alpine", "python:3.11-slim", "node:20-slim"}

// DefaultSandboxAllowedNetworkModes defines standard safe network modes (full is disabled by default).
var DefaultSandboxAllowedNetworkModes = []string{"none", "restricted"}

// BoolPtr returns a pointer to the given bool value.
func BoolPtr(b bool) *bool {
	return &b
}

// ForwarderConfig defines settings for locating and preparing the proxy forwarder binary.
type ForwarderConfig struct {
	DataDir           string
	ProxyFwdPath      string
	AllowRuntimeBuild *bool
}

// Config holds package-level configuration parameters for the sandbox subsystem.
type Config struct {
	Enabled             bool
	Drivers             []string
	DockerSocket        string
	AllowedImages       []string
	AllowedNetworkModes []string
	MaxLifetime         time.Duration
	DefaultExecTimeout  time.Duration
	MaxExecTimeout      time.Duration
	CPULimit            float64
	MemoryLimitMB       int
	DataDir             string
	HostDataDir         string
	ProxyFwdPath        string
	AllowRuntimeBuild   *bool
}

// DefaultConfig returns a sandbox.Config initialized with safe defaults.
func DefaultConfig() Config {
	return Config{
		Enabled:             true,
		Drivers:             []string{"bwrap", "docker"},
		DockerSocket:        "/var/run/docker.sock",
		AllowedImages:       DefaultSandboxAllowedImages,
		AllowedNetworkModes: DefaultSandboxAllowedNetworkModes,
		MaxLifetime:         30 * time.Minute,
		DefaultExecTimeout:  1 * time.Minute,
		MaxExecTimeout:      10 * time.Minute,
		CPULimit:            1.0,
		MemoryLimitMB:       512,
		DataDir:             "./data",
	}
}

// ForwarderConfig derives a ForwarderConfig from the sandbox Config.
func (c *Config) ForwarderConfig() ForwarderConfig {
	return ForwarderConfig{
		DataDir:           c.DataDir,
		ProxyFwdPath:      c.ProxyFwdPath,
		AllowRuntimeBuild: c.AllowRuntimeBuild,
	}
}

// Validate validates the sandbox configuration fields.
func (c *Config) Validate() error {
	if !c.Enabled {
		return nil
	}
	if len(c.Drivers) == 0 {
		return errors.New("SANDBOX_DRIVERS cannot be empty when sandbox is enabled")
	}
	if len(c.AllowedNetworkModes) == 0 {
		return errors.New("SANDBOX_ALLOWED_NETWORK_MODES cannot be empty when sandbox is enabled")
	}
	for _, m := range c.AllowedNetworkModes {
		lower := strings.ToLower(strings.TrimSpace(m))
		if lower != string(NetworkNone) && lower != string(NetworkRestricted) && lower != string(NetworkFull) {
			return fmt.Errorf("invalid network mode in SANDBOX_ALLOWED_NETWORK_MODES: %s", m)
		}
	}
	if c.MaxLifetime <= 0 {
		return errors.New("SANDBOX_MAX_LIFETIME must be greater than 0")
	}
	if c.DefaultExecTimeout <= 0 {
		return errors.New("SANDBOX_DEFAULT_EXEC_TIMEOUT must be greater than 0")
	}
	if c.MaxExecTimeout < c.DefaultExecTimeout {
		return errors.New("SANDBOX_MAX_EXEC_TIMEOUT cannot be less than SANDBOX_DEFAULT_EXEC_TIMEOUT")
	}
	if c.CPULimit <= 0 {
		return errors.New("SANDBOX_CPU_LIMIT must be greater than 0")
	}
	if c.MemoryLimitMB <= 0 {
		return errors.New("SANDBOX_MEMORY_LIMIT_MB must be greater than 0")
	}
	if c.HostDataDir != "" && !filepath.IsAbs(c.HostDataDir) {
		return errors.New("SANDBOX_HOST_DATA_DIR must be an absolute path")
	}
	return nil
}
