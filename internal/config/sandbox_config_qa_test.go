package config

import (
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestSandboxConfig_BoundaryValidation verifies exact boundary conditions for sandbox configuration parameters.
func TestSandboxConfig_BoundaryValidation(t *testing.T) {
	baseCfg := func() *Config {
		return &Config{
			OpenAIAPIKey:               "test-key",
			BesedkaURL:                 "http://127.0.0.1:8080",
			TownhallMaxParagraphs:      2,
			DMMaxParagraphs:            10,
			MsgRingBufferSize:          100,
			SandboxEnabled:             true,
			SandboxDrivers:             []string{"bwrap", "docker"},
			SandboxAllowedNetworkModes: []string{"none", "restricted"},
			SandboxMaxLifetime:         30 * time.Minute,
			SandboxDefaultExecTimeout:  1 * time.Minute,
			SandboxMaxExecTimeout:      10 * time.Minute,
			SandboxCPULimit:            1.0,
			SandboxMemoryLimitMB:       512,
		}
	}

	t.Run("max exec timeout equals default exec timeout is valid", func(t *testing.T) {
		cfg := baseCfg()
		cfg.SandboxDefaultExecTimeout = 5 * time.Minute
		cfg.SandboxMaxExecTimeout = 5 * time.Minute
		assert.NoError(t, cfg.Validate(true))
	})

	t.Run("max exec timeout less than default exec timeout is rejected", func(t *testing.T) {
		cfg := baseCfg()
		cfg.SandboxDefaultExecTimeout = 5 * time.Minute
		cfg.SandboxMaxExecTimeout = 5*time.Minute - 1*time.Nanosecond
		err := cfg.Validate(true)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "SANDBOX_MAX_EXEC_TIMEOUT cannot be less than SANDBOX_DEFAULT_EXEC_TIMEOUT")
	})

	t.Run("sandbox max lifetime boundaries", func(t *testing.T) {
		cfg := baseCfg()
		cfg.SandboxMaxLifetime = 0
		assert.ErrorContains(t, cfg.Validate(true), "SANDBOX_MAX_LIFETIME must be greater than 0")

		cfg.SandboxMaxLifetime = -1 * time.Second
		assert.ErrorContains(t, cfg.Validate(true), "SANDBOX_MAX_LIFETIME must be greater than 0")

		cfg.SandboxMaxLifetime = 1 * time.Nanosecond
		assert.NoError(t, cfg.Validate(true))
	})

	t.Run("sandbox default exec timeout boundaries", func(t *testing.T) {
		cfg := baseCfg()
		cfg.SandboxDefaultExecTimeout = 0
		assert.ErrorContains(t, cfg.Validate(true), "SANDBOX_DEFAULT_EXEC_TIMEOUT must be greater than 0")

		cfg.SandboxDefaultExecTimeout = -1 * time.Millisecond
		assert.ErrorContains(t, cfg.Validate(true), "SANDBOX_DEFAULT_EXEC_TIMEOUT must be greater than 0")

		cfg.SandboxDefaultExecTimeout = 1 * time.Millisecond
		cfg.SandboxMaxExecTimeout = 1 * time.Millisecond
		assert.NoError(t, cfg.Validate(true))
	})

	t.Run("sandbox CPU limit boundaries", func(t *testing.T) {
		cfg := baseCfg()
		cfg.SandboxCPULimit = 0.0
		assert.ErrorContains(t, cfg.Validate(true), "SANDBOX_CPU_LIMIT must be greater than 0")

		cfg.SandboxCPULimit = -0.5
		assert.ErrorContains(t, cfg.Validate(true), "SANDBOX_CPU_LIMIT must be greater than 0")

		cfg.SandboxCPULimit = 0.001
		assert.NoError(t, cfg.Validate(true))

		cfg.SandboxCPULimit = 128.0
		assert.NoError(t, cfg.Validate(true))
	})

	t.Run("sandbox memory limit boundaries", func(t *testing.T) {
		cfg := baseCfg()
		cfg.SandboxMemoryLimitMB = 0
		assert.ErrorContains(t, cfg.Validate(true), "SANDBOX_MEMORY_LIMIT_MB must be greater than 0")

		cfg.SandboxMemoryLimitMB = -512
		assert.ErrorContains(t, cfg.Validate(true), "SANDBOX_MEMORY_LIMIT_MB must be greater than 0")

		cfg.SandboxMemoryLimitMB = 1
		assert.NoError(t, cfg.Validate(true))

		cfg.SandboxMemoryLimitMB = 131072
		assert.NoError(t, cfg.Validate(true))
	})

	t.Run("sandbox network modes case insensitivity and whitespace tolerance", func(t *testing.T) {
		cfg := baseCfg()
		cfg.SandboxAllowedNetworkModes = []string{"  NONE  ", "Restricted", "FULL"}
		assert.NoError(t, cfg.Validate(true))
	})

	t.Run("sandbox network modes invalid values", func(t *testing.T) {
		invalidModes := []string{
			"bridge",
			"host",
			"macvlan",
			"isolated",
			"none,restricted",
			"",
			"   ",
		}
		for _, m := range invalidModes {
			cfg := baseCfg()
			cfg.SandboxAllowedNetworkModes = []string{m}
			err := cfg.Validate(true)
			require.Error(t, err, "mode %q should have been rejected", m)
			assert.Contains(t, err.Error(), "invalid network mode in SANDBOX_ALLOWED_NETWORK_MODES")
		}
	})

	t.Run("sandbox disabled skips all sandbox validations", func(t *testing.T) {
		cfg := baseCfg()
		cfg.SandboxEnabled = false
		cfg.SandboxDrivers = nil
		cfg.SandboxAllowedNetworkModes = []string{"completely_invalid"}
		cfg.SandboxMaxLifetime = -100 * time.Hour
		cfg.SandboxDefaultExecTimeout = -10 * time.Minute
		cfg.SandboxMaxExecTimeout = -50 * time.Minute
		cfg.SandboxCPULimit = -999.0
		cfg.SandboxMemoryLimitMB = -99999
		assert.NoError(t, cfg.Validate(true))
	})
}

// TestSandboxConfig_LoadFromEnvFallbacks verifies robust fallback behavior when environment
// variables contain invalid, negative, or malformed values.
func TestSandboxConfig_LoadFromEnvFallbacks(t *testing.T) {
	// Clean environment
	envVars := []string{
		"SANDBOX_ENABLED", "SANDBOX_DRIVERS", "SANDBOX_DOCKER_SOCKET",
		"SANDBOX_ALLOWED_IMAGES", "SANDBOX_ALLOWED_NETWORK_MODES",
		"SANDBOX_MAX_LIFETIME", "SANDBOX_DEFAULT_EXEC_TIMEOUT",
		"SANDBOX_MAX_EXEC_TIMEOUT", "SANDBOX_CPU_LIMIT", "SANDBOX_MEMORY_LIMIT_MB",
	}
	for _, k := range envVars {
		_ = os.Unsetenv(k)
	}

	t.Run("fallback on invalid durations", func(t *testing.T) {
		t.Setenv("SANDBOX_MAX_LIFETIME", "not-a-duration")
		t.Setenv("SANDBOX_DEFAULT_EXEC_TIMEOUT", "-5m")
		t.Setenv("SANDBOX_MAX_EXEC_TIMEOUT", "0s")

		cfg, err := LoadFromEnv()
		require.NoError(t, err)
		assert.Equal(t, 30*time.Minute, cfg.SandboxMaxLifetime)
		assert.Equal(t, 1*time.Minute, cfg.SandboxDefaultExecTimeout)
		assert.Equal(t, 10*time.Minute, cfg.SandboxMaxExecTimeout)
	})

	t.Run("fallback on invalid numeric limits", func(t *testing.T) {
		t.Setenv("SANDBOX_CPU_LIMIT", "-2.5")
		t.Setenv("SANDBOX_MEMORY_LIMIT_MB", "-1024")

		cfg, err := LoadFromEnv()
		require.NoError(t, err)
		assert.Equal(t, 1.0, cfg.SandboxCPULimit)
		assert.Equal(t, 512, cfg.SandboxMemoryLimitMB)

		t.Setenv("SANDBOX_CPU_LIMIT", "invalid-float")
		t.Setenv("SANDBOX_MEMORY_LIMIT_MB", "invalid-int")

		cfg2, err2 := LoadFromEnv()
		require.NoError(t, err2)
		assert.Equal(t, 1.0, cfg2.SandboxCPULimit)
		assert.Equal(t, 512, cfg2.SandboxMemoryLimitMB)
	})

	t.Run("fallback on empty slices", func(t *testing.T) {
		t.Setenv("SANDBOX_DRIVERS", "")
		t.Setenv("SANDBOX_ALLOWED_IMAGES", "")
		t.Setenv("SANDBOX_ALLOWED_NETWORK_MODES", "")

		cfg, err := LoadFromEnv()
		require.NoError(t, err)
		assert.Equal(t, []string{"bwrap", "docker"}, cfg.SandboxDrivers)
		assert.Equal(t, DefaultSandboxAllowedImages, cfg.SandboxAllowedImages)
		assert.Equal(t, DefaultSandboxAllowedNetworkModes, cfg.SandboxAllowedNetworkModes)
	})
}
