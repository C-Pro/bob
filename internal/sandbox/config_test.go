package sandbox

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSandboxConfig_Validate_Defaults(t *testing.T) {
	cfg := DefaultConfig()
	require.NoError(t, cfg.Validate())
}

func TestSandboxConfig_Validate_Disabled(t *testing.T) {
	cfg := Config{Enabled: false}
	require.NoError(t, cfg.Validate())
}

func TestSandboxConfig_Validate_ValidationErrors(t *testing.T) {
	tests := []struct {
		name        string
		modify      func(c *Config)
		expectedErr string
	}{
		{
			name:        "empty drivers",
			modify:      func(c *Config) { c.Drivers = nil },
			expectedErr: "SANDBOX_DRIVERS cannot be empty",
		},
		{
			name:        "empty network modes",
			modify:      func(c *Config) { c.AllowedNetworkModes = nil },
			expectedErr: "SANDBOX_ALLOWED_NETWORK_MODES cannot be empty",
		},
		{
			name:        "invalid network mode",
			modify:      func(c *Config) { c.AllowedNetworkModes = []string{"invalid"} },
			expectedErr: "invalid network mode in SANDBOX_ALLOWED_NETWORK_MODES",
		},
		{
			name:        "non-positive lifetime",
			modify:      func(c *Config) { c.MaxLifetime = 0 },
			expectedErr: "SANDBOX_MAX_LIFETIME must be greater than 0",
		},
		{
			name:        "non-positive default exec timeout",
			modify:      func(c *Config) { c.DefaultExecTimeout = 0 },
			expectedErr: "SANDBOX_DEFAULT_EXEC_TIMEOUT must be greater than 0",
		},
		{
			name:        "max timeout less than default timeout",
			modify:      func(c *Config) { c.MaxExecTimeout = c.DefaultExecTimeout - time.Second },
			expectedErr: "SANDBOX_MAX_EXEC_TIMEOUT cannot be less than SANDBOX_DEFAULT_EXEC_TIMEOUT",
		},
		{
			name:        "non-positive cpu limit",
			modify:      func(c *Config) { c.CPULimit = 0 },
			expectedErr: "SANDBOX_CPU_LIMIT must be greater than 0",
		},
		{
			name:        "non-positive memory limit",
			modify:      func(c *Config) { c.MemoryLimitMB = 0 },
			expectedErr: "SANDBOX_MEMORY_LIMIT_MB must be greater than 0",
		},
		{
			name:        "non-absolute host data dir",
			modify:      func(c *Config) { c.HostDataDir = "relative/path" },
			expectedErr: "SANDBOX_HOST_DATA_DIR must be an absolute path",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := DefaultConfig()
			tc.modify(&cfg)
			err := cfg.Validate()
			require.Error(t, err)
			assert.Contains(t, err.Error(), tc.expectedErr)
		})
	}
}

func TestSandboxConfig_ForwarderConfig(t *testing.T) {
	cfg := Config{
		DataDir:           "/data",
		ProxyFwdPath:      "/bin/fwd",
		AllowRuntimeBuild: BoolPtr(true),
	}
	fwdCfg := cfg.ForwarderConfig()
	assert.Equal(t, "/data", fwdCfg.DataDir)
	assert.Equal(t, "/bin/fwd", fwdCfg.ProxyFwdPath)
	require.NotNil(t, fwdCfg.AllowRuntimeBuild)
	assert.True(t, *fwdCfg.AllowRuntimeBuild)
}
