package config

import (
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLoadFromEnvDefaults(t *testing.T) {
	// Clear relevant env vars
	_ = os.Unsetenv("BOT_HANDLE")
	_ = os.Unsetenv("BESEDKA_URL")
	_ = os.Unsetenv("BESEDKA_API_KEY")
	_ = os.Unsetenv("OPENAI_API_KEY")
	_ = os.Unsetenv("OPENAI_MODEL")
	_ = os.Unsetenv("OPENAI_BASE_URL")
	_ = os.Unsetenv("GEMINI_API_KEY")
	_ = os.Unsetenv("GEMINI_MODEL")
	_ = os.Unsetenv("GEMINI_BASE_URL")
	_ = os.Unsetenv("TOWNHALL_MAX_PARAGRAPHS")
	_ = os.Unsetenv("DM_MAX_PARAGRAPHS")
	_ = os.Unsetenv("MSG_RING_BUFFER_SIZE")
	_ = os.Unsetenv("TAVILY_API_KEY")
	_ = os.Unsetenv("TAVILY_BASE_URL")
	_ = os.Unsetenv("DATA_DIR")
	_ = os.Unsetenv("EMBEDDING_MODEL")
	_ = os.Unsetenv("EMBEDDING_PRECISION")

	_ = os.Unsetenv("SECRET")
	_ = os.Unsetenv("AUTH_SECRET")
	_ = os.Unsetenv("S3_ENDPOINT")
	_ = os.Unsetenv("S3_REGION")
	_ = os.Unsetenv("S3_BUCKET")
	_ = os.Unsetenv("S3_ACCESS_KEY")
	_ = os.Unsetenv("S3_SECRET_KEY")
	_ = os.Unsetenv("S3_PATH_STYLE")
	_ = os.Unsetenv("S3_BACKUP_INTERVAL")
	_ = os.Unsetenv("S3_BACKUP_KEEP")
	_ = os.Unsetenv("S3_BACKUP_PREFIX")
	_ = os.Unsetenv("SANDBOX_ENABLED")
	_ = os.Unsetenv("SANDBOX_DRIVERS")
	_ = os.Unsetenv("SANDBOX_DOCKER_SOCKET")
	_ = os.Unsetenv("SANDBOX_ALLOWED_IMAGES")
	_ = os.Unsetenv("SANDBOX_ALLOWED_NETWORK_MODES")
	_ = os.Unsetenv("SANDBOX_MAX_LIFETIME")
	_ = os.Unsetenv("SANDBOX_DEFAULT_EXEC_TIMEOUT")
	_ = os.Unsetenv("SANDBOX_MAX_EXEC_TIMEOUT")
	_ = os.Unsetenv("SANDBOX_CPU_LIMIT")
	_ = os.Unsetenv("SANDBOX_MEMORY_LIMIT_MB")
	_ = os.Unsetenv("SANDBOX_HOST_DATA_DIR")

	cfg, err := LoadFromEnv()
	require.NoError(t, err)
	assert.Equal(t, "@bot", cfg.BotHandle)
	assert.Equal(t, "http://127.0.0.1:8080", cfg.BesedkaURL)
	assert.Equal(t, "", cfg.BesedkaAPIKey)
	assert.Equal(t, "", cfg.OpenAIAPIKey)
	assert.Equal(t, "gemini-3.7-flash", cfg.OpenAIModel)
	assert.Equal(t, "https://generativelanguage.googleapis.com/v1beta/openai/", cfg.OpenAIBaseURL)
	assert.Equal(t, "", cfg.TavilyAPIKey)
	assert.Equal(t, "https://api.tavily.com", cfg.TavilyBaseURL)
	assert.Equal(t, 2, cfg.TownhallMaxParagraphs)
	assert.Equal(t, 10, cfg.DMMaxParagraphs)
	assert.Equal(t, 100, cfg.MsgRingBufferSize)
	assert.Equal(t, "./data", cfg.DataDir)
	assert.Equal(t, "", cfg.EmbeddingModel)
	assert.Equal(t, "bf16", cfg.EmbeddingPrecision)
	assert.Equal(t, "data/bob.db", cfg.DBPath("bob.db"))
	assert.False(t, cfg.S3Enabled())
	assert.Equal(t, "bob_agent/", cfg.S3BackupPrefix)
	assert.Equal(t, "us-east-1", cfg.S3Region)
	assert.True(t, cfg.S3PathStyle)
	assert.Equal(t, 7, cfg.S3BackupKeep)
	assert.True(t, cfg.SandboxEnabled)
	assert.Equal(t, []string{"bwrap", "docker"}, cfg.SandboxDrivers)
	assert.Equal(t, "/var/run/docker.sock", cfg.SandboxDockerSocket)
	assert.Equal(t, []string{"alpine:latest", "golang:alpine", "python:3.11-slim", "node:20-slim"}, cfg.SandboxAllowedImages)
	assert.Equal(t, []string{"none", "restricted"}, cfg.SandboxAllowedNetworkModes)
	assert.Equal(t, 30*time.Minute, cfg.SandboxMaxLifetime)
	assert.Equal(t, 1*time.Minute, cfg.SandboxDefaultExecTimeout)
	assert.Equal(t, 10*time.Minute, cfg.SandboxMaxExecTimeout)
	assert.Equal(t, 1.0, cfg.SandboxCPULimit)
	assert.Equal(t, 512, cfg.SandboxMemoryLimitMB)
	assert.Equal(t, "", cfg.SandboxHostDataDir)
}

func TestLoadFromEnvStandardOpenAI(t *testing.T) {
	_ = os.Unsetenv("GEMINI_API_KEY")
	_ = os.Unsetenv("GEMINI_MODEL")
	_ = os.Unsetenv("GEMINI_BASE_URL")

	t.Setenv("BOT_HANDLE", "assistant")
	t.Setenv("BESEDKA_URL", "http://besedka.local:8080")
	t.Setenv("OPENAI_API_KEY", "sk-openai-test-key")
	t.Setenv("OPENAI_MODEL", "gpt-4o-mini")
	t.Setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
	t.Setenv("TOWNHALL_MAX_PARAGRAPHS", "3")
	t.Setenv("DM_MAX_PARAGRAPHS", "15")
	t.Setenv("MSG_RING_BUFFER_SIZE", "50")
	t.Setenv("TAVILY_API_KEY", "tvly-test-key-12345")
	t.Setenv("TAVILY_BASE_URL", "https://custom.tavily.api/v1")
	t.Setenv("DATA_DIR", "/var/lib/bob")
	t.Setenv("EMBEDDING_MODEL", "gemini-embedding-2")
	t.Setenv("EMBEDDING_PRECISION", "int8")

	cfg, err := LoadFromEnv()
	require.NoError(t, err)
	assert.Equal(t, "@assistant", cfg.BotHandle)
	assert.Equal(t, "http://besedka.local:8080", cfg.BesedkaURL)
	assert.Equal(t, "sk-openai-test-key", cfg.OpenAIAPIKey)
	assert.Equal(t, "gpt-4o-mini", cfg.OpenAIModel)
	assert.Equal(t, "https://api.openai.com/v1/", cfg.OpenAIBaseURL)
	assert.Equal(t, "tvly-test-key-12345", cfg.TavilyAPIKey)
	assert.Equal(t, "https://custom.tavily.api/v1", cfg.TavilyBaseURL)
	assert.Equal(t, 3, cfg.TownhallMaxParagraphs)
	assert.Equal(t, 15, cfg.DMMaxParagraphs)
	assert.Equal(t, 50, cfg.MsgRingBufferSize)
	assert.Equal(t, "/var/lib/bob", cfg.DataDir)
	assert.Equal(t, "gemini-embedding-2", cfg.EmbeddingModel)
	assert.Equal(t, "int8", cfg.EmbeddingPrecision)
	assert.Equal(t, "/var/lib/bob/custom.db", cfg.DBPath("custom.db"))
}

func TestLoadFromEnvS3(t *testing.T) {
	t.Setenv("SECRET", "custom-secret")
	t.Setenv("S3_ENDPOINT", "http://127.0.0.1:9000")
	t.Setenv("S3_REGION", "eu-central-1")
	t.Setenv("S3_BUCKET", "bob-backups")
	t.Setenv("S3_ACCESS_KEY", "access-123")
	t.Setenv("S3_SECRET_KEY", "secret-456")
	t.Setenv("S3_PATH_STYLE", "false")
	t.Setenv("S3_BACKUP_INTERVAL", "2h")
	t.Setenv("S3_BACKUP_KEEP", "14")
	t.Setenv("S3_BACKUP_PREFIX", "custom_bob")

	cfg, err := LoadFromEnv()
	require.NoError(t, err)
	assert.True(t, cfg.S3Enabled())
	assert.Equal(t, "custom-secret", cfg.Secret)
	assert.Equal(t, "http://127.0.0.1:9000", cfg.S3Endpoint)
	assert.Equal(t, "eu-central-1", cfg.S3Region)
	assert.Equal(t, "bob-backups", cfg.S3Bucket)
	assert.Equal(t, "access-123", cfg.S3AccessKey)
	assert.Equal(t, "secret-456", cfg.S3SecretKey)
	assert.False(t, cfg.S3PathStyle)
	assert.Equal(t, 2*time.Hour, cfg.S3BackupInterval)
	assert.Equal(t, 14, cfg.S3BackupKeep)
	assert.Equal(t, "custom_bob/", cfg.S3BackupPrefix)
}

func TestLoadFromEnvGeminiFallback(t *testing.T) {
	_ = os.Unsetenv("OPENAI_API_KEY")
	_ = os.Unsetenv("OPENAI_MODEL")
	_ = os.Unsetenv("OPENAI_BASE_URL")

	t.Setenv("BOT_HANDLE", "assistant")
	t.Setenv("BESEDKA_URL", "http://besedka.local:8080")
	t.Setenv("GEMINI_API_KEY", "gemini-legacy-key")
	t.Setenv("GEMINI_MODEL", "gemini-2.5-pro")
	t.Setenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")

	cfg, err := LoadFromEnv()
	require.NoError(t, err)
	assert.Equal(t, "gemini-legacy-key", cfg.OpenAIAPIKey)
	assert.Equal(t, "gemini-legacy-key", cfg.GeminiAPIKey)
	assert.Equal(t, "gemini-2.5-pro", cfg.OpenAIModel)
	assert.Equal(t, "gemini-2.5-pro", cfg.GeminiModel)
	assert.Equal(t, "https://generativelanguage.googleapis.com/v1beta/openai/", cfg.OpenAIBaseURL)
	assert.Equal(t, "https://generativelanguage.googleapis.com/v1beta/openai/", cfg.GeminiBaseURL)
}

func TestConfigValidation(t *testing.T) {
	cfg := &Config{
		BesedkaURL:            "http://localhost:8080",
		TownhallMaxParagraphs: 2,
		DMMaxParagraphs:       10,
		MsgRingBufferSize:     100,
	}

	err := cfg.Validate(true)
	assert.ErrorContains(t, err, "OPENAI_API_KEY (or GEMINI_API_KEY) is required")

	cfg.OpenAIAPIKey = "valid-key"
	err = cfg.Validate(true)
	assert.NoError(t, err)

	cfg.BesedkaURL = ""
	err = cfg.Validate(false)
	assert.ErrorContains(t, err, "BESEDKA_URL cannot be empty")

	cfg.BesedkaURL = "http://localhost:8080"
	cfg.MsgRingBufferSize = 0
	err = cfg.Validate(false)
	assert.ErrorContains(t, err, "invalid MSG_RING_BUFFER_SIZE: 0")

	// S3 validation tests
	cfg.MsgRingBufferSize = 100
	cfg.S3Bucket = "my-bucket"
	cfg.S3Endpoint = ""
	err = cfg.Validate(false)
	assert.ErrorContains(t, err, "S3_BUCKET and S3_ENDPOINT must be set together")

	cfg.S3Endpoint = "http://localhost:9000"
	err = cfg.Validate(false)
	assert.ErrorContains(t, err, "SECRET (or AUTH_SECRET) is required")

	cfg.Secret = "my-secret-key"
	err = cfg.Validate(false)
	assert.ErrorContains(t, err, "S3_ACCESS_KEY and S3_SECRET_KEY are required")

	cfg.S3AccessKey = "access"
	cfg.S3SecretKey = "secret"
	cfg.S3BackupInterval = 1 * time.Hour
	cfg.S3BackupKeep = 0
	err = cfg.Validate(false)
	assert.ErrorContains(t, err, "S3_BACKUP_KEEP must be greater than 0")

	cfg.S3BackupKeep = 7
	cfg.S3BackupInterval = 0
	err = cfg.Validate(false)
	assert.ErrorContains(t, err, "S3_BACKUP_INTERVAL must be greater than 0")

	cfg.S3BackupInterval = 1 * time.Hour
	err = cfg.Validate(false)
	assert.NoError(t, err)
}

func TestLoadDotEnv(t *testing.T) {
	tempFile, err := os.CreateTemp(t.TempDir(), ".env*")
	require.NoError(t, err)
	defer func() { _ = tempFile.Close() }()

	content := `
# Comment line
PLAIN_KEY=plain_val
export EXPORTED_KEY=exported_val
export DOUBLE_QUOTED="hello world"
export SINGLE_QUOTED='single world'
export PRE_EXISTING=new_val
`
	_, err = tempFile.WriteString(content)
	require.NoError(t, err)

	_ = os.Unsetenv("PLAIN_KEY")
	_ = os.Unsetenv("EXPORTED_KEY")
	_ = os.Unsetenv("DOUBLE_QUOTED")
	_ = os.Unsetenv("SINGLE_QUOTED")
	t.Setenv("PRE_EXISTING", "original_val")

	LoadDotEnv(tempFile.Name())

	assert.Equal(t, "plain_val", os.Getenv("PLAIN_KEY"))
	assert.Equal(t, "exported_val", os.Getenv("EXPORTED_KEY"))
	assert.Equal(t, "hello world", os.Getenv("DOUBLE_QUOTED"))
	assert.Equal(t, "single world", os.Getenv("SINGLE_QUOTED"))
	assert.Equal(t, "original_val", os.Getenv("PRE_EXISTING"))
}

func TestSandboxConfigValidation(t *testing.T) {
	cfg := &Config{
		OpenAIAPIKey:               "test-key",
		BesedkaURL:                 "http://127.0.0.1:8080",
		TownhallMaxParagraphs:      2,
		DMMaxParagraphs:            10,
		MsgRingBufferSize:          100,
		SandboxEnabled:             true,
		SandboxDrivers:             []string{"bwrap"},
		SandboxAllowedNetworkModes: []string{"none", "restricted"},
		SandboxMaxLifetime:         30 * time.Minute,
		SandboxDefaultExecTimeout:  1 * time.Minute,
		SandboxMaxExecTimeout:      10 * time.Minute,
		SandboxCPULimit:            1.0,
		SandboxMemoryLimitMB:       512,
	}
	require.NoError(t, cfg.Validate(true))

	// Drivers cannot be empty
	cfg.SandboxDrivers = nil
	err := cfg.Validate(true)
	assert.ErrorContains(t, err, "SANDBOX_DRIVERS cannot be empty")
	cfg.SandboxDrivers = []string{"docker"}

	// Allowed network modes cannot be empty
	cfg.SandboxAllowedNetworkModes = nil
	err = cfg.Validate(true)
	assert.ErrorContains(t, err, "SANDBOX_ALLOWED_NETWORK_MODES cannot be empty")
	cfg.SandboxAllowedNetworkModes = []string{"invalid_mode"}
	err = cfg.Validate(true)
	assert.ErrorContains(t, err, "invalid network mode in SANDBOX_ALLOWED_NETWORK_MODES")
	cfg.SandboxAllowedNetworkModes = []string{"none", "restricted"}

	// Max lifetime > 0
	cfg.SandboxMaxLifetime = 0
	err = cfg.Validate(true)
	assert.ErrorContains(t, err, "SANDBOX_MAX_LIFETIME must be greater than 0")
	cfg.SandboxMaxLifetime = 30 * time.Minute

	// Default exec timeout > 0
	cfg.SandboxDefaultExecTimeout = 0
	err = cfg.Validate(true)
	assert.ErrorContains(t, err, "SANDBOX_DEFAULT_EXEC_TIMEOUT must be greater than 0")
	cfg.SandboxDefaultExecTimeout = 1 * time.Minute

	// Max exec timeout >= default
	cfg.SandboxMaxExecTimeout = 30 * time.Second
	err = cfg.Validate(true)
	assert.ErrorContains(t, err, "SANDBOX_MAX_EXEC_TIMEOUT cannot be less than SANDBOX_DEFAULT_EXEC_TIMEOUT")
	cfg.SandboxMaxExecTimeout = 10 * time.Minute

	// CPU limit > 0
	cfg.SandboxCPULimit = 0
	err = cfg.Validate(true)
	assert.ErrorContains(t, err, "SANDBOX_CPU_LIMIT must be greater than 0")
	cfg.SandboxCPULimit = 1.5

	// Memory limit > 0
	cfg.SandboxMemoryLimitMB = 0
	err = cfg.Validate(true)
	assert.ErrorContains(t, err, "SANDBOX_MEMORY_LIMIT_MB must be greater than 0")
	cfg.SandboxMemoryLimitMB = 1024

	// Valid now
	assert.NoError(t, cfg.Validate(true))

	// When disabled, invalid sandbox values are ignored
	cfg.SandboxEnabled = false
	cfg.SandboxDrivers = nil
	cfg.SandboxAllowedNetworkModes = nil
	assert.NoError(t, cfg.Validate(true))
}

func TestSandboxCustomEnv(t *testing.T) {
	t.Setenv("SANDBOX_ENABLED", "false")
	t.Setenv("SANDBOX_DRIVERS", "docker, bwrap")
	t.Setenv("SANDBOX_DOCKER_SOCKET", "/custom/docker.sock")
	t.Setenv("SANDBOX_ALLOWED_IMAGES", "custom/image:tag, other:latest")
	t.Setenv("SANDBOX_ALLOWED_NETWORK_MODES", "none, restricted, full")
	t.Setenv("SANDBOX_MAX_LIFETIME", "1h")
	t.Setenv("SANDBOX_DEFAULT_EXEC_TIMEOUT", "2m")
	t.Setenv("SANDBOX_MAX_EXEC_TIMEOUT", "15m")
	t.Setenv("SANDBOX_CPU_LIMIT", "2.5")
	t.Setenv("SANDBOX_MEMORY_LIMIT_MB", "1024")

	cfg, err := LoadFromEnv()
	require.NoError(t, err)
	assert.False(t, cfg.SandboxEnabled)
	assert.Equal(t, []string{"docker", "bwrap"}, cfg.SandboxDrivers)
	assert.Equal(t, "/custom/docker.sock", cfg.SandboxDockerSocket)
	assert.Equal(t, []string{"custom/image:tag", "other:latest"}, cfg.SandboxAllowedImages)
	assert.Equal(t, []string{"none", "restricted", "full"}, cfg.SandboxAllowedNetworkModes)
	assert.Equal(t, 1*time.Hour, cfg.SandboxMaxLifetime)
	assert.Equal(t, 2*time.Minute, cfg.SandboxDefaultExecTimeout)
	assert.Equal(t, 15*time.Minute, cfg.SandboxMaxExecTimeout)
	assert.Equal(t, 2.5, cfg.SandboxCPULimit)
	assert.Equal(t, 1024, cfg.SandboxMemoryLimitMB)
}

func TestLoadFromEnv_SandboxEnabledParsing(t *testing.T) {
	falsyValues := []string{"false", "0", "no", "n", "off", "disable", "disabled", "OFF", "Disabled"}
	for _, v := range falsyValues {
		t.Setenv("SANDBOX_ENABLED", v)
		cfg, err := LoadFromEnv()
		require.NoErrorf(t, err, "SANDBOX_ENABLED=%q should parse as false", v)
		assert.Falsef(t, cfg.SandboxEnabled, "SANDBOX_ENABLED=%q must result in false", v)
	}

	truthyValues := []string{"true", "1", "yes", "y", "on", "enable", "enabled", "ON", "Enabled"}
	for _, v := range truthyValues {
		t.Setenv("SANDBOX_ENABLED", v)
		cfg, err := LoadFromEnv()
		require.NoErrorf(t, err, "SANDBOX_ENABLED=%q should parse as true", v)
		assert.Truef(t, cfg.SandboxEnabled, "SANDBOX_ENABLED=%q must result in true", v)
	}

	// Invalid value must return an error and fail closed
	t.Setenv("SANDBOX_ENABLED", "invalid_value")
	_, err := LoadFromEnv()
	require.Error(t, err)
	assert.Contains(t, err.Error(), "invalid boolean value \"invalid_value\" for SANDBOX_ENABLED")
}

func TestLoadFromEnv_SandboxHostDataDir(t *testing.T) {
	t.Setenv("SANDBOX_HOST_DATA_DIR", "/var/besedka-data/bob/")
	cfg, err := LoadFromEnv()
	require.NoError(t, err)
	assert.Equal(t, "/var/besedka-data/bob", cfg.SandboxHostDataDir)
}

func TestConfig_Validate_SandboxHostDataDir(t *testing.T) {
	baseCfg := Config{
		OpenAIAPIKey:               "test-key",
		BesedkaURL:                 "http://127.0.0.1:8080",
		TownhallMaxParagraphs:      2,
		DMMaxParagraphs:            10,
		MsgRingBufferSize:          100,
		SandboxEnabled:             true,
		SandboxDrivers:             []string{"docker"},
		SandboxAllowedNetworkModes: []string{"none"},
		SandboxMaxLifetime:         30 * time.Minute,
		SandboxDefaultExecTimeout:  1 * time.Minute,
		SandboxMaxExecTimeout:      10 * time.Minute,
		SandboxCPULimit:            1.0,
		SandboxMemoryLimitMB:       512,
	}

	// Valid absolute path passes validation
	cfgValid := baseCfg
	cfgValid.SandboxHostDataDir = "/tmp/besedka-data/opt/besedka/bob"
	require.NoError(t, cfgValid.Validate(true))

	// Empty path passes validation
	cfgEmpty := baseCfg
	cfgEmpty.SandboxHostDataDir = ""
	require.NoError(t, cfgEmpty.Validate(true))

	// Relative path fails validation
	cfgInvalid := baseCfg
	cfgInvalid.SandboxHostDataDir = "relative/path"
	err := cfgInvalid.Validate(true)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "SANDBOX_HOST_DATA_DIR must be an absolute path")
}
