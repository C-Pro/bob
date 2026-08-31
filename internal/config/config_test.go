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

	cfg, err := LoadFromEnv()
	require.NoError(t, err)
	assert.Equal(t, "@bot", cfg.BotHandle)
	assert.Equal(t, "http://127.0.0.1:8080", cfg.BesedkaURL)
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
