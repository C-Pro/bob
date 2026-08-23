package config

import (
	"os"
	"testing"

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
}
