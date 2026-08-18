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
	_ = os.Unsetenv("GEMINI_API_KEY")
	_ = os.Unsetenv("GEMINI_MODEL")
	_ = os.Unsetenv("GEMINI_BASE_URL")
	_ = os.Unsetenv("TOWNHALL_MAX_PARAGRAPHS")
	_ = os.Unsetenv("DM_MAX_PARAGRAPHS")
	_ = os.Unsetenv("MSG_RING_BUFFER_SIZE")

	cfg, err := LoadFromEnv()
	require.NoError(t, err)
	assert.Equal(t, "@bot", cfg.BotHandle)
	assert.Equal(t, "http://127.0.0.1:8080", cfg.BesedkaURL)
	assert.Equal(t, "gemini-3.7-flash", cfg.GeminiModel)
	assert.Equal(t, "https://generativelanguage.googleapis.com/v1beta/openai/", cfg.GeminiBaseURL)
	assert.Equal(t, 2, cfg.TownhallMaxParagraphs)
	assert.Equal(t, 10, cfg.DMMaxParagraphs)
	assert.Equal(t, 100, cfg.MsgRingBufferSize)
}

func TestLoadFromEnvCustom(t *testing.T) {
	t.Setenv("BOT_HANDLE", "assistant")
	t.Setenv("BESEDKA_URL", "http://besedka.local:8080")
	t.Setenv("GEMINI_API_KEY", "test-key-123")
	t.Setenv("GEMINI_MODEL", "gemini-2.5-pro")
	t.Setenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")
	t.Setenv("TOWNHALL_MAX_PARAGRAPHS", "3")
	t.Setenv("DM_MAX_PARAGRAPHS", "15")
	t.Setenv("MSG_RING_BUFFER_SIZE", "50")

	cfg, err := LoadFromEnv()
	require.NoError(t, err)
	assert.Equal(t, "@assistant", cfg.BotHandle)
	assert.Equal(t, "http://besedka.local:8080", cfg.BesedkaURL)
	assert.Equal(t, "test-key-123", cfg.GeminiAPIKey)
	assert.Equal(t, "gemini-2.5-pro", cfg.GeminiModel)
	assert.Equal(t, "https://generativelanguage.googleapis.com/v1beta/openai/", cfg.GeminiBaseURL)
	assert.Equal(t, 3, cfg.TownhallMaxParagraphs)
	assert.Equal(t, 15, cfg.DMMaxParagraphs)
	assert.Equal(t, 50, cfg.MsgRingBufferSize)
}

func TestConfigValidation(t *testing.T) {
	cfg := &Config{
		BesedkaURL:            "http://localhost:8080",
		TownhallMaxParagraphs: 2,
		DMMaxParagraphs:       10,
		MsgRingBufferSize:     100,
	}

	err := cfg.Validate(true)
	assert.ErrorContains(t, err, "GEMINI_API_KEY is required")

	cfg.GeminiAPIKey = "valid-key"
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
