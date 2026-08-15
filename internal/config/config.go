package config

import (
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"
)

// Config holds runtime configuration settings for the agent.
type Config struct {
	BotHandle             string
	BesedkaURL            string
	BesedkaAPIKey         string
	GeminiAPIKey          string
	GeminiModel           string
	GeminiBaseURL         string
	TownhallMaxParagraphs int
	DMMaxParagraphs       int
}

// LoadFromEnv loads configuration from environment variables (or .env file) with sensible defaults.
func LoadFromEnv() (*Config, error) {
	LoadDotEnv(".env")

	cfg := &Config{
		BotHandle:             getEnvOrDefault("BOT_HANDLE", "@bot"),
		BesedkaURL:            getEnvOrDefault("BESEDKA_URL", "http://localhost:8080"),
		BesedkaAPIKey:         os.Getenv("BESEDKA_API_KEY"),
		GeminiAPIKey:          os.Getenv("GEMINI_API_KEY"),
		GeminiModel:           getEnvOrDefault("GEMINI_MODEL", "gemini-3.7-flash"),
		GeminiBaseURL:         getEnvOrDefault("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"),
		TownhallMaxParagraphs: getEnvIntOrDefault("TOWNHALL_MAX_PARAGRAPHS", 2),
		DMMaxParagraphs:       getEnvIntOrDefault("DM_MAX_PARAGRAPHS", 10),
	}

	// Normalize bot handle to ensure it starts with @
	if !strings.HasPrefix(cfg.BotHandle, "@") {
		cfg.BotHandle = "@" + cfg.BotHandle
	}

	// Normalize base URL to ensure trailing slash
	if !strings.HasSuffix(cfg.GeminiBaseURL, "/") {
		cfg.GeminiBaseURL = cfg.GeminiBaseURL + "/"
	}

	return cfg, nil
}

// Validate checks required fields for runtime readiness.
func (c *Config) Validate(requireGeminiKey bool) error {
	if requireGeminiKey && strings.TrimSpace(c.GeminiAPIKey) == "" {
		return errors.New("GEMINI_API_KEY is required")
	}
	if strings.TrimSpace(c.BesedkaURL) == "" {
		return errors.New("BESEDKA_URL cannot be empty")
	}
	if c.TownhallMaxParagraphs <= 0 {
		return fmt.Errorf("invalid TOWNHALL_MAX_PARAGRAPHS: %d", c.TownhallMaxParagraphs)
	}
	if c.DMMaxParagraphs <= 0 {
		return fmt.Errorf("invalid DM_MAX_PARAGRAPHS: %d", c.DMMaxParagraphs)
	}
	return nil
}

func getEnvOrDefault(key, defaultValue string) string {
	if val := strings.TrimSpace(os.Getenv(key)); val != "" {
		return val
	}
	return defaultValue
}

func getEnvIntOrDefault(key string, defaultValue int) int {
	if valStr := strings.TrimSpace(os.Getenv(key)); valStr != "" {
		if val, err := strconv.Atoi(valStr); err == nil && val > 0 {
			return val
		}
	}
	return defaultValue
}

// LoadDotEnv parses a simple .env file and populates environment variables that are not yet set.
func LoadDotEnv(filename string) {
	data, err := os.ReadFile(filename)
	if err != nil {
		return
	}
	lines := strings.Split(string(data), "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		parts := strings.SplitN(line, "=", 2)
		if len(parts) == 2 {
			key := strings.TrimSpace(parts[0])
			val := strings.TrimSpace(parts[1])
			if os.Getenv(key) == "" {
				_ = os.Setenv(key, val)
			}
		}
	}
}

