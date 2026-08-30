package config

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// DefaultUserAgent is the standard User-Agent header used for outgoing HTTP requests.
const DefaultUserAgent = "Besedka-Bot/1.0"

// Config holds runtime configuration settings for the agent.
type Config struct {
	BotHandle             string
	BesedkaURL            string
	BesedkaAPIKey         string
	OpenAIAPIKey          string
	OpenAIModel           string
	OpenAIBaseURL         string
	GeminiAPIKey          string // Deprecated: backward-compatible fallback for OpenAIAPIKey
	GeminiModel           string // Deprecated: backward-compatible fallback for OpenAIModel
	GeminiBaseURL         string // Deprecated: backward-compatible fallback for OpenAIBaseURL
	TownhallMaxParagraphs int
	DMMaxParagraphs       int
	MsgRingBufferSize     int
	TavilyAPIKey          string
	TavilyBaseURL         string
	DataDir               string
	EmbeddingModel        string
	EmbeddingPrecision    string
}

// LoadFromEnv loads configuration from environment variables (or .env file) with sensible defaults.
func LoadFromEnv() (*Config, error) {
	LoadDotEnv(".env")

	defaultBesedkaURL := os.Getenv("BASE_URL")
	if defaultBesedkaURL == "" {
		defaultBesedkaURL = "http://127.0.0.1:8080"
	}

	apiKey := getEnvOrDefault("OPENAI_API_KEY", os.Getenv("GEMINI_API_KEY"))
	model := getEnvOrDefault("OPENAI_MODEL", getEnvOrDefault("GEMINI_MODEL", "gemini-3.7-flash"))
	baseURL := getEnvOrDefault("OPENAI_BASE_URL", getEnvOrDefault("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"))
	tavilyBaseURL := strings.TrimSuffix(getEnvOrDefault("TAVILY_BASE_URL", "https://api.tavily.com"), "/")

	cfg := &Config{
		BotHandle:             getEnvOrDefault("BOT_HANDLE", "@bot"),
		BesedkaURL:            getEnvOrDefault("BESEDKA_URL", defaultBesedkaURL),
		BesedkaAPIKey:         os.Getenv("BESEDKA_API_KEY"),
		OpenAIAPIKey:          apiKey,
		OpenAIModel:           model,
		OpenAIBaseURL:         baseURL,
		GeminiAPIKey:          apiKey,
		GeminiModel:           model,
		GeminiBaseURL:         baseURL,
		TavilyAPIKey:          getEnvOrDefault("TAVILY_API_KEY", ""),
		TavilyBaseURL:         tavilyBaseURL,
		TownhallMaxParagraphs: getEnvIntOrDefault("TOWNHALL_MAX_PARAGRAPHS", 2),
		DMMaxParagraphs:       getEnvIntOrDefault("DM_MAX_PARAGRAPHS", 10),
		MsgRingBufferSize:     getEnvIntOrDefault("MSG_RING_BUFFER_SIZE", 100),
		DataDir:               getEnvOrDefault("DATA_DIR", "./data"),
		EmbeddingModel:        getEnvOrDefault("EMBEDDING_MODEL", ""),
		EmbeddingPrecision:    strings.ToLower(getEnvOrDefault("EMBEDDING_PRECISION", "bf16")),
	}

	// Normalize bot handle to ensure it starts with @
	if !strings.HasPrefix(cfg.BotHandle, "@") {
		cfg.BotHandle = "@" + cfg.BotHandle
	}

	// Avoid IPv6 loopback connection reset issues in containerized environments by normalizing localhost to 127.0.0.1
	if strings.Contains(cfg.BesedkaURL, "localhost") {
		cfg.BesedkaURL = strings.ReplaceAll(cfg.BesedkaURL, "localhost", "127.0.0.1")
	}

	// Normalize base URLs to ensure trailing slash
	if cfg.OpenAIBaseURL != "" && !strings.HasSuffix(cfg.OpenAIBaseURL, "/") {
		cfg.OpenAIBaseURL = cfg.OpenAIBaseURL + "/"
	}
	cfg.GeminiBaseURL = cfg.OpenAIBaseURL

	return cfg, nil
}

// DBPath returns the joined path to a SQLite database file inside DataDir.
func (c *Config) DBPath(filename string) string {
	return filepath.Join(c.DataDir, filename)
}

// Validate checks required fields for runtime readiness.
func (c *Config) Validate(requireAPIKey bool) error {
	if requireAPIKey && strings.TrimSpace(c.OpenAIAPIKey) == "" && strings.TrimSpace(c.GeminiAPIKey) == "" {
		return errors.New("OPENAI_API_KEY (or GEMINI_API_KEY) is required")
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
	if c.MsgRingBufferSize <= 0 {
		return fmt.Errorf("invalid MSG_RING_BUFFER_SIZE: %d", c.MsgRingBufferSize)
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
			if _, exists := os.LookupEnv(key); !exists {
				_ = os.Setenv(key, val)
			}
		}
	}
}
