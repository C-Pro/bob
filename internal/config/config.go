package config

import (
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"bob/internal/sandbox"
)

// DefaultUserAgent is the standard User-Agent header used for outgoing HTTP requests.
const DefaultUserAgent = "Besedka-Bot/1.0"

// Config holds runtime configuration settings for the agent.
type Config struct {
	BotHandle                  string
	BesedkaURL                 string
	BesedkaAPIKey              string
	OpenAIAPIKey               string
	OpenAIModel                string
	OpenAIBaseURL              string
	GeminiAPIKey               string // Deprecated: backward-compatible fallback for OpenAIAPIKey
	GeminiModel                string // Deprecated: backward-compatible fallback for OpenAIModel
	GeminiBaseURL              string // Deprecated: backward-compatible fallback for OpenAIBaseURL
	TownhallMaxParagraphs      int
	DMMaxParagraphs            int
	MsgRingBufferSize          int
	TavilyAPIKey               string
	TavilyBaseURL              string
	DataDir                    string
	EmbeddingModel             string
	EmbeddingPrecision         string
	Secret                     string
	S3Endpoint                 string
	S3Region                   string
	S3Bucket                   string
	S3AccessKey                string
	S3SecretKey                string
	S3PathStyle                bool
	S3BackupInterval           time.Duration
	S3BackupKeep               int
	S3BackupPrefix             string
	SandboxEnabled             bool
	SandboxDrivers             []string
	SandboxDockerSocket        string
	SandboxAllowedImages       []string
	SandboxAllowedNetworkModes []string
	SandboxMaxLifetime         time.Duration
	SandboxDefaultExecTimeout  time.Duration
	SandboxMaxExecTimeout      time.Duration
	SandboxCPULimit            float64
	SandboxMemoryLimitMB       int
	SandboxHostDataDir         string
	SandboxProxyFwdPath        string
	SandboxAllowRuntimeBuild   *bool
}

// DefaultSandboxAllowedImages defines standard safe container images.
var DefaultSandboxAllowedImages = sandbox.DefaultSandboxAllowedImages

// DefaultSandboxAllowedNetworkModes defines standard safe network modes (full is disabled by default).
var DefaultSandboxAllowedNetworkModes = sandbox.DefaultSandboxAllowedNetworkModes

// S3Enabled reports whether object-storage backup is configured.
func (c *Config) S3Enabled() bool {
	return c.S3Bucket != "" && c.S3Endpoint != ""
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
	backupIntervalStr := getEnvOrDefault("S3_BACKUP_INTERVAL", "1h")
	backupInterval, err := time.ParseDuration(backupIntervalStr)
	if err != nil {
		return nil, fmt.Errorf("invalid S3_BACKUP_INTERVAL: %w", err)
	}

	secret := getEnvOrDefault("SECRET", os.Getenv("AUTH_SECRET"))
	s3Prefix := getEnvOrDefault("S3_BACKUP_PREFIX", "bob_agent/")
	if s3Prefix != "" && !strings.HasSuffix(s3Prefix, "/") {
		s3Prefix = s3Prefix + "/"
	}
	s3PathStyleStr := getEnvOrDefault("S3_PATH_STYLE", "true")
	s3PathStyle := s3PathStyleStr == "true" || s3PathStyleStr == "1"

	sandboxEnabled, err := getEnvBoolOrDefault("SANDBOX_ENABLED", true)
	if err != nil {
		return nil, err
	}

	defaultRuntimeBuild := os.Getenv("ENV") == "development"
	for _, arg := range os.Args {
		if strings.HasPrefix(arg, "-test.") {
			defaultRuntimeBuild = true
			break
		}
	}
	sandboxAllowRuntimeBuild, err := getEnvBoolOrDefault("BOB_ALLOW_RUNTIME_BUILD", defaultRuntimeBuild)
	if err != nil {
		return nil, err
	}

	cfg := &Config{
		BotHandle:                  getEnvOrDefault("BOT_HANDLE", "@bot"),
		BesedkaURL:                 getEnvOrDefault("BESEDKA_URL", defaultBesedkaURL),
		BesedkaAPIKey:              os.Getenv("BESEDKA_API_KEY"),
		OpenAIAPIKey:               apiKey,
		OpenAIModel:                model,
		OpenAIBaseURL:              baseURL,
		GeminiAPIKey:               apiKey,
		GeminiModel:                model,
		GeminiBaseURL:              baseURL,
		TavilyAPIKey:               getEnvOrDefault("TAVILY_API_KEY", ""),
		TavilyBaseURL:              tavilyBaseURL,
		TownhallMaxParagraphs:      getEnvIntOrDefault("TOWNHALL_MAX_PARAGRAPHS", 2),
		DMMaxParagraphs:            getEnvIntOrDefault("DM_MAX_PARAGRAPHS", 10),
		MsgRingBufferSize:          getEnvIntOrDefault("MSG_RING_BUFFER_SIZE", 100),
		DataDir:                    getEnvOrDefault("DATA_DIR", "./data"),
		EmbeddingModel:             getEnvOrDefault("EMBEDDING_MODEL", ""),
		EmbeddingPrecision:         strings.ToLower(getEnvOrDefault("EMBEDDING_PRECISION", "bf16")),
		Secret:                     secret,
		S3Endpoint:                 os.Getenv("S3_ENDPOINT"),
		S3Region:                   getEnvOrDefault("S3_REGION", "us-east-1"),
		S3Bucket:                   os.Getenv("S3_BUCKET"),
		S3AccessKey:                os.Getenv("S3_ACCESS_KEY"),
		S3SecretKey:                os.Getenv("S3_SECRET_KEY"),
		S3PathStyle:                s3PathStyle,
		S3BackupInterval:           backupInterval,
		S3BackupKeep:               getEnvIntOrDefault("S3_BACKUP_KEEP", 7),
		S3BackupPrefix:             s3Prefix,
		SandboxEnabled:             sandboxEnabled,
		SandboxDrivers:             getEnvSliceOrDefault("SANDBOX_DRIVERS", []string{"bwrap", "docker"}),
		SandboxDockerSocket:        getEnvOrDefault("SANDBOX_DOCKER_SOCKET", "/var/run/docker.sock"),
		SandboxAllowedImages:       getEnvSliceOrDefault("SANDBOX_ALLOWED_IMAGES", DefaultSandboxAllowedImages),
		SandboxAllowedNetworkModes: getEnvSliceOrDefault("SANDBOX_ALLOWED_NETWORK_MODES", DefaultSandboxAllowedNetworkModes),
		SandboxMaxLifetime:         getEnvDurationOrDefault("SANDBOX_MAX_LIFETIME", 30*time.Minute),
		SandboxDefaultExecTimeout:  getEnvDurationOrDefault("SANDBOX_DEFAULT_EXEC_TIMEOUT", 1*time.Minute),
		SandboxMaxExecTimeout:      getEnvDurationOrDefault("SANDBOX_MAX_EXEC_TIMEOUT", 10*time.Minute),
		SandboxCPULimit:            getEnvFloatOrDefault("SANDBOX_CPU_LIMIT", 1.0),
		SandboxMemoryLimitMB:       getEnvIntOrDefault("SANDBOX_MEMORY_LIMIT_MB", 512),
		SandboxHostDataDir:         getEnvOrDefault("SANDBOX_HOST_DATA_DIR", ""),
		SandboxProxyFwdPath:        getEnvOrDefault("SANDBOX_PROXY_FWD_PATH", ""),
		SandboxAllowRuntimeBuild:   &sandboxAllowRuntimeBuild,
	}

	if cfg.SandboxHostDataDir != "" {
		cfg.SandboxHostDataDir = filepath.Clean(cfg.SandboxHostDataDir)
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

// SandboxConfig derives a sandbox.Config from the application Config.
func (c *Config) SandboxConfig() sandbox.Config {
	return sandbox.Config{
		Enabled:             c.SandboxEnabled,
		Drivers:             c.SandboxDrivers,
		DockerSocket:        c.SandboxDockerSocket,
		AllowedImages:       c.SandboxAllowedImages,
		AllowedNetworkModes: c.SandboxAllowedNetworkModes,
		MaxLifetime:         c.SandboxMaxLifetime,
		DefaultExecTimeout:  c.SandboxDefaultExecTimeout,
		MaxExecTimeout:      c.SandboxMaxExecTimeout,
		CPULimit:            c.SandboxCPULimit,
		MemoryLimitMB:       c.SandboxMemoryLimitMB,
		DataDir:             c.DataDir,
		HostDataDir:         c.SandboxHostDataDir,
		ProxyFwdPath:        c.SandboxProxyFwdPath,
		AllowRuntimeBuild:   c.SandboxAllowRuntimeBuild,
	}
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
	if (c.S3Bucket == "") != (c.S3Endpoint == "") {
		return errors.New("S3_BUCKET and S3_ENDPOINT must be set together")
	}
	if c.S3Enabled() {
		if strings.TrimSpace(c.Secret) == "" {
			return errors.New("SECRET (or AUTH_SECRET) is required when S3 backup is enabled")
		}
		if strings.TrimSpace(c.S3AccessKey) == "" || strings.TrimSpace(c.S3SecretKey) == "" {
			return errors.New("S3_ACCESS_KEY and S3_SECRET_KEY are required when S3 backup is enabled")
		}
		if c.S3BackupInterval <= 0 {
			return errors.New("S3_BACKUP_INTERVAL must be greater than 0")
		}
		if c.S3BackupKeep <= 0 {
			return errors.New("S3_BACKUP_KEEP must be greater than 0")
		}
	}
	sbxCfg := c.SandboxConfig()
	if err := sbxCfg.Validate(); err != nil {
		return err
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

func getEnvBoolOrDefault(key string, defaultValue bool) (bool, error) {
	val := strings.TrimSpace(os.Getenv(key))
	if val == "" {
		return defaultValue, nil
	}
	lower := strings.ToLower(val)
	switch lower {
	case "true", "1", "yes", "y", "on", "enable", "enabled":
		return true, nil
	case "false", "0", "no", "n", "off", "disable", "disabled":
		return false, nil
	default:
		return false, fmt.Errorf("invalid boolean value %q for %s: must be true/false, yes/no, on/off, enabled/disabled, 1/0", val, key)
	}
}

func getEnvSliceOrDefault(key string, defaultValue []string) []string {
	if val := strings.TrimSpace(os.Getenv(key)); val != "" {
		parts := strings.Split(val, ",")
		var res []string
		for _, p := range parts {
			if trimmed := strings.TrimSpace(p); trimmed != "" {
				res = append(res, trimmed)
			}
		}
		if len(res) > 0 {
			return res
		}
	}
	return defaultValue
}

func getEnvDurationOrDefault(key string, defaultValue time.Duration) time.Duration {
	if val := strings.TrimSpace(os.Getenv(key)); val != "" {
		if d, err := time.ParseDuration(val); err == nil && d > 0 {
			return d
		}
	}
	return defaultValue
}

func getEnvFloatOrDefault(key string, defaultValue float64) float64 {
	if valStr := strings.TrimSpace(os.Getenv(key)); valStr != "" {
		if val, err := strconv.ParseFloat(valStr, 64); err == nil && val > 0 {
			return val
		}
	}
	return defaultValue
}

// LoadDotEnv parses a simple .env file and populates environment variables that are not yet set.
// Supports both 'KEY=VAL' and 'export KEY=VAL' syntax, including quoted values.
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
		if strings.HasPrefix(line, "export ") {
			line = strings.TrimSpace(strings.TrimPrefix(line, "export "))
		}
		parts := strings.SplitN(line, "=", 2)
		if len(parts) == 2 {
			key := strings.TrimSpace(parts[0])
			val := strings.TrimSpace(parts[1])
			if len(val) >= 2 && ((strings.HasPrefix(val, `"`) && strings.HasSuffix(val, `"`)) || (strings.HasPrefix(val, `'`) && strings.HasSuffix(val, `'`))) {
				val = val[1 : len(val)-1]
			}
			if _, exists := os.LookupEnv(key); !exists {
				if err := os.Setenv(key, val); err != nil {
					slog.Warn("failed to set environment variable from dot env", "key", key, "err", err)
				}
			}
		}
	}
}
