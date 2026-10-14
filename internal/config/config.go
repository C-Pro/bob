package config

import (
	"errors"
	"fmt"
	"log/slog"
	"math"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"bob/internal/sandbox"
)

// DefaultUserAgent is the standard User-Agent header used for outgoing HTTP requests.
const DefaultUserAgent = "Besedka-Bot/1.0"

// DefaultMaxAttachmentSizeBytes is the default maximum size for attachment downloads/uploads (25MB).
const DefaultMaxAttachmentSizeBytes int64 = 25 * 1024 * 1024

// Config holds runtime configuration settings for the agent.
type Config struct {
	BotHandle                      string
	BesedkaURL                     string
	BesedkaAPIKey                  string
	OpenAIAPIKey                   string
	OpenAIModel                    string
	OpenAIBaseURL                  string
	GeminiAPIKey                   string // Deprecated: backward-compatible fallback for OpenAIAPIKey
	GeminiModel                    string // Deprecated: backward-compatible fallback for OpenAIModel
	GeminiBaseURL                  string // Deprecated: backward-compatible fallback for OpenAIBaseURL
	LLMRequestTimeout              time.Duration
	LLMFirstByteTimeout            time.Duration
	TownhallMaxParagraphs          int
	DMMaxParagraphs                int
	MsgRingBufferSize              int
	TownhallToolMaxIterations      int
	DMToolMaxIterations            int
	FSMRetentionDays               int
	TavilyAPIKey                   string
	TavilyBaseURL                  string
	DataDir                        string
	EmbeddingModel                 string
	EmbeddingPrecision             string
	Secret                         string
	S3Endpoint                     string
	S3Region                       string
	S3Bucket                       string
	S3AccessKey                    string
	S3SecretKey                    string
	S3PathStyle                    bool
	S3BackupInterval               time.Duration
	S3BackupKeep                   int
	S3BackupPrefix                 string
	SandboxEnabled                 bool
	SandboxDrivers                 []string
	SandboxDockerSocket            string
	SandboxAllowedImages           []string
	SandboxAllowedNetworkModes     []string
	SandboxMaxLifetime             time.Duration
	SandboxDefaultExecTimeout      time.Duration
	SandboxMaxExecTimeout          time.Duration
	SandboxCPULimit                float64
	SandboxMemoryLimitMB           int
	SandboxTasksMax                int
	SandboxHostDataDir             string
	SandboxProxyFwdPath            string
	SandboxAllowRuntimeBuild       *bool
	SchedulerMinRunTimeout         time.Duration
	SchedulerMaxRunTimeout         time.Duration
	SchedulerMinMaxTurns           int
	SchedulerMaxMaxTurns           int
	MaxAttachmentSizeBytes         int64
	KnowledgeMaxLoadedMemories     int
	KnowledgeMaxLoadedMemoryBytes  int
	KnowledgeMaxLoadedSkills       int
	KnowledgeMaxLoadedSkillBytes   int
	KnowledgeDefaultDiscoveryLimit int
	KnowledgeMaxDiscoveryLimit     int
	KnowledgeReconcileInterval     time.Duration
	CLIMaxIterations               int
	SandboxDefaultDriver           string
	SandboxDockerImage             string
	SandboxExtraPaths              []string
	SandboxAutoToolchains          bool
}

// DefaultSandboxAllowedImages defines standard safe container images.
var DefaultSandboxAllowedImages = sandbox.DefaultSandboxAllowedImages

// DefaultSandboxAllowedNetworkModes defines standard safe network modes (full is disabled by default).
var DefaultSandboxAllowedNetworkModes = sandbox.DefaultSandboxAllowedNetworkModes

// S3Enabled reports whether object-storage backup is configured.
func (c *Config) S3Enabled() bool {
	return c.S3Bucket != "" && c.S3Endpoint != ""
}

// Defaults returns a Config populated with sensible base defaults.
// Slices and pointers returned are freshly allocated and isolated.
func Defaults() *Config {
	return &Config{
		BotHandle:                      "@bot",
		BesedkaURL:                     "http://127.0.0.1:8080",
		OpenAIModel:                    "gemini-3.8-flash",
		OpenAIBaseURL:                  "https://generativelanguage.googleapis.com/v1beta/openai/",
		GeminiModel:                    "gemini-3.8-flash",
		GeminiBaseURL:                  "https://generativelanguage.googleapis.com/v1beta/openai/",
		LLMRequestTimeout:              10 * time.Minute,
		LLMFirstByteTimeout:            1 * time.Minute,
		TavilyBaseURL:                  "https://api.tavily.com",
		TownhallMaxParagraphs:          2,
		DMMaxParagraphs:                10,
		MsgRingBufferSize:              100,
		TownhallToolMaxIterations:      10,
		DMToolMaxIterations:            20,
		CLIMaxIterations:               100,
		FSMRetentionDays:               7,
		DataDir:                        "./data",
		EmbeddingModel:                 "",
		EmbeddingPrecision:             "bf16",
		S3Region:                       "us-east-1",
		S3BackupInterval:               1 * time.Hour,
		S3BackupKeep:                   7,
		S3BackupPrefix:                 "bob_agent/",
		S3PathStyle:                    true,
		SandboxEnabled:                 true,
		SandboxDefaultDriver:           "bwrap",
		SandboxDrivers:                 []string{"bwrap", "docker"},
		SandboxDockerSocket:            "/var/run/docker.sock",
		SandboxDockerImage:             "python:3.12-slim",
		SandboxAllowedImages:           append([]string(nil), DefaultSandboxAllowedImages...),
		SandboxAllowedNetworkModes:     append([]string(nil), DefaultSandboxAllowedNetworkModes...),
		SandboxMaxLifetime:             30 * time.Minute,
		SandboxDefaultExecTimeout:      1 * time.Minute,
		SandboxMaxExecTimeout:          10 * time.Minute,
		SandboxCPULimit:                1.0,
		SandboxMemoryLimitMB:           512,
		SandboxTasksMax:                64,
		SchedulerMinRunTimeout:         1 * time.Minute,
		SchedulerMaxRunTimeout:         1 * time.Hour,
		SchedulerMinMaxTurns:           10,
		SchedulerMaxMaxTurns:           100,
		MaxAttachmentSizeBytes:         DefaultMaxAttachmentSizeBytes,
		KnowledgeMaxLoadedMemories:     8,
		KnowledgeMaxLoadedMemoryBytes:  16384,
		KnowledgeMaxLoadedSkills:       3,
		KnowledgeMaxLoadedSkillBytes:   24576,
		KnowledgeDefaultDiscoveryLimit: 5,
		KnowledgeMaxDiscoveryLimit:     10,
		KnowledgeReconcileInterval:     1 * time.Minute,
	}
}

// LoadFromEnv loads configuration from environment variables (or .env file) with sensible defaults.
func LoadFromEnv() (*Config, error) {
	LoadDotEnv(".env")

	cfg := Defaults()

	defaultBesedkaURL := os.Getenv("BASE_URL")
	if defaultBesedkaURL == "" {
		defaultBesedkaURL = cfg.BesedkaURL
	}
	cfg.BesedkaURL = getEnvOrDefault("BESEDKA_URL", defaultBesedkaURL)
	cfg.BesedkaAPIKey = os.Getenv("BESEDKA_API_KEY")

	apiKey := getEnvOrDefault("OPENAI_API_KEY", os.Getenv("GEMINI_API_KEY"))
	if apiKey != "" {
		cfg.OpenAIAPIKey = apiKey
		cfg.GeminiAPIKey = apiKey
	}

	model := getEnvOrDefault("OPENAI_MODEL", getEnvOrDefault("GEMINI_MODEL", cfg.OpenAIModel))
	cfg.OpenAIModel = model
	cfg.GeminiModel = model

	baseURL := getEnvOrDefault("OPENAI_BASE_URL", getEnvOrDefault("GEMINI_BASE_URL", cfg.OpenAIBaseURL))
	cfg.OpenAIBaseURL = baseURL
	cfg.GeminiBaseURL = baseURL

	cfg.LLMRequestTimeout = getEnvDurationOrDefault("LLM_REQUEST_TIMEOUT", getEnvDurationOrDefault("OPENAI_REQUEST_TIMEOUT", cfg.LLMRequestTimeout))
	cfg.LLMFirstByteTimeout = getEnvDurationOrDefault("LLM_FIRST_BYTE_TIMEOUT", getEnvDurationOrDefault("OPENAI_FIRST_BYTE_TIMEOUT", cfg.LLMFirstByteTimeout))

	cfg.BotHandle = getEnvOrDefault("BOT_HANDLE", cfg.BotHandle)
	cfg.TavilyAPIKey = getEnvOrDefault("TAVILY_API_KEY", "")
	cfg.TavilyBaseURL = strings.TrimSuffix(getEnvOrDefault("TAVILY_BASE_URL", cfg.TavilyBaseURL), "/")

	backupIntervalStr := getEnvOrDefault("S3_BACKUP_INTERVAL", "1h")
	backupInterval, err := time.ParseDuration(backupIntervalStr)
	if err != nil {
		return nil, fmt.Errorf("invalid S3_BACKUP_INTERVAL: %w", err)
	}
	cfg.S3BackupInterval = backupInterval

	cfg.Secret = getEnvOrDefault("SECRET", os.Getenv("AUTH_SECRET"))
	s3Prefix := getEnvOrDefault("S3_BACKUP_PREFIX", cfg.S3BackupPrefix)
	if s3Prefix != "" && !strings.HasSuffix(s3Prefix, "/") {
		s3Prefix = s3Prefix + "/"
	}
	cfg.S3BackupPrefix = s3Prefix

	s3PathStyleStr := getEnvOrDefault("S3_PATH_STYLE", "true")
	cfg.S3PathStyle = s3PathStyleStr == "true" || s3PathStyleStr == "1"

	sandboxEnabled, err := getEnvBoolOrDefault("SANDBOX_ENABLED", cfg.SandboxEnabled)
	if err != nil {
		return nil, err
	}
	cfg.SandboxEnabled = sandboxEnabled

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
	cfg.SandboxAllowRuntimeBuild = &sandboxAllowRuntimeBuild

	schedMinTimeout, err := getEnvDurationStrict("SCHEDULER_MIN_RUN_TIMEOUT", cfg.SchedulerMinRunTimeout)
	if err != nil {
		return nil, err
	}
	cfg.SchedulerMinRunTimeout = schedMinTimeout

	schedMaxTimeout, err := getEnvDurationStrict("SCHEDULER_MAX_RUN_TIMEOUT", cfg.SchedulerMaxRunTimeout)
	if err != nil {
		return nil, err
	}
	cfg.SchedulerMaxRunTimeout = schedMaxTimeout

	schedMinTurns, err := getEnvIntStrict("SCHEDULER_MIN_MAX_TURNS", cfg.SchedulerMinMaxTurns)
	if err != nil {
		return nil, err
	}
	cfg.SchedulerMinMaxTurns = schedMinTurns

	schedMaxTurns, err := getEnvIntStrict("SCHEDULER_MAX_MAX_TURNS", cfg.SchedulerMaxMaxTurns)
	if err != nil {
		return nil, err
	}
	cfg.SchedulerMaxMaxTurns = schedMaxTurns

	maxAttachmentSizeBytes, err := parseAttachmentSize(os.Getenv("MAX_ATTACHMENT_SIZE"), cfg.MaxAttachmentSizeBytes)
	if err != nil {
		return nil, fmt.Errorf("invalid MAX_ATTACHMENT_SIZE: %w", err)
	}
	cfg.MaxAttachmentSizeBytes = maxAttachmentSizeBytes

	knowledgeMaxMemories, err := getEnvIntStrict("KNOWLEDGE_MAX_LOADED_MEMORIES", cfg.KnowledgeMaxLoadedMemories)
	if err != nil {
		return nil, err
	}
	cfg.KnowledgeMaxLoadedMemories = knowledgeMaxMemories

	knowledgeMaxMemoryBytes, err := getEnvIntStrict("KNOWLEDGE_MAX_LOADED_MEMORY_BYTES", cfg.KnowledgeMaxLoadedMemoryBytes)
	if err != nil {
		return nil, err
	}
	cfg.KnowledgeMaxLoadedMemoryBytes = knowledgeMaxMemoryBytes

	knowledgeMaxSkills, err := getEnvIntStrict("KNOWLEDGE_MAX_LOADED_SKILLS", cfg.KnowledgeMaxLoadedSkills)
	if err != nil {
		return nil, err
	}
	cfg.KnowledgeMaxLoadedSkills = knowledgeMaxSkills

	knowledgeMaxSkillBytes, err := getEnvIntStrict("KNOWLEDGE_MAX_LOADED_SKILL_BYTES", cfg.KnowledgeMaxLoadedSkillBytes)
	if err != nil {
		return nil, err
	}
	cfg.KnowledgeMaxLoadedSkillBytes = knowledgeMaxSkillBytes

	knowledgeDefaultLimit, err := getEnvIntStrict("KNOWLEDGE_DEFAULT_DISCOVERY_LIMIT", cfg.KnowledgeDefaultDiscoveryLimit)
	if err != nil {
		return nil, err
	}
	cfg.KnowledgeDefaultDiscoveryLimit = knowledgeDefaultLimit

	knowledgeMaxLimit, err := getEnvIntStrict("KNOWLEDGE_MAX_DISCOVERY_LIMIT", cfg.KnowledgeMaxDiscoveryLimit)
	if err != nil {
		return nil, err
	}
	cfg.KnowledgeMaxDiscoveryLimit = knowledgeMaxLimit

	knowledgeReconcileInterval, err := getEnvDurationStrict("KNOWLEDGE_RECONCILE_INTERVAL", cfg.KnowledgeReconcileInterval)
	if err != nil {
		return nil, err
	}
	cfg.KnowledgeReconcileInterval = knowledgeReconcileInterval

	cfg.TownhallMaxParagraphs = getEnvIntOrDefault("TOWNHALL_MAX_PARAGRAPHS", cfg.TownhallMaxParagraphs)
	cfg.DMMaxParagraphs = getEnvIntOrDefault("DM_MAX_PARAGRAPHS", cfg.DMMaxParagraphs)
	cfg.MsgRingBufferSize = getEnvIntOrDefault("MSG_RING_BUFFER_SIZE", cfg.MsgRingBufferSize)
	cfg.TownhallToolMaxIterations = getEnvIntOrDefault("TOWNHALL_TOOL_MAX_ITERATIONS", cfg.TownhallToolMaxIterations)
	cfg.DMToolMaxIterations = getEnvIntOrDefault("DM_TOOL_MAX_ITERATIONS", cfg.DMToolMaxIterations)
	cfg.FSMRetentionDays = getEnvIntOrDefault("FSM_RETENTION_DAYS", cfg.FSMRetentionDays)
	cfg.DataDir = getEnvOrDefault("DATA_DIR", cfg.DataDir)
	cfg.EmbeddingModel = getEnvOrDefault("EMBEDDING_MODEL", cfg.EmbeddingModel)
	cfg.EmbeddingPrecision = strings.ToLower(getEnvOrDefault("EMBEDDING_PRECISION", cfg.EmbeddingPrecision))

	cfg.S3Endpoint = os.Getenv("S3_ENDPOINT")
	cfg.S3Region = getEnvOrDefault("S3_REGION", cfg.S3Region)
	cfg.S3Bucket = os.Getenv("S3_BUCKET")
	cfg.S3AccessKey = os.Getenv("S3_ACCESS_KEY")
	cfg.S3SecretKey = os.Getenv("S3_SECRET_KEY")
	cfg.S3BackupKeep = getEnvIntOrDefault("S3_BACKUP_KEEP", cfg.S3BackupKeep)

	cfg.SandboxDrivers = getEnvSliceOrDefault("SANDBOX_DRIVERS", cfg.SandboxDrivers)
	cfg.SandboxDockerSocket = getEnvOrDefault("SANDBOX_DOCKER_SOCKET", cfg.SandboxDockerSocket)
	cfg.SandboxAllowedImages = getEnvSliceOrDefault("SANDBOX_ALLOWED_IMAGES", cfg.SandboxAllowedImages)
	cfg.SandboxAllowedNetworkModes = getEnvSliceOrDefault("SANDBOX_ALLOWED_NETWORK_MODES", cfg.SandboxAllowedNetworkModes)
	cfg.SandboxMaxLifetime = getEnvDurationOrDefault("SANDBOX_MAX_LIFETIME", cfg.SandboxMaxLifetime)
	cfg.SandboxDefaultExecTimeout = getEnvDurationOrDefault("SANDBOX_DEFAULT_EXEC_TIMEOUT", cfg.SandboxDefaultExecTimeout)
	cfg.SandboxMaxExecTimeout = getEnvDurationOrDefault("SANDBOX_MAX_EXEC_TIMEOUT", cfg.SandboxMaxExecTimeout)
	cfg.SandboxCPULimit = getEnvFloatOrDefault("SANDBOX_CPU_LIMIT", cfg.SandboxCPULimit)
	cfg.SandboxMemoryLimitMB = getEnvIntOrDefault("SANDBOX_MEMORY_LIMIT_MB", cfg.SandboxMemoryLimitMB)
	if val := os.Getenv("BOB_SANDBOX_TASKS_MAX"); strings.TrimSpace(val) != "" {
		cfg.SandboxTasksMax = getEnvIntOrDefault("BOB_SANDBOX_TASKS_MAX", cfg.SandboxTasksMax)
	} else if val := os.Getenv("SANDBOX_TASKS_MAX"); strings.TrimSpace(val) != "" {
		cfg.SandboxTasksMax = getEnvIntOrDefault("SANDBOX_TASKS_MAX", cfg.SandboxTasksMax)
	}
	cfg.SandboxHostDataDir = getEnvOrDefault("SANDBOX_HOST_DATA_DIR", cfg.SandboxHostDataDir)
	cfg.SandboxProxyFwdPath = getEnvOrDefault("SANDBOX_PROXY_FWD_PATH", cfg.SandboxProxyFwdPath)

	if val := os.Getenv("BOB_SANDBOX_EXTRA_PATHS"); strings.TrimSpace(val) != "" {
		cfg.SandboxExtraPaths = parsePathList(val)
	} else if val := os.Getenv("SANDBOX_EXTRA_PATHS"); strings.TrimSpace(val) != "" {
		cfg.SandboxExtraPaths = parsePathList(val)
	}

	if val := os.Getenv("BOB_SANDBOX_AUTO_TOOLCHAINS"); strings.TrimSpace(val) != "" {
		b, err := strconv.ParseBool(strings.TrimSpace(val))
		if err == nil {
			cfg.SandboxAutoToolchains = b
		}
	} else if val := os.Getenv("SANDBOX_AUTO_TOOLCHAINS"); strings.TrimSpace(val) != "" {
		b, err := strconv.ParseBool(strings.TrimSpace(val))
		if err == nil {
			cfg.SandboxAutoToolchains = b
		}
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
		TasksMax:            c.SandboxTasksMax,
		DataDir:             c.DataDir,
		HostDataDir:         c.SandboxHostDataDir,
		ProxyFwdPath:        c.SandboxProxyFwdPath,
		AllowRuntimeBuild:   c.SandboxAllowRuntimeBuild,
		ExtraPaths:          append([]string(nil), c.SandboxExtraPaths...),
		AutoToolchains:      c.SandboxAutoToolchains,
	}
}

// Validate checks required fields for runtime readiness.
func (c *Config) Validate(requireAPIKey bool) error {
	if requireAPIKey {
		if strings.TrimSpace(c.OpenAIAPIKey) == "" && strings.TrimSpace(c.GeminiAPIKey) == "" {
			return errors.New("OPENAI_API_KEY (or GEMINI_API_KEY) is required")
		}
		if strings.TrimSpace(c.OpenAIModel) == "" && strings.TrimSpace(c.GeminiModel) == "" {
			return errors.New("OPENAI_MODEL (or GEMINI_MODEL) cannot be empty")
		}
	}
	if strings.TrimSpace(c.BesedkaURL) != "" {
		if err := validateURL(c.BesedkaURL); err != nil {
			return fmt.Errorf("invalid BESEDKA_URL: %w", err)
		}
	}
	if c.LLMRequestTimeout < 0 {
		return fmt.Errorf("invalid LLM request timeout: %v", c.LLMRequestTimeout)
	}
	if c.LLMRequestTimeout == 0 {
		c.LLMRequestTimeout = 10 * time.Minute
	}
	if c.LLMFirstByteTimeout < 0 {
		return fmt.Errorf("invalid LLM first byte timeout: %v", c.LLMFirstByteTimeout)
	}
	if c.LLMFirstByteTimeout == 0 {
		c.LLMFirstByteTimeout = 1 * time.Minute
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
	if c.TownhallToolMaxIterations <= 0 {
		c.TownhallToolMaxIterations = 10
	}
	if c.DMToolMaxIterations <= 0 {
		c.DMToolMaxIterations = 20
	}
	if c.FSMRetentionDays <= 0 {
		c.FSMRetentionDays = 7
	}
	if c.SchedulerMinRunTimeout <= 0 {
		c.SchedulerMinRunTimeout = 1 * time.Minute
	}
	if c.SchedulerMaxRunTimeout <= 0 {
		c.SchedulerMaxRunTimeout = 1 * time.Hour
	}
	if c.SchedulerMaxRunTimeout < c.SchedulerMinRunTimeout {
		return fmt.Errorf("SCHEDULER_MAX_RUN_TIMEOUT (%v) cannot be less than SCHEDULER_MIN_RUN_TIMEOUT (%v)", c.SchedulerMaxRunTimeout, c.SchedulerMinRunTimeout)
	}
	if c.SchedulerMinMaxTurns <= 0 {
		c.SchedulerMinMaxTurns = 10
	}
	if c.SchedulerMaxMaxTurns <= 0 {
		c.SchedulerMaxMaxTurns = 100
	}
	if c.SchedulerMaxMaxTurns < c.SchedulerMinMaxTurns {
		return fmt.Errorf("SCHEDULER_MAX_MAX_TURNS (%d) cannot be less than SCHEDULER_MIN_MAX_TURNS (%d)", c.SchedulerMaxMaxTurns, c.SchedulerMinMaxTurns)
	}
	if c.MaxAttachmentSizeBytes <= 0 {
		c.MaxAttachmentSizeBytes = DefaultMaxAttachmentSizeBytes
	}
	if c.KnowledgeMaxLoadedMemories <= 0 {
		c.KnowledgeMaxLoadedMemories = 8
	}
	if c.KnowledgeMaxLoadedMemoryBytes <= 0 {
		c.KnowledgeMaxLoadedMemoryBytes = 16384
	}
	if c.KnowledgeMaxLoadedSkills <= 0 {
		c.KnowledgeMaxLoadedSkills = 3
	}
	if c.KnowledgeMaxLoadedSkillBytes <= 0 {
		c.KnowledgeMaxLoadedSkillBytes = 24576
	}
	if c.KnowledgeDefaultDiscoveryLimit <= 0 {
		c.KnowledgeDefaultDiscoveryLimit = 5
	}
	if c.KnowledgeMaxDiscoveryLimit <= 0 {
		c.KnowledgeMaxDiscoveryLimit = 10
	}
	if c.KnowledgeMaxDiscoveryLimit < c.KnowledgeDefaultDiscoveryLimit {
		return fmt.Errorf("KNOWLEDGE_MAX_DISCOVERY_LIMIT (%d) cannot be less than KNOWLEDGE_DEFAULT_DISCOVERY_LIMIT (%d)", c.KnowledgeMaxDiscoveryLimit, c.KnowledgeDefaultDiscoveryLimit)
	}
	if c.KnowledgeReconcileInterval <= 0 {
		c.KnowledgeReconcileInterval = 1 * time.Minute
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

// ValidateBesedka checks both general runtime configuration and daemon-specific Besedka requirements.
func (c *Config) ValidateBesedka(requireAPIKey ...bool) error {
	reqKey := true
	if len(requireAPIKey) > 0 {
		reqKey = requireAPIKey[0]
	}
	if err := c.Validate(reqKey); err != nil {
		return err
	}
	if strings.TrimSpace(c.BesedkaURL) == "" {
		return errors.New("BESEDKA_URL cannot be empty")
	}
	return nil
}

func validateURL(rawURL string) error {
	u, err := url.ParseRequestURI(rawURL)
	if err != nil {
		return fmt.Errorf("invalid URL: %w", err)
	}
	if u.Scheme != "http" && u.Scheme != "https" {
		return fmt.Errorf("invalid URL scheme %q, must be http or https", u.Scheme)
	}
	if u.Host == "" {
		return errors.New("URL must specify a host")
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

func getEnvDurationStrict(key string, defaultValue time.Duration) (time.Duration, error) {
	val := strings.TrimSpace(os.Getenv(key))
	if val == "" {
		return defaultValue, nil
	}
	d, err := time.ParseDuration(val)
	if err != nil || d <= 0 {
		return 0, fmt.Errorf("invalid duration for %s: %q", key, val)
	}
	return d, nil
}

func getEnvIntStrict(key string, defaultValue int) (int, error) {
	val := strings.TrimSpace(os.Getenv(key))
	if val == "" {
		return defaultValue, nil
	}
	i, err := strconv.Atoi(val)
	if err != nil || i <= 0 {
		return 0, fmt.Errorf("invalid integer for %s: %q", key, val)
	}
	return i, nil
}

func parsePathList(val string) []string {
	var sep string
	if strings.Contains(val, ":") {
		sep = ":"
	} else {
		sep = ","
	}
	var out []string
	for _, p := range strings.Split(val, sep) {
		trimmed := strings.TrimSpace(p)
		if trimmed != "" {
			out = append(out, trimmed)
		}
	}
	return out
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

// parseAttachmentSize parses a byte size string like "25MB", "10M", "1024", or returns defaultValue if empty.
func parseAttachmentSize(val string, defaultValue int64) (int64, error) {
	s := strings.TrimSpace(val)
	if s == "" {
		return defaultValue, nil
	}

	var numPart strings.Builder
	var unitPart strings.Builder
	seenUnit := false
	for _, r := range s {
		if !seenUnit && r >= '0' && r <= '9' {
			numPart.WriteRune(r)
		} else {
			seenUnit = true
			unitPart.WriteRune(r)
		}
	}

	if numPart.Len() == 0 {
		return 0, fmt.Errorf("invalid byte size %q: missing numeric value", val)
	}

	base, err := strconv.ParseInt(numPart.String(), 10, 64)
	if err != nil || base <= 0 {
		return 0, fmt.Errorf("invalid byte size %q: value must be a positive integer", val)
	}

	unit := strings.ToUpper(strings.TrimSpace(unitPart.String()))
	var multiplier int64
	switch unit {
	case "", "B":
		multiplier = 1
	case "K", "KB", "KIB":
		multiplier = 1024
	case "M", "MB", "MIB":
		multiplier = 1024 * 1024
	case "G", "GB", "GIB":
		multiplier = 1024 * 1024 * 1024
	default:
		return 0, fmt.Errorf("invalid byte size unit %q in %q", unit, val)
	}

	if multiplier > 1 && base > math.MaxInt64/multiplier {
		return 0, fmt.Errorf("byte size %q causes int64 overflow", val)
	}

	return base * multiplier, nil
}
