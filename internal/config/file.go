package config

import (
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	"gopkg.in/yaml.v3"
)

// DefaultPaths returns the default configuration file path and default data directory for CLI / user mode.
func DefaultPaths() (configPath, dataDir string, err error) {
	homeDir, err := os.UserHomeDir()
	if err != nil {
		return "", "", fmt.Errorf("failed to determine user home directory: %w", err)
	}

	// 1. Config path: $XDG_CONFIG_HOME/bob/config.yaml or ~/.config/bob/config.yaml
	xdgConfig := strings.TrimSpace(os.Getenv("XDG_CONFIG_HOME"))
	if xdgConfig != "" {
		expanded, expErr := ExpandPath(xdgConfig)
		if expErr != nil {
			return "", "", fmt.Errorf("failed to expand XDG_CONFIG_HOME: %w", expErr)
		}
		configPath = filepath.Join(expanded, "bob", "config.yaml")
	} else {
		userConfig, confErr := os.UserConfigDir()
		if confErr == nil && userConfig != "" {
			configPath = filepath.Join(userConfig, "bob", "config.yaml")
		} else {
			configPath = filepath.Join(homeDir, ".config", "bob", "config.yaml")
		}
	}

	// 2. Data dir: $XDG_DATA_HOME/bob or ~/.local/share/bob
	xdgData := strings.TrimSpace(os.Getenv("XDG_DATA_HOME"))
	if xdgData != "" {
		expanded, expErr := ExpandPath(xdgData)
		if expErr != nil {
			return "", "", fmt.Errorf("failed to expand XDG_DATA_HOME: %w", expErr)
		}
		dataDir = filepath.Join(expanded, "bob")
	} else {
		dataDir = filepath.Join(homeDir, ".local", "share", "bob")
	}

	return configPath, dataDir, nil
}

// ExpandPath expands leading '~/' or '~' with the user's home directory.
func ExpandPath(p string) (string, error) {
	if !strings.HasPrefix(p, "~") {
		return p, nil
	}
	if p != "~" && !strings.HasPrefix(p, "~/") {
		return p, nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("cannot expand home directory: %w", err)
	}
	if p == "~" {
		return home, nil
	}
	return filepath.Join(home, p[2:]), nil
}

type fileConfig struct {
	LLM struct {
		APIKey             string         `yaml:"api_key"`
		BaseURL            *string        `yaml:"base_url"`
		Model              *string        `yaml:"model"`
		RequestTimeout     *time.Duration `yaml:"request_timeout"`
		FirstByteTimeout   *time.Duration `yaml:"first_byte_timeout"`
		Timeout            *time.Duration `yaml:"timeout"`
		EmbeddingModel     *string        `yaml:"embedding_model"`
		EmbeddingPrecision *string        `yaml:"embedding_precision"`

		// Backwards-compatible aliases
		OpenAIAPIKey  string  `yaml:"openai_api_key,omitempty"`
		OpenAIModel   *string `yaml:"openai_model,omitempty"`
		OpenAIBaseURL *string `yaml:"openai_base_url,omitempty"`
		GeminiAPIKey  string  `yaml:"gemini_api_key,omitempty"`
		GeminiModel   *string `yaml:"gemini_model,omitempty"`
		GeminiBaseURL *string `yaml:"gemini_base_url,omitempty"`
	} `yaml:"llm"`
	Web struct {
		TavilyAPIKey  *string `yaml:"tavily_api_key"`
		TavilyBaseURL *string `yaml:"tavily_base_url"`
	} `yaml:"web"`
	DataDir *string `yaml:"data_dir"`
	CLI     struct {
		MaxIterations *int `yaml:"max_iterations"`
	} `yaml:"cli"`
	Knowledge struct {
		MaxLoadedMemories     *int `yaml:"max_loaded_memories"`
		MaxLoadedMemoryBytes  *int `yaml:"max_loaded_memory_bytes"`
		MaxLoadedSkills       *int `yaml:"max_loaded_skills"`
		MaxLoadedSkillBytes   *int `yaml:"max_loaded_skill_bytes"`
		DefaultDiscoveryLimit *int `yaml:"default_discovery_limit"`
		MaxDiscoveryLimit     *int `yaml:"max_discovery_limit"`
	} `yaml:"knowledge"`
	Sandbox struct {
		DefaultDriver       *string        `yaml:"default_driver"`
		DockerSocket        *string        `yaml:"docker_socket"`
		DockerImage         *string        `yaml:"docker_image"`
		AllowedImages       *[]string      `yaml:"allowed_images"`
		AllowedNetworkModes *[]string      `yaml:"allowed_network_modes"`
		DefaultExecTimeout  *time.Duration `yaml:"default_exec_timeout"`
		MaxExecTimeout      *time.Duration `yaml:"max_exec_timeout"`
		MaxLifetime         *time.Duration `yaml:"max_lifetime"`
		CPULimit            *float64       `yaml:"cpu_limit"`
		MemoryLimitMB       *int           `yaml:"memory_limit_mb"`
		TasksMax            *int           `yaml:"tasks_max"`
		ExtraPaths          *[]string      `yaml:"extra_paths"`
		AutoToolchains      *bool          `yaml:"auto_toolchains"`
	} `yaml:"sandbox"`
}

// checkFileSecurity verifies that a file is regular, has safe permissions, and is owned by expectedUID.
func checkFileSecurity(fi os.FileInfo, path string, expectedUID int) error {
	if !fi.Mode().IsRegular() {
		return fmt.Errorf("config path %s is not a regular file", path)
	}

	// Group and other access must be 0 (e.g. 0600 or 0400)
	if fi.Mode().Perm()&0o077 != 0 {
		return fmt.Errorf("insecure permissions %04o on config file %s: group/other access forbidden, run 'chmod 600 %s'", fi.Mode().Perm(), path, path)
	}

	// Verify UID ownership on unix
	if stat, ok := fi.Sys().(*syscall.Stat_t); ok {
		if expectedUID >= 0 && stat.Uid != uint32(expectedUID) {
			return fmt.Errorf("config file %s is owned by UID %d, expected current user UID %d", path, stat.Uid, expectedUID)
		}
	}

	return nil
}

// LoadFromFile loads and parses a YAML configuration file from path, enforcing strict security and schema validation.
func LoadFromFile(path string) (*Config, error) {
	expandedPath, err := ExpandPath(path)
	if err != nil {
		return nil, fmt.Errorf("failed to expand config path %s: %w", path, err)
	}

	f, err := os.Open(expandedPath)
	if err != nil {
		return nil, fmt.Errorf("cannot open config file %s: %w", path, err)
	}
	defer func() { _ = f.Close() }()

	fi, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("cannot stat config file %s: %w", path, err)
	}

	if err := checkFileSecurity(fi, path, os.Getuid()); err != nil {
		return nil, err
	}

	dec := yaml.NewDecoder(f)
	dec.KnownFields(true)

	var fc fileConfig
	if err := dec.Decode(&fc); err != nil {
		if errors.Is(err, io.EOF) {
			return nil, fmt.Errorf("config file %s is empty", path)
		}
		return nil, fmt.Errorf("failed to decode config %s: %w", path, err)
	}

	var extra any
	if err := dec.Decode(&extra); !errors.Is(err, io.EOF) {
		return nil, fmt.Errorf("multiple YAML documents in %s are not supported", path)
	}

	cfg := Defaults()

	// Apply CLI-specific baseline defaults
	_, defaultDataDir, _ := DefaultPaths()
	if defaultDataDir != "" {
		cfg.DataDir = defaultDataDir
	}
	cfg.SandboxEnabled = false
	cfg.SandboxAllowedNetworkModes = []string{"none", "restricted", "full"}
	cfg.SandboxTasksMax = 1024

	// 1. LLM Settings
	apiKey := strings.TrimSpace(fc.LLM.APIKey)
	if apiKey == "" {
		apiKey = strings.TrimSpace(fc.LLM.OpenAIAPIKey)
	}
	if apiKey == "" {
		apiKey = strings.TrimSpace(fc.LLM.GeminiAPIKey)
	}
	cfg.OpenAIAPIKey = apiKey
	cfg.GeminiAPIKey = apiKey

	if fc.LLM.Model != nil {
		cfg.OpenAIModel = *fc.LLM.Model
		cfg.GeminiModel = *fc.LLM.Model
	} else if fc.LLM.OpenAIModel != nil {
		cfg.OpenAIModel = *fc.LLM.OpenAIModel
		cfg.GeminiModel = *fc.LLM.OpenAIModel
	} else if fc.LLM.GeminiModel != nil {
		cfg.OpenAIModel = *fc.LLM.GeminiModel
		cfg.GeminiModel = *fc.LLM.GeminiModel
	}

	if fc.LLM.BaseURL != nil {
		cfg.OpenAIBaseURL = *fc.LLM.BaseURL
		cfg.GeminiBaseURL = *fc.LLM.BaseURL
	} else if fc.LLM.OpenAIBaseURL != nil {
		cfg.OpenAIBaseURL = *fc.LLM.OpenAIBaseURL
		cfg.GeminiBaseURL = *fc.LLM.OpenAIBaseURL
	} else if fc.LLM.GeminiBaseURL != nil {
		cfg.OpenAIBaseURL = *fc.LLM.GeminiBaseURL
		cfg.GeminiBaseURL = *fc.LLM.GeminiBaseURL
	}
	if cfg.OpenAIBaseURL != "" && !strings.HasSuffix(cfg.OpenAIBaseURL, "/") {
		cfg.OpenAIBaseURL += "/"
		cfg.GeminiBaseURL = cfg.OpenAIBaseURL
	}

	if fc.LLM.RequestTimeout != nil {
		cfg.LLMRequestTimeout = *fc.LLM.RequestTimeout
	} else if fc.LLM.Timeout != nil {
		cfg.LLMRequestTimeout = *fc.LLM.Timeout
	}
	if fc.LLM.FirstByteTimeout != nil {
		cfg.LLMFirstByteTimeout = *fc.LLM.FirstByteTimeout
	}

	if fc.LLM.EmbeddingModel != nil {
		cfg.EmbeddingModel = *fc.LLM.EmbeddingModel
	}
	if fc.LLM.EmbeddingPrecision != nil {
		cfg.EmbeddingPrecision = strings.ToLower(*fc.LLM.EmbeddingPrecision)
	}

	// 2. Web Settings
	if fc.Web.TavilyAPIKey != nil {
		cfg.TavilyAPIKey = *fc.Web.TavilyAPIKey
	}
	if fc.Web.TavilyBaseURL != nil {
		cfg.TavilyBaseURL = strings.TrimSuffix(*fc.Web.TavilyBaseURL, "/")
	}

	// 3. DataDir
	if fc.DataDir != nil {
		expanded, err := ExpandPath(*fc.DataDir)
		if err != nil {
			return nil, fmt.Errorf("failed to expand data_dir: %w", err)
		}
		cfg.DataDir = expanded
	}

	// 4. CLI Settings
	if fc.CLI.MaxIterations != nil {
		cfg.CLIMaxIterations = *fc.CLI.MaxIterations
	}

	// 5. Knowledge Settings
	if fc.Knowledge.MaxLoadedMemories != nil {
		cfg.KnowledgeMaxLoadedMemories = *fc.Knowledge.MaxLoadedMemories
	}
	if fc.Knowledge.MaxLoadedMemoryBytes != nil {
		cfg.KnowledgeMaxLoadedMemoryBytes = *fc.Knowledge.MaxLoadedMemoryBytes
	}
	if fc.Knowledge.MaxLoadedSkills != nil {
		cfg.KnowledgeMaxLoadedSkills = *fc.Knowledge.MaxLoadedSkills
	}
	if fc.Knowledge.MaxLoadedSkillBytes != nil {
		cfg.KnowledgeMaxLoadedSkillBytes = *fc.Knowledge.MaxLoadedSkillBytes
	}
	if fc.Knowledge.DefaultDiscoveryLimit != nil {
		cfg.KnowledgeDefaultDiscoveryLimit = *fc.Knowledge.DefaultDiscoveryLimit
	}
	if fc.Knowledge.MaxDiscoveryLimit != nil {
		cfg.KnowledgeMaxDiscoveryLimit = *fc.Knowledge.MaxDiscoveryLimit
	}

	// 6. Sandbox Settings
	if fc.Sandbox.DefaultDriver != nil {
		cfg.SandboxDefaultDriver = *fc.Sandbox.DefaultDriver
	}
	if fc.Sandbox.DockerSocket != nil {
		cfg.SandboxDockerSocket = *fc.Sandbox.DockerSocket
	}
	if fc.Sandbox.DockerImage != nil {
		cfg.SandboxDockerImage = *fc.Sandbox.DockerImage
	}
	if fc.Sandbox.AllowedImages != nil {
		cfg.SandboxAllowedImages = append([]string(nil), (*fc.Sandbox.AllowedImages)...)
	}
	if fc.Sandbox.AllowedNetworkModes != nil {
		cfg.SandboxAllowedNetworkModes = append([]string(nil), (*fc.Sandbox.AllowedNetworkModes)...)
	}
	if fc.Sandbox.DefaultExecTimeout != nil {
		cfg.SandboxDefaultExecTimeout = *fc.Sandbox.DefaultExecTimeout
	}
	if fc.Sandbox.MaxExecTimeout != nil {
		cfg.SandboxMaxExecTimeout = *fc.Sandbox.MaxExecTimeout
	}
	if fc.Sandbox.MaxLifetime != nil {
		cfg.SandboxMaxLifetime = *fc.Sandbox.MaxLifetime
	}
	if fc.Sandbox.CPULimit != nil {
		cfg.SandboxCPULimit = *fc.Sandbox.CPULimit
	}
	if fc.Sandbox.MemoryLimitMB != nil {
		cfg.SandboxMemoryLimitMB = *fc.Sandbox.MemoryLimitMB
	}
	if fc.Sandbox.TasksMax != nil {
		cfg.SandboxTasksMax = *fc.Sandbox.TasksMax
	} else if val := os.Getenv("BOB_SANDBOX_TASKS_MAX"); strings.TrimSpace(val) != "" {
		if v, err := strconv.Atoi(strings.TrimSpace(val)); err == nil {
			cfg.SandboxTasksMax = v
		}
	} else if val := os.Getenv("SANDBOX_TASKS_MAX"); strings.TrimSpace(val) != "" {
		if v, err := strconv.Atoi(strings.TrimSpace(val)); err == nil {
			cfg.SandboxTasksMax = v
		}
	}
	if fc.Sandbox.ExtraPaths != nil {
		cfg.SandboxExtraPaths = append([]string(nil), (*fc.Sandbox.ExtraPaths)...)
	} else if val := os.Getenv("BOB_SANDBOX_EXTRA_PATHS"); strings.TrimSpace(val) != "" {
		cfg.SandboxExtraPaths = parsePathList(val)
	} else if val := os.Getenv("SANDBOX_EXTRA_PATHS"); strings.TrimSpace(val) != "" {
		cfg.SandboxExtraPaths = parsePathList(val)
	}

	if fc.Sandbox.AutoToolchains != nil {
		cfg.SandboxAutoToolchains = *fc.Sandbox.AutoToolchains
	} else if val := os.Getenv("BOB_SANDBOX_AUTO_TOOLCHAINS"); strings.TrimSpace(val) != "" {
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

	if cfg.SandboxDockerImage == "" && len(cfg.SandboxAllowedImages) > 0 {
		cfg.SandboxDockerImage = cfg.SandboxAllowedImages[0]
	}

	return cfg, nil
}

// ValidateCLI checks that configuration is valid for CLI execution mode.
func (c *Config) ValidateCLI() error {
	if strings.TrimSpace(c.OpenAIAPIKey) == "" && strings.TrimSpace(c.GeminiAPIKey) == "" {
		return errors.New("api_key is required in config")
	}
	if strings.TrimSpace(c.OpenAIModel) == "" && strings.TrimSpace(c.GeminiModel) == "" {
		return errors.New("model is required in config")
	}

	// Normalize into OpenAI* fields
	if c.OpenAIAPIKey == "" {
		c.OpenAIAPIKey = c.GeminiAPIKey
	}
	if c.OpenAIModel == "" {
		c.OpenAIModel = c.GeminiModel
	}
	if c.OpenAIBaseURL == "" {
		c.OpenAIBaseURL = c.GeminiBaseURL
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

	if c.CLIMaxIterations <= 0 {
		return fmt.Errorf("cli.max_iterations must be positive, got %d", c.CLIMaxIterations)
	}

	driver := strings.TrimSpace(c.SandboxDefaultDriver)
	if driver != "" && driver != "bwrap" && driver != "docker" {
		return fmt.Errorf("unknown sandbox default driver %q: must be 'bwrap' or 'docker'", c.SandboxDefaultDriver)
	}

	// Validate sandbox configuration on a copy with Enabled: true to catch invalid settings
	sbxCopy := c.SandboxConfig()
	sbxCopy.Enabled = true
	if math.IsNaN(sbxCopy.CPULimit) || math.IsInf(sbxCopy.CPULimit, 0) || sbxCopy.CPULimit <= 0 {
		return fmt.Errorf("sandbox cpu limit must be a positive finite number, got %v", sbxCopy.CPULimit)
	}
	if err := sbxCopy.Validate(); err != nil {
		return fmt.Errorf("invalid sandbox configuration: %w", err)
	}

	if c.KnowledgeMaxLoadedMemories <= 0 {
		return fmt.Errorf("knowledge.max_loaded_memories must be positive: %d", c.KnowledgeMaxLoadedMemories)
	}
	if c.KnowledgeMaxLoadedMemoryBytes <= 0 {
		return fmt.Errorf("knowledge.max_loaded_memory_bytes must be positive: %d", c.KnowledgeMaxLoadedMemoryBytes)
	}
	if c.KnowledgeMaxLoadedSkills <= 0 {
		return fmt.Errorf("knowledge.max_loaded_skills must be positive: %d", c.KnowledgeMaxLoadedSkills)
	}
	if c.KnowledgeMaxLoadedSkillBytes <= 0 {
		return fmt.Errorf("knowledge.max_loaded_skill_bytes must be positive: %d", c.KnowledgeMaxLoadedSkillBytes)
	}
	if c.KnowledgeDefaultDiscoveryLimit <= 0 {
		return fmt.Errorf("knowledge.default_discovery_limit must be positive: %d", c.KnowledgeDefaultDiscoveryLimit)
	}
	if c.KnowledgeMaxDiscoveryLimit <= 0 {
		return fmt.Errorf("knowledge.max_discovery_limit must be positive: %d", c.KnowledgeMaxDiscoveryLimit)
	}
	if c.KnowledgeMaxDiscoveryLimit < c.KnowledgeDefaultDiscoveryLimit {
		return fmt.Errorf("knowledge.max_discovery_limit (%d) cannot be less than default_discovery_limit (%d)", c.KnowledgeMaxDiscoveryLimit, c.KnowledgeDefaultDiscoveryLimit)
	}

	return nil
}

// ConfigTemplateContent is the default annotated template written by WriteTemplate.
const ConfigTemplateContent = `# Bob Agent Configuration
# Permissions: 0600 (chmod 600 ~/.config/bob/config.yaml)

llm:
  api_key: "" # Required: OpenAI or Google Gemini API key
  # base_url: "https://generativelanguage.googleapis.com/v1beta/openai/"
  model: "gemini-3.8-flash"
  # request_timeout: "10m" # LLM API end-to-end request timeout (default: 10m)
  # first_byte_timeout: "1m" # LLM API initial response header timeout (default: 1m)
  # embedding_model: "" # Local model by default; "none" to disable
  # embedding_precision: "bf16"

web:
  # tavily_api_key: "" # Optional: enables web_search tool (web_fetch is always available)
  # tavily_base_url: "https://api.tavily.com"

# data_dir: "~/.local/share/bob"

cli:
  max_iterations: 20

knowledge:
  # max_loaded_memories: 8
  # max_loaded_memory_bytes: 16384
  # max_loaded_skills: 3
  # max_loaded_skill_bytes: 24576
  # default_discovery_limit: 5
  # max_discovery_limit: 10

sandbox:
  default_driver: "bwrap" # "bwrap" or "docker"
  # docker_socket: "/var/run/docker.sock"
  # docker_image: "python:3.12-slim"
  # allowed_network_modes: ["none", "restricted", "full"]
  # default_exec_timeout: "1m"
  # max_exec_timeout: "10m"
  # max_lifetime: "30m"
  # cpu_limit: 1.0
  # memory_limit_mb: 512
  # tasks_max: 1024
  # extra_paths: [] # Additional host directories to mount read-only and prepend to sandbox $PATH
  # auto_toolchains: false # Automatically discover and mount host dev toolchains (mise, pyenv, cargo)
`

// WriteTemplate creates a commented default configuration file at path with 0600 permissions,
// refusing to overwrite an existing file.
func WriteTemplate(path string) error {
	expandedPath, err := ExpandPath(path)
	if err != nil {
		return fmt.Errorf("failed to expand template path %s: %w", path, err)
	}

	dir := filepath.Dir(expandedPath)
	if err := os.MkdirAll(dir, 0700); err != nil {
		return fmt.Errorf("failed to create directory %s: %w", dir, err)
	}

	f, err := os.OpenFile(expandedPath, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		if os.IsExist(err) {
			return fmt.Errorf("refusing to overwrite existing config file: %s", path)
		}
		return fmt.Errorf("failed to create config file %s: %w", path, err)
	}

	var writeErr error
	defer func() {
		if closeErr := f.Close(); closeErr != nil && writeErr == nil {
			writeErr = closeErr
		}
		if writeErr != nil {
			_ = os.Remove(expandedPath)
		}
	}()

	if _, err := io.WriteString(f, ConfigTemplateContent); err != nil {
		writeErr = fmt.Errorf("failed to write template content to %s: %w", path, err)
		return writeErr
	}

	return nil
}
