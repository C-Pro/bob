package config

import (
	"math"
	"os"
	"path/filepath"
	"syscall"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDefaultPaths(t *testing.T) {
	home, err := os.UserHomeDir()
	require.NoError(t, err)

	t.Run("defaults without XDG env", func(t *testing.T) {
		t.Setenv("XDG_CONFIG_HOME", "")
		t.Setenv("XDG_DATA_HOME", "")

		cfgPath, dataDir, err := DefaultPaths()
		require.NoError(t, err)
		assert.Contains(t, cfgPath, filepath.Join("bob", "config.yaml"))
		assert.Equal(t, filepath.Join(home, ".local", "share", "bob"), dataDir)
	})

	t.Run("custom XDG env vars respected", func(t *testing.T) {
		tempDir := t.TempDir()
		t.Setenv("XDG_CONFIG_HOME", filepath.Join(tempDir, "custom_config"))
		t.Setenv("XDG_DATA_HOME", filepath.Join(tempDir, "custom_data"))

		cfgPath, dataDir, err := DefaultPaths()
		require.NoError(t, err)
		assert.Equal(t, filepath.Join(tempDir, "custom_config", "bob", "config.yaml"), cfgPath)
		assert.Equal(t, filepath.Join(tempDir, "custom_data", "bob"), dataDir)
	})

	t.Run("tilde in XDG env expanded", func(t *testing.T) {
		t.Setenv("XDG_CONFIG_HOME", "~/myconfig")
		t.Setenv("XDG_DATA_HOME", "~/mydata")

		cfgPath, dataDir, err := DefaultPaths()
		require.NoError(t, err)
		assert.Equal(t, filepath.Join(home, "myconfig", "bob", "config.yaml"), cfgPath)
		assert.Equal(t, filepath.Join(home, "mydata", "bob"), dataDir)
	})
}

func TestExpandPath(t *testing.T) {
	home, err := os.UserHomeDir()
	require.NoError(t, err)

	expanded, err := ExpandPath("~/foo/bar")
	require.NoError(t, err)
	assert.Equal(t, filepath.Join(home, "foo", "bar"), expanded)

	expanded, err = ExpandPath("~")
	require.NoError(t, err)
	assert.Equal(t, home, expanded)

	expanded, err = ExpandPath("/absolute/path")
	require.NoError(t, err)
	assert.Equal(t, "/absolute/path", expanded)

	expanded, err = ExpandPath("relative/path")
	require.NoError(t, err)
	assert.Equal(t, "relative/path", expanded)

	expanded, err = ExpandPath("~other/path")
	require.NoError(t, err)
	assert.Equal(t, "~other/path", expanded)
}

func TestCheckFileSecurity(t *testing.T) {
	tempDir := t.TempDir()
	filePath := filepath.Join(tempDir, "config.yaml")

	require.NoError(t, os.WriteFile(filePath, []byte("test"), 0600))
	fi, err := os.Stat(filePath)
	require.NoError(t, err)

	// Valid permissions (0600) and current UID
	assert.NoError(t, checkFileSecurity(fi, filePath, os.Getuid()))

	// Valid permissions (0400)
	require.NoError(t, os.Chmod(filePath, 0400))
	fi, err = os.Stat(filePath)
	require.NoError(t, err)
	assert.NoError(t, checkFileSecurity(fi, filePath, os.Getuid()))

	// Insecure permissions (0644 - group/other readable)
	require.NoError(t, os.Chmod(filePath, 0644))
	fi, err = os.Stat(filePath)
	require.NoError(t, err)
	assert.ErrorContains(t, checkFileSecurity(fi, filePath, os.Getuid()), "insecure permissions")

	// Insecure permissions (0660 - group writable)
	require.NoError(t, os.Chmod(filePath, 0660))
	fi, err = os.Stat(filePath)
	require.NoError(t, err)
	assert.ErrorContains(t, checkFileSecurity(fi, filePath, os.Getuid()), "insecure permissions")

	// Directory rejected
	dirFi, err := os.Stat(tempDir)
	require.NoError(t, err)
	assert.ErrorContains(t, checkFileSecurity(dirFi, tempDir, os.Getuid()), "not a regular file")

	// Foreign UID rejected
	require.NoError(t, os.Chmod(filePath, 0600))
	fi, err = os.Stat(filePath)
	require.NoError(t, err)
	foreignUID := os.Getuid() + 9999
	assert.ErrorContains(t, checkFileSecurity(fi, filePath, foreignUID), "owned by UID")
}

func TestLoadFromFile_Valid(t *testing.T) {
	tempDir := t.TempDir()
	cfgFile := filepath.Join(tempDir, "config.yaml")

	content := `
llm:
  api_key: "test-api-key"
  model: "gemini-3.7-flash"
  base_url: "https://custom.openai.api/v1/"
  request_timeout: "12m"
  first_byte_timeout: "2m"
  embedding_model: "custom-embed"
  embedding_precision: "fp32"

web:
  tavily_api_key: "tav-key"
  tavily_base_url: "https://tavily.example.com"

data_dir: "~/bob-data"

cli:
  max_iterations: 15

knowledge:
  max_loaded_memories: 12
  max_loaded_memory_bytes: 32768
  max_loaded_skills: 5
  max_loaded_skill_bytes: 65536
  default_discovery_limit: 8
  max_discovery_limit: 16

sandbox:
  default_driver: "docker"
  docker_socket: "/custom/docker.sock"
  docker_image: "ubuntu:latest"
  allowed_images:
    - "ubuntu:latest"
    - "alpine:latest"
  allowed_network_modes:
    - "none"
    - "restricted"
  default_exec_timeout: "2m"
  max_exec_timeout: "5m"
  max_lifetime: "45m"
  cpu_limit: 2.0
  memory_limit_mb: 1024
  tasks_max: 2048
`

	require.NoError(t, os.WriteFile(cfgFile, []byte(content), 0600))

	cfg, err := LoadFromFile(cfgFile)
	require.NoError(t, err)
	require.NotNil(t, cfg)

	// LLM settings
	assert.Equal(t, "test-api-key", cfg.OpenAIAPIKey)
	assert.Equal(t, "test-api-key", cfg.GeminiAPIKey)
	assert.Equal(t, "gemini-3.7-flash", cfg.OpenAIModel)
	assert.Equal(t, "https://custom.openai.api/v1/", cfg.OpenAIBaseURL)
	assert.Equal(t, 12*time.Minute, cfg.LLMRequestTimeout)
	assert.Equal(t, 2*time.Minute, cfg.LLMFirstByteTimeout)
	assert.Equal(t, "custom-embed", cfg.EmbeddingModel)
	assert.Equal(t, "fp32", cfg.EmbeddingPrecision)

	// Web settings
	assert.Equal(t, "tav-key", cfg.TavilyAPIKey)
	assert.Equal(t, "https://tavily.example.com", cfg.TavilyBaseURL)

	// Data dir expanded
	home, _ := os.UserHomeDir()
	assert.Equal(t, filepath.Join(home, "bob-data"), cfg.DataDir)

	// CLI
	assert.Equal(t, 15, cfg.CLIMaxIterations)

	// Knowledge
	assert.Equal(t, 12, cfg.KnowledgeMaxLoadedMemories)
	assert.Equal(t, 32768, cfg.KnowledgeMaxLoadedMemoryBytes)
	assert.Equal(t, 5, cfg.KnowledgeMaxLoadedSkills)
	assert.Equal(t, 65536, cfg.KnowledgeMaxLoadedSkillBytes)
	assert.Equal(t, 8, cfg.KnowledgeDefaultDiscoveryLimit)
	assert.Equal(t, 16, cfg.KnowledgeMaxDiscoveryLimit)

	// Sandbox
	assert.False(t, cfg.SandboxEnabled, "CLI sandbox should be disabled by default until --mount")
	assert.Equal(t, "docker", cfg.SandboxDefaultDriver)
	assert.Equal(t, "/custom/docker.sock", cfg.SandboxDockerSocket)
	assert.Equal(t, "ubuntu:latest", cfg.SandboxDockerImage)
	assert.Equal(t, []string{"ubuntu:latest", "alpine:latest"}, cfg.SandboxAllowedImages)
	assert.Equal(t, []string{"none", "restricted"}, cfg.SandboxAllowedNetworkModes)
	assert.Equal(t, 2*time.Minute, cfg.SandboxDefaultExecTimeout)
	assert.Equal(t, 5*time.Minute, cfg.SandboxMaxExecTimeout)
	assert.Equal(t, 45*time.Minute, cfg.SandboxMaxLifetime)
	assert.Equal(t, 2.0, cfg.SandboxCPULimit)
	assert.Equal(t, 1024, cfg.SandboxMemoryLimitMB)
	assert.Equal(t, 2048, cfg.SandboxTasksMax)

	// Passes ValidateCLI
	assert.NoError(t, cfg.ValidateCLI())
}

func TestLoadFromFile_Aliases(t *testing.T) {
	tempDir := t.TempDir()
	cfgFile := filepath.Join(tempDir, "config.yaml")

	content := `
llm:
  gemini_api_key: "gemini-secret"
  gemini_model: "gemini-pro"
  gemini_base_url: "https://gemini.example.com"
`
	require.NoError(t, os.WriteFile(cfgFile, []byte(content), 0600))

	cfg, err := LoadFromFile(cfgFile)
	require.NoError(t, err)
	assert.Equal(t, "gemini-secret", cfg.OpenAIAPIKey)
	assert.Equal(t, "gemini-secret", cfg.GeminiAPIKey)
	assert.Equal(t, "gemini-pro", cfg.OpenAIModel)
	assert.Equal(t, "https://gemini.example.com/", cfg.OpenAIBaseURL)
}

func TestLoadFromFile_StrictValidation(t *testing.T) {
	tempDir := t.TempDir()

	t.Run("unknown field rejected", func(t *testing.T) {
		f := filepath.Join(tempDir, "unknown.yaml")
		require.NoError(t, os.WriteFile(f, []byte("unknown_field: 123\n"), 0600))
		_, err := LoadFromFile(f)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "field unknown_field not found")
	})

	t.Run("missing file rejected", func(t *testing.T) {
		_, err := LoadFromFile(filepath.Join(tempDir, "nonexistent.yaml"))
		require.Error(t, err)
		assert.Contains(t, err.Error(), "cannot open config file")
	})

	t.Run("empty file rejected", func(t *testing.T) {
		f := filepath.Join(tempDir, "empty.yaml")
		require.NoError(t, os.WriteFile(f, []byte(""), 0600))
		_, err := LoadFromFile(f)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "empty")
	})

	t.Run("multiple documents rejected", func(t *testing.T) {
		f := filepath.Join(tempDir, "multi.yaml")
		content := "llm:\n  api_key: key1\n---\nllm:\n  api_key: key2\n"
		require.NoError(t, os.WriteFile(f, []byte(content), 0600))
		_, err := LoadFromFile(f)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "multiple YAML documents")
	})

	t.Run("insecure permissions rejected", func(t *testing.T) {
		f := filepath.Join(tempDir, "insecure.yaml")
		require.NoError(t, os.WriteFile(f, []byte("llm:\n  api_key: key\n"), 0644))
		_, err := LoadFromFile(f)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "insecure permissions")
	})
}

func TestValidateCLI(t *testing.T) {
	valid := Defaults()
	valid.OpenAIAPIKey = "valid-key"
	valid.OpenAIModel = "valid-model"
	require.NoError(t, valid.ValidateCLI())

	t.Run("missing API key fails", func(t *testing.T) {
		c := Defaults()
		c.OpenAIAPIKey = ""
		c.GeminiAPIKey = ""
		assert.ErrorContains(t, c.ValidateCLI(), "api_key is required")
	})

	t.Run("missing model fails", func(t *testing.T) {
		c := Defaults()
		c.OpenAIAPIKey = "key"
		c.OpenAIModel = ""
		c.GeminiModel = ""
		assert.ErrorContains(t, c.ValidateCLI(), "model is required")
	})

	t.Run("non-positive cli iterations fails", func(t *testing.T) {
		c := Defaults()
		c.OpenAIAPIKey = "key"
		c.CLIMaxIterations = 0
		assert.ErrorContains(t, c.ValidateCLI(), "cli.max_iterations must be positive")
	})

	t.Run("unknown sandbox driver fails", func(t *testing.T) {
		c := Defaults()
		c.OpenAIAPIKey = "key"
		c.SandboxDefaultDriver = "virtualbox"
		assert.ErrorContains(t, c.ValidateCLI(), "unknown sandbox default driver")
	})

	t.Run("invalid cpu limit fails", func(t *testing.T) {
		c := Defaults()
		c.OpenAIAPIKey = "key"
		c.SandboxCPULimit = math.NaN()
		assert.ErrorContains(t, c.ValidateCLI(), "sandbox cpu limit must be a positive finite number")

		c.SandboxCPULimit = math.Inf(1)
		assert.ErrorContains(t, c.ValidateCLI(), "sandbox cpu limit must be a positive finite number")

		c.SandboxCPULimit = -1
		assert.ErrorContains(t, c.ValidateCLI(), "sandbox cpu limit must be a positive finite number")
	})

	t.Run("negative and zero timeouts in ValidateCLI", func(t *testing.T) {
		c := Defaults()
		c.OpenAIAPIKey = "key"
		c.OpenAIModel = "model"
		c.LLMRequestTimeout = -1
		assert.ErrorContains(t, c.ValidateCLI(), "invalid LLM request timeout")

		c.LLMRequestTimeout = 0
		c.LLMFirstByteTimeout = -1
		assert.ErrorContains(t, c.ValidateCLI(), "invalid LLM first byte timeout")

		c.LLMFirstByteTimeout = 0
		require.NoError(t, c.ValidateCLI())
		assert.Equal(t, 10*time.Minute, c.LLMRequestTimeout)
		assert.Equal(t, 1*time.Minute, c.LLMFirstByteTimeout)
	})
}

func TestWriteTemplate(t *testing.T) {
	tempDir := t.TempDir()
	target := filepath.Join(tempDir, "sub", "config.yaml")

	// First write succeeds
	err := WriteTemplate(target)
	require.NoError(t, err)

	// Verify file was written with mode 0600
	fi, err := os.Stat(target)
	require.NoError(t, err)
	assert.Equal(t, os.FileMode(0600), fi.Mode().Perm())

	// Second write to same path refuses to overwrite
	err = WriteTemplate(target)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "refusing to overwrite")

	// Template can be loaded by LoadFromFile
	cfg, err := LoadFromFile(target)
	require.NoError(t, err)
	require.NotNil(t, cfg)
	assert.Equal(t, "gemini-3.8-flash", cfg.OpenAIModel)
	assert.Equal(t, 100, cfg.CLIMaxIterations)
	assert.Equal(t, "bwrap", cfg.SandboxDefaultDriver)

	// Template has blank api_key, so ValidateCLI reports missing api_key
	err = cfg.ValidateCLI()
	require.ErrorContains(t, err, "api_key is required")

	// Populating api_key passes ValidateCLI
	cfg.OpenAIAPIKey = "my-test-key"
	assert.NoError(t, cfg.ValidateCLI())
}

func TestLoadFromFile_EnvironmentIsolation(t *testing.T) {
	tempDir := t.TempDir()
	cfgFile := filepath.Join(tempDir, "config.yaml")

	content := `
llm:
  api_key: "file-key"
  model: "file-model"

data_dir: "~/file-data"
`
	require.NoError(t, os.WriteFile(cfgFile, []byte(content), 0600))

	// Hostile environment variables should NOT override YAML file values
	t.Setenv("OPENAI_API_KEY", "hostile-env-key")
	t.Setenv("OPENAI_MODEL", "hostile-env-model")
	t.Setenv("DATA_DIR", "/hostile/data/dir")

	cfg, err := LoadFromFile(cfgFile)
	require.NoError(t, err)
	assert.Equal(t, "file-key", cfg.OpenAIAPIKey)
	assert.Equal(t, "file-model", cfg.OpenAIModel)

	home, _ := os.UserHomeDir()
	assert.Equal(t, filepath.Join(home, "file-data"), cfg.DataDir)
}

func TestCheckFileSecurity_UnusedHelper(t *testing.T) {
	_ = syscall.Stat_t{}
}

func TestLoadFromFile_SandboxExtraPaths(t *testing.T) {
	tempDir := t.TempDir()
	cfgFile := filepath.Join(tempDir, "config.yaml")

	content := `
llm:
  api_key: "test-key"
  model: "test-model"

sandbox:
  extra_paths:
    - "/custom/bin"
    - "/opt/toolchain"
`
	require.NoError(t, os.WriteFile(cfgFile, []byte(content), 0600))

	cfg, err := LoadFromFile(cfgFile)
	require.NoError(t, err)
	assert.Equal(t, []string{"/custom/bin", "/opt/toolchain"}, cfg.SandboxExtraPaths)

	sbxCfg := cfg.SandboxConfig()
	assert.Equal(t, []string{"/custom/bin", "/opt/toolchain"}, sbxCfg.ExtraPaths)
}

func TestLoadFromFile_SandboxExtraPaths_Precedence(t *testing.T) {
	tempDir := t.TempDir()

	t.Run("YAML overrides environment variables", func(t *testing.T) {
		t.Setenv("BOB_SANDBOX_EXTRA_PATHS", "/env/path1:/env/path2")
		t.Setenv("SANDBOX_EXTRA_PATHS", "/env/generic")

		cfgFile := filepath.Join(tempDir, "config_yaml.yaml")
		content := `
llm:
  api_key: "test-key"
  model: "test-model"
sandbox:
  extra_paths:
    - "/yaml/override"
`
		require.NoError(t, os.WriteFile(cfgFile, []byte(content), 0600))
		cfg, err := LoadFromFile(cfgFile)
		require.NoError(t, err)
		assert.Equal(t, []string{"/yaml/override"}, cfg.SandboxExtraPaths)
	})

	t.Run("Environment fallback when omitted from YAML", func(t *testing.T) {
		t.Setenv("BOB_SANDBOX_EXTRA_PATHS", "/env/path1:/env/path2")

		cfgFile := filepath.Join(tempDir, "config_noyaml.yaml")
		content := `
llm:
  api_key: "test-key"
  model: "test-model"
`
		require.NoError(t, os.WriteFile(cfgFile, []byte(content), 0600))
		cfg, err := LoadFromFile(cfgFile)
		require.NoError(t, err)
		assert.Equal(t, []string{"/env/path1", "/env/path2"}, cfg.SandboxExtraPaths)
	})
}

func TestLoadFromFile_SandboxAutoToolchains(t *testing.T) {
	tempDir := t.TempDir()

	t.Run("YAML true parses correctly", func(t *testing.T) {
		cfgFile := filepath.Join(tempDir, "config_auto_true.yaml")
		content := `
llm:
  api_key: "test-key"
  model: "test-model"
sandbox:
  auto_toolchains: true
`
		require.NoError(t, os.WriteFile(cfgFile, []byte(content), 0600))
		cfg, err := LoadFromFile(cfgFile)
		require.NoError(t, err)
		assert.True(t, cfg.SandboxAutoToolchains)
		assert.True(t, cfg.SandboxConfig().AutoToolchains)
	})

	t.Run("YAML overrides env var", func(t *testing.T) {
		t.Setenv("BOB_SANDBOX_AUTO_TOOLCHAINS", "true")
		cfgFile := filepath.Join(tempDir, "config_auto_false.yaml")
		content := `
llm:
  api_key: "test-key"
  model: "test-model"
sandbox:
  auto_toolchains: false
`
		require.NoError(t, os.WriteFile(cfgFile, []byte(content), 0600))
		cfg, err := LoadFromFile(cfgFile)
		require.NoError(t, err)
		assert.False(t, cfg.SandboxAutoToolchains)
	})

	t.Run("Environment fallback when omitted from YAML", func(t *testing.T) {
		t.Setenv("BOB_SANDBOX_AUTO_TOOLCHAINS", "true")
		cfgFile := filepath.Join(tempDir, "config_auto_env.yaml")
		content := `
llm:
  api_key: "test-key"
  model: "test-model"
`
		require.NoError(t, os.WriteFile(cfgFile, []byte(content), 0600))
		cfg, err := LoadFromFile(cfgFile)
		require.NoError(t, err)
		assert.True(t, cfg.SandboxAutoToolchains)
	})
}



