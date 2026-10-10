package main

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	openai "github.com/sashabaranov/go-openai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"bob/internal/bootstrap"
	"bob/internal/config"
	"bob/internal/sandbox"
)

func writeValidConfigFile(t *testing.T, llmURL string) string {
	t.Helper()
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "config.yaml")

	content := `llm:
  api_key: "test-key"
  base_url: "` + llmURL + `"
  model: "test-model"
  embedding_model: "none"

data_dir: "` + filepath.Join(dir, "data") + `"

cli:
  max_iterations: 10

sandbox:
  default_driver: "bwrap"
  allowed_images:
    - "python:3.12-slim"
  allowed_network_modes:
    - "none"
    - "restricted"
    - "full"
`
	err := os.WriteFile(cfgPath, []byte(content), 0600)
	require.NoError(t, err)
	return cfgPath
}

func TestRun_Help(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := run([]string{"-h"}, nil, &stdout, &stderr)
	assert.Equal(t, 0, code)
}

func TestRun_UnexpectedPositionalArg(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := run([]string{"-p", "hello", "extra_arg"}, nil, &stdout, &stderr)
	assert.Equal(t, 2, code)
	assert.Contains(t, stderr.String(), "unexpected argument \"extra_arg\"")
}

func TestRun_MissingPrompt_InteractiveNotImplemented(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := run([]string{}, nil, &stdout, &stderr)
	assert.Equal(t, 2, code)
	assert.Contains(t, stderr.String(), "interactive mode is not implemented yet")
}

func TestRun_EmptyPrompt(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := run([]string{"-p", "   "}, nil, &stdout, &stderr)
	assert.Equal(t, 2, code)
	assert.Contains(t, stderr.String(), "prompt cannot be empty")
}

func TestRun_InitConfig(t *testing.T) {
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "config.yaml")

	var stdout, stderr bytes.Buffer
	code := run([]string{"--init-config", "--config", cfgPath}, nil, &stdout, &stderr)
	assert.Equal(t, 0, code)
	assert.Contains(t, stdout.String(), "Initialized configuration template")

	info, err := os.Stat(cfgPath)
	require.NoError(t, err)
	assert.Equal(t, os.FileMode(0600), info.Mode().Perm())

	// Running again should fail without overwriting
	stdout.Reset()
	stderr.Reset()
	code = run([]string{"--init-config", "--config", cfgPath}, nil, &stdout, &stderr)
	assert.Equal(t, 1, code)
	assert.Contains(t, stderr.String(), "refusing to overwrite existing config file")
}

func TestRun_FlagMatrixValidation(t *testing.T) {
	tests := []struct {
		name        string
		args        []string
		expectedErr string
	}{
		{
			name:        "sandbox without mount",
			args:        []string{"-p", "hi", "--sandbox", "bwrap"},
			expectedErr: "--sandbox requires --mount",
		},
		{
			name:        "net without mount",
			args:        []string{"-p", "hi", "--net", "none"},
			expectedErr: "--net requires --mount",
		},
		{
			name:        "image without mount",
			args:        []string{"-p", "hi", "--image", "python:3.12-slim"},
			expectedErr: "--image requires --mount",
		},
		{
			name:        "allow-domain without net restricted",
			args:        []string{"-p", "hi", "--mount", ".", "--allow-domain", "example.com"},
			expectedErr: "--allow-domain requires --net restricted",
		},
		{
			name:        "allow-domain with net full",
			args:        []string{"-p", "hi", "--mount", ".", "--net", "full", "--allow-domain", "example.com"},
			expectedErr: "--allow-domain requires --net restricted",
		},
		{
			name:        "invalid sandbox driver",
			args:        []string{"-p", "hi", "--mount", ".", "--sandbox", "invalid"},
			expectedErr: "invalid sandbox driver \"invalid\"",
		},
		{
			name:        "invalid net mode",
			args:        []string{"-p", "hi", "--mount", ".", "--net", "invalid"},
			expectedErr: "invalid network mode \"invalid\"",
		},
		{
			name:        "max-iterations <= 0",
			args:        []string{"-p", "hi", "--max-iterations", "0"},
			expectedErr: "--max-iterations must be greater than 0",
		},
		{
			name:        "timeout <= 0",
			args:        []string{"-p", "hi", "--timeout", "0s"},
			expectedErr: "--timeout must be greater than 0",
		},
		{
			name:        "negative timeout",
			args:        []string{"-p", "hi", "--timeout", "-1m"},
			expectedErr: "--timeout must be greater than 0",
		},
		{
			name:        "empty model",
			args:        []string{"-p", "hi", "--model", ""},
			expectedErr: "--model cannot be empty",
		},
		{
			name:        "empty mount",
			args:        []string{"-p", "hi", "--mount", ""},
			expectedErr: "--mount path cannot be empty",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			code := run(tc.args, nil, &stdout, &stderr)
			assert.Equal(t, 2, code)
			assert.Contains(t, stderr.String(), tc.expectedErr)
		})
	}
}

func TestRun_ConfigMissing(t *testing.T) {
	nonExistent := filepath.Join(t.TempDir(), "missing.yaml")
	var stdout, stderr bytes.Buffer
	code := run([]string{"-p", "hi", "--config", nonExistent}, nil, &stdout, &stderr)
	assert.Equal(t, 2, code)
	assert.Contains(t, stderr.String(), "configuration file not found")
}

func TestRun_SuccessPath_Plain(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := openai.ChatCompletionResponse{
			ID:    "chatcmpl-test",
			Model: "test-model",
			Choices: []openai.ChatCompletionChoice{
				{
					Index: 0,
					Message: openai.ChatCompletionMessage{
						Role:    openai.ChatMessageRoleAssistant,
						Content: "One-shot answer here.",
					},
					FinishReason: "stop",
				},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	cfgPath := writeValidConfigFile(t, server.URL)

	var stdout, stderr bytes.Buffer
	code := run([]string{"-p", "Hello Bob", "--config", cfgPath}, nil, &stdout, &stderr)
	assert.Equal(t, 0, code)
	assert.Equal(t, "One-shot answer here.\n", stdout.String())
}

func TestRun_PipedStdin_DashPrompt(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req openai.ChatCompletionRequest
		_ = json.NewDecoder(r.Body).Decode(&req)

		resp := openai.ChatCompletionResponse{
			ID:    "chatcmpl-test",
			Model: "test-model",
			Choices: []openai.ChatCompletionChoice{
				{
					Index: 0,
					Message: openai.ChatCompletionMessage{
						Role:    openai.ChatMessageRoleAssistant,
						Content: "Processed stdin instruction.",
					},
					FinishReason: "stop",
				},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	cfgPath := writeValidConfigFile(t, server.URL)

	stdin := strings.NewReader("Instruction via stdin pipe")
	var stdout, stderr bytes.Buffer
	code := run([]string{"-p", "-", "--config", cfgPath}, stdin, &stdout, &stderr)
	assert.Equal(t, 0, code)
	assert.Equal(t, "Processed stdin instruction.\n", stdout.String())
}

func TestRun_PipedStdin_AppendedBlock(t *testing.T) {
	var userMsg string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req openai.ChatCompletionRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		for _, m := range req.Messages {
			if m.Role == openai.ChatMessageRoleUser {
				userMsg = m.Content
			}
		}

		resp := openai.ChatCompletionResponse{
			ID:    "chatcmpl-test",
			Model: "test-model",
			Choices: []openai.ChatCompletionChoice{
				{
					Index: 0,
					Message: openai.ChatCompletionMessage{
						Role:    openai.ChatMessageRoleAssistant,
						Content: "Diff processed.",
					},
					FinishReason: "stop",
				},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	cfgPath := writeValidConfigFile(t, server.URL)

	stdin := strings.NewReader("diff content here")
	var stdout, stderr bytes.Buffer
	code := run([]string{"-p", "Review this diff", "--config", cfgPath}, stdin, &stdout, &stderr)
	assert.Equal(t, 0, code)
	assert.Equal(t, "Diff processed.\n", stdout.String())
	assert.Contains(t, userMsg, "Review this diff")
	assert.Contains(t, userMsg, "diff content here")
}

func TestRun_WithMount_AndQuietFlag(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := openai.ChatCompletionResponse{
			ID:    "chatcmpl-test",
			Model: "test-model",
			Choices: []openai.ChatCompletionChoice{
				{
					Index: 0,
					Message: openai.ChatCompletionMessage{
						Role:    openai.ChatMessageRoleAssistant,
						Content: "Mounted workspace response.",
					},
					FinishReason: "stop",
				},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	cfgPath := writeValidConfigFile(t, server.URL)
	mountDir := t.TempDir()

	// Mock sandbox driver to avoid requiring bwrap on host during tests
	mockDriver := &mockDriverTest{available: true}
	factory := func(cfg *config.Config, opts bootstrap.Options) (*bootstrap.Runtime, error) {
		rt, err := bootstrap.New(cfg, opts)
		if err != nil {
			return nil, err
		}
		sbxCfg := cfg.SandboxConfig()
		sbxCfg.Enabled = true
		sbxCfg.Drivers = []string{"bwrap"}
		rt.SandboxManager = sandbox.NewManager(sbxCfg, []sandbox.Driver{mockDriver})
		return rt, nil
	}

	var stdout, stderr bytes.Buffer
	code := runWithFactory([]string{
		"-p", "List files",
		"--mount", mountDir + ":ro",
		"-q",
		"--config", cfgPath,
	}, nil, &stdout, &stderr, factory)

	assert.Equal(t, 0, code)
	assert.Equal(t, "Mounted workspace response.\n", stdout.String())
	assert.Empty(t, stderr.String())
}

func TestRun_TimeoutExpired(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(150 * time.Millisecond)
		resp := openai.ChatCompletionResponse{
			ID:    "chatcmpl-test",
			Model: "test-model",
			Choices: []openai.ChatCompletionChoice{
				{
					Index: 0,
					Message: openai.ChatCompletionMessage{
						Role:    openai.ChatMessageRoleAssistant,
						Content: "Too late.",
					},
					FinishReason: "stop",
				},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	cfgPath := writeValidConfigFile(t, server.URL)

	var stdout, stderr bytes.Buffer
	code := run([]string{"-p", "hello", "--config", cfgPath, "--timeout", "30ms"}, nil, &stdout, &stderr)
	assert.Equal(t, 1, code)
	assert.Contains(t, stderr.String(), "Error: execution timed out after 30ms (--timeout)")
}

func TestRun_MaxIterationsFlagAndConfig(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := openai.ChatCompletionResponse{
			ID:    "chatcmpl-test",
			Model: "test-model",
			Choices: []openai.ChatCompletionChoice{
				{
					Index: 0,
					Message: openai.ChatCompletionMessage{
						Role:    openai.ChatMessageRoleAssistant,
						Content: "Done.",
					},
					FinishReason: "stop",
				},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	cfgPath := writeValidConfigFile(t, server.URL)

	var recordedMaxIter int
	factory := func(cfg *config.Config, opts bootstrap.Options) (*bootstrap.Runtime, error) {
		recordedMaxIter = cfg.CLIMaxIterations
		return bootstrap.New(cfg, opts)
	}

	var stdout, stderr bytes.Buffer
	// Without --max-iterations, uses value from config (10 in writeValidConfigFile)
	code := runWithFactory([]string{"-p", "hello", "--config", cfgPath}, nil, &stdout, &stderr, factory)
	assert.Equal(t, 0, code)
	assert.Equal(t, 10, recordedMaxIter)

	// With --max-iterations, overrides value from config
	stdout.Reset()
	stderr.Reset()
	code = runWithFactory([]string{"-p", "hello", "--config", cfgPath, "--max-iterations", "42"}, nil, &stdout, &stderr, factory)
	assert.Equal(t, 0, code)
	assert.Equal(t, 42, recordedMaxIter)
}

type mockDriverTest struct {
	available bool
}

func (m *mockDriverTest) Type() sandbox.DriverType {
	return sandbox.DriverBwrap
}

func (m *mockDriverTest) Available(ctx context.Context) bool {
	return m.available
}

func (m *mockDriverTest) Create(ctx context.Context, sbx *sandbox.UserSandbox, workspace string) error {
	return nil
}

func (m *mockDriverTest) Exec(ctx context.Context, sbx *sandbox.UserSandbox, cmd []string, timeout time.Duration) (*sandbox.ExecResult, error) {
	return &sandbox.ExecResult{ExitCode: 0, Duration: 10 * time.Millisecond}, nil
}

func (m *mockDriverTest) Destroy(ctx context.Context, sbx *sandbox.UserSandbox) error {
	return nil
}

func TestRun_LLMTimeoutMessage(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(150 * time.Millisecond)
	}))
	defer server.Close()

	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "config.yaml")
	content := `llm:
  api_key: "test-key"
  base_url: "` + server.URL + `"
  model: "test-model"
  request_timeout: "50ms"
  embedding_model: "none"
`
	require.NoError(t, os.WriteFile(cfgPath, []byte(content), 0600))

	var stdout, stderr bytes.Buffer
	// CLI overall timeout is default (1h), but LLM request_timeout is 50ms
	code := run([]string{"-p", "hello", "--config", cfgPath}, nil, &stdout, &stderr)
	assert.Equal(t, 1, code)
	assert.Contains(t, stderr.String(), "Error: LLM API timeout (50ms): request exceeded deadline")
}
