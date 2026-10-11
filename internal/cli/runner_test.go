package cli

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	openai "github.com/sashabaranov/go-openai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"bob/internal/bootstrap"
	"bob/internal/config"
	"bob/internal/fsm"
	"bob/internal/sandbox"
)

type failWriter struct{}

func (f *failWriter) Write(p []byte) (n int, err error) {
	return 0, errors.New("disk full")
}

type mockSandboxDriver struct {
	available   bool
	lastCmd     []string
	lastTimeout time.Duration
	result      *sandbox.ExecResult
	err         error
}

func (m *mockSandboxDriver) Type() sandbox.DriverType {
	return sandbox.DriverBwrap
}

func (m *mockSandboxDriver) Available(ctx context.Context) bool {
	return m.available
}

func (m *mockSandboxDriver) Create(ctx context.Context, sbx *sandbox.UserSandbox, workspace string) error {
	return nil
}

func (m *mockSandboxDriver) Exec(ctx context.Context, sbx *sandbox.UserSandbox, cmd []string, timeout time.Duration) (*sandbox.ExecResult, error) {
	m.lastCmd = cmd
	m.lastTimeout = timeout
	if m.err != nil {
		return nil, m.err
	}
	return m.result, nil
}

func (m *mockSandboxDriver) Destroy(ctx context.Context, sbx *sandbox.UserSandbox) error {
	return nil
}

func setupTestRuntime(t *testing.T, handler http.HandlerFunc) (*bootstrap.Runtime, *httptest.Server) {
	t.Helper()
	server := httptest.NewServer(handler)
	cfg := &config.Config{
		DataDir:                        t.TempDir(),
		OpenAIModel:                    "test-model",
		OpenAIAPIKey:                   "test-key",
		OpenAIBaseURL:                  server.URL,
		TownhallToolMaxIterations:      10,
		DMToolMaxIterations:            20,
		KnowledgeMaxLoadedMemories:     5,
		KnowledgeMaxLoadedMemoryBytes:  1024,
		KnowledgeMaxLoadedSkills:       2,
		KnowledgeMaxLoadedSkillBytes:   2048,
		KnowledgeDefaultDiscoveryLimit: 3,
		KnowledgeMaxDiscoveryLimit:     5,
		SandboxEnabled:                 false,
		EmbeddingModel:                 "none",
		MaxAttachmentSizeBytes:         1024 * 1024,
	}
	rt, err := bootstrap.New(cfg, bootstrap.Options{
		HTTPClient: server.Client(),
	})
	require.NoError(t, err)
	return rt, server
}

func TestRun_Validation(t *testing.T) {
	ctx := context.Background()

	t.Run("nil streams", func(t *testing.T) {
		assert.Equal(t, 1, Run(ctx, nil, Options{Prompt: "test"}, nil, nil))
		var buf bytes.Buffer
		assert.Equal(t, 1, Run(ctx, nil, Options{Prompt: "test"}, &buf, nil))
		assert.Equal(t, 1, Run(ctx, nil, Options{Prompt: "test"}, nil, &buf))
	})

	t.Run("nil runtime", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		code := Run(ctx, nil, Options{Prompt: "test"}, &stdout, &stderr)
		assert.Equal(t, 1, code)
		assert.Contains(t, stderr.String(), "runtime is not initialized")
	})

	t.Run("empty prompt and stdin", func(t *testing.T) {
		rt, srv := setupTestRuntime(t, func(w http.ResponseWriter, r *http.Request) {})
		defer srv.Close()
		defer func() { _ = rt.Close() }()

		var stdout, stderr bytes.Buffer
		code := Run(ctx, rt, Options{Prompt: "   "}, &stdout, &stderr)
		assert.Equal(t, 2, code)
		assert.Contains(t, stderr.String(), "prompt cannot be empty")
	})

	t.Run("stdin exceeds size cap", func(t *testing.T) {
		rt, srv := setupTestRuntime(t, func(w http.ResponseWriter, r *http.Request) {})
		defer srv.Close()
		defer func() { _ = rt.Close() }()

		var stdout, stderr bytes.Buffer
		bigInput := strings.Repeat("A", int(rt.Config.MaxAttachmentSizeBytes)+1)
		code := Run(ctx, rt, Options{Prompt: "test", Stdin: bigInput}, &stdout, &stderr)
		assert.Equal(t, 2, code)
		assert.Contains(t, stderr.String(), "stdin exceeds maximum allowed size")
	})

	t.Run("sandbox requested but manager nil", func(t *testing.T) {
		rt, srv := setupTestRuntime(t, func(w http.ResponseWriter, r *http.Request) {})
		defer srv.Close()
		defer func() { _ = rt.Close() }()

		var stdout, stderr bytes.Buffer
		code := Run(ctx, rt, Options{
			Prompt: "test",
			Sandbox: &SandboxSpec{
				WorkspaceDir: t.TempDir(),
			},
		}, &stdout, &stderr)
		assert.Equal(t, 1, code)
		assert.Contains(t, stderr.String(), "sandbox manager is not enabled or available")
	})
}

func TestRun_PlainAnswer(t *testing.T) {
	ctx := context.Background()

	rt, srv := setupTestRuntime(t, func(w http.ResponseWriter, r *http.Request) {
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
						Content: "Bob says hello!",
					},
					FinishReason: "stop",
				},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	})
	defer srv.Close()
	defer func() { _ = rt.Close() }()

	var stdout, stderr bytes.Buffer
	code := Run(ctx, rt, Options{Prompt: "hello"}, &stdout, &stderr)
	assert.Equal(t, 0, code)
	assert.Equal(t, "Bob says hello!\n", stdout.String())
	assert.Empty(t, stderr.String())
}

func TestRun_StdinWithBackticks(t *testing.T) {
	ctx := context.Background()
	var receivedContent string

	rt, srv := setupTestRuntime(t, func(w http.ResponseWriter, r *http.Request) {
		var req openai.ChatCompletionRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		for _, m := range req.Messages {
			if m.Role == openai.ChatMessageRoleUser {
				receivedContent = m.Content
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
						Content: "Processed diff.",
					},
					FinishReason: "stop",
				},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	})
	defer srv.Close()
	defer func() { _ = rt.Close() }()

	var stdout, stderr bytes.Buffer
	code := Run(ctx, rt, Options{
		Prompt: "Review diff",
		Stdin:  "```go\nfunc test() {}\n```",
	}, &stdout, &stderr)

	assert.Equal(t, 0, code)
	assert.Equal(t, "Processed diff.\n", stdout.String())
	assert.Contains(t, receivedContent, "````")
	assert.Contains(t, receivedContent, "Review diff")
}

func TestRun_StdinWithDashPrompt(t *testing.T) {
	ctx := context.Background()
	var receivedContent string

	rt, srv := setupTestRuntime(t, func(w http.ResponseWriter, r *http.Request) {
		var req openai.ChatCompletionRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		for _, m := range req.Messages {
			if m.Role == openai.ChatMessageRoleUser {
				receivedContent = m.Content
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
						Content: "Prompt from stdin handled.",
					},
					FinishReason: "stop",
				},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	})
	defer srv.Close()
	defer func() { _ = rt.Close() }()

	var stdout, stderr bytes.Buffer
	code := Run(ctx, rt, Options{
		Prompt: "-",
		Stdin:  "Direct instruction via stdin",
	}, &stdout, &stderr)

	assert.Equal(t, 0, code)
	assert.Equal(t, "Prompt from stdin handled.\n", stdout.String())
	assert.Equal(t, "Direct instruction via stdin", receivedContent)
}

func TestRun_ToolCall_ProgressAndQuiet(t *testing.T) {
	ctx := context.Background()

	// Web page mock
	pageServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain")
		_, _ = w.Write([]byte("Go 1.26 release notes content"))
	}))
	defer pageServer.Close()

	var callCount int32
	handler := func(w http.ResponseWriter, r *http.Request) {
		count := atomic.AddInt32(&callCount, 1)
		w.Header().Set("Content-Type", "application/json")

		if count == 1 {
			// First turn: call web_fetch
			resp := openai.ChatCompletionResponse{
				ID:    "chatcmpl-tool",
				Model: "test-model",
				Choices: []openai.ChatCompletionChoice{
					{
						Index: 0,
						Message: openai.ChatCompletionMessage{
							Role: openai.ChatMessageRoleAssistant,
							ToolCalls: []openai.ToolCall{
								{
									ID:   "call_fetch_1",
									Type: "function",
									Function: openai.FunctionCall{
										Name: "web_fetch",
										Arguments: fmt.Sprintf(`{"url": "%s"}`, pageServer.URL),
									},
								},
							},
						},
						FinishReason: "tool_calls",
					},
				},
			}
			_ = json.NewEncoder(w).Encode(resp)
		} else {
			// Second turn: reply with final answer
			resp := openai.ChatCompletionResponse{
				ID:    "chatcmpl-final",
				Model: "test-model",
				Choices: []openai.ChatCompletionChoice{
					{
						Index: 0,
						Message: openai.ChatCompletionMessage{
							Role:    openai.ChatMessageRoleAssistant,
							Content: "Found release notes for Go 1.26.",
						},
						FinishReason: "stop",
					},
				},
			}
			_ = json.NewEncoder(w).Encode(resp)
		}
	}

	t.Run("non-quiet progress output", func(t *testing.T) {
		atomic.StoreInt32(&callCount, 0)
		rt, srv := setupTestRuntime(t, handler)
		defer srv.Close()
		defer func() { _ = rt.Close() }()

		var stdout, stderr bytes.Buffer
		code := Run(ctx, rt, Options{Prompt: "Fetch notes", Quiet: false}, &stdout, &stderr)
		assert.Equal(t, 0, code)
		outStr := stdout.String()

		// Should contain progress step line, a blank separator line, then final answer
		assert.Contains(t, outStr, "web_fetch")
		assert.Contains(t, outStr, "\n\nFound release notes for Go 1.26.\n")
	})

	t.Run("quiet mode suppresses progress", func(t *testing.T) {
		atomic.StoreInt32(&callCount, 0)
		rt, srv := setupTestRuntime(t, handler)
		defer srv.Close()
		defer func() { _ = rt.Close() }()

		var stdout, stderr bytes.Buffer
		code := Run(ctx, rt, Options{Prompt: "Fetch notes", Quiet: true}, &stdout, &stderr)
		assert.Equal(t, 0, code)
		assert.Equal(t, "Found release notes for Go 1.26.\n", stdout.String())
		assert.NotContains(t, stdout.String(), "web_fetch")
	})
}

func TestRun_RejectNonAllowlistedTool(t *testing.T) {
	ctx := context.Background()

	var callCount int32
	handler := func(w http.ResponseWriter, r *http.Request) {
		count := atomic.AddInt32(&callCount, 1)
		w.Header().Set("Content-Type", "application/json")

		if count == 1 {
			// LLM attempts to call a non-allowlisted tool
			resp := openai.ChatCompletionResponse{
				ID:    "chatcmpl-disallowed",
				Model: "test-model",
				Choices: []openai.ChatCompletionChoice{
					{
						Index: 0,
						Message: openai.ChatCompletionMessage{
							Role: openai.ChatMessageRoleAssistant,
							ToolCalls: []openai.ToolCall{
								{
									ID:   "call_sched_1",
									Type: "function",
									Function: openai.FunctionCall{
										Name:      "create_schedule",
										Arguments: `{"schedule_name": "test"}`,
									},
								},
							},
						},
						FinishReason: "tool_calls",
					},
				},
			}
			_ = json.NewEncoder(w).Encode(resp)
		} else {
			resp := openai.ChatCompletionResponse{
				ID:    "chatcmpl-final",
				Model: "test-model",
				Choices: []openai.ChatCompletionChoice{
					{
						Index: 0,
						Message: openai.ChatCompletionMessage{
							Role:    openai.ChatMessageRoleAssistant,
							Content: "Understood, scheduling is not allowed.",
						},
						FinishReason: "stop",
					},
				},
			}
			_ = json.NewEncoder(w).Encode(resp)
		}
	}

	rt, srv := setupTestRuntime(t, handler)
	defer srv.Close()
	defer func() { _ = rt.Close() }()

	var stdout, stderr bytes.Buffer
	code := Run(ctx, rt, Options{Prompt: "Schedule task", Quiet: true}, &stdout, &stderr)
	assert.Equal(t, 1, code)
	assert.Contains(t, stderr.String(), "unauthorized tool call")
}

func TestRun_WithSandbox_Execute(t *testing.T) {
	ctx := context.Background()

	driver := &mockSandboxDriver{
		available: true,
		result: &sandbox.ExecResult{
			ExitCode: 0,
			Duration: 50 * time.Millisecond,
			Stdout:   "total 0\n",
		},
	}

	var callCount int32
	handler := func(w http.ResponseWriter, r *http.Request) {
		count := atomic.AddInt32(&callCount, 1)
		w.Header().Set("Content-Type", "application/json")

		if count == 1 {
			resp := openai.ChatCompletionResponse{
				ID:    "chatcmpl-exec",
				Model: "test-model",
				Choices: []openai.ChatCompletionChoice{
					{
						Index: 0,
						Message: openai.ChatCompletionMessage{
							Role: openai.ChatMessageRoleAssistant,
							ToolCalls: []openai.ToolCall{
								{
									ID:   "call_exec_1",
									Type: "function",
									Function: openai.FunctionCall{
										Name:      "execute",
										Arguments: `{"command": "ls -la"}`,
									},
								},
							},
						},
						FinishReason: "tool_calls",
					},
				},
			}
			_ = json.NewEncoder(w).Encode(resp)
		} else {
			resp := openai.ChatCompletionResponse{
				ID:    "chatcmpl-final",
				Model: "test-model",
				Choices: []openai.ChatCompletionChoice{
					{
						Index: 0,
						Message: openai.ChatCompletionMessage{
							Role:    openai.ChatMessageRoleAssistant,
							Content: "Workspace is empty.",
						},
						FinishReason: "stop",
					},
				},
			}
			_ = json.NewEncoder(w).Encode(resp)
		}
	}

	rt, srv := setupTestRuntime(t, handler)
	defer srv.Close()
	defer func() { _ = rt.Close() }()

	// Inject sandbox manager with mock driver
	rt.Config.SandboxDefaultExecTimeout = 3 * time.Minute
	sbxCfg := config.Defaults().SandboxConfig()
	sbxCfg.Enabled = true
	sbxCfg.Drivers = []string{"bwrap"}
	rt.SandboxManager = sandbox.NewManager(sbxCfg, []sandbox.Driver{driver})

	workspaceDir := t.TempDir()
	var stdout, stderr bytes.Buffer
	code := Run(ctx, rt, Options{
		Prompt: "list files",
		Sandbox: &SandboxSpec{
			WorkspaceDir: workspaceDir,
			ReadOnly:     true,
			Driver:       "bwrap",
			NetworkMode:  "none",
		},
		Quiet: false,
	}, &stdout, &stderr)

	assert.Equal(t, 0, code)
	assert.Contains(t, stdout.String(), "• execute ls -la")
	assert.Contains(t, stdout.String(), "Workspace is empty.")
	assert.Equal(t, []string{"sh", "-c", "ls -la"}, driver.lastCmd)
	assert.Equal(t, 3*time.Minute, driver.lastTimeout, "SandboxDefaultExecTimeout from config must be propagated to ExecTool")
}

func TestRun_WithSandbox_ExecuteNonZeroExit(t *testing.T) {
	ctx := context.Background()

	driver := &mockSandboxDriver{
		available: true,
		result: &sandbox.ExecResult{
			ExitCode: 42,
			Duration: 25 * time.Millisecond,
			Stderr:   "command failed with 42\n",
		},
	}

	var callCount int32
	handler := func(w http.ResponseWriter, r *http.Request) {
		count := atomic.AddInt32(&callCount, 1)
		w.Header().Set("Content-Type", "application/json")

		if count == 1 {
			resp := openai.ChatCompletionResponse{
				ID:    "chatcmpl-exec-fail",
				Model: "test-model",
				Choices: []openai.ChatCompletionChoice{
					{
						Index: 0,
						Message: openai.ChatCompletionMessage{
							Role: openai.ChatMessageRoleAssistant,
							ToolCalls: []openai.ToolCall{
								{
									ID:   "call_exec_fail",
									Type: "function",
									Function: openai.FunctionCall{
										Name:      "execute",
										Arguments: `{"command": "exit 42"}`,
									},
								},
							},
						},
						FinishReason: "tool_calls",
					},
				},
			}
			_ = json.NewEncoder(w).Encode(resp)
		} else {
			resp := openai.ChatCompletionResponse{
				ID:    "chatcmpl-final",
				Model: "test-model",
				Choices: []openai.ChatCompletionChoice{
					{
						Index: 0,
						Message: openai.ChatCompletionMessage{
							Role:    openai.ChatMessageRoleAssistant,
							Content: "Command exited with code 42.",
						},
						FinishReason: "stop",
					},
				},
			}
			_ = json.NewEncoder(w).Encode(resp)
		}
	}

	rt, srv := setupTestRuntime(t, handler)
	defer srv.Close()
	defer func() { _ = rt.Close() }()

	sbxCfg := config.Defaults().SandboxConfig()
	sbxCfg.Enabled = true
	sbxCfg.Drivers = []string{"bwrap"}
	rt.SandboxManager = sandbox.NewManager(sbxCfg, []sandbox.Driver{driver})

	workspaceDir := t.TempDir()
	var stdout, stderr bytes.Buffer
	code := Run(ctx, rt, Options{
		Prompt: "run failing command",
		Sandbox: &SandboxSpec{
			WorkspaceDir: workspaceDir,
			ReadOnly:     false,
			Driver:       "bwrap",
			NetworkMode:  "none",
		},
		Quiet: false,
	}, &stdout, &stderr)

	assert.Equal(t, 0, code)
	assert.Contains(t, stdout.String(), "✗ execute exit 42")
	assert.Contains(t, stdout.String(), "exit 42")
	assert.Contains(t, stdout.String(), "Command exited with code 42.")
}

func TestRun_ContextCanceled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())

	handler := func(w http.ResponseWriter, r *http.Request) {
		// Cancel context during LLM call
		cancel()
		time.Sleep(50 * time.Millisecond)
		http.Error(w, "context canceled", http.StatusGatewayTimeout)
	}

	rt, srv := setupTestRuntime(t, handler)
	defer srv.Close()
	defer func() { _ = rt.Close() }()

	var stdout, stderr bytes.Buffer
	code := Run(ctx, rt, Options{Prompt: "Will be canceled"}, &stdout, &stderr)
	assert.Equal(t, 130, code)

	// Verify FSM run in SQLite store is terminal (terminated/failed, NOT running)
	store, err := rt.StoreProvider.GetStore(context.Background(), "cli", true)
	require.NoError(t, err)

	active, err := store.ListActiveRuns(context.Background())
	require.NoError(t, err)
	assert.Empty(t, active, "there must be no active runs remaining in SQLite")

	var status string
	err = store.DB().QueryRowContext(context.Background(), "SELECT status FROM fsm_runs").Scan(&status)
	require.NoError(t, err)
	assert.True(t, fsm.RunStatus(status).IsTerminal(), "run status must be terminal, got %s", status)
}

func TestRun_OutputWriteError(t *testing.T) {
	ctx := context.Background()

	rt, srv := setupTestRuntime(t, func(w http.ResponseWriter, r *http.Request) {
		resp := openai.ChatCompletionResponse{
			ID:    "chatcmpl-test",
			Model: "test-model",
			Choices: []openai.ChatCompletionChoice{
				{
					Index: 0,
					Message: openai.ChatCompletionMessage{
						Role:    openai.ChatMessageRoleAssistant,
						Content: "Result to write",
					},
					FinishReason: "stop",
				},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	})
	defer srv.Close()
	defer func() { _ = rt.Close() }()

	var stderr bytes.Buffer
	code := Run(ctx, rt, Options{Prompt: "hello"}, &failWriter{}, &stderr)
	assert.Equal(t, 1, code)
	assert.Contains(t, stderr.String(), "Error writing output")
}
