package tools

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"bob/internal/agentapi"
	"bob/internal/config"
	"bob/internal/sandbox"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestFormatExecResult(t *testing.T) {
	t.Run("nil result", func(t *testing.T) {
		assert.Equal(t, "(No result)", FormatExecResult(nil))
	})

	t.Run("stdout only", func(t *testing.T) {
		res := &sandbox.ExecResult{
			ExitCode: 0,
			Duration: 123 * time.Millisecond,
			Stdout:   "hello world\n",
		}
		expected := "Exit Code: 0\nDuration: 123ms\n\n[stdout]\nhello world\n"
		assert.Equal(t, expected, FormatExecResult(res))
	})

	t.Run("stderr only", func(t *testing.T) {
		res := &sandbox.ExecResult{
			ExitCode: 1,
			Duration: 50 * time.Millisecond,
			Stderr:   "command not found\n",
		}
		expected := "Exit Code: 1\nDuration: 50ms\n\n[stderr]\ncommand not found\n"
		assert.Equal(t, expected, FormatExecResult(res))
	})

	t.Run("both stdout and stderr", func(t *testing.T) {
		res := &sandbox.ExecResult{
			ExitCode: 2,
			Duration: 250 * time.Millisecond,
			Stdout:   "partial output\n",
			Stderr:   "warning emitted\n",
		}
		expected := "Exit Code: 2\nDuration: 250ms\n\n[stdout]\npartial output\n\n[stderr]\nwarning emitted\n"
		assert.Equal(t, expected, FormatExecResult(res))
	})

	t.Run("no output", func(t *testing.T) {
		res := &sandbox.ExecResult{
			ExitCode: 0,
			Duration: 10 * time.Millisecond,
		}
		expected := "Exit Code: 0\nDuration: 10ms\n\n(No output)"
		assert.Equal(t, expected, FormatExecResult(res))
	})
}

type execTestMockDriver struct {
	available bool
	lastCmd   []string
	lastTO    time.Duration
	result    *sandbox.ExecResult
	err       error
}

func (m *execTestMockDriver) Type() sandbox.DriverType {
	return sandbox.DriverBwrap
}

func (m *execTestMockDriver) Available(ctx context.Context) bool {
	return m.available
}

func (m *execTestMockDriver) Create(ctx context.Context, sbx *sandbox.UserSandbox, workspace string) error {
	return nil
}

func (m *execTestMockDriver) Exec(ctx context.Context, sbx *sandbox.UserSandbox, cmd []string, timeout time.Duration) (*sandbox.ExecResult, error) {
	m.lastCmd = cmd
	m.lastTO = timeout
	if m.err != nil {
		return nil, m.err
	}
	return m.result, nil
}

func (m *execTestMockDriver) Destroy(ctx context.Context, sbx *sandbox.UserSandbox) error {
	return nil
}

func TestExecTool(t *testing.T) {
	ctx := context.Background()

	cfg := config.Defaults().SandboxConfig()
	cfg.Enabled = true
	cfg.Drivers = []string{"bwrap"}

	info := ExecToolInfo{
		Driver:             "bwrap",
		WorkspaceMount:     "rw",
		Network:            "none",
		DefaultExecTimeout: 10 * time.Second,
		MaxExecTimeout:     30 * time.Second,
	}

	t.Run("Definitions schema", func(t *testing.T) {
		driver := &execTestMockDriver{available: true}
		mgr := sandbox.NewManager(cfg, []sandbox.Driver{driver})

		tool := NewExecTool(mgr, "user-123", info)
		defs, err := tool.Definitions(ctx)
		require.NoError(t, err)
		require.Len(t, defs, 1)

		assert.Equal(t, "execute", defs[0].Schema.Function.Name)
		assert.False(t, defs[0].ReadOnly)
		assert.Contains(t, defs[0].Schema.Function.Description, "driver: bwrap")
		assert.Contains(t, defs[0].Schema.Function.Description, "workspace: rw")
		assert.Contains(t, defs[0].Schema.Function.Description, "network: none")
	})

	t.Run("Execute unknown tool name", func(t *testing.T) {
		driver := &execTestMockDriver{available: true}
		mgr := sandbox.NewManager(cfg, []sandbox.Driver{driver})
		tool := NewExecTool(mgr, "user-123", info)

		_, err := tool.Execute(ctx, agentapi.ToolCall{
			Name:      "other_tool",
			Arguments: `{"command":"ls"}`,
		})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "unknown tool")
	})

	t.Run("Execute nil sandbox manager", func(t *testing.T) {
		tool := NewExecTool(nil, "user-123", info)
		_, err := tool.Execute(ctx, agentapi.ToolCall{
			Name:      "execute",
			Arguments: `{"command":"ls"}`,
		})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "sandbox manager is not configured")
	})

	t.Run("Execute invalid json arguments", func(t *testing.T) {
		driver := &execTestMockDriver{available: true}
		mgr := sandbox.NewManager(cfg, []sandbox.Driver{driver})
		tool := NewExecTool(mgr, "user-123", info)

		_, err := tool.Execute(ctx, agentapi.ToolCall{
			Name:      "execute",
			Arguments: `{invalid json}`,
		})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "failed to parse execute arguments")
	})

	t.Run("Execute empty command", func(t *testing.T) {
		driver := &execTestMockDriver{available: true}
		mgr := sandbox.NewManager(cfg, []sandbox.Driver{driver})
		tool := NewExecTool(mgr, "user-123", info)

		_, err := tool.Execute(ctx, agentapi.ToolCall{
			Name:      "execute",
			Arguments: `{"command":"   "}`,
		})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "command cannot be empty")
	})

	t.Run("Execute success and default timeout", func(t *testing.T) {
		driver := &execTestMockDriver{
			available: true,
			result: &sandbox.ExecResult{
				ExitCode: 0,
				Duration: 50 * time.Millisecond,
				Stdout:   "file1.txt\nfile2.txt\n",
			},
		}
		mgr := sandbox.NewManager(cfg, []sandbox.Driver{driver})
		// Create sandbox for user-123
		_, err := mgr.RequestSandbox(ctx, "user-123", "chat-1", sandbox.RequestParams{
			Driver:      sandbox.DriverBwrap,
			NetworkMode: sandbox.NetworkNone,
		})
		require.NoError(t, err)
		_, err = mgr.ApproveSandbox(ctx, "user-123")
		require.NoError(t, err)

		tool := NewExecTool(mgr, "user-123", info)

		res, err := tool.Execute(ctx, agentapi.ToolCall{
			Name:      "execute",
			Arguments: `{"command":"ls -la"}`,
		})
		require.NoError(t, err)
		assert.Contains(t, res.Content, "Exit Code: 0")
		assert.Contains(t, res.Content, "file1.txt")
		assert.Equal(t, []string{"sh", "-c", "ls -la"}, driver.lastCmd)
		assert.Equal(t, 10*time.Second, driver.lastTO)
	})

	t.Run("Execute clamped max timeout", func(t *testing.T) {
		driver := &execTestMockDriver{
			available: true,
			result: &sandbox.ExecResult{
				ExitCode: 0,
				Duration: 10 * time.Millisecond,
				Stdout:   "done\n",
			},
		}
		mgr := sandbox.NewManager(cfg, []sandbox.Driver{driver})
		_, err := mgr.RequestSandbox(ctx, "user-123", "chat-1", sandbox.RequestParams{
			Driver:      sandbox.DriverBwrap,
			NetworkMode: sandbox.NetworkNone,
		})
		require.NoError(t, err)
		_, err = mgr.ApproveSandbox(ctx, "user-123")
		require.NoError(t, err)

		tool := NewExecTool(mgr, "user-123", info)

		_, err = tool.Execute(ctx, agentapi.ToolCall{
			Name:      "execute",
			Arguments: `{"command":"sleep 100", "timeout_seconds": 60}`,
		})
		require.NoError(t, err)
		assert.Equal(t, 30*time.Second, driver.lastTO, "timeout should be clamped to MaxExecTimeout")
	})

	t.Run("Execute execution failure / missing sandbox", func(t *testing.T) {
		driver := &execTestMockDriver{available: true}
		mgr := sandbox.NewManager(cfg, []sandbox.Driver{driver})
		// Do not provision/approve sandbox for user-999

		tool := NewExecTool(mgr, "user-999", info)
		res, err := tool.Execute(ctx, agentapi.ToolCall{
			Name:      "execute",
			Arguments: `{"command":"echo hello"}`,
		})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "execution failed")
		assert.Contains(t, res.Content, "execution failed")
	})

	t.Run("Execute driver error", func(t *testing.T) {
		driver := &execTestMockDriver{
			available: true,
			err:       errors.New("kernel cgroup failure"),
		}
		mgr := sandbox.NewManager(cfg, []sandbox.Driver{driver})
		_, err := mgr.RequestSandbox(ctx, "user-123", "chat-1", sandbox.RequestParams{
			Driver:      sandbox.DriverBwrap,
			NetworkMode: sandbox.NetworkNone,
		})
		require.NoError(t, err)
		_, err = mgr.ApproveSandbox(ctx, "user-123")
		require.NoError(t, err)

		tool := NewExecTool(mgr, "user-123", info)
		_, err = tool.Execute(ctx, agentapi.ToolCall{
			Name:      "execute",
			Arguments: `{"command":"echo fail"}`,
		})
		require.Error(t, err)
		assert.True(t, strings.Contains(err.Error(), "cgroup failure"))
	})
}
