package tools

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"bob/internal/agentapi"
	"bob/internal/sandbox"

	openai "github.com/sashabaranov/go-openai"
)

// FormatExecResult formats a sandbox execution result into standard human-readable text.
func FormatExecResult(res *sandbox.ExecResult) string {
	if res == nil {
		return "(No result)"
	}
	var b strings.Builder
	fmt.Fprintf(&b, "Exit Code: %d\nDuration: %s\n", res.ExitCode, res.Duration.Round(time.Millisecond))
	if res.Stdout != "" {
		b.WriteString("\n[stdout]\n" + res.Stdout)
	}
	if res.Stderr != "" {
		b.WriteString("\n[stderr]\n" + res.Stderr)
	}
	if res.Stdout == "" && res.Stderr == "" {
		b.WriteString("\n(No output)")
	}
	return b.String()
}

// ExecToolInfo describes the pre-provisioned sandbox execution environment.
type ExecToolInfo struct {
	Driver             string
	WorkspaceMount     string // "ro", "rw", or "none"
	Network            string
	DefaultExecTimeout time.Duration
	MaxExecTimeout     time.Duration
}

// ExecTool implements agentapi.Toolset for running shell commands in a pre-provisioned sandbox.
type ExecTool struct {
	sandboxManager *sandbox.Manager
	userID         string
	info           ExecToolInfo
}

// NewExecTool creates a new ExecTool instance.
func NewExecTool(mgr *sandbox.Manager, userID string, info ExecToolInfo) *ExecTool {
	return &ExecTool{
		sandboxManager: mgr,
		userID:         userID,
		info:           info,
	}
}

// ExecuteArgs describes the arguments for the execute tool.
type ExecuteArgs struct {
	Command        string `json:"command"`
	Description    string `json:"description,omitempty"`
	TimeoutSeconds int    `json:"timeout_seconds,omitempty"`
}

// Definitions returns the execute tool schema. ReadOnly is false (sequential dispatch).
func (e *ExecTool) Definitions(ctx context.Context) ([]agentapi.ToolDefinition, error) {
	wsDesc := "none"
	if e.info.WorkspaceMount != "" {
		wsDesc = e.info.WorkspaceMount
	}
	netDesc := "none"
	if e.info.Network != "" {
		netDesc = e.info.Network
	}
	driverDesc := "bwrap"
	if e.info.Driver != "" {
		driverDesc = e.info.Driver
	}

	description := fmt.Sprintf("Runs a shell command inside the pre-provisioned sandbox (driver: %s, workspace: %s, network: %s).",
		driverDesc, wsDesc, netDesc)

	schema := openai.Tool{
		Type: openai.ToolTypeFunction,
		Function: &openai.FunctionDefinition{
			Name:        "execute",
			Description: description,
			Parameters: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"command": map[string]interface{}{
						"type":        "string",
						"description": "Shell command to execute via sh -c",
					},
					"description": map[string]interface{}{
						"type":        "string",
						"description": "Brief explanation of what the command does",
					},
					"timeout_seconds": map[string]interface{}{
						"type":        "integer",
						"description": "Optional execution timeout in seconds (clamped to sandbox max timeout)",
					},
				},
				"required": []string{"command"},
			},
		},
	}

	return []agentapi.ToolDefinition{
		{
			Schema:   schema,
			ReadOnly: false,
		},
	}, nil
}

// Execute runs the shell command inside the pre-provisioned sandbox.
func (e *ExecTool) Execute(ctx context.Context, call agentapi.ToolCall) (agentapi.ToolResult, error) {
	if call.Name != "execute" {
		return agentapi.ToolResult{
			Content: fmt.Sprintf("unknown tool %q", call.Name),
		}, fmt.Errorf("unknown tool %q", call.Name)
	}

	if e.sandboxManager == nil {
		return agentapi.ToolResult{
			Content: "sandbox manager is not configured",
		}, errors.New("sandbox manager is not configured")
	}

	var args ExecuteArgs
	if err := json.Unmarshal([]byte(call.Arguments), &args); err != nil {
		return agentapi.ToolResult{
			Content: fmt.Sprintf("invalid arguments: %v", err),
		}, fmt.Errorf("failed to parse execute arguments: %w", err)
	}

	cmd := strings.TrimSpace(args.Command)
	if cmd == "" {
		return agentapi.ToolResult{
			Content: "command cannot be empty",
		}, errors.New("command cannot be empty")
	}

	timeout := time.Duration(args.TimeoutSeconds) * time.Second
	if timeout <= 0 {
		timeout = e.info.DefaultExecTimeout
		if timeout <= 0 {
			timeout = 1 * time.Minute
		}
	}
	if e.info.MaxExecTimeout > 0 && timeout > e.info.MaxExecTimeout {
		timeout = e.info.MaxExecTimeout
	}

	res, err := e.sandboxManager.Exec(ctx, e.userID, []string{"sh", "-c", cmd}, timeout)
	if err != nil {
		return agentapi.ToolResult{
			Content: fmt.Sprintf("execution failed: %v", err),
		}, fmt.Errorf("execution failed: %w", err)
	}

	return agentapi.ToolResult{
		Content: FormatExecResult(res),
	}, nil
}
