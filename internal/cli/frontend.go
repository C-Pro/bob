package cli

import (
	"context"
	"fmt"
	"strings"

	"bob/internal/agentapi"
	"bob/internal/sandbox"
	"bob/internal/tools"
)

// Frontend implements agentapi.Frontend for the CLI runner.
type Frontend struct {
	toolsRegistry  *tools.Registry
	sandboxManager *sandbox.Manager
	execTool       *tools.ExecTool
	progress       *ProgressRenderer
	networkMode    string
}

// NewFrontend creates a new CLI Frontend.
func NewFrontend(reg *tools.Registry, mgr *sandbox.Manager, exec *tools.ExecTool, progress *ProgressRenderer, networkMode string) *Frontend {
	return &Frontend{
		toolsRegistry:  reg,
		sandboxManager: mgr,
		execTool:       exec,
		progress:       progress,
		networkMode:    networkMode,
	}
}

// Bind returns the execution-scoped toolset and progress observer for a CLI turn.
func (f *Frontend) Bind(ctx context.Context, run agentapi.RunDescriptor) (agentapi.Bindings, error) {
	normFE := strings.ToLower(strings.TrimSpace(run.Session.FrontendID))
	if normFE != "cli" {
		return agentapi.Bindings{}, fmt.Errorf("invalid frontend %q, expected \"cli\"", run.Session.FrontendID)
	}

	if f.toolsRegistry == nil {
		return agentapi.Bindings{}, fmt.Errorf("tools registry is not configured")
	}

	sessionCtx := tools.NewChatSessionContext("cli", "local", true)
	sessionCtx.FSMRunID = run.RunID

	allowedTools := []string{
		"discover_memories", "load_memory",
		"discover_skills", "load_skill",
	}

	// Host web tools (web_search, web_fetch) are enabled when:
	// - No sandbox workspace is mounted (standalone CLI queries), OR
	// - Sandbox is configured with full network access (--net full).
	// When a workspace is mounted with --net none or --net restricted, host web tools
	// are omitted to prevent model prompt-injection data exfiltration.
	if f.execTool == nil || strings.EqualFold(f.networkMode, "full") {
		allowedTools = append(allowedTools, "web_search", "web_fetch")
	}

	baseToolset := f.toolsRegistry.SessionToolset(sessionCtx)
	allowedToolset := tools.AllowlistToolset(baseToolset, allowedTools...)

	var finalToolset = allowedToolset
	if f.execTool != nil {
		composite, err := tools.CompositeToolset(allowedToolset, f.execTool)
		if err != nil {
			return agentapi.Bindings{}, fmt.Errorf("failed to compose CLI toolsets: %w", err)
		}
		finalToolset = composite
	}

	// Validate definitions
	if _, err := finalToolset.Definitions(ctx); err != nil {
		return agentapi.Bindings{}, fmt.Errorf("failed to retrieve tool definitions: %w", err)
	}

	return agentapi.Bindings{
		Tools:    finalToolset,
		Progress: f.progress,
	}, nil
}

// Deliver returns ErrDetachedDeliveryUnsupported as one-shot CLI execution prints directly to stdout.
func (f *Frontend) Deliver(ctx context.Context, c agentapi.Completion) error {
	return agentapi.ErrDetachedDeliveryUnsupported
}
