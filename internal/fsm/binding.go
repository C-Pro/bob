package fsm

import (
	"context"
	"fmt"
	"strings"

	"bob/internal/agentapi"

	openai "github.com/sashabaranov/go-openai"
)

// ToolBinding binds an agentapi.Toolset for a single execution run.
type ToolBinding struct {
	toolset     agentapi.Toolset
	definitions []agentapi.ToolDefinition
	schemas     []openai.Tool
	byName      map[string]agentapi.ToolDefinition
}

// NewToolBinding validates and indexes the tool definitions provided by a Toolset.
func NewToolBinding(ctx context.Context, toolset agentapi.Toolset) (*ToolBinding, error) {
	if toolset == nil {
		return &ToolBinding{
			byName: make(map[string]agentapi.ToolDefinition),
		}, nil
	}
	defs, err := toolset.Definitions(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to get tool definitions: %w", err)
	}
	b := &ToolBinding{
		toolset:     toolset,
		definitions: defs,
		schemas:     make([]openai.Tool, 0, len(defs)),
		byName:      make(map[string]agentapi.ToolDefinition, len(defs)),
	}
	for _, d := range defs {
		if d.Schema.Function == nil || strings.TrimSpace(d.Schema.Function.Name) == "" {
			return nil, fmt.Errorf("invalid tool definition: missing function name")
		}
		name := strings.TrimSpace(d.Schema.Function.Name)
		if _, exists := b.byName[name]; exists {
			return nil, fmt.Errorf("duplicate tool definition: %q", name)
		}
		b.byName[name] = d
		b.schemas = append(b.schemas, d.Schema)
	}
	return b, nil
}

// Schemas returns the OpenAI tool schemas advertised to the LLM.
func (b *ToolBinding) Schemas() []openai.Tool {
	if b == nil {
		return nil
	}
	return b.schemas
}

// HasTool reports whether a tool name is authorized in this binding.
func (b *ToolBinding) HasTool(name string) bool {
	if b == nil || b.byName == nil {
		return false
	}
	_, ok := b.byName[name]
	return ok
}

// Definition returns the definition for a tool name.
func (b *ToolBinding) Definition(name string) (agentapi.ToolDefinition, bool) {
	if b == nil || b.byName == nil {
		return agentapi.ToolDefinition{}, false
	}
	def, ok := b.byName[name]
	return def, ok
}

// ClassifyStepExecutionMode inspects a slice of FSMSteps and determines the batch execution mode.
// If all tools are bound and ReadOnly, returns ExecutionModeParallel.
// Otherwise returns ExecutionModeSequential.
func (b *ToolBinding) ClassifyStepExecutionMode(steps []*FSMStep) ExecutionMode {
	if len(steps) <= 1 {
		return ExecutionModeSequential
	}
	for _, s := range steps {
		def, ok := b.Definition(s.ToolName)
		if !ok || !def.ReadOnly {
			return ExecutionModeSequential
		}
	}
	return ExecutionModeParallel
}

// Execute dispatches a tool call through the bound Toolset after verifying authorization.
func (b *ToolBinding) Execute(ctx context.Context, call agentapi.ToolCall) (agentapi.ToolResult, error) {
	if b == nil || b.toolset == nil {
		return agentapi.ToolResult{}, fmt.Errorf("no tools configured for execution")
	}
	if !b.HasTool(call.Name) {
		return agentapi.ToolResult{}, fmt.Errorf("unauthorized tool execution: %q is not in the allowed toolset", call.Name)
	}
	return b.toolset.Execute(ctx, call)
}

// legacyToolset adapts a slice of openai.Tool and a ToolInvoker into an agentapi.Toolset.
type legacyToolset struct {
	tools   []openai.Tool
	invoker ToolInvoker
}

// NewLegacyToolset creates an agentapi.Toolset from legacy tools and an invoker.
func NewLegacyToolset(tools []openai.Tool, invoker ToolInvoker) agentapi.Toolset {
	return &legacyToolset{
		tools:   tools,
		invoker: invoker,
	}
}

func (l *legacyToolset) Definitions(ctx context.Context) ([]agentapi.ToolDefinition, error) {
	defs := make([]agentapi.ToolDefinition, 0, len(l.tools))
	for _, t := range l.tools {
		if t.Function == nil || t.Function.Name == "" {
			continue
		}
		defs = append(defs, agentapi.ToolDefinition{
			Schema:   t,
			ReadOnly: IsReadOnlyTool(t.Function.Name),
		})
	}
	return defs, nil
}

func (l *legacyToolset) Execute(ctx context.Context, call agentapi.ToolCall) (agentapi.ToolResult, error) {
	if l.invoker == nil {
		return agentapi.ToolResult{}, fmt.Errorf("tool invoker is not configured")
	}
	content, err := l.invoker.Execute(ctx, call.Name, call.Arguments)
	return agentapi.ToolResult{Content: content}, err
}

// providerToolset adapts a ToolDefinitionProvider and ToolInvoker into an agentapi.Toolset.
type providerToolset struct {
	provider ToolDefinitionProvider
	invoker  ToolInvoker
	chatID   string
	isDM     bool
}

// NewProviderToolset creates an agentapi.Toolset dynamically backed by a ToolDefinitionProvider.
func NewProviderToolset(provider ToolDefinitionProvider, invoker ToolInvoker, chatID string, isDM bool) agentapi.Toolset {
	return &providerToolset{
		provider: provider,
		invoker:  invoker,
		chatID:   chatID,
		isDM:     isDM,
	}
}

func (p *providerToolset) Definitions(ctx context.Context) ([]agentapi.ToolDefinition, error) {
	if p.provider == nil {
		return nil, nil
	}
	tools := p.provider.ToolDefinitions(ctx, p.chatID, p.isDM)
	defs := make([]agentapi.ToolDefinition, 0, len(tools))
	for _, t := range tools {
		if t.Function == nil || t.Function.Name == "" {
			continue
		}
		defs = append(defs, agentapi.ToolDefinition{
			Schema:   t,
			ReadOnly: IsReadOnlyTool(t.Function.Name),
		})
	}
	return defs, nil
}

func (p *providerToolset) Execute(ctx context.Context, call agentapi.ToolCall) (agentapi.ToolResult, error) {
	if p.invoker == nil {
		return agentapi.ToolResult{}, fmt.Errorf("tool invoker is not configured")
	}
	content, err := p.invoker.Execute(ctx, call.Name, call.Arguments)
	return agentapi.ToolResult{Content: content}, err
}

