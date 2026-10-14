package tools

import (
	"context"
	"fmt"

	"bob/internal/agentapi"
)

// allowlistToolset wraps a base Toolset, exposing only the allowlisted tool names.
// Invocations of non-allowlisted tools are rejected immediately without invoking the base toolset.
type allowlistToolset struct {
	base      agentapi.Toolset
	allowlist map[string]struct{}
}

// AllowlistToolset creates a Toolset filter that exposes and authorizes only tools named in names.
func AllowlistToolset(base agentapi.Toolset, names ...string) agentapi.Toolset {
	allowMap := make(map[string]struct{}, len(names))
	for _, n := range names {
		allowMap[n] = struct{}{}
	}
	return &allowlistToolset{
		base:      base,
		allowlist: allowMap,
	}
}

func (a *allowlistToolset) Definitions(ctx context.Context) ([]agentapi.ToolDefinition, error) {
	if a.base == nil {
		return nil, nil
	}
	defs, err := a.base.Definitions(ctx)
	if err != nil {
		return nil, err
	}
	filtered := make([]agentapi.ToolDefinition, 0, len(defs))
	for _, def := range defs {
		if def.Schema.Function == nil {
			continue
		}
		if _, ok := a.allowlist[def.Schema.Function.Name]; ok {
			filtered = append(filtered, def)
		}
	}
	return filtered, nil
}

func (a *allowlistToolset) Execute(ctx context.Context, call agentapi.ToolCall) (agentapi.ToolResult, error) {
	if _, ok := a.allowlist[call.Name]; !ok {
		return agentapi.ToolResult{
			Content: fmt.Sprintf("tool %q is not authorized in this session", call.Name),
		}, fmt.Errorf("tool %q is not authorized in this session", call.Name)
	}
	if a.base == nil {
		return agentapi.ToolResult{}, fmt.Errorf("base toolset is nil")
	}
	return a.base.Execute(ctx, call)
}

// compositeToolset combines multiple Toolsets into one.
type compositeToolset struct {
	sets []agentapi.Toolset
}

// CompositeToolset merges multiple toolsets. Nil toolsets are ignored.
func CompositeToolset(sets ...agentapi.Toolset) (agentapi.Toolset, error) {
	validSets := make([]agentapi.Toolset, 0, len(sets))
	for _, s := range sets {
		if s != nil {
			validSets = append(validSets, s)
		}
	}
	return &compositeToolset{sets: validSets}, nil
}

func (c *compositeToolset) Definitions(ctx context.Context) ([]agentapi.ToolDefinition, error) {
	var merged []agentapi.ToolDefinition
	seen := make(map[string]int)

	for setIdx, set := range c.sets {
		defs, err := set.Definitions(ctx)
		if err != nil {
			return nil, err
		}
		for _, def := range defs {
			if def.Schema.Function == nil {
				continue
			}
			name := def.Schema.Function.Name
			if prevIdx, exists := seen[name]; exists && prevIdx != setIdx {
				return nil, fmt.Errorf("duplicate tool %q defined across composite toolsets", name)
			}
			seen[name] = setIdx
			merged = append(merged, def)
		}
	}
	return merged, nil
}

func (c *compositeToolset) Execute(ctx context.Context, call agentapi.ToolCall) (agentapi.ToolResult, error) {
	for _, set := range c.sets {
		defs, err := set.Definitions(ctx)
		if err != nil {
			return agentapi.ToolResult{}, err
		}
		for _, def := range defs {
			if def.Schema.Function.Name == call.Name {
				return set.Execute(ctx, call)
			}
		}
	}
	return agentapi.ToolResult{
		Content: fmt.Sprintf("tool %q not found", call.Name),
	}, fmt.Errorf("tool %q not found in composite toolset", call.Name)
}
