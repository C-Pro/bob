package tools

import (
	"context"
	"testing"

	"bob/internal/agentapi"

	openai "github.com/sashabaranov/go-openai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type mockToolset struct {
	defs     []agentapi.ToolDefinition
	calls    []agentapi.ToolCall
	execFunc func(ctx context.Context, call agentapi.ToolCall) (agentapi.ToolResult, error)
}

func (m *mockToolset) Definitions(ctx context.Context) ([]agentapi.ToolDefinition, error) {
	return m.defs, nil
}

func (m *mockToolset) Execute(ctx context.Context, call agentapi.ToolCall) (agentapi.ToolResult, error) {
	m.calls = append(m.calls, call)
	if m.execFunc != nil {
		return m.execFunc(ctx, call)
	}
	return agentapi.ToolResult{Content: "mock result for " + call.Name}, nil
}

func TestAllowlistToolset(t *testing.T) {
	ctx := context.Background()

	base := &mockToolset{
		defs: []agentapi.ToolDefinition{
			{
				Schema:   openai.Tool{Function: &openai.FunctionDefinition{Name: "web_search"}},
				ReadOnly: true,
			},
			{
				Schema:   openai.Tool{Function: &openai.FunctionDefinition{Name: "web_fetch"}},
				ReadOnly: true,
			},
			{
				Schema:   openai.Tool{Function: &openai.FunctionDefinition{Name: "dangerous_exec"}},
				ReadOnly: false,
			},
		},
	}

	al := AllowlistToolset(base, "web_search", "web_fetch")

	// 1. Filtering Definitions
	defs, err := al.Definitions(ctx)
	require.NoError(t, err)
	require.Len(t, defs, 2)
	assert.Equal(t, "web_search", defs[0].Schema.Function.Name)
	assert.True(t, defs[0].ReadOnly)
	assert.Equal(t, "web_fetch", defs[1].Schema.Function.Name)
	assert.True(t, defs[1].ReadOnly)

	// 2. Allowed tool execution succeeds and delegates to base
	res, err := al.Execute(ctx, agentapi.ToolCall{Name: "web_search"})
	require.NoError(t, err)
	assert.Equal(t, "mock result for web_search", res.Content)
	require.Len(t, base.calls, 1)

	// 3. Non-allowlisted tool execution fails immediately without invoking base
	res, err = al.Execute(ctx, agentapi.ToolCall{Name: "dangerous_exec"})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "not authorized")
	require.Len(t, base.calls, 1, "base must not be called for non-allowlisted tool")

	// 4. Nil base returns empty definitions
	nilAl := AllowlistToolset(nil, "foo")
	nilDefs, err := nilAl.Definitions(ctx)
	require.NoError(t, err)
	assert.Empty(t, nilDefs)
	_, err = nilAl.Execute(ctx, agentapi.ToolCall{Name: "foo"})
	require.Error(t, err)
}

func TestCompositeToolset(t *testing.T) {
	ctx := context.Background()

	set1 := &mockToolset{
		defs: []agentapi.ToolDefinition{
			{
				Schema:   openai.Tool{Function: &openai.FunctionDefinition{Name: "tool_1"}},
				ReadOnly: true,
			},
		},
	}
	set2 := &mockToolset{
		defs: []agentapi.ToolDefinition{
			{
				Schema:   openai.Tool{Function: &openai.FunctionDefinition{Name: "tool_2"}},
				ReadOnly: false,
			},
		},
	}

	comp, err := CompositeToolset(set1, set2)
	require.NoError(t, err)

	// 1. Merged definitions with ReadOnly flags preserved
	defs, err := comp.Definitions(ctx)
	require.NoError(t, err)
	require.Len(t, defs, 2)
	assert.Equal(t, "tool_1", defs[0].Schema.Function.Name)
	assert.True(t, defs[0].ReadOnly)
	assert.Equal(t, "tool_2", defs[1].Schema.Function.Name)
	assert.False(t, defs[1].ReadOnly)

	// 2. Routing to appropriate child toolset
	res1, err := comp.Execute(ctx, agentapi.ToolCall{Name: "tool_1"})
	require.NoError(t, err)
	assert.Equal(t, "mock result for tool_1", res1.Content)
	require.Len(t, set1.calls, 1)
	require.Empty(t, set2.calls)

	res2, err := comp.Execute(ctx, agentapi.ToolCall{Name: "tool_2"})
	require.NoError(t, err)
	assert.Equal(t, "mock result for tool_2", res2.Content)
	require.Len(t, set2.calls, 1)

	// 3. Unknown tool rejected
	_, err = comp.Execute(ctx, agentapi.ToolCall{Name: "unknown"})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "not found")

	// 4. Duplicate tool name detection
	dupSet := &mockToolset{
		defs: []agentapi.ToolDefinition{
			{
				Schema: openai.Tool{Function: &openai.FunctionDefinition{Name: "tool_1"}},
			},
		},
	}
	compDup, err := CompositeToolset(set1, dupSet)
	require.NoError(t, err)
	_, err = compDup.Definitions(ctx)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "duplicate tool \"tool_1\"")
}
