package cli

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"bob/internal/agentapi"
	"bob/internal/sandbox"
	"bob/internal/tools"
)

func TestFrontend_NetworkToolFiltering(t *testing.T) {
	reg := tools.NewRegistry(nil, nil)
	sbxMgr := sandbox.NewManager(sandbox.Config{}, nil)
	defer func() { _ = sbxMgr.Close() }()

	execTool := tools.NewExecTool(sbxMgr, "test-user", tools.ExecToolInfo{
		Driver:  "bwrap",
		Network: "none",
	})

	runDesc := agentapi.RunDescriptor{
		RunID: "run-1",
		Session: agentapi.SessionRef{
			FrontendID: "cli",
			SessionID:  "local",
		},
	}

	t.Run("no mounted workspace exposes web tools", func(t *testing.T) {
		fe := NewFrontend(reg, sbxMgr, nil, nil, "none")
		bindings, err := fe.Bind(context.Background(), runDesc)
		require.NoError(t, err)

		defs, err := bindings.Tools.Definitions(context.Background())
		require.NoError(t, err)

		var names []string
		for _, d := range defs {
			names = append(names, d.Schema.Function.Name)
		}
		assert.Contains(t, names, "web_search")
		assert.Contains(t, names, "web_fetch")
		assert.NotContains(t, names, "execute")
	})

	t.Run("mounted workspace with network full exposes web tools and execute", func(t *testing.T) {
		fe := NewFrontend(reg, sbxMgr, execTool, nil, "full")
		bindings, err := fe.Bind(context.Background(), runDesc)
		require.NoError(t, err)

		defs, err := bindings.Tools.Definitions(context.Background())
		require.NoError(t, err)

		var names []string
		for _, d := range defs {
			names = append(names, d.Schema.Function.Name)
		}
		assert.Contains(t, names, "web_search")
		assert.Contains(t, names, "web_fetch")
		assert.Contains(t, names, "execute")
	})

	t.Run("mounted workspace with network none filters out web tools", func(t *testing.T) {
		fe := NewFrontend(reg, sbxMgr, execTool, nil, "none")
		bindings, err := fe.Bind(context.Background(), runDesc)
		require.NoError(t, err)

		defs, err := bindings.Tools.Definitions(context.Background())
		require.NoError(t, err)

		var names []string
		for _, d := range defs {
			names = append(names, d.Schema.Function.Name)
		}
		assert.NotContains(t, names, "web_search")
		assert.NotContains(t, names, "web_fetch")
		assert.Contains(t, names, "execute")
	})

	t.Run("mounted workspace with network restricted filters out web tools", func(t *testing.T) {
		fe := NewFrontend(reg, sbxMgr, execTool, nil, "restricted")
		bindings, err := fe.Bind(context.Background(), runDesc)
		require.NoError(t, err)

		defs, err := bindings.Tools.Definitions(context.Background())
		require.NoError(t, err)

		var names []string
		for _, d := range defs {
			names = append(names, d.Schema.Function.Name)
		}
		assert.NotContains(t, names, "web_search")
		assert.NotContains(t, names, "web_fetch")
		assert.Contains(t, names, "execute")
	})

	t.Run("invalid frontend returns error", func(t *testing.T) {
		fe := NewFrontend(reg, sbxMgr, nil, nil, "none")
		_, err := fe.Bind(context.Background(), agentapi.RunDescriptor{
			RunID: "run-1",
			Session: agentapi.SessionRef{
				FrontendID: "besedka",
			},
		})
		require.Error(t, err)
		assert.Contains(t, err.Error(), `invalid frontend "besedka"`)
	})
}
