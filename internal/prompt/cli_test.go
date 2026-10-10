package prompt

import (
	"testing"
	"time"

	"bob/internal/agentapi"

	openai "github.com/sashabaranov/go-openai"
	"github.com/stretchr/testify/assert"
)

func TestRenderCLI(t *testing.T) {
	fixedTime := time.Date(2026, 10, 8, 12, 34, 56, 0, time.UTC)

	t.Run("unsandboxed with base tools", func(t *testing.T) {
		ctx := CLIPromptContext{
			Now: fixedTime,
			TZ:  "UTC",
			OS:  "linux",
			Tools: []agentapi.ToolDefinition{
				{
					Schema: openai.Tool{
						Function: &openai.FunctionDefinition{
							Name:        "web_search",
							Description: "Search the web for current information",
						},
					},
				},
				{
					Schema: openai.Tool{
						Function: &openai.FunctionDefinition{
							Name:        "web_fetch",
							Description: "Fetch content of a web page",
						},
					},
				},
			},
			Sandbox: nil,
		}

		p := RenderCLI(ctx)
		assert.Contains(t, p, "Operating system: linux")
		assert.Contains(t, p, "Current date and time: 2026-10-08 12:34:56 UTC (UTC)")
		assert.Contains(t, p, "- web_search: Search the web for current information")
		assert.Contains(t, p, "- web_fetch: Fetch content of a web page")
		assert.Contains(t, p, "You do NOT have access to local files on the user's filesystem.")
		assert.NotContains(t, p, "Sandbox Environment:")
	})

	t.Run("sandboxed read-only", func(t *testing.T) {
		ctx := CLIPromptContext{
			Now: fixedTime,
			TZ:  "America/New_York",
			OS:  "linux",
			Tools: []agentapi.ToolDefinition{
				{
					Schema: openai.Tool{
						Function: &openai.FunctionDefinition{
							Name:        "execute",
							Description: "Runs a shell command inside the pre-provisioned sandbox",
						},
					},
				},
			},
			Sandbox: &CLISandboxInfo{
				HostWorkspace:     "/home/user/myproject",
				WorkspaceReadOnly: true,
				Driver:            "bwrap",
				Network:           "restricted",
			},
		}

		p := RenderCLI(ctx)
		assert.Contains(t, p, "Current date and time: 2026-10-08 12:34:56 UTC (America/New_York)")
		assert.Contains(t, p, "- execute: Runs a shell command inside the pre-provisioned sandbox")
		assert.Contains(t, p, "Sandbox Environment:")
		assert.Contains(t, p, "driver: bwrap, network: restricted")
		assert.Contains(t, p, `"/home/user/myproject" is mounted at '/workspace' in read-only mode`)
		assert.Contains(t, p, "Note: The workspace is read-only. Commands cannot modify files in /workspace.")
		assert.NotContains(t, p, "You do NOT have access to local files")
	})

	t.Run("sandboxed read-write", func(t *testing.T) {
		ctx := CLIPromptContext{
			Now: fixedTime,
			TZ:  "UTC",
			OS:  "linux",
			Sandbox: &CLISandboxInfo{
				HostWorkspace:     "/home/user/workspace",
				WorkspaceReadOnly: false,
				Driver:            "docker",
				Network:           "full",
			},
		}

		p := RenderCLI(ctx)
		assert.Contains(t, p, "Sandbox Environment:")
		assert.Contains(t, p, "driver: docker, network: full")
		assert.Contains(t, p, `"/home/user/workspace" is mounted at '/workspace' in read-write mode`)
		assert.NotContains(t, p, "Note: The workspace is read-only")
	})
}
