package prompt

import (
	"fmt"
	"strings"
	"time"

	"bob/internal/agentapi"
)

// CLISandboxInfo holds sandbox execution environment details for prompt rendering.
type CLISandboxInfo struct {
	HostWorkspace     string
	WorkspaceReadOnly bool
	Driver            string
	Network           string
}

// CLIPromptContext specifies the environment and available capabilities for the CLI runner.
type CLIPromptContext struct {
	Now     time.Time
	TZ      string
	OS      string
	Tools   []agentapi.ToolDefinition
	Sandbox *CLISandboxInfo
}

// RenderCLI renders the system prompt for the CLI runner.
func RenderCLI(ctx CLIPromptContext) string {
	var b strings.Builder

	b.WriteString("You are Bob, an AI assistant operating in command-line mode.\n\n")

	// Environment & Date/Time
	tz := ctx.TZ
	if tz == "" && !ctx.Now.IsZero() {
		tz = ctx.Now.Location().String()
	}
	if tz == "" {
		tz = "UTC"
	}
	osName := ctx.OS
	if osName == "" {
		osName = "linux"
	}

	nowStr := "unknown"
	if !ctx.Now.IsZero() {
		nowStr = ctx.Now.Format("2006-01-02 15:04:05 MST")
	}

	fmt.Fprintf(&b, "Current date and time: %s (%s)\n", nowStr, tz)
	fmt.Fprintf(&b, "Operating system: %s\n\n", osName)

	b.WriteString("Provide direct, helpful, and concise answers using clean Markdown formatting suitable for a terminal output. Do not include conversational filler, preamble, or sign-offs.\n\n")

	// Capabilities and Tools
	if len(ctx.Tools) > 0 {
		b.WriteString("Available tools:\n")
		for _, tool := range ctx.Tools {
			if tool.Schema.Function != nil && tool.Schema.Function.Name != "" {
				name := tool.Schema.Function.Name
				desc := tool.Schema.Function.Description
				if desc != "" {
					fmt.Fprintf(&b, "- %s: %s\n", name, desc)
				} else {
					fmt.Fprintf(&b, "- %s\n", name)
				}
			}
		}
		b.WriteString("\n")
	}

	// Filesystem / Sandbox access
	if ctx.Sandbox != nil {
		modeStr := "read-write"
		if ctx.Sandbox.WorkspaceReadOnly {
			modeStr = "read-only"
		}
		driver := ctx.Sandbox.Driver
		if driver == "" {
			driver = "bwrap"
		}
		net := ctx.Sandbox.Network
		if net == "" {
			net = "none"
		}

		b.WriteString("Sandbox Environment:\n")
		fmt.Fprintf(&b, "You have access to an isolated execution sandbox (driver: %s, network: %s).\n", driver, net)
		fmt.Fprintf(&b, "The host directory %q is mounted at '/workspace' in %s mode.\n", ctx.Sandbox.HostWorkspace, modeStr)
		b.WriteString("The 'execute' tool runs shell commands via 'sh -c' inside this sandbox with '/workspace' as the working directory.\n")
		if ctx.Sandbox.WorkspaceReadOnly {
			b.WriteString("Note: The workspace is read-only. Commands cannot modify files in /workspace.\n")
		}
	} else {
		b.WriteString("Local Filesystem:\n")
		b.WriteString("You do NOT have access to local files on the user's filesystem. There are no tools available to read, write, or inspect local files or directories.\n")
	}

	return strings.TrimSpace(b.String())
}
