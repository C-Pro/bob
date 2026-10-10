package cli

import (
	"context"
	"crypto/rand"
	"errors"
	"fmt"
	"io"
	"runtime"
	"strings"
	"time"

	openai "github.com/sashabaranov/go-openai"

	"bob/internal/agentapi"
	"bob/internal/bootstrap"
	"bob/internal/llm"
	"bob/internal/prompt"
	"bob/internal/sandbox"
	"bob/internal/tools"
)

// SandboxSpec defines the parameters for provisioning a CLI execution sandbox.
type SandboxSpec struct {
	WorkspaceDir   string
	ReadOnly       bool
	Driver         string
	DockerImage    string
	NetworkMode    string
	AllowedDomains []string
}

// Options configures one-shot CLI execution.
type Options struct {
	Prompt        string
	Stdin         string
	Model         string
	MaxIterations int
	Sandbox       *SandboxSpec
	Quiet         bool
	Timeout       time.Duration
}

func handleContextError(ctx context.Context, err error, timeout time.Duration, stderr io.Writer) (int, bool) {
	if ctx.Err() != nil {
		if errors.Is(ctx.Err(), context.Canceled) {
			return 130, true
		}
		if errors.Is(ctx.Err(), context.DeadlineExceeded) {
			if timeout > 0 {
				_, _ = fmt.Fprintf(stderr, "Error: execution timed out after %v (--timeout)\n", timeout)
			} else {
				_, _ = fmt.Fprintln(stderr, "Error: execution timed out")
			}
			return 1, true
		}
	}
	if errors.Is(err, context.Canceled) {
		return 130, true
	}
	return 0, false
}

func generateRunID() string {
	b := make([]byte, 8)
	if _, err := rand.Read(b); err != nil {
		return fmt.Sprintf("run_%d_%d", time.Now().UnixNano(), time.Now().Unix())
	}
	return fmt.Sprintf("run_%d_%x", time.Now().Unix(), b)
}

// Run executes a one-shot CLI turn and returns an exit code:
// 0: success
// 1: runtime failure (tool error, LLM failure, sandbox provisioning error, etc.)
// 2: usage / validation error (empty prompt, invalid parameters)
// 130: interrupted / canceled (SIGINT / SIGTERM / context cancellation)
func Run(ctx context.Context, rt *bootstrap.Runtime, opts Options, stdout, stderr io.Writer) int {
	if stdout == nil || stderr == nil {
		return 1
	}
	if rt == nil || rt.AgentEngine == nil || rt.ToolsRegistry == nil {
		_, _ = fmt.Fprintln(stderr, "Error: runtime is not initialized")
		return 1
	}

	userPrompt := opts.Prompt
	if userPrompt == "-" {
		userPrompt = opts.Stdin
	} else if opts.Stdin != "" {
		fence := "```"
		for strings.Contains(opts.Stdin, fence) {
			fence += "`"
		}
		if userPrompt != "" {
			userPrompt = userPrompt + "\n\n" + fence + "\n" + opts.Stdin + "\n" + fence
		} else {
			userPrompt = fence + "\n" + opts.Stdin + "\n" + fence
		}
	}

	if strings.TrimSpace(userPrompt) == "" {
		_, _ = fmt.Fprintln(stderr, "Error: prompt cannot be empty")
		return 2
	}

	if rt.Config != nil && rt.Config.MaxAttachmentSizeBytes > 0 && int64(len(opts.Stdin)) > rt.Config.MaxAttachmentSizeBytes {
		_, _ = fmt.Fprintf(stderr, "Error: stdin exceeds maximum allowed size (%d bytes)\n", rt.Config.MaxAttachmentSizeBytes)
		return 2
	}

	outWriter := NewOutputWriter(stdout)
	progress := NewProgressRenderer(outWriter, opts.Quiet)

	// Authorize knowledge tools for CLI session
	rt.ToolsRegistry.SetDMAuthorizer(func(ctx context.Context, chatID, userID string) error {
		return nil
	})

	var execTool *tools.ExecTool
	if opts.Sandbox != nil {
		if rt.SandboxManager == nil {
			_, _ = fmt.Fprintln(stderr, "Error: sandbox manager is not enabled or available")
			return 1
		}

		driver := opts.Sandbox.Driver
		if driver == "" && rt.Config != nil {
			driver = rt.Config.SandboxDefaultDriver
		}
		if driver == "" {
			driver = "bwrap"
		}
		image := opts.Sandbox.DockerImage
		if image == "" && rt.Config != nil {
			image = rt.Config.SandboxDockerImage
		}
		netMode := opts.Sandbox.NetworkMode
		if netMode == "" {
			netMode = "none"
		}
		lifetime := 30 * time.Minute
		if rt.Config != nil && rt.Config.SandboxMaxLifetime > 0 {
			lifetime = rt.Config.SandboxMaxLifetime
		}

		params := sandbox.ProvisionParams{
			Driver:            sandbox.DriverType(driver),
			DockerImage:       image,
			NetworkMode:       sandbox.NetworkMode(netMode),
			AllowedDomains:    opts.Sandbox.AllowedDomains,
			WorkspaceDir:      opts.Sandbox.WorkspaceDir,
			WorkspaceReadOnly: opts.Sandbox.ReadOnly,
			Lifetime:          lifetime,
		}

		_, err := rt.SandboxManager.Provision(ctx, "local", "cli", params)
		if err != nil {
			if code, handled := handleContextError(ctx, err, opts.Timeout, stderr); handled {
				return code
			}
			_, _ = fmt.Fprintf(stderr, "Error provisioning sandbox: %v\n", err)
			return 1
		}
		defer func() {
			if rt.SandboxManager != nil {
				_ = rt.SandboxManager.Destroy(context.Background(), "local")
			}
		}()

		maxTimeout := 10 * time.Minute
		if rt.Config != nil && rt.Config.SandboxMaxExecTimeout > 0 {
			maxTimeout = rt.Config.SandboxMaxExecTimeout
		}
		mountMode := "ro"
		if !opts.Sandbox.ReadOnly {
			mountMode = "rw"
		}
		execTool = tools.NewExecTool(rt.SandboxManager, "local", tools.ExecToolInfo{
			Driver:         driver,
			WorkspaceMount: mountMode,
			Network:        netMode,
			MaxExecTimeout: maxTimeout,
		})
	}

	fe := NewFrontend(rt.ToolsRegistry, rt.SandboxManager, execTool, progress)
	if err := rt.RegisterFrontend("cli", fe); err != nil {
		_, _ = fmt.Fprintf(stderr, "Error registering frontend: %v\n", err)
		return 1
	}

	model := opts.Model
	if model == "" && rt.Config != nil {
		model = rt.Config.OpenAIModel
	}
	maxIter := opts.MaxIterations
	if maxIter <= 0 {
		if rt.Config != nil && rt.Config.CLIMaxIterations > 0 {
			maxIter = rt.Config.CLIMaxIterations
		} else {
			maxIter = 100
		}
	}

	runID := generateRunID()
	runDesc := agentapi.RunDescriptor{
		RunID: runID,
		Session: agentapi.SessionRef{
			FrontendID: "cli",
			SessionID:  "cli",
			ScopeID:    "cli",
			IsDM:       true,
		},
		Actor: agentapi.Actor{
			ID: "local",
		},
		Kind:          agentapi.Interactive,
		Model:         model,
		MaxIterations: maxIter,
		Metadata: map[string]string{
			"disable_volatile_fallback": "true",
		},
	}

	bindings, err := fe.Bind(ctx, runDesc)
	if err != nil {
		if code, handled := handleContextError(ctx, err, opts.Timeout, stderr); handled {
			return code
		}
		_, _ = fmt.Fprintf(stderr, "Error binding frontend: %v\n", err)
		return 1
	}

	toolDefs, err := bindings.Tools.Definitions(ctx)
	if err != nil {
		if code, handled := handleContextError(ctx, err, opts.Timeout, stderr); handled {
			return code
		}
		_, _ = fmt.Fprintf(stderr, "Error getting tool definitions: %v\n", err)
		return 1
	}

	var sbInfo *prompt.CLISandboxInfo
	if opts.Sandbox != nil {
		driver := opts.Sandbox.Driver
		if driver == "" && rt.Config != nil {
			driver = rt.Config.SandboxDefaultDriver
		}
		if driver == "" {
			driver = "bwrap"
		}
		netMode := opts.Sandbox.NetworkMode
		if netMode == "" {
			netMode = "none"
		}
		sbInfo = &prompt.CLISandboxInfo{
			HostWorkspace:     opts.Sandbox.WorkspaceDir,
			WorkspaceReadOnly: opts.Sandbox.ReadOnly,
			Driver:            driver,
			Network:           netMode,
		}
	}

	now := time.Now()
	zone, _ := now.Zone()
	sysPrompt := prompt.RenderCLI(prompt.CLIPromptContext{
		Now:     now,
		TZ:      zone,
		OS:      runtime.GOOS,
		Tools:   toolDefs,
		Sandbox: sbInfo,
	})

	turn := agentapi.Turn{
		Run:          runDesc,
		SystemPrompt: sysPrompt,
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleUser,
				Content: userPrompt,
			},
		},
		Deliver: false,
	}

	res, err := rt.AgentEngine.Run(ctx, turn)
	if err != nil {
		if code, handled := handleContextError(ctx, err, opts.Timeout, stderr); handled {
			return code
		}
		var timeoutErr *llm.TimeoutError
		if errors.As(err, &timeoutErr) {
			_, _ = fmt.Fprintf(stderr, "Error: %v\n", timeoutErr)
			return 1
		}
		_, _ = fmt.Fprintf(stderr, "Error: %v\n", err)
		return 1
	}

	if err := outWriter.WriteFinalAnswer(res.Content); err != nil {
		_, _ = fmt.Fprintf(stderr, "Error writing output: %v\n", err)
		return 1
	}
	if err := outWriter.Err(); err != nil {
		_, _ = fmt.Fprintf(stderr, "Error writing output: %v\n", err)
		return 1
	}

	return 0
}
