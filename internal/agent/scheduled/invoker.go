package scheduled

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"time"

	"bob/internal/agent"
	"bob/internal/agentapi"
	"bob/internal/chatcontext"
	"bob/internal/fsm"
	"bob/internal/models"
	"bob/internal/sandbox"
	"bob/internal/scheduler"
	"bob/internal/tools"

	openai "github.com/sashabaranov/go-openai"
)

// MessageSender sends output messages to chat platforms.
type MessageSender interface {
	SendMessage(chatID, content string) error
}

// AttachmentMessageSender extends MessageSender with support for attachments.
type AttachmentMessageSender interface {
	MessageSender
	SendMessageWithAttachments(chatID, content string, attachments []models.Attachment) error
}

// AttachmentProcessor processes attachments to return context formatting and images.
type AttachmentProcessor interface {
	ProcessAttachments(ctx context.Context, attachments []models.Attachment) (string, []chatcontext.ImageAttachment)
}

// MessageSenderFunc is an adapter for MessageSender.
type MessageSenderFunc func(chatID, content string) error

// SendMessage calls f(chatID, content).
func (f MessageSenderFunc) SendMessage(chatID, content string) error {
	return f(chatID, content)
}

// FSMRunner executes tool loops.
type FSMRunner interface {
	RunToolLoop(ctx context.Context, req fsm.ToolLoopRequest) (*fsm.ToolLoopResult, error)
}

// FSMRunnerFunc is an adapter allowing a function to be used as FSMRunner.
type FSMRunnerFunc func(ctx context.Context, req fsm.ToolLoopRequest) (*fsm.ToolLoopResult, error)

// RunToolLoop calls f(ctx, req).
func (f FSMRunnerFunc) RunToolLoop(ctx context.Context, req fsm.ToolLoopRequest) (*fsm.ToolLoopResult, error) {
	return f(ctx, req)
}

// Config configures Invoker.
type Config struct {
	Runner              agentapi.TurnRunner
	Frontend            agentapi.Frontend
	FSM                 FSMRunner
	Tools               *tools.Registry
	Sandbox             *sandbox.Manager
	Sender              MessageSender
	AttachmentProcessor AttachmentProcessor
	ContextMgr          *chatcontext.Manager
	Model               string
	BotID               string
	BotName             string
	LockTimeout         time.Duration
	SchedulerStoreProv  func(ctx context.Context, chatID string, isDM bool) (*scheduler.Store, error)
}

// Invoker implements scheduler.Invoker for executing schedules with FSM/TurnRunner and ephemeral sandboxes.
type Invoker struct {
	cfg           Config
	sessionLocker *agent.SessionLocker
}

// NewInvoker creates a new Invoker.
func NewInvoker(cfg Config, locker ...*agent.SessionLocker) *Invoker {
	if cfg.BotID == "" {
		cfg.BotID = "bot"
	}
	if cfg.BotName == "" {
		cfg.BotName = "Bob"
	}
	var sessionLocker *agent.SessionLocker
	if len(locker) > 0 && locker[0] != nil {
		sessionLocker = locker[0]
	} else {
		sessionLocker = agent.NewSessionLocker()
	}
	return &Invoker{
		cfg:           cfg,
		sessionLocker: sessionLocker,
	}
}

// ChatLocker returns the invoker's SessionLocker.
func (inv *Invoker) ChatLocker() *agent.SessionLocker {
	return inv.sessionLocker
}

// SetFSM replaces the FSM runner used for scheduled executions.
func (inv *Invoker) SetFSM(f FSMRunner) {
	inv.cfg.FSM = f
}

// SetRunner replaces the TurnRunner used for scheduled executions.
func (inv *Invoker) SetRunner(r agentapi.TurnRunner) {
	inv.cfg.Runner = r
}

// SetTools replaces the tool registry used for scheduled executions.
func (inv *Invoker) SetTools(r *tools.Registry) {
	inv.cfg.Tools = r
}

// SetSandbox replaces the sandbox manager used for scheduled executions.
func (inv *Invoker) SetSandbox(sm *sandbox.Manager) {
	inv.cfg.Sandbox = sm
}

// ExecuteSchedule executes a triggered schedule task with bounded chat lock, ephemeral sandbox, and tool loop.
func (inv *Invoker) ExecuteSchedule(ctx context.Context, sched *scheduler.Schedule) error {
	if sched == nil {
		return errors.New("schedule cannot be nil")
	}

	// 1. Acquire per-session lock when Runner is not configured
	// When Runner is configured, Runner.Run will acquire the session lock.
	lockTimeout := inv.cfg.LockTimeout
	if lockTimeout <= 0 {
		lockTimeout = 30 * time.Second
	}
	if inv.cfg.Runner == nil && inv.sessionLocker != nil {
		release, err := inv.sessionLocker.TryAcquire(ctx, sched.ChatID, lockTimeout)
		if err != nil {
			return err
		}
		defer release()
	}

	isDM := sched.ChatID != "townhall"

	// 2. Resolve scheduler store and verify grant if exists
	var schedStore *scheduler.Store
	var err error
	if inv.cfg.SchedulerStoreProv != nil {
		schedStore, err = inv.cfg.SchedulerStoreProv(ctx, sched.ChatID, isDM)
		if err != nil {
			return fmt.Errorf("failed to get scheduler store: %w", err)
		}
	}

	var grant *scheduler.ScheduleGrant
	if schedStore != nil {
		grant, err = schedStore.GetGrant(ctx, sched.ID)
		if err != nil && !errors.Is(err, scheduler.ErrGrantNotFound) {
			return fmt.Errorf("failed to query schedule grant: %w", err)
		}
	}

	hasSandboxGrant := false

	// 3. Ephemeral Sandbox provisioning if pre-approved grant exists
	if grant != nil {
		if !grant.Verify() {
			return errors.New("security violation: schedule grant verification failed (params_hash mismatch)")
		}
		now := time.Now().Unix()
		if grant.ValidUntil > 0 && grant.ValidUntil <= now {
			return errors.New("schedule grant has expired")
		}

		if inv.cfg.Sandbox != nil {
			var env scheduler.PermissionEnvelope
			if jsonErr := json.Unmarshal([]byte(grant.PermissionRequestJSON), &env); jsonErr == nil && env.Sandbox != nil {
				if !isDM {
					return errors.New("sandbox execution is not permitted in Townhall")
				}

				driver := sandbox.DriverType(strings.ToLower(env.Sandbox.Driver))
				netMode := sandbox.NetworkMode(strings.ToLower(env.Sandbox.Network))
				if netMode == "" {
					netMode = sandbox.NetworkNone
				}

				var mounts []sandbox.UserMount
				for _, m := range env.Sandbox.Mounts {
					mounts = append(mounts, sandbox.UserMount{
						RelativePath: m.Path,
						ReadOnly:     m.ReadOnly,
					})
				}

				req := sandbox.RequestParams{
					Driver:         driver,
					DockerImage:    env.Sandbox.Image,
					NetworkMode:    netMode,
					AllowedDomains: env.Sandbox.Domains,
					Mounts:         mounts,
					Reason:         fmt.Sprintf("Scheduled run: %s", sched.Name),
				}

				_, reqErr := inv.cfg.Sandbox.RequestSandbox(ctx, sched.UserID, sched.ChatID, req)
				if reqErr != nil {
					return fmt.Errorf("failed to request ephemeral sandbox: %w", reqErr)
				}
				_, appErr := inv.cfg.Sandbox.ApproveSandbox(ctx, sched.UserID)
				if appErr != nil {
					_ = inv.cfg.Sandbox.Destroy(ctx, sched.UserID)
					return fmt.Errorf("failed to activate ephemeral sandbox: %w", appErr)
				}
				hasSandboxGrant = true
				defer func() {
					cleanupCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
					defer cancel()
					_ = inv.cfg.Sandbox.Destroy(cleanupCtx, sched.UserID)
				}()
			}
		}
	}

	// 4. Prepare chat context and tool definitions
	var budgetLimits tools.KnowledgeBudgetLimits
	if inv.cfg.Tools != nil {
		budgetLimits = inv.cfg.Tools.KnowledgeLimits()
	} else {
		budgetLimits = tools.DefaultKnowledgeBudgetLimits()
	}
	sessionCtx := tools.NewChatSessionContext(sched.ChatID, sched.UserID, isDM, budgetLimits)
	sessionCtx.IsScheduled = true
	if inv.cfg.Sender != nil {
		sessionCtx.Notifier = inv.cfg.Sender.SendMessage
	}
	runCtx := tools.WithChatSession(ctx, sessionCtx)

	var toolDefs []openai.Tool
	allowedTools := make(map[string]bool)
	if inv.cfg.Tools != nil {
		allDefs := inv.cfg.Tools.ToolDefinitionsForSession(sessionCtx)
		for _, t := range allDefs {
			name := ""
			if t.Function != nil {
				name = t.Function.Name
			}
			if name == "schedule_task" || name == "cancel_schedule" || name == "list_schedules" {
				continue
			}
			if !hasSandboxGrant && strings.HasPrefix(name, "sandbox_") {
				continue
			}
			if hasSandboxGrant && (name == "sandbox_request" || name == "sandbox_destroy") {
				continue
			}
			toolDefs = append(toolDefs, t)
			allowedTools[name] = true
		}
	}

	// 5. Build conversation prompt for scheduled execution
	systemPrompt := fmt.Sprintf("You are Bob executing a scheduled autonomous task.\nSchedule Name: %s\nRun Instructions: %s",
		sched.Name, sched.Instruction)
	llmMsgs := []openai.ChatCompletionMessage{
		{
			Role:    openai.ChatMessageRoleSystem,
			Content: systemPrompt,
		},
		{
			Role:    openai.ChatMessageRoleUser,
			Content: fmt.Sprintf("Execute scheduled task '%s'. Follow the instructions carefully.", sched.Name),
		},
	}

	maxTurns := sched.MaxTurns
	if maxTurns <= 0 {
		maxTurns = 15
	}

	// 6. Execute via Runner (if configured) or direct FSM fallback
	var resultContent string
	var resultAttachments []models.Attachment

	if inv.cfg.Runner != nil {
		runID := fmt.Sprintf("sched_%s_%d", sched.ID, time.Now().UnixNano())
		turn := agentapi.Turn{
			Run: agentapi.RunDescriptor{
				RunID: runID,
				Session: agentapi.SessionRef{
					FrontendID: "besedka",
					SessionID:  sched.ChatID,
					ScopeID:    sched.ChatID,
				},
				Actor: agentapi.Actor{
					ID: sched.UserID,
				},
				Kind:          agentapi.Scheduled,
				Model:         inv.cfg.Model,
				MaxIterations: maxTurns,
			},
			SystemPrompt: systemPrompt,
			Messages: []openai.ChatCompletionMessage{
				{
					Role:    openai.ChatMessageRoleUser,
					Content: fmt.Sprintf("Execute scheduled task '%s'. Follow the instructions carefully.", sched.Name),
				},
			},
		}

		res, runErr := inv.cfg.Runner.Run(runCtx, turn)
		if runErr != nil {
			return fmt.Errorf("scheduled turn execution error: %w", runErr)
		}
		resultContent = res.Content
		for _, a := range res.Attachments {
			resultAttachments = append(resultAttachments, models.Attachment{
				FileID:   a.ID,
				Name:     a.Name,
				MimeType: a.MIMEType,
			})
		}
	} else if inv.cfg.FSM != nil {
		var toolset agentapi.Toolset
		if inv.cfg.Tools != nil {
			rawToolset := fsm.NewLegacyToolset(toolDefs, inv.cfg.Tools)
			toolset = &restrictedToolset{
				underlying: rawToolset,
				allowed:    allowedTools,
			}
		}
		fsmReq := fsm.ToolLoopRequest{
			RunID:         fmt.Sprintf("sched_%s_%d", sched.ID, time.Now().UnixNano()),
			ChatID:        sched.ChatID,
			UserID:        sched.UserID,
			IsDM:          isDM,
			Model:         inv.cfg.Model,
			Messages:      llmMsgs,
			Tools:         toolDefs,
			Toolset:       toolset,
			MaxIterations: maxTurns,
		}

		res, loopErr := inv.cfg.FSM.RunToolLoop(runCtx, fsmReq)
		if loopErr != nil {
			return fmt.Errorf("fsm execution error: %w", loopErr)
		}
		if res != nil {
			resultContent = res.Content
			resultAttachments = res.Attachments
		}
	}

	// 7. Deliver completion to frontend / output channels
	if resultContent != "" {
		resultText := fmt.Sprintf("⏱️ **Scheduled Task [%s] completed:**\n%s", sched.Name, resultContent)
		var stagedAtts []models.Attachment
		if sessionCtx.StagedAttachments != nil {
			stagedAtts = sessionCtx.GetStagedAttachments()
		}
		if len(stagedAtts) == 0 && len(resultAttachments) > 0 {
			stagedAtts = resultAttachments
		}

		if inv.cfg.Sender != nil {
			if attSender, ok := inv.cfg.Sender.(AttachmentMessageSender); ok && len(stagedAtts) > 0 {
				if sendErr := attSender.SendMessageWithAttachments(sched.ChatID, resultText, stagedAtts); sendErr != nil {
					slog.Warn("failed to send scheduled task result message with attachments", "error", sendErr)
				}
			} else if sendErr := inv.cfg.Sender.SendMessage(sched.ChatID, resultText); sendErr != nil {
				slog.Warn("failed to send scheduled task result message", "error", sendErr)
			}
		}

		if inv.cfg.ContextMgr != nil {
			pushContent, pushImages := resultText, []chatcontext.ImageAttachment(nil)
			if len(stagedAtts) > 0 {
				var attProc AttachmentProcessor
				if ap, ok := inv.cfg.Sender.(AttachmentProcessor); ok {
					attProc = ap
				} else if inv.cfg.AttachmentProcessor != nil {
					attProc = inv.cfg.AttachmentProcessor
				}

				if attProc != nil {
					extraText, images := attProc.ProcessAttachments(ctx, stagedAtts)
					pushContent = strings.TrimSpace(resultText + extraText)
					pushImages = images
				} else {
					var extraText strings.Builder
					for _, att := range stagedAtts {
						attName := strings.TrimSpace(att.Name)
						if attName == "" {
							attName = "attachment"
						}
						fileID := strings.TrimSpace(att.FileID)
						mimeStr := att.MimeType
						if mimeStr == "" {
							mimeStr = "application/octet-stream"
						}
						fmt.Fprintf(&extraText, "\n\n[Attachment: %s (id: %s, type: %s)]", attName, fileID, mimeStr)
					}
					pushContent = strings.TrimSpace(resultText + extraText.String())
				}
			}

			inv.cfg.ContextMgr.Push(sched.ChatID, chatcontext.Entry{
				Role:       "assistant",
				SenderID:   inv.cfg.BotID,
				SenderName: inv.cfg.BotName,
				Content:    pushContent,
				Images:     pushImages,
				Timestamp:  time.Now().Unix(),
			})
		}
	}

	return nil
}

// restrictedToolset enforces tool allowlists on both definitions and execution dispatch.
type restrictedToolset struct {
	underlying agentapi.Toolset
	allowed    map[string]bool
}

func (rt *restrictedToolset) Definitions(ctx context.Context) ([]agentapi.ToolDefinition, error) {
	if rt.underlying == nil {
		return nil, nil
	}
	defs, err := rt.underlying.Definitions(ctx)
	if err != nil {
		return nil, err
	}
	filtered := make([]agentapi.ToolDefinition, 0, len(defs))
	for _, d := range defs {
		if rt.allowed[d.Schema.Function.Name] {
			filtered = append(filtered, d)
		}
	}
	return filtered, nil
}

func (rt *restrictedToolset) Execute(ctx context.Context, call agentapi.ToolCall) (agentapi.ToolResult, error) {
	if !rt.allowed[call.Name] {
		return agentapi.ToolResult{
			Content: fmt.Sprintf("Error: tool %q is not authorized in scheduled execution", call.Name),
		}, nil
	}
	if rt.underlying == nil {
		return agentapi.ToolResult{
			Content: "Error: no toolset available",
		}, nil
	}
	return rt.underlying.Execute(ctx, call)
}
