package gateway

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"sync"
	"time"

	"bob/internal/chatcontext"
	"bob/internal/fsm"
	"bob/internal/sandbox"
	"bob/internal/scheduler"
	"bob/internal/tools"

	openai "github.com/sashabaranov/go-openai"
)

// MessageSender sends output messages to chat platforms.
type MessageSender interface {
	SendMessage(chatID, content string) error
}

// MessageSenderFunc is an adapter for MessageSender.
type MessageSenderFunc func(chatID, content string) error

// SendMessage calls f(chatID, content).
func (f MessageSenderFunc) SendMessage(chatID, content string) error {
	return f(chatID, content)
}

// ChatLocker provides per-chat mutex locks with timeouts.
type ChatLocker struct {
	mu    sync.Mutex
	locks map[string]*sync.Mutex
}

// NewChatLocker creates a new ChatLocker.
func NewChatLocker() *ChatLocker {
	return &ChatLocker{
		locks: make(map[string]*sync.Mutex),
	}
}

func (l *ChatLocker) getLock(chatID string) *sync.Mutex {
	l.mu.Lock()
	defer l.mu.Unlock()
	mu, ok := l.locks[chatID]
	if !ok {
		mu = &sync.Mutex{}
		l.locks[chatID] = mu
	}
	return mu
}

// TryAcquire attempts to acquire the lock for chatID within the given timeout.
// Returns a release function on success, or an error wrapping scheduler.ErrChatBusy on timeout.
func (l *ChatLocker) TryAcquire(ctx context.Context, chatID string, timeout time.Duration) (func(), error) {
	mu := l.getLock(chatID)
	if mu.TryLock() {
		return func() { mu.Unlock() }, nil
	}

	deadline := time.After(timeout)
	ticker := time.NewTicker(300 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-deadline:
			return nil, fmt.Errorf("%w: timed out waiting for chat %s execution lock after %v", scheduler.ErrChatBusy, chatID, timeout)
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-ticker.C:
			if mu.TryLock() {
				return func() { mu.Unlock() }, nil
			}
		}
	}
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

// InvokerConfig configures ScheduleInvoker.
type InvokerConfig struct {
	FSM                FSMRunner
	Tools              *tools.Registry
	Sandbox            *sandbox.Manager
	Sender             MessageSender
	ContextMgr         *chatcontext.Manager
	Model              string
	BotID              string
	BotName            string
	LockTimeout        time.Duration
	SchedulerStoreProv func(ctx context.Context, chatID string, isDM bool) (*scheduler.Store, error)
}

// ScheduleInvoker implements scheduler.Invoker for executing schedules with FSM and ephemeral sandboxes.
type ScheduleInvoker struct {
	cfg        InvokerConfig
	chatLocker *ChatLocker
}

// NewScheduleInvoker creates a new ScheduleInvoker.
func NewScheduleInvoker(cfg InvokerConfig, locker ...*ChatLocker) *ScheduleInvoker {
	if cfg.BotID == "" {
		cfg.BotID = "bot"
	}
	if cfg.BotName == "" {
		cfg.BotName = "Bob"
	}
	var chatLocker *ChatLocker
	if len(locker) > 0 && locker[0] != nil {
		chatLocker = locker[0]
	} else {
		chatLocker = NewChatLocker()
	}
	return &ScheduleInvoker{
		cfg:        cfg,
		chatLocker: chatLocker,
	}
}

// ChatLocker returns the invoker's ChatLocker.
func (inv *ScheduleInvoker) ChatLocker() *ChatLocker {
	return inv.chatLocker
}

// SetFSM replaces the FSM runner used for scheduled executions.
func (inv *ScheduleInvoker) SetFSM(f FSMRunner) {
	inv.cfg.FSM = f
}

// SetTools replaces the tool registry used for scheduled executions.
func (inv *ScheduleInvoker) SetTools(r *tools.Registry) {
	inv.cfg.Tools = r
}

// SetSandbox replaces the sandbox manager used for scheduled executions.
func (inv *ScheduleInvoker) SetSandbox(sm *sandbox.Manager) {
	inv.cfg.Sandbox = sm
}

// ExecuteSchedule executes a triggered schedule task with bounded chat lock, ephemeral sandbox, and FSM tool loop.
func (inv *ScheduleInvoker) ExecuteSchedule(ctx context.Context, sched *scheduler.Schedule) error {
	if sched == nil {
		return errors.New("schedule cannot be nil")
	}

	// 1. Acquire per-chat lock bounded by timeout (default 30 seconds)
	lockTimeout := inv.cfg.LockTimeout
	if lockTimeout <= 0 {
		lockTimeout = 30 * time.Second
	}
	release, err := inv.chatLocker.TryAcquire(ctx, sched.ChatID, lockTimeout)
	if err != nil {
		return err
	}
	defer release()

	isDM := sched.ChatID != "townhall"

	// 2. Resolve scheduler store and verify grant if exists
	var schedStore *scheduler.Store
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
					_ = inv.cfg.Sandbox.Destroy(context.Background(), sched.UserID)
				}()
			}
		}
	}

	// 4. Prepare chat context and tool definitions
	sessionCtx := tools.ChatSessionContext{
		ChatID: sched.ChatID,
		UserID: sched.UserID,
		IsDM:   isDM,
	}
	if inv.cfg.Sender != nil {
		sessionCtx.Notifier = inv.cfg.Sender.SendMessage
	}
	runCtx := tools.WithChatSession(ctx, sessionCtx)

	var toolDefs []openai.Tool
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

	// 6. Execute FSM Tool Loop
	if inv.cfg.FSM != nil {
		fsmReq := fsm.ToolLoopRequest{
			RunID:         fmt.Sprintf("sched_%s_%d", sched.ID, time.Now().UnixNano()),
			ChatID:        sched.ChatID,
			UserID:        sched.UserID,
			IsDM:          isDM,
			Model:         inv.cfg.Model,
			Messages:      llmMsgs,
			Tools:         toolDefs,
			MaxIterations: maxTurns,
		}

		res, loopErr := inv.cfg.FSM.RunToolLoop(runCtx, fsmReq)
		if loopErr != nil {
			return fmt.Errorf("fsm execution error: %w", loopErr)
		}

		if res != nil && res.Content != "" && inv.cfg.Sender != nil {
			resultText := fmt.Sprintf("⏱️ **Scheduled Task [%s] completed:**\n%s", sched.Name, res.Content)
			if sendErr := inv.cfg.Sender.SendMessage(sched.ChatID, resultText); sendErr != nil {
				slog.Warn("failed to send scheduled task result message", "error", sendErr)
			}
			if inv.cfg.ContextMgr != nil {
				inv.cfg.ContextMgr.Push(sched.ChatID, chatcontext.Entry{
					Role:       "assistant",
					SenderID:   inv.cfg.BotID,
					SenderName: inv.cfg.BotName,
					Content:    resultText,
					Timestamp:  time.Now().Unix(),
				})
			}
		}
	}

	return nil
}
