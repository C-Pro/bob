package agent

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	openai "github.com/sashabaranov/go-openai"

	"bob/internal/agentapi"
	"bob/internal/fsm"
	"bob/internal/llm"
)

// LLMClient abstracts language model completions and interactive tool execution.
type LLMClient interface {
	GenerateChatResponse(ctx context.Context, messages []openai.ChatCompletionMessage) (string, error)
	GenerateChatResponseWithToolLoop(
		ctx context.Context,
		messages []openai.ChatCompletionMessage,
		tools []openai.Tool,
		executor llm.ToolExecutor,
		maxIterations int,
	) (string, error)
}

// FSMEngine coordinates durable workflow execution.
type FSMEngine interface {
	RunToolLoop(ctx context.Context, req fsm.ToolLoopRequest) (*fsm.ToolLoopResult, error)
}

// Config configures the core Agent Engine.
type Config struct {
	DefaultModel  string
	MaxIterations int
	LockTimeout   time.Duration
}

// Dependencies contains the required and optional subsystems needed by the Engine.
type Dependencies struct {
	Config        Config
	LLMClient     LLMClient
	FSMEngine     FSMEngine
	SessionLocker *SessionLocker
	Frontends     map[string]agentapi.Frontend
}

// Engine coordinates conversational turns, session concurrency, frontend binding, and tool execution.
type Engine struct {
	cfg           Config
	llmClient     LLMClient
	fsmEngine     FSMEngine
	sessionLocker *SessionLocker

	frontendsMu sync.RWMutex
	frontends   map[string]agentapi.Frontend

	closed atomic.Bool
}

// NewEngine constructs an initialized Engine.
func NewEngine(deps Dependencies) (*Engine, error) {
	locker := deps.SessionLocker
	if locker == nil {
		locker = NewSessionLocker()
	}

	frontends := make(map[string]agentapi.Frontend)
	for id, fe := range deps.Frontends {
		if strings.TrimSpace(id) == "" || fe == nil {
			continue
		}
		frontends[strings.ToLower(strings.TrimSpace(id))] = fe
	}

	return &Engine{
		cfg:           deps.Config,
		llmClient:     deps.LLMClient,
		fsmEngine:     deps.FSMEngine,
		sessionLocker: locker,
		frontends:     frontends,
	}, nil
}

// RegisterFrontend registers or updates a frontend binding.
func (e *Engine) RegisterFrontend(id string, fe agentapi.Frontend) error {
	normID := strings.ToLower(strings.TrimSpace(id))
	if normID == "" {
		return errors.New("frontend ID cannot be empty")
	}
	if fe == nil {
		return errors.New("frontend implementation cannot be nil")
	}

	e.frontendsMu.Lock()
	defer e.frontendsMu.Unlock()
	e.frontends[normID] = fe
	return nil
}

// DeliveryError indicates that turn execution succeeded, but delivery to the frontend failed.
type DeliveryError struct {
	Result agentapi.Result
	Err    error
}

func (e *DeliveryError) Error() string {
	return fmt.Sprintf("turn executed successfully, but delivery failed: %v", e.Err)
}

func (e *DeliveryError) Unwrap() error {
	return e.Err
}

// Run executes a single turn, implementing agentapi.TurnRunner.
func (e *Engine) Run(ctx context.Context, turn agentapi.Turn) (agentapi.Result, error) {
	if e.closed.Load() {
		return agentapi.Result{}, errors.New("engine is closed")
	}

	// Normalize session reference
	feID := strings.ToLower(strings.TrimSpace(turn.Run.Session.FrontendID))
	if feID == "" {
		feID = "besedka"
		turn.Run.Session.FrontendID = feID
	}
	if turn.Run.Session.ScopeID == "" {
		turn.Run.Session.ScopeID = turn.Run.Session.SessionID
	}

	// Acquire per-session concurrency lock
	if e.sessionLocker != nil {
		timeout := e.cfg.LockTimeout
		if timeout <= 0 {
			timeout = 30 * time.Second
		}
		key := NormalizeSessionKey(turn.Run.Session)
		release, err := e.sessionLocker.TryAcquire(ctx, key, timeout)
		if err != nil {
			return agentapi.Result{}, fmt.Errorf("failed to acquire session lock: %w", err)
		}
		defer release()
	}

	// Lookup frontend
	e.frontendsMu.RLock()
	fe, ok := e.frontends[feID]
	e.frontendsMu.RUnlock()
	if !ok {
		return agentapi.Result{}, fmt.Errorf("unregistered frontend %q", feID)
	}

	// Bind frontend dependencies for this run
	bindings, err := fe.Bind(ctx, turn.Run)
	if err != nil {
		return agentapi.Result{}, fmt.Errorf("frontend %q bind failed: %w", feID, err)
	}

	// Prepare messages without mutating input slice
	llmMsgs := make([]openai.ChatCompletionMessage, 0, len(turn.Messages)+2)
	if strings.TrimSpace(turn.SystemPrompt) != "" {
		llmMsgs = append(llmMsgs, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleSystem,
			Content: turn.SystemPrompt,
		})
	}
	llmMsgs = append(llmMsgs, turn.Messages...)

	// Defense-in-depth: Ensure message sequence never terminates with an assistant turn
	if len(llmMsgs) > 0 && llmMsgs[len(llmMsgs)-1].Role == openai.ChatMessageRoleAssistant {
		llmMsgs = append(llmMsgs, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: "Please continue.",
		})
	}

	model := turn.Run.Model
	if model == "" {
		model = e.cfg.DefaultModel
	}
	maxIterations := turn.Run.MaxIterations
	if maxIterations <= 0 {
		maxIterations = e.cfg.MaxIterations
		if maxIterations <= 0 {
			maxIterations = 10
		}
	}

	var result agentapi.Result
	var executed bool
	var runErr error

	// 1. Primary execution via FSM Engine (durable tool loop)
	if e.fsmEngine != nil {
		var fsmProgObs fsm.ProgressObserver
		if bindings.Progress != nil {
			if po, ok := bindings.Progress.(fsm.ProgressObserver); ok {
				fsmProgObs = po
			}
		}

		fsmReq := fsm.ToolLoopRequest{
			RunID:            turn.Run.RunID,
			ChatID:           turn.Run.Session.SessionID,
			UserID:           turn.Run.Actor.ID,
			FrontendID:       feID,
			ScopeID:          turn.Run.Session.ScopeID,
			ExecutionKind:    turn.Run.Kind,
			Model:            model,
			Messages:         llmMsgs,
			Toolset:          bindings.Tools,
			MaxIterations:    maxIterations,
			ProgressObserver: fsmProgObs,
		}

		fsmRes, err := e.fsmEngine.RunToolLoop(ctx, fsmReq)
		if err == nil && fsmRes != nil {
			result = agentapi.Result{
				RunID:       fsmRes.RunID,
				Content:     fsmRes.Content,
				Iterations:  fsmRes.TotalIterations,
				Attachments: fsmRes.AgentAttachments,
				Actions:     fsmRes.Actions,
			}
			executed = true
		} else {
			if ctx.Err() != nil {
				return agentapi.Result{}, ctx.Err()
			}
			slog.Warn("fsm tool loop execution failed, evaluating volatile fallback", "session", turn.Run.Session.SessionID, "error", err)
			runErr = err
		}
	}

	// 2. Volatile execution fallback (or standalone when FSM is absent or failed)
	if !executed {
		if e.llmClient == nil {
			if runErr != nil {
				return agentapi.Result{}, runErr
			}
			return agentapi.Result{}, errors.New("no execution engine or llm client available")
		}

		var reply string
		if bindings.Tools != nil {
			defs, defErr := bindings.Tools.Definitions(ctx)
			if defErr != nil {
				return agentapi.Result{}, fmt.Errorf("failed to retrieve tool definitions: %w", defErr)
			}

			if len(defs) > 0 {
				openaiTools := make([]openai.Tool, 0, len(defs))
				for _, d := range defs {
					openaiTools = append(openaiTools, d.Schema)
				}
				executor := &toolsetExecutor{toolset: bindings.Tools}
				reply, err = e.llmClient.GenerateChatResponseWithToolLoop(ctx, llmMsgs, openaiTools, executor, maxIterations)
				if err != nil {
					return agentapi.Result{}, fmt.Errorf("volatile tool loop failed: %w", err)
				}
			} else {
				reply, err = e.llmClient.GenerateChatResponse(ctx, llmMsgs)
				if err != nil {
					return agentapi.Result{}, fmt.Errorf("llm completion failed: %w", err)
				}
			}
		} else {
			reply, err = e.llmClient.GenerateChatResponse(ctx, llmMsgs)
			if err != nil {
				return agentapi.Result{}, fmt.Errorf("llm completion failed: %w", err)
			}
		}

		result = agentapi.Result{
			RunID:      turn.Run.RunID,
			Content:    reply,
			Iterations: 1,
		}
	}

	// 3. Frontend delivery under session lock
	if turn.Deliver && fe != nil {
		deliverCtx := ctx
		if _, hasDeadline := ctx.Deadline(); !hasDeadline {
			var cancel context.CancelFunc
			deliverCtx, cancel = context.WithTimeout(ctx, 30*time.Second)
			defer cancel()
		}

		comp := agentapi.Completion{
			Run:    turn.Run,
			Result: result,
			Status: agentapi.RunCompleted,
		}
		if deliverErr := fe.Deliver(deliverCtx, comp); deliverErr != nil {
			return result, &DeliveryError{
				Result: result,
				Err:    deliverErr,
			}
		}
	}

	return result, nil
}

// Close gracefully closes the engine.
func (e *Engine) Close() error {
	e.closed.Store(true)
	return nil
}

type toolsetExecutor struct {
	toolset agentapi.Toolset
}

func (e *toolsetExecutor) Execute(ctx context.Context, name string, argsJSON string) (string, error) {
	res, err := e.toolset.Execute(ctx, agentapi.ToolCall{
		Name:      name,
		Arguments: argsJSON,
	})
	if err != nil {
		return "", err
	}
	return res.Content, nil
}
