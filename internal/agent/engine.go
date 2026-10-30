package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
	openai "github.com/sashabaranov/go-openai"

	"bob/internal/agentapi"
	"bob/internal/fsm"
	"bob/internal/llm"
	"bob/internal/trajectory"
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

// TrajectoryStoreProvider resolves isolated trajectory stores for chat sessions.
type TrajectoryStoreProvider interface {
	GetTrajectoryStore(ctx context.Context, chatID string, isDM bool) (*trajectory.SQLiteStore, error)
}

// Config configures the core Agent Engine.
type Config struct {
	DefaultModel  string
	MaxIterations int
	LockTimeout   time.Duration
}

// Dependencies contains the required and optional subsystems needed by the Engine.
type Dependencies struct {
	Config             Config
	LLMClient          LLMClient
	FSMEngine          FSMEngine
	SessionLocker      *SessionLocker
	Frontends          map[string]agentapi.Frontend
	TrajectoryStore    trajectory.Store
	TrajectoryProvider TrajectoryStoreProvider
	TrajectoryDelay    time.Duration
}

// Engine coordinates conversational turns, session concurrency, frontend binding, and tool execution.
type Engine struct {
	cfg           Config
	llmClient     LLMClient
	fsmMu         sync.RWMutex
	fsmEngine     FSMEngine
	sessionLocker *SessionLocker

	frontendsMu sync.RWMutex
	frontends   map[string]agentapi.Frontend

	trajectoryStore    trajectory.Store
	trajectoryProvider TrajectoryStoreProvider
	trajectoryDelay    time.Duration

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

	delay := deps.TrajectoryDelay
	if delay <= 0 {
		delay = 15 * time.Minute
	}

	return &Engine{
		cfg:                deps.Config,
		llmClient:          deps.LLMClient,
		fsmEngine:          deps.FSMEngine,
		sessionLocker:      locker,
		frontends:          frontends,
		trajectoryStore:    deps.TrajectoryStore,
		trajectoryProvider: deps.TrajectoryProvider,
		trajectoryDelay:    delay,
	}, nil
}

func (e *Engine) resolveTrajectoryStore(ctx context.Context, chatID string, isDM bool) trajectory.Store {
	if e == nil {
		return nil
	}
	if e.trajectoryStore != nil {
		return e.trajectoryStore
	}
	if e.trajectoryProvider != nil {
		tStore, err := e.trajectoryProvider.GetTrajectoryStore(ctx, chatID, isDM)
		if err == nil && tStore != nil {
			return tStore
		}
	}
	return nil
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

// SetFSMEngine sets the durable FSM engine instance.
// It is protected by a mutex for thread-safety, but callers should typically invoke it
// during bootstrap before the engine begins accepting turns.
func (e *Engine) SetFSMEngine(fe FSMEngine) {
	e.fsmMu.Lock()
	defer e.fsmMu.Unlock()
	e.fsmEngine = fe
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
	if e == nil {
		return agentapi.Result{}, errors.New("engine is nil")
	}
	if e.closed.Load() {
		return agentapi.Result{}, errors.New("engine is closed")
	}

	if turn.Run.RunID == "" {
		turn.Run.RunID = uuid.NewString()
	}
	trajID := turn.Run.RunID

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

	// Extract user input and active skills for trajectory recording
	var userInput string
	for i := len(turn.Messages) - 1; i >= 0; i-- {
		if turn.Messages[i].Role == openai.ChatMessageRoleUser {
			userInput = turn.Messages[i].Content
			break
		}
	}
	var activeSkills []string
	if turn.Run.Metadata != nil {
		if s, ok := turn.Run.Metadata["active_skills"]; ok && s != "" {
			activeSkills = strings.Split(s, ",")
		}
	}

	trajStore := e.resolveTrajectoryStore(ctx, turn.Run.Session.SessionID, turn.Run.Session.IsDM)
	privacy, exportApproved := trajectory.DerivePrivacyLevel(turn.Run.Session.IsDM, userInput, turn.SystemPrompt, activeSkills)

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

	startPayload := trajectory.TurnStartPayload{
		RunID:          turn.Run.RunID,
		FrontendID:     feID,
		SessionID:      turn.Run.Session.SessionID,
		ScopeID:        turn.Run.Session.ScopeID,
		ActorID:        turn.Run.Actor.ID,
		Model:          model,
		ExecutionKind:  string(turn.Run.Kind),
		PrivacyLevel:   privacy,
		ExportApproved: exportApproved,
		Generation:     1,
		Input:          userInput,
		TaskState:      turn.SystemPrompt,
		ActiveSkills:   activeSkills,
	}

	if trajStore != nil {
		if startEv, evErr := trajectory.MakeTurnStartEvent(trajID, turn.Run.RunID, 1, startPayload); evErr == nil {
			_ = trajStore.RecordEvent(ctx, startEv)
		}
	}

	var result agentapi.Result
	var executed bool
	var runErr error

	disableFallback := turn.Run.Metadata != nil && turn.Run.Metadata["disable_volatile_fallback"] == "true"

	e.fsmMu.RLock()
	fsmEngine := e.fsmEngine
	e.fsmMu.RUnlock()

	// 1. Primary execution via FSM Engine (durable tool loop)
	if fsmEngine != nil {
		var fsmProgObs fsm.ProgressObserver
		if bindings.Progress != nil {
			if po, ok := bindings.Progress.(fsm.ProgressObserver); ok {
				fsmProgObs = po
			}
		}

		var sourceMsgSeq *int64
		if turn.Run.Metadata != nil {
			if s, ok := turn.Run.Metadata["source_message_seq"]; ok && s != "" {
				if seq, err := strconv.ParseInt(s, 10, 64); err == nil && seq > 0 {
					sourceMsgSeq = &seq
				}
			}
		}

		fsmReq := fsm.ToolLoopRequest{
			RunID:            turn.Run.RunID,
			ChatID:           turn.Run.Session.SessionID,
			UserID:           turn.Run.Actor.ID,
			IsDM:             turn.Run.Session.IsDM,
			FrontendID:       feID,
			ScopeID:          turn.Run.Session.ScopeID,
			ExecutionKind:    turn.Run.Kind,
			Model:            model,
			Messages:         llmMsgs,
			Toolset:          bindings.Tools,
			MaxIterations:    maxIterations,
			SourceMessageSeq: sourceMsgSeq,
			ProgressObserver: fsmProgObs,
		}

		fsmRes, err := fsmEngine.RunToolLoop(ctx, fsmReq)
		if err == nil && fsmRes != nil {
			result = agentapi.Result{
				RunID:       fsmRes.RunID,
				Content:     fsmRes.Content,
				Iterations:  fsmRes.TotalIterations,
				Attachments: fsmRes.AgentAttachments,
				Actions:     fsmRes.Actions,
			}
			executed = true
			if trajStore != nil {
				termPayload := trajectory.TurnTerminalPayload{
					Generation: 1,
					Reason:     "completed",
					Failed:     false,
					FinalReply: fsmRes.Content,
				}
				if termEv, evErr := trajectory.MakeTurnTerminalEvent(trajID, turn.Run.RunID, 1, termPayload); evErr == nil {
					_ = trajStore.RecordEvent(ctx, termEv)
				}
				_ = trajStore.ReconcileTerminalRuns(ctx)
			}
		} else {
			if ctx.Err() != nil {
				if trajStore != nil {
					cleanupCtx, cleanupCancel := context.WithTimeout(context.Background(), 5*time.Second)
					termPayload := trajectory.TurnTerminalPayload{
						Generation: 1,
						Reason:     "cancelled",
						Failed:     true,
					}
					if termEv, evErr := trajectory.MakeTurnTerminalEvent(trajID, turn.Run.RunID, 1, termPayload); evErr == nil {
						_ = trajStore.RecordEvent(cleanupCtx, termEv)
					}
					_ = trajStore.ReconcileTerminalRuns(cleanupCtx)
					cleanupCancel()
				}
				return agentapi.Result{}, ctx.Err()
			}
			runErr = err
			// Check if failure was a database persistence error (StoreError)
			var storeErr *fsm.StoreError
			if errors.As(err, &storeErr) {
				slog.Error("fsm tool loop persistence failure, blocking volatile fallback", "session", turn.Run.Session.SessionID, "error", err)
				if trajStore != nil {
					termPayload := trajectory.TurnTerminalPayload{
						Generation: 1,
						Reason:     "fsm_failed",
						Failed:     true,
					}
					if termEv, evErr := trajectory.MakeTurnTerminalEvent(trajID, turn.Run.RunID, 1, termPayload); evErr == nil {
						_ = trajStore.RecordEvent(ctx, termEv)
					}
					_ = trajStore.ReconcileTerminalRuns(ctx)
				}
				return agentapi.Result{}, runErr
			}
			var timeoutErr *llm.TimeoutError
			isTimeout := errors.As(err, &timeoutErr)
			if disableFallback || isTimeout {
				slog.Warn("fsm tool loop execution failed", "session", turn.Run.Session.SessionID, "error", err)
				if trajStore != nil {
					reason := "fsm_failed"
					if isTimeout {
						reason = "timeout"
					}
					termPayload := trajectory.TurnTerminalPayload{
						Generation: 1,
						Reason:     reason,
						Failed:     true,
					}
					if termEv, evErr := trajectory.MakeTurnTerminalEvent(trajID, turn.Run.RunID, 1, termPayload); evErr == nil {
						_ = trajStore.RecordEvent(ctx, termEv)
					}
					_ = trajStore.ReconcileTerminalRuns(ctx)
				}
				return agentapi.Result{}, runErr
			}
			slog.Warn("fsm tool loop execution failed, evaluating volatile fallback", "session", turn.Run.Session.SessionID, "error", err)
		}
	}

	fallbackGen := 2
	if fsmEngine == nil {
		fallbackGen = 1
	}

	// 2. Volatile execution fallback (or standalone when FSM is absent or failed)
	if !executed {
		if disableFallback && runErr != nil {
			return agentapi.Result{}, runErr
		}

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
				rawExecutor := &toolsetExecutor{toolset: bindings.Tools}
				recExecutor := &recordingExecutor{
					inner:     rawExecutor,
					trajStore: trajStore,
					trajID:    trajID,
					runID:     turn.Run.RunID,
					gen:       fallbackGen,
				}
				reply, err = e.llmClient.GenerateChatResponseWithToolLoop(ctx, llmMsgs, openaiTools, recExecutor, maxIterations)
				if err != nil {
					if trajStore != nil {
						termPayload := trajectory.TurnTerminalPayload{
							Generation: fallbackGen,
							Reason:     "volatile_failed",
							Failed:     true,
						}
						if termEv, evErr := trajectory.MakeTurnTerminalEvent(trajID, turn.Run.RunID, fallbackGen, termPayload); evErr == nil {
							_ = trajStore.RecordEvent(ctx, termEv)
						}
						_ = trajStore.ReconcileTerminalRuns(ctx)
					}
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

		if trajStore != nil {
			termPayload := trajectory.TurnTerminalPayload{
				Generation: fallbackGen,
				Reason:     "completed",
				Failed:     false,
				FinalReply: reply,
			}
			if termEv, evErr := trajectory.MakeTurnTerminalEvent(trajID, turn.Run.RunID, fallbackGen, termPayload); evErr == nil {
				_ = trajStore.RecordEvent(ctx, termEv)
			}
			_ = trajStore.ReconcileTerminalRuns(ctx)
		}
	}

	finalGen := 1
	if !executed {
		finalGen = 2
	}
	if fsmEngine == nil {
		finalGen = 1
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
			if trajStore != nil {
				termPayload := trajectory.TurnTerminalPayload{
					Generation: finalGen,
					Reason:     "delivery_failed",
					Failed:     true,
					FinalReply: result.Content,
				}
				if termEv, evErr := trajectory.MakeTurnTerminalEvent(trajID, turn.Run.RunID, finalGen, termPayload, "delivery"); evErr == nil {
					_ = trajStore.RecordEvent(deliverCtx, termEv)
				}
				_ = trajStore.ReconcileTerminalRuns(deliverCtx)
			}
			return result, &DeliveryError{
				Result: result,
				Err:    deliverErr,
			}
		}
	}

	return result, nil
}

type recordingExecutor struct {
	inner     llm.ToolExecutor
	trajStore trajectory.Store
	trajID    string
	runID     string
	gen       int
	mu        sync.Mutex
	stepIndex int
}

func (r *recordingExecutor) Execute(ctx context.Context, name string, argsJSON string) (string, error) {
	startTime := time.Now()
	res, err := r.inner.Execute(ctx, name, argsJSON)
	durationMs := time.Since(startTime).Milliseconds()

	if r.trajStore != nil {
		r.mu.Lock()
		idx := r.stepIndex
		r.stepIndex++
		r.mu.Unlock()

		exitCode := 0
		obs := res
		if err != nil {
			exitCode = 1
			obs = err.Error()
		} else if parsed := trajectory.ParseExitCode(res); parsed != nil {
			exitCode = *parsed
		}

		step := trajectory.Step{
			Index:       idx,
			Iteration:   1,
			Attempt:     1,
			Action:      name,
			ActionArgs:  json.RawMessage(argsJSON),
			Observation: obs,
			ExitCode:    &exitCode,
			DurationMs:  durationMs,
		}
		stepID := fmt.Sprintf("step_vol_%d", idx)
		if ev, evErr := trajectory.MakeToolAttemptEvent(r.trajID, r.runID, r.gen, 1, "loop", stepID, 1, step); evErr == nil {
			_ = r.trajStore.RecordEvent(ctx, ev)
		}
	}
	return res, err
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
