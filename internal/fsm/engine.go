package fsm

import (
	"context"
	"crypto/rand"
	"errors"
	"fmt"
	"log/slog"
	"sync"
	"sync/atomic"
	"time"

	"bob/internal/tools"

	openai "github.com/sashabaranov/go-openai"
)

// LLMClient defines the chat completion interface required by the FSM engine.
type LLMClient interface {
	CreateChatCompletion(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error)
}

// StoreProvider resolves or enumerates SQLite stores for chats.
type StoreProvider interface {
	GetStore(ctx context.Context, chatID string, isDM bool) (*Store, error)
	ActiveStores(ctx context.Context) ([]*Store, error)
}

// StaticStoreProvider provides a single store for all requests (useful for tests or single-DB setups).
type StaticStoreProvider struct {
	store *Store
}

// NewStaticStoreProvider creates a StoreProvider returning a static store.
func NewStaticStoreProvider(store *Store) *StaticStoreProvider {
	return &StaticStoreProvider{store: store}
}

// GetStore returns the static store.
func (p *StaticStoreProvider) GetStore(ctx context.Context, chatID string, isDM bool) (*Store, error) {
	return p.store, nil
}

// ActiveStores returns the static store as the only active store.
func (p *StaticStoreProvider) ActiveStores(ctx context.Context) ([]*Store, error) {
	return []*Store{p.store}, nil
}

// ToolDefinitionProvider supplies tool definitions for a given chat session.
type ToolDefinitionProvider interface {
	ToolDefinitions(ctx context.Context, chatID string, isDM bool) []openai.Tool
}

// Runner defines the interface for executing an FSM workflow.
type Runner interface {
	Execute(ctx context.Context, run *FSMRun, store *Store, tools []openai.Tool, model string) error
}

// ToolLoopRequest encapsulates all parameters needed to execute a tool loop.
type ToolLoopRequest struct {
	RunID         string
	ChatID        string
	UserID        string
	IsDM          bool
	Model         string
	Messages      []openai.ChatCompletionMessage
	Tools         []openai.Tool
	MaxIterations int
	OnTransition  func(state RunState, run *FSMRun)
}

// ToolLoopResult contains the outcome of a completed tool loop run.
type ToolLoopResult struct {
	RunID           string
	Content         string
	TotalIterations int
}

// EngineOption configures an Engine instance.
type EngineOption func(*Engine)

// WithPollInterval sets the polling interval for due waiting runs.
func WithPollInterval(d time.Duration) EngineOption {
	return func(e *Engine) {
		if d > 0 {
			e.pollInterval = d
		}
	}
}

// WithDefaultModel sets the default LLM model identifier.
func WithDefaultModel(m string) EngineOption {
	return func(e *Engine) {
		if m != "" {
			e.defaultModel = m
		}
	}
}

// WithEngineStepExecutor configures a custom StepExecutor for the engine.
func WithEngineStepExecutor(exec *StepExecutor) EngineOption {
	return func(e *Engine) {
		if exec != nil {
			e.stepExecutor = exec
		}
	}
}

// WithToolDefinitionProvider sets the provider for chat session tool definitions.
func WithToolDefinitionProvider(p ToolDefinitionProvider) EngineOption {
	return func(e *Engine) {
		e.toolDefProvider = p
	}
}

// ResultSink defines the interface for delivering completed or failed workflow results out-of-band.
type ResultSink interface {
	Deliver(ctx context.Context, run *FSMRun) error
}

// WithResultSink configures a ResultSink for delivering async/recovered run results.
func WithResultSink(sink ResultSink) EngineOption {
	return func(e *Engine) {
		e.resultSink = sink
	}
}

// Engine coordinates durable FSM workflow executions, state dispatching, delayed transitions, and crash recovery.
type Engine struct {
	storeProvider   StoreProvider
	llmClient       LLMClient
	invoker         ToolInvoker
	stepExecutor    *StepExecutor
	toolDefProvider ToolDefinitionProvider
	resultSink      ResultSink
	pollInterval    time.Duration
	defaultModel    string

	runnersMu sync.RWMutex
	runners   map[FSMType]Runner

	runningMu sync.Mutex
	running   map[string]context.CancelFunc

	wakeCh chan struct{}
	stopCh chan struct{}
	wg     sync.WaitGroup

	closed atomic.Bool
}

// NewEngine creates and initializes a new FSM Engine.
func NewEngine(storeProvider StoreProvider, llmClient LLMClient, invoker ToolInvoker, opts ...EngineOption) *Engine {
	e := &Engine{
		storeProvider: storeProvider,
		llmClient:     llmClient,
		invoker:       invoker,
		pollInterval:  500 * time.Millisecond,
		defaultModel:  "gemini-3.7-flash",
		runners:       make(map[FSMType]Runner),
		running:       make(map[string]context.CancelFunc),
		wakeCh:        make(chan struct{}, 16),
		stopCh:        make(chan struct{}),
	}

	for _, opt := range opts {
		opt(e)
	}

	if e.stepExecutor == nil {
		e.stepExecutor = NewStepExecutor(invoker, nil)
	}

	if e.toolDefProvider == nil && invoker != nil {
		if p, ok := invoker.(ToolDefinitionProvider); ok {
			e.toolDefProvider = p
		} else if inv, ok := invoker.(interface {
			ToolDefinitionsForChat(ctx context.Context, chatID string, isDM bool) []openai.Tool
		}); ok {
			e.toolDefProvider = ToolDefinitionProviderFunc(inv.ToolDefinitionsForChat)
		}
	}

	// Register default ToolLoopRunner
	e.RegisterRunner(FSMTypeToolLoop, NewToolLoopRunner(llmClient, e.stepExecutor, e.defaultModel))

	return e
}

// RegisterRunner registers a workflow runner for the given FSMType.
func (e *Engine) RegisterRunner(fsmType FSMType, runner Runner) {
	e.runnersMu.Lock()
	defer e.runnersMu.Unlock()
	e.runners[fsmType] = runner
}

// SetToolDefinitionProvider configures or updates the tool definition provider for the engine.
func (e *Engine) SetToolDefinitionProvider(p ToolDefinitionProvider) {
	e.toolDefProvider = p
}

// ToolDefinitionProviderFunc adapts a function to ToolDefinitionProvider.
type ToolDefinitionProviderFunc func(ctx context.Context, chatID string, isDM bool) []openai.Tool

// ToolDefinitions calls the underlying function to return tool definitions.
func (f ToolDefinitionProviderFunc) ToolDefinitions(ctx context.Context, chatID string, isDM bool) []openai.Tool {
	return f(ctx, chatID, isDM)
}

// SetResultSink configures or updates the result delivery sink for the engine.
func (e *Engine) SetResultSink(sink ResultSink) {
	e.resultSink = sink
}

func (e *Engine) getRunner(fsmType FSMType) (Runner, error) {
	e.runnersMu.RLock()
	defer e.runnersMu.RUnlock()
	runner, ok := e.runners[fsmType]
	if !ok {
		return nil, fmt.Errorf("no runner registered for fsm type: %s", fsmType)
	}
	return runner, nil
}

func (e *Engine) acquireRun(runID string, cancel context.CancelFunc) bool {
	e.runningMu.Lock()
	defer e.runningMu.Unlock()
	if _, exists := e.running[runID]; exists {
		return false
	}
	e.running[runID] = cancel
	return true
}

func (e *Engine) releaseRun(runID string) {
	e.runningMu.Lock()
	defer e.runningMu.Unlock()
	delete(e.running, runID)
}

func (e *Engine) isRunActive(runID string) bool {
	e.runningMu.Lock()
	defer e.runningMu.Unlock()
	_, exists := e.running[runID]
	return exists
}

func (e *Engine) signalWake() {
	select {
	case e.wakeCh <- struct{}{}:
	default:
	}
}

// Start initiates background delayed-transition polling and crash recovery.
func (e *Engine) Start(ctx context.Context) error {
	if err := e.Recover(ctx); err != nil {
		slog.Error("fsm engine crash recovery encountered errors", "error", err)
	}

	e.wg.Add(1)
	go func() {
		defer e.wg.Done()
		ticker := time.NewTicker(e.pollInterval)
		defer ticker.Stop()

		for {
			select {
			case <-ctx.Done():
				return
			case <-e.stopCh:
				return
			case <-ticker.C:
				if err := e.PollDueWaitingRuns(ctx); err != nil {
					slog.Error("fsm poller failed to poll due waiting runs", "error", err)
				}
			case <-e.wakeCh:
				if err := e.PollDueWaitingRuns(ctx); err != nil {
					slog.Error("fsm poller failed to process wake signal", "error", err)
				}
			}
		}
	}()

	return nil
}

// Stop terminates the engine's background poller and cancels any active running runs.
func (e *Engine) Stop() {
	if e.closed.Swap(true) {
		return
	}
	close(e.stopCh)

	e.runningMu.Lock()
	for _, cancel := range e.running {
		cancel()
	}
	e.runningMu.Unlock()

	e.wg.Wait()
}

// Recover scans all active stores for interrupted runs and resumes them.
func (e *Engine) Recover(ctx context.Context) error {
	stores, err := e.storeProvider.ActiveStores(ctx)
	if err != nil {
		return fmt.Errorf("failed to list active stores for recovery: %w", err)
	}

	now := time.Now().Unix()
	var allErrs error

	for _, store := range stores {
		runs, err := store.ListActiveRuns(ctx)
		if err != nil {
			allErrs = errors.Join(allErrs, err)
			continue
		}

		for _, r := range runs {
			run := r
			if e.isRunActive(run.ID) {
				continue
			}

			// If it's WAITING and not yet due, schedule an in-memory timer
			if run.Status == RunStatusWaiting && run.ResumeAt != nil && *run.ResumeAt > now {
				delay := time.Until(time.Unix(*run.ResumeAt, 0))
				time.AfterFunc(delay, func() {
					e.signalWake()
				})
				continue
			}

			// Interrupted in RUNNING, PENDING, or due WAITING: resume execution
			e.wg.Add(1)
			go func(runID string, s *Store) {
				defer e.wg.Done()
				runCtx, cancel := context.WithCancel(ctx)
				defer cancel()

				if !e.acquireRun(runID, cancel) {
					return
				}
				defer e.releaseRun(runID)

				fresh, err := s.GetRun(runCtx, runID)
				if err != nil {
					slog.Error("failed to get fresh run on recovery", "run_id", runID, "error", err)
					return
				}
				if fresh.Status.IsTerminal() {
					return
				}
				if fresh.Status == RunStatusWaiting && fresh.ResumeAt != nil && *fresh.ResumeAt > time.Now().Unix() {
					delay := time.Until(time.Unix(*fresh.ResumeAt, 0))
					time.AfterFunc(delay, func() {
						e.signalWake()
					})
					return
				}
				runToRecover := *fresh

				if runToRecover.ChatID == "" {
					slog.Error("refusing to recover run with empty chat_id", "run_id", runToRecover.ID)
					return
				}

				runCtx = tools.WithChatSession(runCtx, tools.ChatSessionContext{
					ChatID: runToRecover.ChatID,
					UserID: runToRecover.UserID,
					IsDM:   runToRecover.IsDM,
				})

				runToRecover.Status = RunStatusRunning
				runToRecover.ResumeAt = nil
				if runToRecover.CurrentState == StateWaiting {
					runToRecover.WaitCycles++
					if runToRecover.WaitCycles >= 10 {
						runToRecover.Status = RunStatusFailed
						runToRecover.CurrentState = StateFailed
						runToRecover.ErrorText = fmt.Sprintf("recovered run %s exceeded maximum wait cycles (10)", runToRecover.ID)
						if updateErr := s.UpdateRun(runCtx, &runToRecover); updateErr != nil {
							slog.Error("failed to persist wait cycles failure for recovered run", "run_id", runToRecover.ID, "error", updateErr)
						}
						return
					}
					runToRecover.CurrentState = StateExecuteSteps
				}
				if err := s.UpdateRun(runCtx, &runToRecover); err != nil {
					slog.Error("failed to update recovered run", "run_id", runToRecover.ID, "error", err)
					return
				}

				runner, err := e.getRunner(runToRecover.FSMType)
				if err != nil {
					slog.Error("no runner for recovered run", "fsm_type", runToRecover.FSMType, "error", err)
					return
				}

				var tools []openai.Tool
				if e.toolDefProvider != nil {
					tools = e.toolDefProvider.ToolDefinitions(runCtx, runToRecover.ChatID, runToRecover.IsDM)
				}

				if err := runner.Execute(runCtx, &runToRecover, s, tools, e.defaultModel); err != nil {
					slog.Error("error executing recovered run", "run_id", runToRecover.ID, "error", err)
				}
				if e.resultSink != nil && (runToRecover.Status == RunStatusCompleted || runToRecover.Status == RunStatusFailed) {
					if err := e.resultSink.Deliver(runCtx, &runToRecover); err != nil {
						slog.Error("failed to deliver recovered run result", "run_id", runToRecover.ID, "error", err)
					}
				}
			}(run.ID, store)
		}
	}

	return allErrs
}

// PollDueWaitingRuns checks active stores for runs in WAITING status whose resume_at has arrived and resumes them.
func (e *Engine) PollDueWaitingRuns(ctx context.Context) error {
	stores, err := e.storeProvider.ActiveStores(ctx)
	if err != nil {
		return fmt.Errorf("failed to list active stores: %w", err)
	}

	now := time.Now().Unix()
	var allErrs error

	for _, store := range stores {
		runs, err := store.ListDueWaitingRuns(ctx, now)
		if err != nil {
			allErrs = errors.Join(allErrs, err)
			continue
		}

		for _, r := range runs {
			run := r
			if e.isRunActive(run.ID) {
				continue
			}

			e.wg.Add(1)
			go func(runID string, s *Store) {
				defer e.wg.Done()
				runCtx, cancel := context.WithCancel(ctx)
				defer cancel()

				if !e.acquireRun(runID, cancel) {
					return
				}
				defer e.releaseRun(runID)

				fresh, err := s.GetRun(runCtx, runID)
				if err != nil {
					slog.Error("failed to get fresh run on resume", "run_id", runID, "error", err)
					return
				}
				if fresh.Status.IsTerminal() {
					return
				}
				if fresh.Status != RunStatusWaiting {
					return
				}
				if fresh.ResumeAt != nil && *fresh.ResumeAt > time.Now().Unix() {
					return
				}
				runToResume := *fresh

				if runToResume.ChatID == "" {
					slog.Error("refusing to resume run with empty chat_id", "run_id", runToResume.ID)
					return
				}

				runCtx = tools.WithChatSession(runCtx, tools.ChatSessionContext{
					ChatID: runToResume.ChatID,
					UserID: runToResume.UserID,
					IsDM:   runToResume.IsDM,
				})

				runToResume.Status = RunStatusRunning
				runToResume.ResumeAt = nil
				if runToResume.CurrentState == StateWaiting {
					runToResume.WaitCycles++
					if runToResume.WaitCycles >= 10 {
						runToResume.Status = RunStatusFailed
						runToResume.CurrentState = StateFailed
						runToResume.ErrorText = fmt.Sprintf("waiting run %s exceeded maximum wait cycles (10)", runToResume.ID)
						if updateErr := s.UpdateRun(runCtx, &runToResume); updateErr != nil {
							slog.Error("failed to persist wait cycles failure for waiting run", "run_id", runToResume.ID, "error", updateErr)
						}
						return
					}
					runToResume.CurrentState = StateExecuteSteps
				}
				if err := s.UpdateRun(runCtx, &runToResume); err != nil {
					slog.Error("failed to update waiting run to running", "run_id", runToResume.ID, "error", err)
					return
				}

				runner, err := e.getRunner(runToResume.FSMType)
				if err != nil {
					slog.Error("no runner for resumed run", "fsm_type", runToResume.FSMType, "error", err)
					return
				}

				var tools []openai.Tool
				if e.toolDefProvider != nil {
					tools = e.toolDefProvider.ToolDefinitions(runCtx, runToResume.ChatID, runToResume.IsDM)
				}

				if err := runner.Execute(runCtx, &runToResume, s, tools, e.defaultModel); err != nil {
					slog.Error("error executing resumed run", "run_id", runToResume.ID, "error", err)
				}
				if e.resultSink != nil && (runToResume.Status == RunStatusCompleted || runToResume.Status == RunStatusFailed) {
					if err := e.resultSink.Deliver(runCtx, &runToResume); err != nil {
						slog.Error("failed to deliver resumed run result", "run_id", runToResume.ID, "error", err)
					}
				}
			}(run.ID, store)
		}
	}

	return allErrs
}

type contextKey string

const transitionCallbackKey contextKey = "fsm_transition_callback"

// WithTransitionCallback attaches a state transition callback to the context.
func WithTransitionCallback(ctx context.Context, cb func(state RunState, run *FSMRun)) context.Context {
	return context.WithValue(ctx, transitionCallbackKey, cb)
}

// GetTransitionCallback returns the state transition callback from context if set.
func GetTransitionCallback(ctx context.Context) func(state RunState, run *FSMRun) {
	if cb, ok := ctx.Value(transitionCallbackKey).(func(state RunState, run *FSMRun)); ok {
		return cb
	}
	return nil
}

func generateRunID() string {
	b := make([]byte, 8)
	if _, err := rand.Read(b); err != nil {
		return fmt.Sprintf("run_%d_%d", time.Now().UnixNano(), time.Now().Unix())
	}
	return fmt.Sprintf("run_%d_%x", time.Now().Unix(), b)
}

// RunToolLoop executes a tool loop workflow run synchronously until completion or suspension.
func (e *Engine) RunToolLoop(ctx context.Context, req ToolLoopRequest) (*ToolLoopResult, error) {
	if req.OnTransition != nil {
		ctx = WithTransitionCallback(ctx, req.OnTransition)
	}
	if req.ChatID == "" {
		return nil, errors.New("chat_id is required")
	}
	if len(req.Messages) == 0 {
		return nil, errors.New("messages cannot be empty")
	}
	if req.RunID == "" {
		req.RunID = generateRunID()
	}
	if req.MaxIterations <= 0 {
		if req.IsDM {
			req.MaxIterations = 20
		} else {
			req.MaxIterations = 10
		}
	}

	store, err := e.storeProvider.GetStore(ctx, req.ChatID, req.IsDM)
	if err != nil {
		return nil, fmt.Errorf("failed to get store for chat %s: %w", req.ChatID, err)
	}

	contextJSON, err := EncodeMessages(req.Messages)
	if err != nil {
		return nil, fmt.Errorf("failed to encode messages: %w", err)
	}

	run := &FSMRun{
		ID:            req.RunID,
		ChatID:        req.ChatID,
		UserID:        req.UserID,
		IsDM:          req.IsDM,
		FSMType:       FSMTypeToolLoop,
		Status:        RunStatusRunning,
		CurrentState:  StateInit,
		Iteration:     0,
		MaxIterations: req.MaxIterations,
		ContextJSON:   contextJSON,
	}

	if err := store.CreateRun(ctx, run); err != nil {
		return nil, fmt.Errorf("failed to create fsm run: %w", err)
	}

	runCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	sess, ok := tools.ChatSessionFromContext(ctx)
	if !ok {
		sess = tools.ChatSessionContext{}
	}
	sess.ChatID = req.ChatID
	sess.UserID = req.UserID
	sess.IsDM = req.IsDM
	runCtx = tools.WithChatSession(runCtx, sess)

	if !e.acquireRun(run.ID, cancel) {
		return nil, fmt.Errorf("run %s is already executing", run.ID)
	}
	defer e.releaseRun(run.ID)

	tools := req.Tools
	if len(tools) == 0 && e.toolDefProvider != nil {
		tools = e.toolDefProvider.ToolDefinitions(ctx, req.ChatID, req.IsDM)
	}

	model := req.Model
	if model == "" {
		model = e.defaultModel
	}

	runner, err := e.getRunner(FSMTypeToolLoop)
	if err != nil {
		return nil, err
	}

	execErr := runner.Execute(runCtx, run, store, tools, model)
	if execErr != nil && runCtx.Err() != nil {
		return nil, runCtx.Err()
	}
	if execErr != nil {
		return nil, execErr
	}

	// Handle WAITING state: wait for resume_at or timer wakeup
	const maxWaitCycles = 10
	localWaitCycles := 0
	for run.Status == RunStatusWaiting {
		localWaitCycles++
		run.WaitCycles++
		if run.WaitCycles >= maxWaitCycles || localWaitCycles >= maxWaitCycles {
			run.Status = RunStatusFailed
			run.CurrentState = StateFailed
			run.ErrorText = fmt.Sprintf("run exceeded maximum wait cycles (%d)", maxWaitCycles)
			if updateErr := store.UpdateRun(runCtx, run); updateErr != nil {
				return nil, errors.Join(fmt.Errorf("run %s exceeded maximum wait cycles (%d)", run.ID, maxWaitCycles), updateErr)
			}
			return nil, fmt.Errorf("run %s exceeded maximum wait cycles (%d)", run.ID, maxWaitCycles)
		}
		if updateErr := store.UpdateRun(runCtx, run); updateErr != nil {
			return nil, updateErr
		}

		if runCtx.Err() != nil {
			return nil, runCtx.Err()
		}

		waitDuration := 500 * time.Millisecond
		if run.ResumeAt != nil {
			delta := time.Until(time.Unix(*run.ResumeAt, 0))
			if delta > 0 {
				waitDuration = delta
			} else {
				waitDuration = 20 * time.Millisecond
			}
		}

		select {
		case <-runCtx.Done():
			return nil, runCtx.Err()
		case <-time.After(waitDuration):
		}

		updatedRun, getErr := store.GetRun(runCtx, run.ID)
		if getErr != nil {
			return nil, getErr
		}
		if updatedRun.WaitCycles < run.WaitCycles {
			updatedRun.WaitCycles = run.WaitCycles
		}
		run = updatedRun
		if run.Status == RunStatusWaiting {
			now := time.Now().Unix()
			if run.ResumeAt == nil || *run.ResumeAt <= now {
				run.Status = RunStatusRunning
				run.ResumeAt = nil
				run.CurrentState = StateExecuteSteps
				if updateErr := store.UpdateRun(runCtx, run); updateErr != nil {
					return nil, updateErr
				}
				execErr = runner.Execute(runCtx, run, store, tools, model)
				if execErr != nil {
					return nil, execErr
				}
			}
		}
	}

	if run.Status == RunStatusFailed {
		return nil, fmt.Errorf("run %s failed: %s", run.ID, run.ErrorText)
	}

	return &ToolLoopResult{
		RunID:           run.ID,
		Content:         run.ResultJSON,
		TotalIterations: run.Iteration,
	}, nil
}
