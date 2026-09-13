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

// Engine coordinates durable FSM workflow executions, state dispatching, delayed transitions, and crash recovery.
type Engine struct {
	storeProvider   StoreProvider
	llmClient       LLMClient
	invoker         ToolInvoker
	stepExecutor    *StepExecutor
	toolDefProvider ToolDefinitionProvider
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
			go func(runToRecover FSMRun, s *Store) {
				defer e.wg.Done()
				runCtx, cancel := context.WithCancel(ctx)
				defer cancel()

				if !e.acquireRun(runToRecover.ID, cancel) {
					return
				}
				defer e.releaseRun(runToRecover.ID)

				runToRecover.Status = RunStatusRunning
				runToRecover.ResumeAt = nil
				if runToRecover.CurrentState == StateWaiting {
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
					tools = e.toolDefProvider.ToolDefinitions(runCtx, runToRecover.ChatID, runToRecover.ChatID != "townhall")
				}

				if err := runner.Execute(runCtx, &runToRecover, s, tools, e.defaultModel); err != nil {
					slog.Error("error executing recovered run", "run_id", runToRecover.ID, "error", err)
				}
			}(run, store)
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
			go func(runToResume FSMRun, s *Store) {
				defer e.wg.Done()
				runCtx, cancel := context.WithCancel(ctx)
				defer cancel()

				if !e.acquireRun(runToResume.ID, cancel) {
					return
				}
				defer e.releaseRun(runToResume.ID)

				runToResume.Status = RunStatusRunning
				runToResume.ResumeAt = nil
				if runToResume.CurrentState == StateWaiting {
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
					tools = e.toolDefProvider.ToolDefinitions(runCtx, runToResume.ChatID, runToResume.ChatID != "townhall")
				}

				if err := runner.Execute(runCtx, &runToResume, s, tools, e.defaultModel); err != nil {
					slog.Error("error executing resumed run", "run_id", runToResume.ID, "error", err)
				}
			}(run, store)
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
	for run.Status == RunStatusWaiting {
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

	return &ToolLoopResult{
		RunID:           run.ID,
		Content:         run.ResultJSON,
		TotalIterations: run.Iteration,
	}, nil
}
