package fsm

import (
	"context"
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"bob/internal/agentapi"
	"bob/internal/knowledge"
	"bob/internal/models"
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
	Execute(ctx context.Context, run *FSMRun, store *Store, toolset agentapi.Toolset, model string) error
}

// ToolLoopRequest encapsulates all parameters needed to execute a tool loop.
type ToolLoopRequest struct {
	RunID            string
	ChatID           string
	UserID           string
	IsDM             bool
	FrontendID       string
	ScopeID          string
	ExecutionKind    agentapi.ExecutionKind
	Model            string
	Messages         []openai.ChatCompletionMessage
	Tools            []openai.Tool    // Legacy tool definitions
	Toolset          agentapi.Toolset // Execution-scoped toolset
	MaxIterations    int
	SourceMessageSeq *int64
	OnTransition     func(state RunState, run *FSMRun)
	ProgressObserver ProgressObserver
}

// ToolLoopResult contains the outcome of a completed tool loop run.
type ToolLoopResult struct {
	RunID            string
	Content          string
	TotalIterations  int
	Attachments      []models.Attachment
	AgentAttachments []agentapi.Attachment
	Actions          []agentapi.Action
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
		e.defaultModel = m
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

// WithFrontend configures a registered frontend for execution binding and recovery delivery.
func WithFrontend(id string, f agentapi.Frontend) EngineOption {
	return func(e *Engine) {
		e.RegisterFrontend(id, f)
	}
}

// WithRecoveryStalenessCutoff sets the maximum staleness duration for interrupted runs on recovery.
func WithRecoveryStalenessCutoff(d time.Duration) EngineOption {
	return func(e *Engine) {
		if d > 0 {
			e.recoveryStalenessCutoff = d
		}
	}
}

// WithMaxRecoveryConcurrency sets the maximum concurrency for crash recovery executions.
func WithMaxRecoveryConcurrency(n int) EngineOption {
	return func(e *Engine) {
		if n > 0 {
			e.maxRecoveryConcurrency = n
		}
	}
}

// WithKnowledgeBudgetLimits configures the knowledge budget limits used during context restoration and execution.
func WithKnowledgeBudgetLimits(limits tools.KnowledgeBudgetLimits) EngineOption {
	return func(e *Engine) {
		e.knowledgeLimits = limits
	}
}

// Engine coordinates durable FSM workflow executions, state dispatching, delayed transitions, and crash recovery.
type Engine struct {
	storeProvider           StoreProvider
	llmClient               LLMClient
	invoker                 ToolInvoker
	stepExecutor            *StepExecutor
	toolDefProvider         ToolDefinitionProvider
	resultSink              ResultSink
	pollInterval            time.Duration
	defaultModel            string
	recoveryStalenessCutoff time.Duration
	maxRecoveryConcurrency  int
	knowledgeLimits         tools.KnowledgeBudgetLimits

	frontendsMu sync.RWMutex
	frontends   map[string]agentapi.Frontend

	runnersMu sync.RWMutex
	runners   map[FSMType]Runner

	runningMu sync.Mutex
	running   map[string]context.CancelFunc

	timersMu sync.Mutex
	timers   map[string]*time.Timer

	wakeCh chan struct{}
	stopCh chan struct{}
	wg     sync.WaitGroup

	closed atomic.Bool
}

// NewEngine creates and initializes a new FSM Engine.
func NewEngine(storeProvider StoreProvider, llmClient LLMClient, invoker ToolInvoker, opts ...EngineOption) *Engine {
	e := &Engine{
		storeProvider:           storeProvider,
		llmClient:               llmClient,
		invoker:                 invoker,
		pollInterval:            2 * time.Second,
		recoveryStalenessCutoff: 15 * time.Minute,
		maxRecoveryConcurrency:  4,
		knowledgeLimits:         tools.DefaultKnowledgeBudgetLimits(),
		frontends:               make(map[string]agentapi.Frontend),
		runners:                 make(map[FSMType]Runner),
		running:                 make(map[string]context.CancelFunc),
		timers:                  make(map[string]*time.Timer),
		wakeCh:                  make(chan struct{}, 16),
		stopCh:                  make(chan struct{}),
	}

	for _, opt := range opts {
		opt(e)
	}

	if e.stepExecutor == nil {
		e.stepExecutor = NewStepExecutor(invoker)
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
	runner := NewToolLoopRunner(llmClient, e.stepExecutor, e.defaultModel)
	if e.toolDefProvider != nil {
		runner.SetToolDefinitionProvider(e.toolDefProvider)
	}
	e.RegisterRunner(FSMTypeToolLoop, runner)

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
	e.runnersMu.Lock()
	e.toolDefProvider = p
	runner := e.runners[FSMTypeToolLoop]
	e.runnersMu.Unlock()
	if tr, ok := runner.(*ToolLoopRunner); ok {
		tr.SetToolDefinitionProvider(p)
	}
}

func (e *Engine) getToolDefProvider() ToolDefinitionProvider {
	e.runnersMu.RLock()
	defer e.runnersMu.RUnlock()
	return e.toolDefProvider
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

// LegacyFrontendID defines the default frontend ID for pre-existing or unspecified runs.
const LegacyFrontendID = "besedka"

// RegisterFrontend registers a frontend by ID.
func (e *Engine) RegisterFrontend(id string, f agentapi.Frontend) {
	e.frontendsMu.Lock()
	defer e.frontendsMu.Unlock()
	if e.frontends == nil {
		e.frontends = make(map[string]agentapi.Frontend)
	}
	e.frontends[id] = f
}

func (e *Engine) getFrontend(id string) (agentapi.Frontend, bool) {
	e.frontendsMu.RLock()
	defer e.frontendsMu.RUnlock()
	if e.frontends == nil {
		return nil, false
	}
	f, ok := e.frontends[id]
	return f, ok
}

func (e *Engine) resolveRunDescriptor(run *FSMRun) agentapi.RunDescriptor {
	frontendID := run.FrontendID
	if frontendID == "" {
		frontendID = LegacyFrontendID
	}
	scopeID := run.ScopeID
	if scopeID == "" {
		scopeID = run.ChatID
	}
	kind := run.ExecutionKind
	if kind == "" {
		if strings.HasPrefix(run.ID, "sched_") {
			kind = agentapi.Scheduled
		} else {
			kind = agentapi.Interactive
		}
	}
	model := run.Model
	if model == "" {
		model = e.defaultModel
	}

	return agentapi.RunDescriptor{
		RunID: run.ID,
		Session: agentapi.SessionRef{
			FrontendID: frontendID,
			SessionID:  run.ChatID,
			ScopeID:    scopeID,
		},
		Actor: agentapi.Actor{
			ID: run.UserID,
		},
		Kind:          kind,
		Model:         model,
		MaxIterations: run.MaxIterations,
	}
}

func (e *Engine) deliverCompletion(ctx context.Context, run *FSMRun, desc agentapi.RunDescriptor, fe agentapi.Frontend, store *Store) {
	if !run.Status.IsTerminal() {
		return
	}

	deliverCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	var attachments []agentapi.Attachment
	var actions []agentapi.Action
	if store != nil && run.ID != "" {
		steps, err := store.ListStepsByRun(deliverCtx, run.ID)
		if err == nil {
			var seenAttIDs = make(map[string]bool)
			for _, s := range steps {
				if s.Status == StepStatusCompleted {
					res, _ := DecodeStoredToolResult(s.ResultJSON)
					for _, att := range res.Attachments {
						if !seenAttIDs[att.ID] {
							seenAttIDs[att.ID] = true
							attachments = append(attachments, att)
						}
					}
					actions = append(actions, res.Actions...)
				}
			}
		}
	}

	completion := agentapi.Completion{
		Run: desc,
		Result: agentapi.Result{
			RunID:       run.ID,
			Content:     run.ResultJSON,
			Iterations:  run.Iteration,
			Attachments: attachments,
			Actions:     actions,
		},
		Status: agentapi.RunStatus(run.Status),
		Error:  run.ErrorText,
	}

	if fe != nil {
		if err := fe.Deliver(deliverCtx, completion); err != nil {
			slog.Error("failed to deliver completion to frontend", "run_id", run.ID, "frontend_id", desc.Session.FrontendID, "error", err)
		}
		return
	}

	if desc.Session.FrontendID == LegacyFrontendID && e.resultSink != nil {
		if err := e.resultSink.Deliver(deliverCtx, run); err != nil {
			slog.Error("failed to deliver recovered run result via legacy sink", "run_id", run.ID, "error", err)
		}
	}
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
	e.cancelWakeTimer(runID)
	return true
}

func (e *Engine) scheduleWakeTimer(runID string, delay time.Duration) {
	if delay <= 0 {
		e.signalWake()
		return
	}

	e.timersMu.Lock()
	defer e.timersMu.Unlock()

	if e.closed.Load() {
		return
	}

	if t, ok := e.timers[runID]; ok {
		t.Stop()
	}

	e.timers[runID] = time.AfterFunc(delay, func() {
		e.timersMu.Lock()
		delete(e.timers, runID)
		e.timersMu.Unlock()

		e.signalWake()
	})
}

func (e *Engine) cancelWakeTimer(runID string) {
	e.timersMu.Lock()
	defer e.timersMu.Unlock()
	if t, ok := e.timers[runID]; ok {
		t.Stop()
		delete(e.timers, runID)
	}
}

// ActiveTimersCount returns the number of active wake timers pending.
func (e *Engine) ActiveTimersCount() int {
	e.timersMu.Lock()
	defer e.timersMu.Unlock()
	return len(e.timers)
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

func (e *Engine) spawn(fn func()) bool {
	e.runningMu.Lock()
	defer e.runningMu.Unlock()
	if e.closed.Load() {
		return false
	}
	e.wg.Add(1)
	go func() {
		defer e.wg.Done()
		fn()
	}()
	return true
}

// Start initiates background delayed-transition polling and crash recovery.
func (e *Engine) Start(ctx context.Context) error {
	if e.defaultModel == "" {
		return errors.New("fsm engine requires a default model")
	}

	if err := e.Recover(ctx); err != nil {
		slog.Error("fsm engine crash recovery encountered errors", "error", err)
	}

	started := e.spawn(func() {
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
	})
	if !started {
		return errors.New("cannot start closed fsm engine")
	}

	return nil
}

// Stop terminates the engine's background poller and cancels any active running runs.
func (e *Engine) Stop() {
	e.runningMu.Lock()
	if e.closed.Swap(true) {
		e.runningMu.Unlock()
		return
	}
	close(e.stopCh)

	for _, cancel := range e.running {
		cancel()
	}
	e.runningMu.Unlock()

	e.timersMu.Lock()
	for id, t := range e.timers {
		t.Stop()
		delete(e.timers, id)
	}
	e.timersMu.Unlock()

	e.wg.Wait()
}

func restoreChatSessionContext(ctx context.Context, store *Store, run *FSMRun, limits ...tools.KnowledgeBudgetLimits) (tools.ChatSessionContext, error) {
	lim := tools.DefaultKnowledgeBudgetLimits()
	if len(limits) > 0 {
		lim = limits[0]
	}
	sess := tools.NewChatSessionContext(run.ChatID, run.UserID, run.IsDM, lim)
	sess.FSMRunID = run.ID
	sess.SourceMessageSeq = run.SourceMessageSeq
	if store == nil || run.ID == "" {
		return sess, nil
	}
	steps, err := store.ListStepsByRun(ctx, run.ID)
	if err != nil {
		return sess, fmt.Errorf("failed to list steps for run %s: %w", run.ID, err)
	}
	for _, step := range steps {
		if step.Status != StepStatusCompleted {
			continue
		}
		decodedRes, _ := DecodeStoredToolResult(step.ResultJSON)
		stepContent := decodedRes.Content

		switch step.ToolName {
		case "sandbox_upload_attachment":
			if len(decodedRes.Attachments) > 0 {
				for _, att := range decodedRes.Attachments {
					var attType models.AttachmentType
					switch att.Type {
					case agentapi.AttachmentImage:
						attType = models.AttachmentTypeImage
					default:
						attType = models.AttachmentTypeFile
					}
					if err := sess.StageAttachment(models.Attachment{
						FileID:   att.ID,
						Name:     att.Name,
						MimeType: att.MIMEType,
						Type:     attType,
					}); err != nil {
						return sess, fmt.Errorf("failed to stage attachment for step %s: %w", step.ID, err)
					}
				}
			} else {
				var payload struct {
					FileID   string `json:"file_id"`
					Name     string `json:"name"`
					MimeType string `json:"mime_type"`
					Type     string `json:"type"`
				}
				if err := json.Unmarshal([]byte(stepContent), &payload); err != nil {
					return sess, fmt.Errorf("failed to parse sandbox_upload_attachment result for step %s: %w", step.ID, err)
				}
				if payload.FileID == "" {
					return sess, fmt.Errorf("invalid sandbox_upload_attachment result for step %s: missing file_id", step.ID)
				}
				if err := sess.StageAttachment(models.Attachment{
					FileID:   payload.FileID,
					Name:     payload.Name,
					MimeType: payload.MimeType,
					Type:     models.AttachmentType(payload.Type),
				}); err != nil {
					return sess, fmt.Errorf("failed to stage attachment for step %s: %w", step.ID, err)
				}
			}
		case "load_memory":
			var payload struct {
				ItemID       string `json:"item_id"`
				VersionID    string `json:"version_id"`
				Content      string `json:"content"`
				ContentBytes int    `json:"content_bytes"`
				ExpiresAt    *int64 `json:"expires_at,omitempty"`
			}
			if err := json.Unmarshal([]byte(stepContent), &payload); err != nil {
				return sess, fmt.Errorf("failed to parse load_memory result for step %s: %w", step.ID, err)
			}
			if payload.ItemID == "" || !strings.HasPrefix(payload.ItemID, "mem_") {
				return sess, fmt.Errorf("invalid load_memory result for step %s: invalid item_id %q", step.ID, payload.ItemID)
			}
			if payload.VersionID == "" {
				return sess, fmt.Errorf("invalid load_memory result for step %s: missing version_id", step.ID)
			}
			itemID, _, err := knowledge.ParseVersionID(payload.VersionID)
			if err != nil || itemID != payload.ItemID {
				return sess, fmt.Errorf("invalid load_memory result for step %s: version_id %q does not match item_id %q", step.ID, payload.VersionID, payload.ItemID)
			}
			if strings.TrimSpace(payload.Content) == "" {
				return sess, fmt.Errorf("invalid load_memory result for step %s: empty content", step.ID)
			}
			actualBytes := len([]byte(payload.Content))
			if payload.ContentBytes != actualBytes {
				return sess, fmt.Errorf("invalid load_memory result for step %s: content_bytes mismatch (recorded %d, actual %d)", step.ID, payload.ContentBytes, actualBytes)
			}
			if payload.ExpiresAt != nil && time.Now().Unix() >= *payload.ExpiresAt {
				// Expired before recovery; skip restoring into budget tracker so stale item is not cached.
				continue
			}
			if err := sess.KnowledgeBudget.RestoreItem("memory", payload.ItemID, payload.VersionID, payload.Content, payload.ContentBytes, stepContent); err != nil {
				return sess, fmt.Errorf("failed to restore memory item for step %s: %w", step.ID, err)
			}
		case "load_skill":
			var payload struct {
				ItemID               string `json:"item_id"`
				VersionID            string `json:"version_id"`
				Name                 string `json:"name"`
				Description          string `json:"description"`
				InstructionsMarkdown string `json:"instructions_markdown"`
				ContentBytes         int    `json:"content_bytes"`
			}
			if err := json.Unmarshal([]byte(stepContent), &payload); err != nil {
				return sess, fmt.Errorf("failed to parse load_skill result for step %s: %w", step.ID, err)
			}
			if payload.ItemID == "" || !strings.HasPrefix(payload.ItemID, "skill_") {
				return sess, fmt.Errorf("invalid load_skill result for step %s: invalid item_id %q", step.ID, payload.ItemID)
			}
			if payload.VersionID == "" {
				return sess, fmt.Errorf("invalid load_skill result for step %s: missing version_id", step.ID)
			}
			itemID, _, err := knowledge.ParseVersionID(payload.VersionID)
			if err != nil || itemID != payload.ItemID {
				return sess, fmt.Errorf("invalid load_skill result for step %s: version_id %q does not match item_id %q", step.ID, payload.VersionID, payload.ItemID)
			}
			if strings.TrimSpace(payload.InstructionsMarkdown) == "" {
				return sess, fmt.Errorf("invalid load_skill result for step %s: empty instructions_markdown", step.ID)
			}
			actualBytes := len([]byte(payload.Name)) + len([]byte(payload.Description)) + len([]byte(payload.InstructionsMarkdown))
			if payload.ContentBytes != actualBytes {
				return sess, fmt.Errorf("invalid load_skill result for step %s: content_bytes mismatch (recorded %d, actual %d)", step.ID, payload.ContentBytes, actualBytes)
			}
			if err := sess.KnowledgeBudget.RestoreItem("skill", payload.ItemID, payload.VersionID, payload.InstructionsMarkdown, payload.ContentBytes, stepContent); err != nil {
				return sess, fmt.Errorf("failed to restore skill item for step %s: %w", step.ID, err)
			}
		}
	}
	return sess, nil
}

// RestoreChatSessionContext reconstructs a ChatSessionContext for a run with any attachments
// previously staged in completed sandbox_upload_attachment steps and budget usage.
func (e *Engine) RestoreChatSessionContext(ctx context.Context, run *FSMRun) (tools.ChatSessionContext, error) {
	if run == nil || e.storeProvider == nil {
		return tools.NewChatSessionContext("", "", false, e.knowledgeLimits), nil
	}
	store, err := e.storeProvider.GetStore(ctx, run.ChatID, run.IsDM)
	if err != nil {
		return tools.NewChatSessionContext(run.ChatID, run.UserID, run.IsDM, e.knowledgeLimits), err
	}
	return restoreChatSessionContext(ctx, store, run, e.knowledgeLimits)
}

// Recover scans all active stores for interrupted runs and resumes them.
func (e *Engine) Recover(ctx context.Context) error {
	stores, err := e.storeProvider.ActiveStores(ctx)
	if err != nil {
		return fmt.Errorf("failed to list active stores for recovery: %w", err)
	}

	maxConc := e.maxRecoveryConcurrency
	if maxConc <= 0 {
		maxConc = 4
	}
	sem := make(chan struct{}, maxConc)

	now := time.Now().Unix()
	var allErrs error
	dispatchedCount := 0

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
				e.scheduleWakeTimer(run.ID, delay)
				continue
			}

			if dispatchedCount > 0 {
				var b [1]byte
				_, _ = rand.Read(b[:])
				jitter := time.Duration(10+int(b[0]%25)) * time.Millisecond
				select {
				case <-ctx.Done():
					return errors.Join(allErrs, ctx.Err())
				case <-time.After(jitter):
				}
			}
			dispatchedCount++

			// Interrupted in RUNNING, PENDING, or due WAITING: resume execution
			runID := run.ID
			s := store
			e.spawn(func() {
				runCtx, cancel := context.WithCancel(ctx)
				defer cancel()

				if !e.acquireRun(runID, cancel) {
					return
				}
				defer e.releaseRun(runID)

				// Acquire recovery concurrency semaphore before executing recovery
				select {
				case sem <- struct{}{}:
					defer func() { <-sem }()
				case <-runCtx.Done():
					return
				}

				fresh, err := s.GetRun(runCtx, runID)
				if err != nil {
					slog.Error("failed to get fresh run on recovery", "run_id", runID, "error", err)
					return
				}
				if fresh.Status.IsTerminal() {
					return
				}

				if e.recoveryStalenessCutoff > 0 && fresh.UpdatedAt > 0 {
					staleness := time.Duration(time.Now().Unix()-fresh.UpdatedAt) * time.Second
					if staleness > e.recoveryStalenessCutoff {
						fresh.Status = RunStatusTerminated
						fresh.CurrentState = StateTerminated
						fresh.ErrorText = fmt.Sprintf("stale run expired before recovery (last updated %s ago, threshold %s)", staleness.Round(time.Second), e.recoveryStalenessCutoff)
						if updateErr := s.UpdateRun(runCtx, fresh); updateErr != nil {
							if errors.Is(updateErr, ErrConcurrentUpdate) {
								slog.Warn("skipping stale run termination due to concurrent update", "run_id", runID)
								return
							}
							slog.Error("failed to mark stale run as terminated", "run_id", runID, "error", updateErr)
						}
						return
					}
				}
				if fresh.Status == RunStatusWaiting && fresh.ResumeAt != nil && *fresh.ResumeAt > time.Now().Unix() {
					delay := time.Until(time.Unix(*fresh.ResumeAt, 0))
					e.scheduleWakeTimer(fresh.ID, delay)
					return
				}
				runToRecover := *fresh

				if runToRecover.ChatID == "" {
					slog.Error("refusing to recover run with empty chat_id", "run_id", runToRecover.ID)
					return
				}

				restoredSess, err := restoreChatSessionContext(runCtx, s, &runToRecover, e.knowledgeLimits)
				if err != nil {
					slog.Error("failed to restore chat session context on recovery", "run_id", runToRecover.ID, "error", err)
					runToRecover.Status = RunStatusFailed
					runToRecover.CurrentState = StateFailed
					runToRecover.ErrorText = fmt.Sprintf("failed to restore session context: %v", err)
					_ = s.UpdateRun(runCtx, &runToRecover)
					return
				}
				runCtx = tools.WithChatSession(runCtx, restoredSess)

				runToRecover.Status = RunStatusRunning
				runToRecover.ResumeAt = nil

				desc := e.resolveRunDescriptor(&runToRecover)
				fe, feFound := e.getFrontend(desc.Session.FrontendID)
				if !feFound && desc.Session.FrontendID != LegacyFrontendID {
					slog.Error("unregistered frontend on recovery", "run_id", runToRecover.ID, "frontend_id", desc.Session.FrontendID)
					runToRecover.Status = RunStatusFailed
					runToRecover.CurrentState = StateFailed
					runToRecover.ErrorText = fmt.Sprintf("unregistered frontend %q on recovery", desc.Session.FrontendID)
					_ = s.UpdateRun(runCtx, &runToRecover)
					return
				}

				var bindings agentapi.Bindings
				if feFound {
					var bindErr error
					bindings, bindErr = fe.Bind(runCtx, desc)
					if bindErr != nil {
						slog.Error("failed to bind frontend on recovery", "run_id", runToRecover.ID, "frontend_id", desc.Session.FrontendID, "error", bindErr)
						runToRecover.Status = RunStatusFailed
						runToRecover.CurrentState = StateFailed
						runToRecover.ErrorText = fmt.Sprintf("failed to bind frontend %q: %v", desc.Session.FrontendID, bindErr)
						_ = s.UpdateRun(runCtx, &runToRecover)
						e.deliverCompletion(runCtx, &runToRecover, desc, fe, s)
						return
					}
				}

				if desc.Kind == agentapi.Scheduled || strings.HasPrefix(runToRecover.ID, "sched_") {
					runToRecover.Status = RunStatusFailed
					runToRecover.CurrentState = StateFailed
					runToRecover.ErrorText = "scheduled run interrupted by service restart; ephemeral environment terminated"
					if updateErr := s.UpdateRun(runCtx, &runToRecover); updateErr != nil {
						slog.Error("failed to mark recovered scheduled run as failed", "run_id", runToRecover.ID, "error", updateErr)
					}
					e.deliverCompletion(runCtx, &runToRecover, desc, fe, s)
					return
				}
				if runToRecover.CurrentState == StateWaiting {
					runToRecover.WaitCycles++
					if runToRecover.WaitCycles >= 10 {
						runToRecover.Status = RunStatusFailed
						runToRecover.CurrentState = StateFailed
						runToRecover.ErrorText = fmt.Sprintf("recovered run %s exceeded maximum wait cycles (10)", runToRecover.ID)
						if updateErr := s.UpdateRun(runCtx, &runToRecover); updateErr != nil {
							if errors.Is(updateErr, ErrConcurrentUpdate) {
								slog.Warn("skipping recovered run wait cycles failure due to concurrent update", "run_id", runToRecover.ID)
								return
							}
							slog.Error("failed to persist wait cycles failure for recovered run", "run_id", runToRecover.ID, "error", updateErr)
						}
						e.deliverCompletion(runCtx, &runToRecover, desc, fe, s)
						return
					}
					runToRecover.CurrentState = StateExecuteSteps
				}
				if err := s.UpdateRun(runCtx, &runToRecover); err != nil {
					if errors.Is(err, ErrConcurrentUpdate) {
						slog.Warn("skipping recovered run due to concurrent update", "run_id", runToRecover.ID)
						return
					}
					slog.Error("failed to update recovered run", "run_id", runToRecover.ID, "error", err)
					return
				}

				runner, err := e.getRunner(runToRecover.FSMType)
				if err != nil {
					slog.Error("no runner for recovered run", "fsm_type", runToRecover.FSMType, "error", err)
					return
				}

				var toolset agentapi.Toolset
				if bindings.Tools != nil {
					toolset = bindings.Tools
				} else if provider := e.getToolDefProvider(); provider != nil {
					var inv ToolInvoker
					if e.stepExecutor != nil {
						inv = e.stepExecutor.invoker
					}
					toolset = NewProviderToolset(provider, inv, runToRecover.ChatID, runToRecover.IsDM)
				}

				if err := runner.Execute(runCtx, &runToRecover, s, toolset, desc.Model); err != nil {
					slog.Error("error executing recovered run", "run_id", runToRecover.ID, "error", err)
				}
				if runToRecover.Status.IsTerminal() {
					e.deliverCompletion(runCtx, &runToRecover, desc, fe, s)
				}
			})
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

			runID := run.ID
			s := store
			e.spawn(func() {
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

				if e.recoveryStalenessCutoff > 0 && fresh.UpdatedAt > 0 {
					staleness := time.Duration(time.Now().Unix()-fresh.UpdatedAt) * time.Second
					if staleness > e.recoveryStalenessCutoff {
						fresh.Status = RunStatusTerminated
						fresh.CurrentState = StateTerminated
						fresh.ErrorText = fmt.Sprintf("stale waiting run expired before resume (last updated %s ago, threshold %s)", staleness.Round(time.Second), e.recoveryStalenessCutoff)
						if updateErr := s.UpdateRun(runCtx, fresh); updateErr != nil {
							if errors.Is(updateErr, ErrConcurrentUpdate) {
								slog.Warn("skipping stale run termination due to concurrent update", "run_id", runID)
								return
							}
							slog.Error("failed to mark stale run as terminated", "run_id", runID, "error", updateErr)
						}
						return
					}
				}

				runToResume := *fresh

				if runToResume.ChatID == "" {
					slog.Error("refusing to resume run with empty chat_id", "run_id", runToResume.ID)
					return
				}

				restoredSess, err := restoreChatSessionContext(runCtx, s, &runToResume, e.knowledgeLimits)
				if err != nil {
					slog.Error("failed to restore chat session context on resume", "run_id", runToResume.ID, "error", err)
					runToResume.Status = RunStatusFailed
					runToResume.CurrentState = StateFailed
					runToResume.ErrorText = fmt.Sprintf("failed to restore session context: %v", err)
					_ = s.UpdateRun(runCtx, &runToResume)
					return
				}
				runCtx = tools.WithChatSession(runCtx, restoredSess)

				runToResume.Status = RunStatusRunning
				runToResume.ResumeAt = nil

				desc := e.resolveRunDescriptor(&runToResume)
				fe, feFound := e.getFrontend(desc.Session.FrontendID)
				if !feFound && desc.Session.FrontendID != LegacyFrontendID {
					slog.Error("unregistered frontend on resume", "run_id", runToResume.ID, "frontend_id", desc.Session.FrontendID)
					runToResume.Status = RunStatusFailed
					runToResume.CurrentState = StateFailed
					runToResume.ErrorText = fmt.Sprintf("unregistered frontend %q on resume", desc.Session.FrontendID)
					_ = s.UpdateRun(runCtx, &runToResume)
					return
				}

				var bindings agentapi.Bindings
				if feFound {
					var bindErr error
					bindings, bindErr = fe.Bind(runCtx, desc)
					if bindErr != nil {
						slog.Error("failed to bind frontend on resume", "run_id", runToResume.ID, "frontend_id", desc.Session.FrontendID, "error", bindErr)
						runToResume.Status = RunStatusFailed
						runToResume.CurrentState = StateFailed
						runToResume.ErrorText = fmt.Sprintf("failed to bind frontend %q: %v", desc.Session.FrontendID, bindErr)
						_ = s.UpdateRun(runCtx, &runToResume)
						e.deliverCompletion(runCtx, &runToResume, desc, fe, s)
						return
					}
				}

				if desc.Kind == agentapi.Scheduled || strings.HasPrefix(runToResume.ID, "sched_") {
					runToResume.Status = RunStatusFailed
					runToResume.CurrentState = StateFailed
					runToResume.ErrorText = "scheduled run interrupted; ephemeral environment terminated"
					if updateErr := s.UpdateRun(runCtx, &runToResume); updateErr != nil {
						slog.Error("failed to mark resumed scheduled run as failed", "run_id", runToResume.ID, "error", updateErr)
					}
					e.deliverCompletion(runCtx, &runToResume, desc, fe, s)
					return
				}
				if runToResume.CurrentState == StateWaiting {
					runToResume.WaitCycles++
					if runToResume.WaitCycles >= 10 {
						runToResume.Status = RunStatusFailed
						runToResume.CurrentState = StateFailed
						runToResume.ErrorText = fmt.Sprintf("waiting run %s exceeded maximum wait cycles (10)", runToResume.ID)
						if updateErr := s.UpdateRun(runCtx, &runToResume); updateErr != nil {
							if errors.Is(updateErr, ErrConcurrentUpdate) {
								slog.Warn("skipping resumed run wait cycles failure due to concurrent update", "run_id", runToResume.ID)
								return
							}
							slog.Error("failed to persist wait cycles failure for waiting run", "run_id", runToResume.ID, "error", updateErr)
						}
						e.deliverCompletion(runCtx, &runToResume, desc, fe, s)
						return
					}
					runToResume.CurrentState = StateExecuteSteps
				}
				if err := s.UpdateRun(runCtx, &runToResume); err != nil {
					if errors.Is(err, ErrConcurrentUpdate) {
						slog.Warn("skipping resumed run due to concurrent update", "run_id", runToResume.ID)
						return
					}
					slog.Error("failed to update waiting run to running", "run_id", runToResume.ID, "error", err)
					return
				}

				runner, err := e.getRunner(runToResume.FSMType)
				if err != nil {
					slog.Error("no runner for resumed run", "fsm_type", runToResume.FSMType, "error", err)
					return
				}

				var toolset agentapi.Toolset
				if bindings.Tools != nil {
					toolset = bindings.Tools
				} else if provider := e.getToolDefProvider(); provider != nil {
					var inv ToolInvoker
					if e.stepExecutor != nil {
						inv = e.stepExecutor.invoker
					}
					toolset = NewProviderToolset(provider, inv, runToResume.ChatID, runToResume.IsDM)
				}

				if err := runner.Execute(runCtx, &runToResume, s, toolset, desc.Model); err != nil {
					slog.Error("error executing resumed run", "run_id", runToResume.ID, "error", err)
				}
				if runToResume.Status.IsTerminal() {
					e.deliverCompletion(runCtx, &runToResume, desc, fe, s)
				}
			})
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
func (e *Engine) RunToolLoop(ctx context.Context, req ToolLoopRequest) (result *ToolLoopResult, retErr error) {
	if req.OnTransition != nil {
		ctx = WithTransitionCallback(ctx, req.OnTransition)
	}
	if req.ProgressObserver != nil {
		ctx = WithProgressObserver(ctx, req.ProgressObserver)
	}
	if req.ChatID == "" {
		return nil, errors.New("chat_id is required")
	}
	if len(req.Messages) == 0 {
		return nil, errors.New("messages cannot be empty")
	}
	model := req.Model
	if model == "" {
		model = e.defaultModel
	}
	if model == "" {
		return nil, errors.New("model cannot be empty")
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
	if len(contextJSON) > MaxContextJSONBytes {
		return nil, fmt.Errorf("initial context size (%d bytes) exceeds maximum allowable limit (%d bytes)", len(contextJSON), MaxContextJSONBytes)
	}

	frontendID := req.FrontendID
	if frontendID == "" {
		frontendID = LegacyFrontendID
	}
	scopeID := req.ScopeID
	if scopeID == "" {
		scopeID = req.ChatID
	}
	execKind := req.ExecutionKind
	if execKind == "" {
		if strings.HasPrefix(req.RunID, "sched_") {
			execKind = agentapi.Scheduled
		} else {
			execKind = agentapi.Interactive
		}
	}

	run := &FSMRun{
		ID:               req.RunID,
		ChatID:           req.ChatID,
		UserID:           req.UserID,
		IsDM:             req.IsDM,
		FrontendID:       frontendID,
		ScopeID:          scopeID,
		Model:            model,
		ExecutionKind:    execKind,
		FSMType:          FSMTypeToolLoop,
		Status:           RunStatusRunning,
		CurrentState:     StateInit,
		Iteration:        0,
		MaxIterations:    req.MaxIterations,
		ContextJSON:      contextJSON,
		SourceMessageSeq: req.SourceMessageSeq,
	}

	if err := store.CreateRun(ctx, run); err != nil {
		return nil, fmt.Errorf("failed to create fsm run: %w", err)
	}

	runCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	defer func() {
		if runCtx.Err() != nil && !run.Status.IsTerminal() {
			bg, c := context.WithTimeout(context.Background(), 5*time.Second)
			defer c()
			run.Status = RunStatusTerminated
			run.CurrentState = StateTerminated
			run.ErrorText = "request context cancelled"
			_ = store.UpdateRun(bg, run)
		}
		if obs := GetProgressObserver(ctx); obs != nil {
			var terminalStatus RunStatus
			var runErr error
			if run.Status == RunStatusCompleted && retErr == nil {
				terminalStatus = RunStatusCompleted
			} else if run.Status.IsTerminal() || retErr != nil {
				terminalStatus = RunStatusFailed
				if retErr != nil {
					runErr = retErr
				} else if run.ErrorText != "" {
					runErr = errors.New(run.ErrorText)
				}
			}
			if terminalStatus != "" {
				obs.OnRunFinished(ctx, terminalStatus, runErr)
			}
		}
	}()

	sess, ok := tools.ChatSessionFromContext(ctx)
	if !ok {
		var err error
		sess, err = restoreChatSessionContext(runCtx, store, run, e.knowledgeLimits)
		if err != nil {
			return nil, fmt.Errorf("failed to restore chat session context: %w", err)
		}
	} else {
		if sess.StagedAttachments == nil {
			sess.StagedAttachments = tools.NewStagedAttachmentCollector()
		}
		if sess.KnowledgeBudget == nil {
			sess.KnowledgeBudget = tools.NewKnowledgeBudgetTracker(e.knowledgeLimits)
		}
		if len(sess.GetStagedAttachments()) == 0 && store != nil && run.ID != "" {
			restored, err := restoreChatSessionContext(runCtx, store, run, e.knowledgeLimits)
			if err != nil {
				return nil, fmt.Errorf("failed to restore chat session context: %w", err)
			}
			for _, att := range restored.GetStagedAttachments() {
				_ = sess.StageAttachment(att)
			}
		}
		sess.ChatID = req.ChatID
		sess.UserID = req.UserID
		sess.IsDM = req.IsDM
	}
	sess.FSMRunID = run.ID
	sess.SourceMessageSeq = run.SourceMessageSeq
	runCtx = tools.WithChatSession(runCtx, sess)

	if !e.acquireRun(run.ID, cancel) {
		return nil, fmt.Errorf("run %s is already executing", run.ID)
	}
	defer e.releaseRun(run.ID)

	runner, err := e.getRunner(FSMTypeToolLoop)
	if err != nil {
		return nil, err
	}

	toolset := req.Toolset
	if toolset == nil && len(req.Tools) > 0 {
		var inv ToolInvoker
		if e.stepExecutor != nil {
			inv = e.stepExecutor.invoker
		}
		toolset = NewLegacyToolset(req.Tools, inv)
	}
	// Note: If toolset is nil, strictly 0 tools are provided. No fallback to toolDefProvider.

	execErr := runner.Execute(runCtx, run, store, toolset, model)
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
				execErr = runner.Execute(runCtx, run, store, toolset, model)
				if execErr != nil {
					return nil, execErr
				}
			}
		}
	}

	if run.Status == RunStatusFailed {
		return nil, fmt.Errorf("run %s failed: %s", run.ID, run.ErrorText)
	}

	var completedAttachments []agentapi.Attachment
	var completedActions []agentapi.Action
	var seenAttIDs = make(map[string]bool)

	allSteps, err := store.ListStepsByRun(ctx, run.ID)
	if err == nil {
		for _, s := range allSteps {
			if s.Status == StepStatusCompleted {
				res, _ := DecodeStoredToolResult(s.ResultJSON)
				for _, att := range res.Attachments {
					if !seenAttIDs[att.ID] {
						seenAttIDs[att.ID] = true
						completedAttachments = append(completedAttachments, att)
					}
				}
				completedActions = append(completedActions, res.Actions...)
			}
		}
	}

	var attachments []models.Attachment
	for _, a := range completedAttachments {
		var attType models.AttachmentType
		switch a.Type {
		case agentapi.AttachmentImage:
			attType = models.AttachmentTypeImage
		default:
			attType = models.AttachmentTypeFile
		}
		attachments = append(attachments, models.Attachment{
			Type:     attType,
			Name:     a.Name,
			MimeType: a.MIMEType,
			FileID:   a.ID,
		})
	}
	if len(attachments) == 0 {
		if sess, ok := tools.ChatSessionFromContext(runCtx); ok {
			attachments = sess.GetStagedAttachments()
		}
	}

	return &ToolLoopResult{
		RunID:            run.ID,
		Content:          run.ResultJSON,
		TotalIterations:  run.Iteration,
		Attachments:      attachments,
		AgentAttachments: completedAttachments,
		Actions:          completedActions,
	}, nil
}
