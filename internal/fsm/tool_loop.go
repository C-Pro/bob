package fsm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"sync"
	"time"

	"bob/internal/agentapi"
	"bob/internal/tools"
	"bob/internal/trajectory"

	openai "github.com/sashabaranov/go-openai"
)

// SynthesisPrompt is appended when the maximum tool execution iterations are exhausted.
const SynthesisPrompt = "You have reached the tool execution limit. Please synthesize and provide the best possible response based on all information gathered so far, including original markdown links to sources found in the search results, without calling any more tools. If any requested actions, scripts, or files could not be completed or executed due to the tool limit, state clearly what was accomplished and what remains to be run; do not claim files were created if they were not."

const (
	// MaxToolResultSizeInContext is the maximum bytes of tool output stored directly inside conversation context.
	MaxToolResultSizeInContext = 16 * 1024
	// MaxContextJSONBytes is the maximum total size in bytes for context_json in a single run.
	MaxContextJSONBytes = 1024 * 1024
)

// ToolLoopRunner executes the simple tool loop finite state machine.
type ToolLoopRunner struct {
	mu                sync.RWMutex
	llmClient         LLMClient
	stepExecutor      *StepExecutor
	defaultModel      string
	toolDefProvider   ToolDefinitionProvider
	compactionCfg     CompactionConfig
	compactionCfgSet  bool
	compactionDecider CompactionDecider
	loopGuardCfg      LoopGuardConfig
}

// NewToolLoopRunner creates a new ToolLoopRunner.
func NewToolLoopRunner(llmClient LLMClient, stepExecutor *StepExecutor, defaultModel string) *ToolLoopRunner {
	if stepExecutor == nil {
		stepExecutor = NewStepExecutor(nil)
	}
	return &ToolLoopRunner{
		llmClient:     llmClient,
		stepExecutor:  stepExecutor,
		defaultModel:  defaultModel,
		compactionCfg: DefaultCompactionConfig(),
		loopGuardCfg:  DefaultLoopGuardConfig(),
	}
}

// SetLoopGuardConfig configures loop guard settings for the runner.
func (r *ToolLoopRunner) SetLoopGuardConfig(cfg LoopGuardConfig) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.loopGuardCfg = cfg
}

// SetCompactionConfig configures adaptive task trajectory compaction for the runner.
func (r *ToolLoopRunner) SetCompactionConfig(cfg CompactionConfig) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.compactionCfg = cfg
	r.compactionCfgSet = true
}

// SetCompactionDecider configures a custom CompactionDecider for evaluating the SelfCompact rubric.
func (r *ToolLoopRunner) SetCompactionDecider(d CompactionDecider) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.compactionDecider = d
}

// SetToolDefinitionProvider configures dynamic tool definition provider for the runner.
func (r *ToolLoopRunner) SetToolDefinitionProvider(p ToolDefinitionProvider) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.toolDefProvider = p
}

// HasToolDefinitionProvider returns true if a dynamic tool definition provider is set.
func (r *ToolLoopRunner) HasToolDefinitionProvider() bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.toolDefProvider != nil
}

func (r *ToolLoopRunner) resolveBinding(ctx context.Context, run *FSMRun, store *Store, toolset agentapi.Toolset) (*ToolBinding, error) {
	if toolset != nil {
		r.mu.RLock()
		cfg := r.compactionCfg
		cfgSet := r.compactionCfgSet
		r.mu.RUnlock()

		if store != nil && run != nil && cfg.Enabled {
			includeDef := cfgSet
			if !includeDef {
				if latest, err := store.GetLatestCompaction(ctx, run.ID); err == nil && latest != nil {
					includeDef = true
				}
			}
			toolset = &taskHistoryBoundToolset{
				base:       toolset,
				store:      store,
				run:        run,
				includeDef: includeDef,
			}
		}
		return NewToolBinding(ctx, toolset)
	}
	return NewToolBinding(ctx, nil)
}

type taskHistoryBoundToolset struct {
	base       agentapi.Toolset
	store      *Store
	run        *FSMRun
	includeDef bool
}

func (t *taskHistoryBoundToolset) Definitions(ctx context.Context) ([]agentapi.ToolDefinition, error) {
	if t.base == nil {
		return nil, nil
	}
	defs, err := t.base.Definitions(ctx)
	if err != nil || len(defs) == 0 {
		return defs, err
	}
	for _, d := range defs {
		if d.Schema.Function != nil && d.Schema.Function.Name == tools.RecallTaskHistoryToolName {
			return defs, nil
		}
	}
	if !t.includeDef {
		return defs, nil
	}
	out := make([]agentapi.ToolDefinition, 0, len(defs)+1)
	out = append(out, defs...)
	out = append(out, agentapi.ToolDefinition{
		Schema:   tools.RecallTaskHistoryToolDefinition(),
		ReadOnly: true,
	})
	return out, nil
}

func (t *taskHistoryBoundToolset) Execute(ctx context.Context, call agentapi.ToolCall) (agentapi.ToolResult, error) {
	if call.Name == tools.RecallTaskHistoryToolName {
		if t.store == nil || t.run == nil {
			return agentapi.ToolResult{}, errors.New("task history store is unavailable")
		}
		chatID := t.run.ChatID
		userID := t.run.UserID
		isDM := t.run.IsDM
		if sess, ok := tools.ChatSessionFromContext(ctx); ok {
			if sess.ChatID != "" && sess.ChatID != chatID {
				return agentapi.ToolResult{}, fmt.Errorf("chat session mismatch for recall_task_history: %q vs %q", sess.ChatID, chatID)
			}
			if sess.FSMRunID != "" && sess.FSMRunID != t.run.ID {
				return agentapi.ToolResult{}, fmt.Errorf("fsm run mismatch for recall_task_history: %q vs %q", sess.FSMRunID, t.run.ID)
			}
			if sess.IsDM != isDM {
				return agentapi.ToolResult{}, fmt.Errorf("channel scope mismatch for recall_task_history: %t vs %t", sess.IsDM, isDM)
			}
			if sess.UserID != userID {
				return agentapi.ToolResult{}, fmt.Errorf("user session mismatch for recall_task_history: %q vs %q", sess.UserID, userID)
			}
			userID = sess.UserID
			isDM = sess.IsDM
		}
		content, err := tools.ExecuteRecallTaskHistoryWithStore(ctx, t.store, chatID, userID, isDM, t.run.ID, call.Arguments)
		return agentapi.ToolResult{Content: content}, err
	}
	if t.base == nil {
		return agentapi.ToolResult{}, errors.New("base toolset is not configured")
	}
	return t.base.Execute(ctx, call)
}

// Execute drives the state machine for the given FSMRun until it reaches a terminal or suspended state.
func (r *ToolLoopRunner) Execute(ctx context.Context, run *FSMRun, store *Store, toolset agentapi.Toolset, model string) error {
	if run == nil {
		return errors.New("run cannot be nil")
	}
	if store == nil {
		return errors.New("store cannot be nil")
	}
	if model == "" {
		model = r.defaultModel
	}
	if model == "" {
		return errors.New("model cannot be empty")
	}
	defer func() {
		if cb := GetTransitionCallback(ctx); cb != nil {
			cb(run.CurrentState, run)
		}
	}()

	for !run.Status.IsTerminal() {
		if ctx.Err() != nil {
			return ctx.Err()
		}

		if cb := GetTransitionCallback(ctx); cb != nil {
			cb(run.CurrentState, run)
		}

		switch run.CurrentState {
		case StateInit:
			if err := r.handleInit(ctx, run, store); err != nil {
				return err
			}

		case StateLLMRequest:
			if err := r.handleLLMRequest(ctx, run, store, toolset, model); err != nil {
				return err
			}

		case StatePrepareSteps:
			if err := r.handlePrepareSteps(ctx, run, store, toolset); err != nil {
				return err
			}

		case StateExecuteSteps:
			waiting, err := r.handleExecuteSteps(ctx, run, store, toolset, model)
			if err != nil {
				return err
			}
			if waiting {
				return nil
			}

		case StateWaiting:
			now := time.Now().Unix()
			if run.ResumeAt != nil && *run.ResumeAt > now {
				return nil
			}
			run.WaitCycles++
			if run.WaitCycles >= 10 {
				run.Status = RunStatusFailed
				run.CurrentState = StateFailed
				run.ErrorText = "run exceeded maximum wait cycles (10)"
				if err := store.UpdateRun(ctx, run); err != nil {
					return errors.Join(fmt.Errorf("run %s exceeded maximum wait cycles (10)", run.ID), err)
				}
				return fmt.Errorf("run %s exceeded maximum wait cycles (10)", run.ID)
			}
			run.Status = RunStatusRunning
			run.ResumeAt = nil
			run.CurrentState = StateExecuteSteps
			if err := store.UpdateRun(ctx, run); err != nil {
				return err
			}

		case StateSynthesis:
			if err := r.handleSynthesis(ctx, run, store, model); err != nil {
				return err
			}

		case StateCompleted, StateFailed, StateTerminated:
			return nil

		default:
			run.Status = RunStatusFailed
			run.ErrorText = fmt.Sprintf("unknown state: %s", run.CurrentState)
			if err := store.UpdateRun(ctx, run); err != nil {
				return errors.Join(fmt.Errorf("unknown fsm state: %s", run.CurrentState), err)
			}
			return fmt.Errorf("unknown fsm state: %s", run.CurrentState)
		}
	}

	return nil
}

func (r *ToolLoopRunner) handleInit(ctx context.Context, run *FSMRun, store *Store) error {
	if run.MaxIterations <= 0 {
		run.MaxIterations = 10
	}
	run.Status = RunStatusRunning
	run.CurrentState = StateLLMRequest
	return store.UpdateRun(ctx, run)
}

func (r *ToolLoopRunner) handleLLMRequest(ctx context.Context, run *FSMRun, store *Store, toolset agentapi.Toolset, model string) error {
	binding, err := r.resolveBinding(ctx, run, store, toolset)
	if err != nil {
		run.Status = RunStatusFailed
		run.CurrentState = StateFailed
		run.ErrorText = fmt.Sprintf("failed to get tool definitions: %v", err)
		_ = store.UpdateRun(ctx, run)
		return err
	}

	messages, err := r.buildChatMessages(ctx, run, store)
	if err != nil {
		run.Status = RunStatusFailed
		run.ErrorText = fmt.Sprintf("failed to decode messages: %v", err)
		_ = store.UpdateRun(ctx, run)
		return err
	}

	var tools []openai.Tool
	if binding != nil && len(binding.Schemas()) > 0 {
		tools = binding.Schemas()
	}

	req := openai.ChatCompletionRequest{
		Model:    model,
		Messages: messages,
		Tools:    tools,
	}

	resp, err := r.llmClient.CreateChatCompletion(ctx, req)
	if err != nil {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		run.Status = RunStatusFailed
		run.CurrentState = StateFailed
		run.ErrorText = fmt.Sprintf("llm request failed: %v", err)
		if updateErr := store.UpdateRun(ctx, run); updateErr != nil {
			return errors.Join(err, updateErr)
		}
		return err
	}

	if len(resp.Choices) == 0 {
		run.Status = RunStatusFailed
		run.ErrorText = "llm returned empty choices"
		if updateErr := store.UpdateRun(ctx, run); updateErr != nil {
			return updateErr
		}
		return errors.New("llm returned empty choices")
	}

	assistantMsg := resp.Choices[0].Message

	if ev, evErr := trajectory.MakeLLMResponseEvent(run.ID, run.ID, 1, run.Iteration, "loop", assistantMsg.Content, assistantMsg.ReasoningContent, assistantMsg.ToolCalls); evErr == nil {
		_ = store.RecordEvent(ctx, ev)
	}

	// If model requested tool calls, enforce strict tool authority
	if len(assistantMsg.ToolCalls) > 0 {
		if binding == nil || len(binding.Schemas()) == 0 {
			run.Status = RunStatusFailed
			run.CurrentState = StateFailed
			run.ErrorText = fmt.Sprintf("unauthorized tool call: model called %q when no tools were authorized", assistantMsg.ToolCalls[0].Function.Name)
			if updateErr := store.UpdateRun(ctx, run); updateErr != nil {
				return errors.Join(errors.New(run.ErrorText), updateErr)
			}
			return errors.New(run.ErrorText)
		}
		for _, tc := range assistantMsg.ToolCalls {
			if !binding.HasTool(tc.Function.Name) {
				run.Status = RunStatusFailed
				run.CurrentState = StateFailed
				run.ErrorText = fmt.Sprintf("unauthorized tool call: %q is not in the allowed toolset", tc.Function.Name)
				if updateErr := store.UpdateRun(ctx, run); updateErr != nil {
					return errors.Join(errors.New(run.ErrorText), updateErr)
				}
				return errors.New(run.ErrorText)
			}
		}
	}

	baseMessages, err := DecodeMessages(run.ContextJSON)
	if err != nil {
		run.Status = RunStatusFailed
		run.ErrorText = fmt.Sprintf("failed to decode context messages: %v", err)
		_ = store.UpdateRun(ctx, run)
		return err
	}
	baseMessages = append(baseMessages, assistantMsg)
	encoded, err := EncodeMessages(baseMessages)
	if err != nil {
		return err
	}
	if len(encoded) > MaxContextJSONBytes {
		return r.failRunWithContextLimit(ctx, run, store, len(encoded))
	}
	run.ContextJSON = encoded

	// If no tool calls were requested, return the assistant text directly
	if len(assistantMsg.ToolCalls) == 0 {
		run.ResultJSON = assistantMsg.Content
		run.Status = RunStatusCompleted
		run.CurrentState = StateCompleted
		return store.UpdateRun(ctx, run)
	}

	// Tool calls requested: evaluate iteration limit
	if run.Iteration >= run.MaxIterations {
		run.CurrentState = StateSynthesis
		return store.UpdateRun(ctx, run)
	}

	run.Iteration++
	run.CurrentState = StatePrepareSteps
	return store.UpdateRun(ctx, run)
}

func (r *ToolLoopRunner) handlePrepareSteps(ctx context.Context, run *FSMRun, store *Store, toolset agentapi.Toolset) error {
	binding, err := r.resolveBinding(ctx, run, store, toolset)
	if err != nil {
		run.Status = RunStatusFailed
		run.CurrentState = StateFailed
		run.ErrorText = fmt.Sprintf("failed to get tool definitions: %v", err)
		_ = store.UpdateRun(ctx, run)
		return err
	}

	messages, err := DecodeMessages(run.ContextJSON)
	if err != nil {
		return err
	}
	if len(messages) == 0 {
		return errors.New("cannot prepare steps: no messages in context")
	}

	lastMsg := messages[len(messages)-1]
	if len(lastMsg.ToolCalls) == 0 {
		run.CurrentState = StateLLMRequest
		return store.UpdateRun(ctx, run)
	}

	// If steps for this (run.ID, run.Iteration) already exist (e.g. recovered after CreateSteps crashed before UpdateRun),
	// skip re-creating and advance directly to StateExecuteSteps.
	existingSteps, err := store.ListStepsByIteration(ctx, run.ID, run.Iteration)
	if err != nil {
		return fmt.Errorf("failed to check existing fsm steps: %w", err)
	}
	if len(existingSteps) > 0 {
		run.CurrentState = StateExecuteSteps
		if obs := GetProgressObserver(ctx); obs != nil {
			obs.OnStepsPrepared(ctx, existingSteps)
		}
		return store.UpdateRun(ctx, run)
	}

	for _, tc := range lastMsg.ToolCalls {
		if binding == nil || !binding.HasTool(tc.Function.Name) {
			run.Status = RunStatusFailed
			run.CurrentState = StateFailed
			run.ErrorText = fmt.Sprintf("unauthorized tool call: %q", tc.Function.Name)
			_ = store.UpdateRun(ctx, run)
			return errors.New(run.ErrorText)
		}
	}

	r.mu.RLock()
	lgCfg := r.loopGuardCfg
	r.mu.RUnlock()

	historySteps, err := store.ListStepsByRun(ctx, run.ID)
	if err != nil {
		return fmt.Errorf("failed to list steps for loop guard: %w", err)
	}

	verdict := DetectLoop(historySteps, lastMsg.ToolCalls, lgCfg)
	if verdict.Action == LoopActionForceSynthesis {
		slog.Warn("loop guard detected stuck loop; forcing synthesis", "run_id", run.ID, "reason", verdict.Reason)
		run.CurrentState = StateSynthesis
		run.ErrorText = fmt.Sprintf("stuck loop detected: %s", verdict.Reason)
		return store.UpdateRun(ctx, run)
	}

	steps := make([]FSMStep, len(lastMsg.ToolCalls))
	stepPtrs := make([]*FSMStep, len(lastMsg.ToolCalls))
	for i, tc := range lastMsg.ToolCalls {
		steps[i] = NewStepFromToolCall(run.ID, run.Iteration, i, tc, ExecutionModeSequential)
		if verdict.Action == LoopActionIntercept {
			if notice, ok := verdict.NoticeByCall[i]; ok {
				now := time.Now().Unix()
				resEnv, _ := EncodeStoredToolResult(agentapi.ToolResult{Content: notice})
				steps[i].Status = StepStatusCompleted
				steps[i].ResultJSON = resEnv
				steps[i].ErrorText = LoopGuardBlockedErrorText
				steps[i].StartedAt = &now
				steps[i].CompletedAt = &now
				zero := 0
				trajStep := trajectory.Step{
					Index:       steps[i].StepIndex,
					Iteration:   steps[i].Iteration,
					Attempt:     1,
					Action:      steps[i].ToolName,
					ActionArgs:  json.RawMessage(steps[i].ArgsJSON),
					Observation: notice,
					ExitCode:    &zero,
				}
				if ev, evErr := trajectory.MakeToolAttemptEvent(run.ID, run.ID, 1, run.Iteration, "loop", steps[i].ID, 1, trajStep); evErr == nil {
					_ = store.RecordEvent(ctx, ev)
				}
			}
		}
		stepPtrs[i] = &steps[i]
	}

	mode := ExecutionModeSequential
	if binding != nil {
		mode = binding.ClassifyStepExecutionMode(stepPtrs)
	}
	for i := range steps {
		steps[i].ExecutionMode = mode
	}

	if err := store.CreateSteps(ctx, steps); err != nil {
		return fmt.Errorf("failed to create fsm steps: %w", err)
	}

	if obs := GetProgressObserver(ctx); obs != nil {
		obs.OnStepsPrepared(ctx, steps)
	}

	run.CurrentState = StateExecuteSteps
	return store.UpdateRun(ctx, run)
}

func (r *ToolLoopRunner) handleExecuteSteps(ctx context.Context, run *FSMRun, store *Store, toolset agentapi.Toolset, model string) (bool, error) {
	steps, err := store.ListStepsByIteration(ctx, run.ID, run.Iteration)
	if err != nil {
		return false, err
	}

	if len(steps) == 0 {
		run.CurrentState = StateLLMRequest
		return false, store.UpdateRun(ctx, run)
	}

	binding, err := r.resolveBinding(ctx, run, store, toolset)
	if err != nil {
		run.Status = RunStatusFailed
		run.CurrentState = StateFailed
		run.ErrorText = fmt.Sprintf("failed to get tool definitions: %v", err)
		_ = store.UpdateRun(ctx, run)
		return false, err
	}

	stepPtrs := make([]*FSMStep, len(steps))
	for i := range steps {
		stepPtrs[i] = &steps[i]
	}

	for _, s := range stepPtrs {
		if !s.Status.IsTerminal() && (binding == nil || !binding.HasTool(s.ToolName)) {
			run.Status = RunStatusFailed
			run.CurrentState = StateFailed
			run.ErrorText = fmt.Sprintf("step %s uses unauthorized tool %q", s.ID, s.ToolName)
			_ = store.UpdateRun(ctx, run)
			return false, errors.New(run.ErrorText)
		}
	}

	executor := r.stepExecutor
	if executor == nil {
		err := errors.New("step executor is not configured")
		run.Status = RunStatusFailed
		run.CurrentState = StateFailed
		run.ErrorText = err.Error()
		if updateErr := store.UpdateRun(ctx, run); updateErr != nil {
			return false, errors.Join(err, updateErr)
		}
		return false, err
	}

	execErr := executor.Execute(ctx, store, stepPtrs, binding)
	if ctx.Err() != nil {
		return false, ctx.Err()
	}

	// If a persistence error occurred during step execution, fail the run immediately.
	var storeErr *StoreError
	if errors.As(execErr, &storeErr) {
		run.Status = RunStatusFailed
		run.CurrentState = StateFailed
		run.ErrorText = fmt.Sprintf("persistence failure during step execution: %v", storeErr)
		if updateErr := store.UpdateRun(ctx, run); updateErr != nil {
			return false, errors.Join(storeErr, updateErr)
		}
		return false, fmt.Errorf("step persistence failed: %w", storeErr)
	}

	// Any step left in a non-terminal status (such as PENDING or RUNNING) after executor.Execute
	// is an invariant violation. We must fail the run rather than reinterpreting it as WAITING.
	for _, s := range stepPtrs {
		if !s.Status.IsTerminal() {
			run.Status = RunStatusFailed
			run.CurrentState = StateFailed
			if execErr != nil {
				run.ErrorText = fmt.Sprintf("step %s unexpectedly left in %s status: %v", s.ID, s.Status, execErr)
			} else {
				run.ErrorText = fmt.Sprintf("step %s unexpectedly left in %s status", s.ID, s.Status)
			}
			if updateErr := store.UpdateRun(ctx, run); updateErr != nil {
				if execErr != nil {
					return false, errors.Join(fmt.Errorf("step %s unexpectedly left in %s: %w", s.ID, s.Status, execErr), updateErr)
				}
				return false, errors.Join(fmt.Errorf("step %s unexpectedly left in %s", s.ID, s.Status), updateErr)
			}
			if execErr != nil {
				return false, fmt.Errorf("step execution failed: %w", execErr)
			}
			return false, fmt.Errorf("step %s unexpectedly left in %s status", s.ID, s.Status)
		}
	}

	if execErr != nil {
		slog.Warn("fsm step execution completed with step failures", "run_id", run.ID, "error", execErr)
	}

	for _, s := range stepPtrs {
		slog.Info("fsm step finished", "tool", s.ToolName, "status", s.Status, "run_id", run.ID)
	}

	if err := r.maybeCompact(ctx, run, store, model); err != nil {
		run.Status = RunStatusFailed
		run.CurrentState = StateFailed
		run.ErrorText = fmt.Sprintf("compaction persistence failure: %v", err)
		if updateErr := store.UpdateRun(ctx, run); updateErr != nil {
			return false, errors.Join(err, updateErr)
		}
		return false, fmt.Errorf("compaction persistence failed: %w", err)
	}

	if run.Iteration >= run.MaxIterations {
		run.CurrentState = StateSynthesis
	} else {
		run.CurrentState = StateLLMRequest
	}
	return false, store.UpdateRun(ctx, run)
}

func (r *ToolLoopRunner) handleSynthesis(ctx context.Context, run *FSMRun, store *Store, model string) error {
	messages, err := r.buildChatMessages(ctx, run, store)
	if err != nil {
		return err
	}

	synthPrompt := SynthesisPrompt
	if strings.Contains(run.ErrorText, "stuck loop detected") {
		synthPrompt = "A repetitive tool execution loop was detected. " + SynthesisPrompt
	}

	messages = append(messages, openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleUser,
		Content: synthPrompt,
	})

	req := openai.ChatCompletionRequest{
		Model:    model,
		Messages: messages,
		Tools:    nil,
	}

	resp, err := r.llmClient.CreateChatCompletion(ctx, req)
	if err != nil {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		run.Status = RunStatusFailed
		run.CurrentState = StateFailed
		run.ErrorText = fmt.Sprintf("failed to generate final synthesis response: %v", err)
		if updateErr := store.UpdateRun(ctx, run); updateErr != nil {
			return errors.Join(err, updateErr)
		}
		return fmt.Errorf("failed to generate final synthesis response: %w", err)
	}

	if len(resp.Choices) == 0 {
		run.Status = RunStatusFailed
		run.CurrentState = StateFailed
		run.ErrorText = "llm returned empty choices during synthesis"
		if updateErr := store.UpdateRun(ctx, run); updateErr != nil {
			return updateErr
		}
		return errors.New("llm returned empty choices during synthesis")
	}

	finalMsg := resp.Choices[0].Message

	if ev, evErr := trajectory.MakeLLMResponseEvent(run.ID, run.ID, 1, run.Iteration, "synthesis", finalMsg.Content, finalMsg.ReasoningContent, nil); evErr == nil {
		_ = store.RecordEvent(ctx, ev)
	}

	baseMessages, err := DecodeMessages(run.ContextJSON)
	if err != nil {
		return err
	}
	baseMessages = append(baseMessages, openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleUser,
		Content: synthPrompt,
	}, finalMsg)

	encoded, err := EncodeMessages(baseMessages)
	if err != nil {
		return err
	}
	if len(encoded) > MaxContextJSONBytes {
		return r.failRunWithContextLimit(ctx, run, store, len(encoded))
	}

	run.ContextJSON = encoded
	run.ResultJSON = finalMsg.Content
	run.Status = RunStatusCompleted
	run.CurrentState = StateCompleted
	return store.UpdateRun(ctx, run)
}

func formatToolStepResult(s FSMStep) string {
	decoded, _ := DecodeStoredToolResult(s.ResultJSON)
	content := decoded.Content
	switch {
	case content != "":
		// use as-is
	case s.ErrorText != "":
		content = fmt.Sprintf(`{"error": %q}`, s.ErrorText)
	default:
		content = fmt.Sprintf(`{"status": %q, "result": ""}`, s.Status)
	}
	if len(content) > MaxToolResultSizeInContext {
		const suffix = "\n[tool output truncated in context]"
		content = truncateUTF8Bytes(content, MaxToolResultSizeInContext-len(suffix)) + suffix
	}
	return content
}

func (r *ToolLoopRunner) buildChatMessages(ctx context.Context, run *FSMRun, store *Store) ([]openai.ChatCompletionMessage, error) {
	messages, err := DecodeMessages(run.ContextJSON)
	if err != nil {
		return nil, err
	}
	if store == nil {
		return messages, nil
	}

	steps, err := store.ListStepsByRun(ctx, run.ID)
	if err != nil {
		return nil, fmt.Errorf("failed to list steps for run %s: %w", run.ID, err)
	}
	latestCompaction, err := store.GetLatestCompaction(ctx, run.ID)
	if err != nil {
		return nil, fmt.Errorf("failed to get latest compaction for run %s: %w", run.ID, err)
	}
	return reconstructChatMessages(messages, steps, latestCompaction), nil
}

func reconstructChatMessages(messages []openai.ChatCompletionMessage, steps []FSMStep, latestCompaction *CompactedTaskState) []openai.ChatCompletionMessage {
	stepsByCallID := make(map[string]FSMStep, len(steps))
	for _, s := range steps {
		if s.ToolCallID != "" {
			stepsByCallID[s.ToolCallID] = s
		}
	}

	toolResponses := make(map[string]openai.ChatCompletionMessage)
	for _, m := range messages {
		if m.Role == openai.ChatMessageRoleTool && m.ToolCallID != "" {
			if _, exists := toolResponses[m.ToolCallID]; !exists {
				toolResponses[m.ToolCallID] = m
			}
		}
	}

	compactedUpTo := 0
	if latestCompaction != nil && latestCompaction.CompactedUpToIter > 0 {
		compactedUpTo = latestCompaction.CompactedUpToIter
	}

	reconstructed := make([]openai.ChatCompletionMessage, 0, len(messages)+len(steps)+1)
	compactedInserted := false

	for _, m := range messages {
		// Tool responses are emitted only with their parent assistant message.
		if m.Role == openai.ChatMessageRoleTool {
			continue
		}
		if m.Role != openai.ChatMessageRoleAssistant || len(m.ToolCalls) == 0 {
			reconstructed = append(reconstructed, m)
			continue
		}

		belongsToRun := false
		allCompacted := compactedUpTo > 0
		for _, tc := range m.ToolCalls {
			s, ok := stepsByCallID[tc.ID]
			if ok && s.Iteration > 0 {
				belongsToRun = true
			} else {
				// Unknown iterations keep the whole group, including prior-turn exchanges.
				allCompacted = false
			}
			if s.Iteration > compactedUpTo {
				allCompacted = false
			}
		}

		if belongsToRun && compactedUpTo > 0 && !compactedInserted {
			reconstructed = append(reconstructed, RenderCompactedStateMessage(latestCompaction))
			compactedInserted = true
		}
		if allCompacted {
			continue
		}

		reconstructed = append(reconstructed, m)
		for _, tc := range m.ToolCalls {
			response, ok := toolResponses[tc.ID]
			if !ok {
				response = openai.ChatCompletionMessage{
					Role:       openai.ChatMessageRoleTool,
					ToolCallID: tc.ID,
					Content:    `{"error":"Tool response unavailable in recovered history; execution outcome is unknown."}`,
				}
				if s, exists := stepsByCallID[tc.ID]; exists {
					response.Content = formatToolStepResult(s)
				}
			}
			reconstructed = append(reconstructed, response)
			delete(toolResponses, tc.ID)
		}
	}

	if compactedUpTo > 0 && !compactedInserted {
		reconstructed = append(reconstructed, RenderCompactedStateMessage(latestCompaction))
	}

	return reconstructed
}

func (r *ToolLoopRunner) failRunWithContextLimit(ctx context.Context, run *FSMRun, store *Store, size int) error {
	run.Status = RunStatusFailed
	run.CurrentState = StateFailed
	run.ErrorText = fmt.Sprintf("context size (%d bytes) exceeds maximum allowable limit (%d bytes)", size, MaxContextJSONBytes)
	if updateErr := store.UpdateRun(ctx, run); updateErr != nil {
		return fmt.Errorf("context size (%d bytes) exceeds maximum allowable limit (%d bytes); failed to persist failure state: %w", size, MaxContextJSONBytes, updateErr)
	}
	return fmt.Errorf("context size (%d bytes) exceeds maximum allowable limit (%d bytes)", size, MaxContextJSONBytes)
}
