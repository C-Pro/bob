package fsm

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"time"

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
	llmClient    LLMClient
	stepExecutor *StepExecutor
	defaultModel string
}

// NewToolLoopRunner creates a new ToolLoopRunner.
func NewToolLoopRunner(llmClient LLMClient, stepExecutor *StepExecutor, defaultModel string) *ToolLoopRunner {
	if defaultModel == "" {
		defaultModel = "gemini-3.7-flash"
	}
	if stepExecutor == nil {
		stepExecutor = NewStepExecutor(nil)
	}
	return &ToolLoopRunner{
		llmClient:    llmClient,
		stepExecutor: stepExecutor,
		defaultModel: defaultModel,
	}
}

// Execute drives the state machine for the given FSMRun until it reaches a terminal or suspended state.
func (r *ToolLoopRunner) Execute(ctx context.Context, run *FSMRun, store *Store, tools []openai.Tool, model string) error {
	if run == nil {
		return errors.New("run cannot be nil")
	}
	if store == nil {
		return errors.New("store cannot be nil")
	}
	if r.stepExecutor == nil {
		r.stepExecutor = NewStepExecutor(nil)
	}
	if model == "" {
		model = r.defaultModel
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
			if err := r.handleLLMRequest(ctx, run, store, tools, model); err != nil {
				return err
			}

		case StatePrepareSteps:
			if err := r.handlePrepareSteps(ctx, run, store); err != nil {
				return err
			}

		case StateExecuteSteps:
			waiting, err := r.handleExecuteSteps(ctx, run, store)
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

func (r *ToolLoopRunner) handleLLMRequest(ctx context.Context, run *FSMRun, store *Store, tools []openai.Tool, model string) error {
	messages, err := DecodeMessages(run.ContextJSON)
	if err != nil {
		run.Status = RunStatusFailed
		run.ErrorText = fmt.Sprintf("failed to decode messages: %v", err)
		_ = store.UpdateRun(ctx, run)
		return err
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

	// If no tool calls were requested, return the assistant text directly
	if len(assistantMsg.ToolCalls) == 0 {
		messages = append(messages, assistantMsg)
		encoded, err := EncodeMessages(messages)
		if err != nil {
			return err
		}
		if len(encoded) > MaxContextJSONBytes {
			return r.failRunWithContextLimit(ctx, run, store, len(encoded))
		}
		run.ContextJSON = encoded
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
	messages = append(messages, assistantMsg)
	encoded, err := EncodeMessages(messages)
	if err != nil {
		return err
	}
	if len(encoded) > MaxContextJSONBytes {
		return r.failRunWithContextLimit(ctx, run, store, len(encoded))
	}
	run.ContextJSON = encoded
	run.CurrentState = StatePrepareSteps
	return store.UpdateRun(ctx, run)
}

func (r *ToolLoopRunner) handlePrepareSteps(ctx context.Context, run *FSMRun, store *Store) error {
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
		return store.UpdateRun(ctx, run)
	}

	mode := ClassifyExecutionMode(lastMsg.ToolCalls)
	steps := make([]FSMStep, len(lastMsg.ToolCalls))
	for i, tc := range lastMsg.ToolCalls {
		steps[i] = NewStepFromToolCall(run.ID, run.Iteration, i, tc, mode)
	}

	if err := store.CreateSteps(ctx, steps); err != nil {
		return fmt.Errorf("failed to create fsm steps: %w", err)
	}

	run.CurrentState = StateExecuteSteps
	return store.UpdateRun(ctx, run)
}

func (r *ToolLoopRunner) handleExecuteSteps(ctx context.Context, run *FSMRun, store *Store) (bool, error) {
	steps, err := store.ListStepsByIteration(ctx, run.ID, run.Iteration)
	if err != nil {
		return false, err
	}

	if len(steps) == 0 {
		run.CurrentState = StateLLMRequest
		return false, store.UpdateRun(ctx, run)
	}

	stepPtrs := make([]*FSMStep, len(steps))
	for i := range steps {
		stepPtrs[i] = &steps[i]
	}

	// Use shared executor directly, passing per-chat store to Execute
	executor := r.stepExecutor
	if executor == nil {
		executor = NewStepExecutor(nil)
	}

	execErr := executor.Execute(ctx, store, stepPtrs)
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

	// Collect tool results and append them as tool messages
	messages, err := DecodeMessages(run.ContextJSON)
	if err != nil {
		return false, err
	}

	for _, s := range stepPtrs {
		slog.Info("fsm step finished", "tool", s.ToolName, "status", s.Status, "run_id", run.ID)
		content := s.ResultJSON
		if content == "" && s.ErrorText != "" {
			content = fmt.Sprintf(`{"error": %q}`, s.ErrorText)
		}
		if len(content) > MaxToolResultSizeInContext {
			content = content[:MaxToolResultSizeInContext] + "\n[tool output truncated in context]"
		}
		messages = append(messages, openai.ChatCompletionMessage{
			Role:       openai.ChatMessageRoleTool,
			Content:    content,
			ToolCallID: s.ToolCallID,
		})
	}

	encoded, err := EncodeMessages(messages)
	if err != nil {
		return false, err
	}
	if len(encoded) > MaxContextJSONBytes {
		return false, r.failRunWithContextLimit(ctx, run, store, len(encoded))
	}

	run.ContextJSON = encoded
	if run.Iteration >= run.MaxIterations {
		run.CurrentState = StateSynthesis
	} else {
		run.CurrentState = StateLLMRequest
	}
	return false, store.UpdateRun(ctx, run)
}

func (r *ToolLoopRunner) handleSynthesis(ctx context.Context, run *FSMRun, store *Store, model string) error {
	messages, err := DecodeMessages(run.ContextJSON)
	if err != nil {
		return err
	}

	messages = append(messages, openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleUser,
		Content: SynthesisPrompt,
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
	messages = append(messages, finalMsg)
	encoded, err := EncodeMessages(messages)
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

func (r *ToolLoopRunner) failRunWithContextLimit(ctx context.Context, run *FSMRun, store *Store, size int) error {
	run.Status = RunStatusFailed
	run.CurrentState = StateFailed
	run.ErrorText = fmt.Sprintf("context size (%d bytes) exceeds maximum allowable limit (%d bytes)", size, MaxContextJSONBytes)
	if updateErr := store.UpdateRun(ctx, run); updateErr != nil {
		return fmt.Errorf("context size (%d bytes) exceeds maximum allowable limit (%d bytes); failed to persist failure state: %w", size, MaxContextJSONBytes, updateErr)
	}
	return fmt.Errorf("context size (%d bytes) exceeds maximum allowable limit (%d bytes)", size, MaxContextJSONBytes)
}
