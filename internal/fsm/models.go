package fsm

import (
	"encoding/json"
	"fmt"

	openai "github.com/sashabaranov/go-openai"
)

// RunStatus defines the high-level lifecycle status of an FSM run.
type RunStatus string

const (
	RunStatusPending    RunStatus = "PENDING"
	RunStatusRunning    RunStatus = "RUNNING"
	RunStatusWaiting    RunStatus = "WAITING"
	RunStatusCompleted  RunStatus = "COMPLETED"
	RunStatusFailed     RunStatus = "FAILED"
	RunStatusTerminated RunStatus = "TERMINATED"
)

// IsTerminal reports whether the run status represents a completed/failed end state.
func (s RunStatus) IsTerminal() bool {
	return s == RunStatusCompleted || s == RunStatusFailed || s == RunStatusTerminated
}

// RunState defines the specific execution state within an FSM run.
type RunState string

const (
	StateInit         RunState = "INIT"
	StateLLMRequest   RunState = "LLM_REQUEST"
	StatePrepareSteps RunState = "PREPARE_STEPS"
	StateExecuteSteps RunState = "EXECUTE_STEPS"
	StateWaiting      RunState = "WAITING"
	StateSynthesis    RunState = "SYNTHESIS"
	StateCompleted    RunState = "COMPLETED"
	StateFailed       RunState = "FAILED"
	StateTerminated   RunState = "TERMINATED"
)

// StepStatus defines the lifecycle status of an individual step/tool execution.
type StepStatus string

const (
	StepStatusPending   StepStatus = "PENDING"
	StepStatusRunning   StepStatus = "RUNNING"
	StepStatusCompleted StepStatus = "COMPLETED"
	StepStatusFailed    StepStatus = "FAILED"
	StepStatusTimedOut  StepStatus = "TIMED_OUT"
	StepStatusSkipped   StepStatus = "SKIPPED"
)

// IsTerminal reports whether the step status represents a finished end state.
func (s StepStatus) IsTerminal() bool {
	return s == StepStatusCompleted || s == StepStatusFailed || s == StepStatusTimedOut || s == StepStatusSkipped
}

// ExecutionMode defines whether steps are executed sequentially or concurrently.
type ExecutionMode string

const (
	ExecutionModeSequential ExecutionMode = "sequential"
	ExecutionModeParallel   ExecutionMode = "parallel"
)

// FSMType defines the workflow type.
type FSMType string

const (
	FSMTypeToolLoop        FSMType = "tool_loop"
	FSMTypeAgenticWorkflow FSMType = "agentic_workflow"
)

// FSMRun represents a persistent workflow execution instance.
type FSMRun struct {
	ID            string    `json:"id"`
	ChatID        string    `json:"chat_id"`
	UserID        string    `json:"user_id"`
	IsDM          bool      `json:"is_dm"`
	FSMType       FSMType   `json:"fsm_type"`
	Status        RunStatus `json:"status"`
	CurrentState  RunState  `json:"current_state"`
	Iteration     int       `json:"iteration"`
	MaxIterations int       `json:"max_iterations"`
	WaitCycles    int       `json:"wait_cycles"`
	ContextJSON   string    `json:"context_json"`
	ResultJSON    string    `json:"result_json"`
	ErrorText     string    `json:"error_text"`
	ResumeAt      *int64    `json:"resume_at"` // Unix timestamp in seconds; nil if not waiting
	Version       int       `json:"version"`
	CreatedAt     int64     `json:"created_at"`
	UpdatedAt     int64     `json:"updated_at"`

	// Unexported fields tracking database synchronization to avoid unnecessary rewrites of context_json.
	lastSavedContextJSON string
	contextSynced        bool
}

// MarkContextSynced marks the current ContextJSON as synchronized with persistent storage.
func (r *FSMRun) MarkContextSynced() {
	r.lastSavedContextJSON = r.ContextJSON
	r.contextSynced = true
}

// IsContextDirty reports whether ContextJSON has changed since the last database sync.
func (r *FSMRun) IsContextDirty() bool {
	return !r.contextSynced || r.ContextJSON != r.lastSavedContextJSON
}

// FSMStep represents an atomic step or tool invocation within a workflow run.
type FSMStep struct {
	ID             string        `json:"id"`
	RunID          string        `json:"run_id"`
	Iteration      int           `json:"iteration"`
	StepIndex      int           `json:"step_index"`
	ToolName       string        `json:"tool_name"`
	ToolCallID     string        `json:"tool_call_id"`
	ArgsJSON       string        `json:"args_json"`
	ResultJSON     string        `json:"result_json"`
	ExecutionMode  ExecutionMode `json:"execution_mode"`
	Status         StepStatus    `json:"status"`
	Attempt        int           `json:"attempt"`
	MaxAttempts    int           `json:"max_attempts"`
	TimeoutSeconds int           `json:"timeout_seconds"`
	StartedAt      *int64        `json:"started_at"`
	CompletedAt    *int64        `json:"completed_at"`
	ErrorText      string        `json:"error_text"`
}

// EncodeMessages serializes OpenAI chat completion messages into JSON string.
func EncodeMessages(messages []openai.ChatCompletionMessage) (string, error) {
	b, err := json.Marshal(messages)
	if err != nil {
		return "", fmt.Errorf("failed to encode messages: %w", err)
	}
	return string(b), nil
}

// DecodeMessages deserializes a JSON string into OpenAI chat completion messages.
func DecodeMessages(data string) ([]openai.ChatCompletionMessage, error) {
	if data == "" || data == "{}" {
		return nil, nil
	}
	var messages []openai.ChatCompletionMessage
	if err := json.Unmarshal([]byte(data), &messages); err != nil {
		return nil, fmt.Errorf("failed to decode messages: %w", err)
	}
	return messages, nil
}

// NewStepFromToolCall constructs an FSMStep from an OpenAI ToolCall.
func NewStepFromToolCall(runID string, iteration, stepIndex int, tc openai.ToolCall, mode ExecutionMode) FSMStep {
	return FSMStep{
		ID:             fmt.Sprintf("%s_i%d_s%d", runID, iteration, stepIndex),
		RunID:          runID,
		Iteration:      iteration,
		StepIndex:      stepIndex,
		ToolName:       tc.Function.Name,
		ToolCallID:     tc.ID,
		ArgsJSON:       tc.Function.Arguments,
		ExecutionMode:  mode,
		Status:         StepStatusPending,
		Attempt:        0,
		MaxAttempts:    3,
		TimeoutSeconds: 30,
	}
}
