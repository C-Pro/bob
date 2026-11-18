package fsm

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"

	"bob/internal/agentapi"

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
	ID               string                 `json:"id"`
	ChatID           string                 `json:"chat_id"`
	UserID           string                 `json:"user_id"`
	IsDM             bool                   `json:"is_dm"`
	FrontendID       string                 `json:"frontend_id,omitempty"`
	ScopeID          string                 `json:"scope_id,omitempty"`
	Model            string                 `json:"model,omitempty"`
	ExecutionKind    agentapi.ExecutionKind `json:"execution_kind,omitempty"`
	FSMType          FSMType                `json:"fsm_type"`
	Status           RunStatus              `json:"status"`
	CurrentState     RunState               `json:"current_state"`
	Iteration        int                    `json:"iteration"`
	MaxIterations    int                    `json:"max_iterations"`
	WaitCycles       int                    `json:"wait_cycles"`
	ContextJSON      string                 `json:"context_json"`
	ResultJSON       string                 `json:"result_json"`
	ErrorText        string                 `json:"error_text"`
	ResumeAt         *int64                 `json:"resume_at"` // Unix timestamp in seconds; nil if not waiting
	Version          int                    `json:"version"`
	SourceMessageSeq *int64                 `json:"source_message_seq,omitempty"`
	CreatedAt        int64                  `json:"created_at"`
	UpdatedAt        int64                  `json:"updated_at"`

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
	maxAttempts := 3
	if tc.Function.Name == "propose_memory" || tc.Function.Name == "propose_skill" {
		maxAttempts = 1
	}
	timeoutSeconds := 30
	if tc.Function.Name == "execute" || tc.Function.Name == "sandbox_exec" {
		timeoutSeconds = 60
		var timeoutArg struct {
			TimeoutSeconds *int `json:"timeout_seconds"`
		}
		if err := json.Unmarshal([]byte(tc.Function.Arguments), &timeoutArg); err == nil && timeoutArg.TimeoutSeconds != nil {
			if *timeoutArg.TimeoutSeconds > 0 {
				timeoutSeconds = *timeoutArg.TimeoutSeconds
				if timeoutSeconds > 600 {
					timeoutSeconds = 600
				}
				if timeoutSeconds < 5 {
					timeoutSeconds = 5
				}
			}
		}
	}
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
		MaxAttempts:    maxAttempts,
		TimeoutSeconds: timeoutSeconds,
	}
}

// ToToolCall reconstructs an OpenAI ToolCall from the FSMStep.
func (s FSMStep) ToToolCall() openai.ToolCall {
	return openai.ToolCall{
		ID:   s.ToolCallID,
		Type: openai.ToolTypeFunction,
		Function: openai.FunctionCall{
			Name:      s.ToolName,
			Arguments: s.ArgsJSON,
		},
	}
}

const (
	// CompactionVerdictProbedCompact indicates that a rubric probe decided to compact the trajectory.
	CompactionVerdictProbedCompact = "probed_compact"
	// CompactionVerdictProbedSkip indicates that a rubric probe decided not to compact yet.
	CompactionVerdictProbedSkip = "probed_skip"
	// CompactionVerdictBackstopForced indicates that compaction was forced by the hard byte backstop.
	CompactionVerdictBackstopForced = "backstop_forced"
)

// VerifiedFinding represents a verified fact or intermediate result distilled from compacted steps.
type VerifiedFinding struct {
	Summary       string   `json:"summary"`
	SourceStepIDs []string `json:"source_step_ids"`
}

// CompactedTaskState represents a versioned structured task state snapshot for an FSM run.
type CompactedTaskState struct {
	ID                   string            `json:"id,omitempty"`
	Version              int               `json:"version"`
	ParentVersion        int               `json:"parent_version"`
	RunID                string            `json:"run_id"`
	CompactedUpToIter    int               `json:"compacted_up_to_iter"`
	CompactedStepIDs     []string          `json:"compacted_step_ids"`
	OriginalObjective    string            `json:"original_objective"`
	ImmutableConstraints []string          `json:"immutable_constraints,omitempty"`
	VerifiedFindings     []VerifiedFinding `json:"verified_findings,omitempty"`
	DecisionsAndRejected []string          `json:"decisions_and_rejected,omitempty"`
	CurrentSubgoal       string            `json:"current_subgoal"`
	NextIntendedActions  []string          `json:"next_intended_actions,omitempty"`
	UnresolvedBlockers   []string          `json:"unresolved_blockers,omitempty"`
	ProducedArtifacts    []string          `json:"produced_artifacts,omitempty"`
	ActiveSkillVersions  []string          `json:"active_skill_versions,omitempty"`
	ActiveMemoryVersions []string          `json:"active_memory_versions,omitempty"`
	RubricVerdict        string            `json:"rubric_verdict"`
	Model                string            `json:"model"`
	InputBytes           int               `json:"-"`
	OutputBytes          int               `json:"-"`
	CreatedAt            int64             `json:"created_at"`
}

// UnmarshalJSON implements custom unmarshaling for CompactedTaskState to gracefully accept
// string fields from LLMs that return a single string where an array of strings was requested.
func (s *CompactedTaskState) UnmarshalJSON(data []byte) error {
	type rawState CompactedTaskState
	type flexibleState struct {
		rawState
		ImmutableConstraints json.RawMessage `json:"immutable_constraints,omitempty"`
		DecisionsAndRejected json.RawMessage `json:"decisions_and_rejected,omitempty"`
		NextIntendedActions  json.RawMessage `json:"next_intended_actions,omitempty"`
		UnresolvedBlockers   json.RawMessage `json:"unresolved_blockers,omitempty"`
		ProducedArtifacts    json.RawMessage `json:"produced_artifacts,omitempty"`
		ActiveSkillVersions  json.RawMessage `json:"active_skill_versions,omitempty"`
		ActiveMemoryVersions json.RawMessage `json:"active_memory_versions,omitempty"`
	}
	var f flexibleState
	if err := json.Unmarshal(data, &f); err != nil {
		return err
	}
	*s = CompactedTaskState(f.rawState)
	s.ImmutableConstraints = parseFlexibleStringSlice(f.ImmutableConstraints)
	s.DecisionsAndRejected = parseFlexibleStringSlice(f.DecisionsAndRejected)
	s.NextIntendedActions = parseFlexibleStringSlice(f.NextIntendedActions)
	s.UnresolvedBlockers = parseFlexibleStringSlice(f.UnresolvedBlockers)
	s.ProducedArtifacts = parseFlexibleStringSlice(f.ProducedArtifacts)
	s.ActiveSkillVersions = parseFlexibleStringSlice(f.ActiveSkillVersions)
	s.ActiveMemoryVersions = parseFlexibleStringSlice(f.ActiveMemoryVersions)
	return nil
}

func parseFlexibleStringSlice(raw json.RawMessage) []string {
	if len(raw) == 0 || bytes.Equal(raw, []byte("null")) {
		return nil
	}
	var list []string
	if err := json.Unmarshal(raw, &list); err == nil {
		return list
	}
	var single string
	if err := json.Unmarshal(raw, &single); err == nil {
		single = strings.TrimSpace(single)
		if single == "" {
			return nil
		}
		return []string{single}
	}
	return nil
}

// CompactionProbeResult represents the structured decision returned by a SelfCompact rubric probe.
type CompactionProbeResult struct {
	ShouldCompact bool                `json:"should_compact"`
	Reason        string              `json:"reason"`
	State         *CompactedTaskState `json:"state,omitempty"`
}
