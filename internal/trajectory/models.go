package trajectory

import (
	"encoding/json"
	"time"
)

// Event represents an immutable lifecycle event in the execution outbox.
type Event struct {
	ID           string          `json:"id"`            // UUID
	TrajectoryID string          `json:"trajectory_id"` // Trajectory UUID
	RunID        string          `json:"run_id"`        // FSM Run ID
	Generation   int             `json:"generation"`    // Durable execution generation
	Iteration    int             `json:"iteration"`     // FSM loop iteration counter
	Phase        string          `json:"phase"`         // "start", "loop", "synthesis", "terminal"
	StepID       string          `json:"step_id"`       // FSM step identifier
	Attempt      int             `json:"attempt"`       // 1-based retry attempt
	EventType    string          `json:"event_type"`    // "turn_start", "tool_attempt", "llm_response", "turn_terminal"
	Payload      json.RawMessage `json:"payload"`       // Type-specific event details
	CreatedAt    time.Time       `json:"created_at"`
}

// Step represents a single tool execution action-observation pair within a turn.
type Step struct {
	Index       int             `json:"index"`
	Iteration   int             `json:"iteration"`           // FSM iteration
	Attempt     int             `json:"attempt"`             // 1-based attempt counter for retries
	Thought     string          `json:"thought,omitempty"`   // z: reasoning trace / provider reasoning content
	Action      string          `json:"action"`              // a: tool name
	ActionArgs  json.RawMessage `json:"action_args"`         // a: tool input arguments JSON
	Observation string          `json:"observation"`         // o: raw untruncated tool output / stdout / stderr
	ExitCode    *int            `json:"exit_code,omitempty"` // nullable: exit code for execute/sandbox tools
	DurationMs  int64           `json:"duration_ms"`         // tool execution duration
}

// EvalBreakdown decomposes the components contributing to R_eval.
type EvalBreakdown struct {
	TotalToolCalls      int     `json:"total_tool_calls"`      // Total logical tool calls attempted
	SuccessfulToolCalls int     `json:"successful_tool_calls"` // Tool calls completing without error / exit 0
	FailedToolCalls     int     `json:"failed_tool_calls"`     // Tool calls returning errors or non-zero exits
	ToolSuccessRate     float64 `json:"tool_success_rate"`     // [0.0, 1.0] fraction of successful invocations
	BudgetExhausted     bool    `json:"budget_exhausted"`     // true if iteration or token budget was hit
	FSMFailed           bool    `json:"fsm_failed"`           // true if FSM terminated with unhandled error
	TerminationReason   string  `json:"termination_reason"`  // "completed", "budget_exhausted", "fsm_failed", "cancelled", "delivery_failed"
	HasUserFeedback     bool    `json:"has_user_feedback"`    // true if explicit user feedback was received
	UserFeedback        float64 `json:"user_feedback"`        // -1.0 (bad), 0.0 (neutral), +1.0 (good)
	FeedbackNote        string  `json:"feedback_note,omitempty"`
	ImplicitSignal      float64 `json:"implicit_signal"`               // -1.0 to +1.0 derived from follow-up turns
	ImplicitProvenance  string  `json:"implicit_provenance,omitempty"` // explanation/source of implicit score
	RewardVersion       string  `json:"reward_version"`               // e.g. "v1"
}

// Trajectory represents the canonical rollout record for a single logical turn.
type Trajectory struct {
	ID             string        `json:"id"`              // UUID / unique trajectory ID
	RunID          string        `json:"run_id"`          // FSM Run ID or Turn ID
	FrontendID     string        `json:"frontend_id"`     // "besedka", "cli", etc.
	SessionID      string        `json:"session_id"`      // Chat ID or CLI session string
	ScopeID        string        `json:"scope_id"`        // Channel / DM / workspace identifier
	ActorID        string        `json:"actor_id"`        // User ID string
	Model          string        `json:"model"`           // LLM model identifier
	ExecutionKind  string        `json:"execution_kind"`  // "interactive" or "scheduled"
	PrivacyLevel   string        `json:"privacy_level"`   // "public", "private", "internal"
	ExportApproved bool          `json:"export_approved"` // true if approved for objectstore upload
	Generation     int           `json:"generation"`      // Durable execution generation (e.g. 1=FSM, 2=volatile fallback)
	CreatedAt      time.Time     `json:"created_at"`      // Turn start timestamp
	CompletedAt    time.Time     `json:"completed_at"`    // Turn completion timestamp
	UserInput      string        `json:"x"`               // x: raw stripped user input
	TaskState      string        `json:"phi_task"`        // \Phi_{task}: active memory/task context
	ActiveSkills   []string      `json:"active_skills,omitempty"`
	Steps          []Step        `json:"steps"` // (z, a, o) sequence
	FinalOutput    string        `json:"y"`     // y: final assistant response
	Reward         float64       `json:"r_eval"`
	Breakdown      EvalBreakdown `json:"eval_breakdown"`
	Revision       int           `json:"revision"` // Monotonic revision counter (starts at 1)
}

// FeedbackRequest encapsulates an incoming user feedback command or reaction.
type FeedbackRequest struct {
	MessageID    string  // Source message idempotency key
	FrontendID   string  // Frontend identifier
	SessionID    string  // Session/chat identifier
	ScopeID      string  // Scope identifier (e.g. channel or dm)
	ActorID      string  // User providing feedback
	TrajectoryID string  // Optional explicit ID; if empty, targets actor's latest interactive turn
	Rating       float64 // +1.0, -1.0, 0.0
	Note         string  // Optional user feedback text
}

// TurnStartPayload contains initial session and task context recorded at turn start.
type TurnStartPayload struct {
	RunID          string   `json:"run_id"`
	FrontendID     string   `json:"frontend_id"`
	SessionID      string   `json:"session_id"`
	ScopeID        string   `json:"scope_id"`
	ActorID        string   `json:"actor_id"`
	Model          string   `json:"model"`
	ExecutionKind  string   `json:"execution_kind"`
	PrivacyLevel   string   `json:"privacy_level"`
	ExportApproved bool     `json:"export_approved"`
	Generation     int      `json:"generation"`
	Input          string   `json:"input"`
	TaskState      string   `json:"task_state"`
	ActiveSkills   []string `json:"active_skills,omitempty"`
}

// TurnTerminalPayload records the final outcome when a turn finishes.
type TurnTerminalPayload struct {
	Generation int    `json:"generation"`
	Reason     string `json:"reason"` // "completed", "fsm_failed", "cancelled", "budget_exhausted"
	Failed     bool   `json:"failed"`
	FinalReply string `json:"final_reply"`
}
