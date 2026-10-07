package agentapi

import "context"

// RunStatus represents the lifecycle state of an agent run.
type RunStatus string

const (
	RunPending    RunStatus = "PENDING"
	RunRunning    RunStatus = "RUNNING"
	RunWaiting    RunStatus = "WAITING"
	RunCompleted  RunStatus = "COMPLETED"
	RunFailed     RunStatus = "FAILED"
	RunTerminated RunStatus = "TERMINATED"
)

// StepStatus represents the state of an individual tool step execution.
type StepStatus string

const (
	StepPending   StepStatus = "PENDING"
	StepRunning   StepStatus = "RUNNING"
	StepCompleted StepStatus = "COMPLETED"
	StepFailed    StepStatus = "FAILED"
	StepTimedOut  StepStatus = "TIMED_OUT"
	StepSkipped   StepStatus = "SKIPPED"
)

// ProgressEventType distinguishes between different progress updates.
type ProgressEventType string

const (
	ProgressStepsPrepared ProgressEventType = "steps_prepared"
	ProgressStepUpdated   ProgressEventType = "step_updated"
	ProgressRunFinished   ProgressEventType = "run_finished"
)

// StepProgress records the state of a single step within a tool iteration.
type StepProgress struct {
	ID        string     `json:"id"`
	Iteration int        `json:"iteration"`
	StepIndex int        `json:"step_index"`
	Call      ToolCall   `json:"call"`
	Status    StepStatus `json:"status"`
	Attempt   int        `json:"attempt"`
	Error     string     `json:"error,omitempty"`
}

// ProgressEvent communicates execution progress to a ProgressObserver.
type ProgressEvent struct {
	RunID  string            `json:"run_id"`
	Type   ProgressEventType `json:"type"`
	Steps  []StepProgress    `json:"steps,omitempty"`
	Step   *StepProgress     `json:"step,omitempty"`
	Status RunStatus         `json:"status,omitempty"`
	Error  string            `json:"error,omitempty"`
}

// ProgressObserver observes execution progress updates without mutating state.
type ProgressObserver interface {
	Observe(context.Context, ProgressEvent)
}
