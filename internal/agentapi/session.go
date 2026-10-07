package agentapi

// SessionRef uniquely identifies a conversation session across frontends and scopes.
type SessionRef struct {
	FrontendID string `json:"frontend_id"`
	SessionID  string `json:"session_id"`
	ScopeID    string `json:"scope_id"`
}

// Actor identifies the initiator or author of a turn.
type Actor struct {
	ID string `json:"id"`
}

// ExecutionKind distinguishes interactive user turns from scheduled executions.
type ExecutionKind string

const (
	Interactive ExecutionKind = "interactive"
	Scheduled   ExecutionKind = "scheduled"
)

// RunDescriptor captures the execution context and constraints of an agent run.
type RunDescriptor struct {
	RunID         string        `json:"run_id"`
	Session       SessionRef    `json:"session"`
	Actor         Actor         `json:"actor"`
	Kind          ExecutionKind `json:"kind"`
	Model         string            `json:"model"`
	MaxIterations int               `json:"max_iterations"`
	Metadata      map[string]string `json:"metadata,omitempty"`
}
