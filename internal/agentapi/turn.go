package agentapi

import (
	"context"
	"encoding/json"

	openai "github.com/sashabaranov/go-openai"
)

// ActionType defines the classification of a side effect or control signal requested during execution.
type ActionType string

const (
	ActionApprovalRequest ActionType = "approval_request"
	ActionSuppressReply   ActionType = "suppress_reply"
)

// Action represents a control signal, intent, or side effect produced during execution or tool calls.
type Action struct {
	ID      string          `json:"id,omitempty"`
	Type    ActionType      `json:"type"`
	Content string          `json:"content,omitempty"`
	Payload json.RawMessage `json:"payload,omitempty"`
}

// Turn contains all inputs needed to execute one agent turn.
type Turn struct {
	Run          RunDescriptor                  `json:"run"`
	SystemPrompt string                         `json:"system_prompt"`
	Messages     []openai.ChatCompletionMessage `json:"messages"`
	Deliver      bool                           `json:"deliver,omitempty"`
}

// Result contains the output produced by an agent turn.
type Result struct {
	RunID       string       `json:"run_id"`
	Content     string       `json:"content"`
	Iterations  int          `json:"iterations"`
	Attachments []Attachment `json:"attachments,omitempty"`
	Actions     []Action     `json:"actions,omitempty"`
}

// Completion represents the final delivery payload for an agent run, used for crash recovery and out-of-band delivery.
type Completion struct {
	Run    RunDescriptor `json:"run"`
	Result Result        `json:"result"`
	Status RunStatus     `json:"status"`
	Error  string        `json:"error,omitempty"`
}

// TurnRunner executes an agent turn within the configured environment.
type TurnRunner interface {
	Run(ctx context.Context, turn Turn) (Result, error)
}
