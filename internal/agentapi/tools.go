package agentapi

import (
	"context"

	openai "github.com/sashabaranov/go-openai"
)

// Toolset defines the execution interface for tools provided to an agent run.
type Toolset interface {
	Definitions(context.Context) ([]ToolDefinition, error)
	Execute(context.Context, ToolCall) (ToolResult, error)
}

// ToolDefinition describes a callable tool and its execution characteristics.
type ToolDefinition struct {
	Schema   openai.Tool `json:"schema"`
	ReadOnly bool        `json:"read_only"`
}

// ToolCall represents a model-generated invocation of a tool.
type ToolCall struct {
	ID        string `json:"id"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// ToolResult encapsulates the outcome of a tool execution.
type ToolResult struct {
	Content     string       `json:"content"`
	Attachments []Attachment `json:"attachments,omitempty"`
	Actions     []Action     `json:"actions,omitempty"`
}
