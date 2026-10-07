package agentapi

import "context"

// Frontend represents an external interaction surface (e.g. Besedka, CLI, TUI).
type Frontend interface {
	Bind(context.Context, RunDescriptor) (Bindings, error)
	Deliver(context.Context, Completion) error
}

// Bindings provides execution-scoped dependencies supplied by the hosting frontend.
type Bindings struct {
	Tools         Toolset           `json:"-"`
	Attachments   AttachmentHandler `json:"-"`
	Progress      ProgressObserver  `json:"-"`
	Notifications NotificationSink  `json:"-"`
}

// Notification represents an urgent or asynchronous message sent to the frontend during execution.
type Notification struct {
	ID          string       `json:"id,omitempty"`
	Content     string       `json:"content"`
	Attachments []Attachment `json:"attachments,omitempty"`
	Actions     []Action     `json:"actions,omitempty"`
}

// NotificationSink receives urgent notifications during tool execution (e.g. sandbox approval cards).
type NotificationSink interface {
	Notify(context.Context, Notification) error
}
