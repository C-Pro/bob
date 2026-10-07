package commands

import (
	"context"

	"bob/internal/agentapi"
)

// Authorizer checks whether an actor is authorized to execute a command.
type Authorizer interface {
	Authorize(ctx context.Context, session agentapi.SessionRef, actor agentapi.Actor, command, subcommand string) error
}

// AuthorizerFunc adapts a function to the Authorizer interface.
type AuthorizerFunc func(ctx context.Context, session agentapi.SessionRef, actor agentapi.Actor, command, subcommand string) error

// Authorize calls f(ctx, session, actor, command, subcommand).
func (f AuthorizerFunc) Authorize(ctx context.Context, session agentapi.SessionRef, actor agentapi.Actor, command, subcommand string) error {
	return f(ctx, session, actor, command, subcommand)
}

// UserResolver resolves user display names for presentation.
type UserResolver interface {
	GetDisplayName(ctx context.Context, userID string) string
}

// UserResolverFunc adapts a function to the UserResolver interface.
type UserResolverFunc func(ctx context.Context, userID string) string

// GetDisplayName calls f(ctx, userID).
func (f UserResolverFunc) GetDisplayName(ctx context.Context, userID string) string {
	return f(ctx, userID)
}

// SandboxContinuation carries information when a sandbox command triggers an agent continuation turn.
type SandboxContinuation struct {
	ActorID string
	Reason  string
}

// Request encapsulates a slash command invocation across any frontend.
type Request struct {
	Session    agentapi.SessionRef
	Actor      agentapi.Actor
	ActorName  string
	IsDirect   bool
	Command    string
	Subcommand string
	Args       []string
	RawText    string
}

// Result encapsulates the outcome of a slash command.
type Result struct {
	Reply                string
	IsIgnored            bool
	RecordAssistantEntry bool
	Continuation         *SandboxContinuation
}
