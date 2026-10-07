package gateway

import (
	"bob/internal/agent"
	"bob/internal/agent/scheduled"
)

// MessageSender sends output messages to chat platforms.
type MessageSender = scheduled.MessageSender

// AttachmentMessageSender extends MessageSender with support for attachments.
type AttachmentMessageSender = scheduled.AttachmentMessageSender

// AttachmentProcessor processes attachments to return context formatting and images.
type AttachmentProcessor = scheduled.AttachmentProcessor

// ChatLocker is a transitional alias for agent.SessionLocker.
type ChatLocker = agent.SessionLocker

// NewChatLocker creates a new ChatLocker.
var NewChatLocker = agent.NewSessionLocker

// FSMRunner executes tool loops.
type FSMRunner = scheduled.FSMRunner

// FSMRunnerFunc is an adapter allowing a function to be used as FSMRunner.
type FSMRunnerFunc = scheduled.FSMRunnerFunc

// InvokerConfig configures ScheduleInvoker.
type InvokerConfig = scheduled.Config

// ScheduleInvoker implements scheduler.Invoker for executing schedules with FSM and ephemeral sandboxes.
type ScheduleInvoker = scheduled.Invoker

// NewScheduleInvoker creates a new ScheduleInvoker.
var NewScheduleInvoker = scheduled.NewInvoker
