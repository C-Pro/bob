package tools

import (
	"context"
	"fmt"
	"strings"

	"bob/internal/agentapi"
	"bob/internal/models"
)

// IsReadOnlyTool reports whether a tool provided by this registry is read-only and side-effect free.
func IsReadOnlyTool(name string) bool {
	switch strings.TrimSpace(name) {
	case "web_search", "web_fetch", "recall_memory", "discover_memories", "discover_skills":
		return true
	default:
		return false
	}
}

// SessionToolset wraps a Registry and a ChatSessionContext as an agentapi.Toolset.
type SessionToolset struct {
	registry *Registry
	session  ChatSessionContext
}

// SessionToolset returns an agentapi.Toolset bound to the given session context.
func (r *Registry) SessionToolset(session ChatSessionContext) agentapi.Toolset {
	return &SessionToolset{
		registry: r,
		session:  session,
	}
}

// Definitions returns the authorized tool definitions for the bound session.
func (s *SessionToolset) Definitions(ctx context.Context) ([]agentapi.ToolDefinition, error) {
	if s.registry == nil {
		return nil, nil
	}
	openaiTools := s.registry.ToolDefinitionsForSession(s.session)
	defs := make([]agentapi.ToolDefinition, 0, len(openaiTools))
	for _, t := range openaiTools {
		if t.Function == nil || t.Function.Name == "" {
			continue
		}
		defs = append(defs, agentapi.ToolDefinition{
			Schema:   t,
			ReadOnly: IsReadOnlyTool(t.Function.Name),
		})
	}
	return defs, nil
}

// Execute dispatches a tool call with the bound chat session in context.
func (s *SessionToolset) Execute(ctx context.Context, call agentapi.ToolCall) (agentapi.ToolResult, error) {
	if s.registry == nil {
		return agentapi.ToolResult{}, fmt.Errorf("tools registry is not configured")
	}

	callSession := s.session
	if ctxSess, ok := ChatSessionFromContext(ctx); ok {
		if ctxSess.ChatID == s.session.ChatID && (s.session.UserID == "" || ctxSess.UserID == s.session.UserID) {
			if ctxSess.FSMRunID != "" {
				callSession.FSMRunID = ctxSess.FSMRunID
			}
			if ctxSess.SourceMessageSeq != nil {
				callSession.SourceMessageSeq = ctxSess.SourceMessageSeq
			}
			if ctxSess.StagedAttachments != nil && callSession.StagedAttachments == nil {
				callSession.StagedAttachments = ctxSess.StagedAttachments
			}
			if ctxSess.KnowledgeBudget != nil && callSession.KnowledgeBudget == nil {
				callSession.KnowledgeBudget = ctxSess.KnowledgeBudget
			}
			if ctxSess.Notifier != nil && callSession.Notifier == nil {
				callSession.Notifier = ctxSess.Notifier
			}
			if ctxSess.SandboxRequestCreated != nil && callSession.SandboxRequestCreated == nil {
				callSession.SandboxRequestCreated = ctxSess.SandboxRequestCreated
			}
		}
	}

	callCtx := WithChatSession(ctx, callSession)
	content, err := s.registry.Execute(callCtx, call.Name, call.Arguments)

	var attachments []agentapi.Attachment
	if callSession.StagedAttachments != nil {
		staged := callSession.StagedAttachments.All()
		for _, att := range staged {
			var attType agentapi.AttachmentType
			switch att.Type {
			case models.AttachmentTypeImage:
				attType = agentapi.AttachmentImage
			default:
				attType = agentapi.AttachmentFile
			}
			attachments = append(attachments, agentapi.Attachment{
				ID:       att.FileID,
				Type:     attType,
				Name:     att.Name,
				MIMEType: att.MimeType,
			})
		}
	}

	var actions []agentapi.Action
	if s.session.SandboxRequestCreated != nil && *s.session.SandboxRequestCreated {
		actions = append(actions, agentapi.Action{
			Type: agentapi.ActionSuppressReply,
		})
	}

	return agentapi.ToolResult{
		Content:     content,
		Attachments: attachments,
		Actions:     actions,
	}, err
}
