package gateway

import (
	"context"
	"fmt"
	"log/slog"
	"strings"

	"bob/internal/agentapi"
	"bob/internal/commands"
	"bob/internal/models"
)

func (g *Gateway) handleSkillCommand(ctx context.Context, msg models.Message, text, senderName string, isDM bool) error {
	if err := g.VerifyDMOwner(ctx, msg.ChatID, msg.UserID); err != nil {
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ %v", err))
	}

	kStore, err := g.getKnowledgeStore(ctx, msg.ChatID, isDM)
	if err != nil {
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Knowledge storage is unavailable: %v", err))
	}

	indexer, idxErr := g.getKnowledgeIndexer(ctx, msg.ChatID, isDM)
	if idxErr != nil {
		slog.Warn("knowledge indexer unavailable for skill command", "chat_id", msg.ChatID, "error", idxErr)
	}

	g.mu.Lock()
	botID := g.botUserID
	g.mu.Unlock()

	parts := strings.Fields(strings.TrimSpace(text))
	subcmd := ""
	var args []string
	if len(parts) > 1 {
		subcmd = parts[1]
		args = parts[2:]
	}

	req := commands.Request{
		Session: agentapi.SessionRef{
			FrontendID: "besedka",
			SessionID:  msg.ChatID,
			ScopeID:    msg.ChatID,
		},
		Actor: agentapi.Actor{
			ID: msg.UserID,
		},
		ActorName:  senderName,
		IsDirect:   isDM,
		Command:    "skill",
		Subcommand: subcmd,
		Args:       args,
		RawText:    text,
	}

	auth := commands.AuthorizerFunc(func(c context.Context, s agentapi.SessionRef, a agentapi.Actor, _, _ string) error {
		return g.VerifyDMOwner(c, s.ScopeID, a.ID)
	})

	handler := commands.NewSkillHandler(kStore, indexer, auth, botID)
	res, err := handler.Handle(ctx, req)
	if err != nil {
		return err
	}
	if res.IsIgnored {
		return nil
	}
	return g.SendMessage(msg.ChatID, res.Reply)
}
