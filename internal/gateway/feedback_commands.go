package gateway

import (
	"context"
	"fmt"
	"strings"

	"bob/internal/agentapi"
	"bob/internal/commands"
	"bob/internal/models"
)

func (g *Gateway) handleFeedbackCommand(ctx context.Context, msg models.Message, text, senderName string, isDM bool) error {
	g.mu.Lock()
	sp := g.storeProvider
	g.mu.Unlock()

	parts := strings.Fields(strings.TrimSpace(text))
	if len(parts) == 0 {
		return nil
	}

	cmd := strings.ToLower(strings.TrimPrefix(parts[0], "/"))
	var args []string
	if len(parts) > 1 {
		args = parts[1:]
	}

	msgID := ""
	if msg.Seq > 0 {
		msgID = fmt.Sprintf("besedka:%s:%d", msg.ChatID, msg.Seq)
	}

	req := commands.Request{
		MessageID: msgID,
		Session: agentapi.SessionRef{
			FrontendID: "besedka",
			SessionID:  msg.ChatID,
			ScopeID:    msg.ChatID,
			IsDM:       isDM,
		},
		Actor: agentapi.Actor{
			ID: msg.UserID,
		},
		ActorName: senderName,
		IsDirect:  isDM,
		Command:   cmd,
		Args:      args,
		RawText:   text,
	}

	handler := commands.NewFeedbackHandler(sp)
	res, err := handler.Handle(ctx, req)
	if err != nil {
		return err
	}

	if res.Reply != "" {
		return g.SendMessage(msg.ChatID, res.Reply)
	}
	return nil
}
