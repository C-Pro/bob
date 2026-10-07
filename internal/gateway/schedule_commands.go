package gateway

import (
	"context"
	"errors"
	"strings"
	"time"

	"bob/internal/agentapi"
	"bob/internal/chatcontext"
	"bob/internal/commands"
	"bob/internal/models"
	"bob/internal/scheduler"
)

func (g *Gateway) getSchedulerStore(ctx context.Context, chatID string, isDM bool) (*scheduler.Store, error) {
	g.mu.Lock()
	sp := g.storeProvider
	g.mu.Unlock()
	if sp == nil {
		return nil, errors.New("scheduler store provider is not initialized")
	}
	return sp.GetSchedulerStore(ctx, chatID, isDM)
}

func (g *Gateway) handleScheduleCommand(ctx context.Context, msg models.Message, text, senderName string, isDM bool) error {
	g.mu.Lock()
	botID := g.botUserID
	botUser := g.botUser
	if botUser.ID == "" && botID != "" {
		if u, ok := g.userCache.Get(botID); ok {
			botUser = u
		}
	}
	g.mu.Unlock()

	parts := strings.Fields(strings.TrimSpace(text))
	subcmd := ""
	var args []string
	if len(parts) > 1 {
		subcmd = parts[1]
		args = parts[2:]
	}

	schedStore, err := g.getSchedulerStore(ctx, msg.ChatID, isDM)
	if err != nil {
		return g.SendMessage(msg.ChatID, "⚠️ Scheduler storage is currently unavailable.")
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
		Command:    "schedule",
		Subcommand: subcmd,
		Args:       args,
		RawText:    text,
	}

	resolver := commands.UserResolverFunc(func(_ context.Context, id string) string {
		return g.userCache.GetDisplayName(id)
	})

	handler := commands.NewScheduleHandler(schedStore, resolver, botID)
	res, err := handler.Handle(ctx, req)
	if err != nil {
		return err
	}

	if res.RecordAssistantEntry {
		g.contextManager.Push(msg.ChatID, chatcontext.Entry{
			Role:       "assistant",
			SenderID:   botID,
			SenderName: botUser.GetDisplayName(),
			Content:    res.Reply,
			Timestamp:  time.Now().Unix(),
		})
	}

	return g.SendMessage(msg.ChatID, res.Reply)
}
