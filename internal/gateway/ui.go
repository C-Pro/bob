package gateway

import (
	"context"
	"strings"

	"bob/internal/agentapi"
	"bob/internal/chatcontext"
	"bob/internal/models"
)

// ToBesedkaUIData converts frontend-agnostic agentapi UI elements into a Besedka UIData payload.
func ToBesedkaUIData(elements []agentapi.UIElement) *models.UIData {
	if len(elements) == 0 {
		return nil
	}
	out := make([]models.UIElement, len(elements))
	for i, el := range elements {
		out[i] = models.UIElement{
			ID:          el.ID,
			Type:        models.UIElementType(el.Type),
			Label:       el.Label,
			Description: el.Description,
			Style:       models.UIElementStyle(el.Style),
			Default:     el.Default,
		}
	}
	return &models.UIData{Elements: out}
}

func formatIncomingMessageContent(msg models.Message, extraText string) string {
	base := ExtractMessageText(msg)
	if msg.Type == models.MessageTypeUIResponse && msg.UIResponse != nil {
		base = models.FormatUIResponseForContext(base, msg.UIResponse)
	} else if msg.UI != nil && len(msg.UI.Elements) > 0 {
		uiNote := models.FormatUIForContext(msg.UI)
		if base == "" {
			base = uiNote
		} else {
			base = base + "\n\n" + uiNote
		}
	}
	return strings.TrimSpace(base + extraText)
}

// FormatOutgoingMessageContent mirrors formatIncomingMessageContent for outgoing bot messages
// so ring buffer deduplication matches incoming WebSocket self-echoes.
func FormatOutgoingMessageContent(content string, ui *models.UIData, extraText string) string {
	base := strings.TrimSpace(content)
	if ui != nil && len(ui.Elements) > 0 {
		uiNote := models.FormatUIForContext(ui)
		if base == "" {
			base = uiNote
		} else {
			base = base + "\n\n" + uiNote
		}
	}
	return strings.TrimSpace(base + extraText)
}

func isUIResponseForBot(entries []chatcontext.Entry, parentSeq int64, botID string) (known bool, isForBot bool) {
	if parentSeq <= 0 || botID == "" {
		return false, false
	}
	for i := len(entries) - 1; i >= 0; i-- {
		e := entries[i]
		if e.Seq == parentSeq {
			return true, e.Role == "assistant" && e.SenderID == botID
		}
	}
	return false, false
}

func (g *Gateway) resolveUIResponseTarget(ctx context.Context, entries []chatcontext.Entry, chatID string, parentSeq int64, botID string) (known bool, isForBot bool) {
	if parentSeq <= 0 || botID == "" {
		return false, false
	}
	if known, forBot := isUIResponseForBot(entries, parentSeq, botID); known {
		return true, forBot
	}
	if g == nil || g.httpClient == nil || g.cfg == nil || strings.TrimSpace(g.cfg.BesedkaURL) == "" {
		return false, false
	}
	msgs, err := g.FetchChatMessages(ctx, chatID, parentSeq, parentSeq)
	if err != nil {
		return false, false
	}
	for _, m := range msgs {
		if m.Seq == parentSeq {
			return true, m.UserID == botID
		}
	}
	return false, false
}
