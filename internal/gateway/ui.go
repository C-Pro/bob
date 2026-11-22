package gateway

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"time"

	"bob/internal/agentapi"
	"bob/internal/chatcontext"
	"bob/internal/models"
	"bob/internal/sandbox"
	"bob/internal/scheduler"
	"bob/internal/tools"
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

type followUpApprovalCard struct {
	Text string
	UI   *models.UIData
}

type approvalItem struct {
	payload agentapi.ApprovalRequestPayload
	ui      *models.UIData
}

type promptItem struct {
	msg string
	ui  *models.UIData
}

func buildApprovalUIData(p agentapi.ApprovalRequestPayload) *models.UIData {
	switch p.Kind {
	case agentapi.ApprovalKindSandbox:
		return models.NewSandboxApprovalUI(p.TargetID)
	case agentapi.ApprovalKindMemory:
		return models.NewMemoryApprovalUI(p.TargetID)
	case agentapi.ApprovalKindSkill:
		return models.NewSkillApprovalUI(p.TargetID)
	case agentapi.ApprovalKindSchedule:
		return models.NewScheduleApprovalUI(p.TargetID)
	default:
		return nil
	}
}

func formatApprovalCardText(p agentapi.ApprovalRequestPayload) string {
	var b strings.Builder
	switch p.Kind {
	case agentapi.ApprovalKindMemory:
		b.WriteString("🧠 **Memory Approval Requested**\n")
		if p.Label != "" {
			fmt.Fprintf(&b, "• **Type:** %s\n", p.Label)
		}
		if p.Summary != "" {
			fmt.Fprintf(&b, "• **Summary:** %s\n", p.Summary)
		}
		if p.TargetID != "" {
			fmt.Fprintf(&b, "• **Version ID:** `%s`\n", p.TargetID)
			fmt.Fprintf(&b, "\nUse the buttons below or reply `/memory approve %s` or `/memory reject %s`.", p.TargetID, p.TargetID)
		}
	case agentapi.ApprovalKindSkill:
		b.WriteString("🛠️ **Skill Approval Requested**\n")
		if p.Label != "" {
			fmt.Fprintf(&b, "• **Name:** %s\n", p.Label)
		}
		if p.Summary != "" {
			fmt.Fprintf(&b, "• **Summary:** %s\n", p.Summary)
		}
		if p.TargetID != "" {
			fmt.Fprintf(&b, "• **Version ID:** `%s`\n", p.TargetID)
			fmt.Fprintf(&b, "\nUse the buttons below or reply `/skill approve %s` or `/skill reject %s`.", p.TargetID, p.TargetID)
		}
	case agentapi.ApprovalKindSchedule:
		b.WriteString("📅 **Schedule Approval Requested**\n")
		if p.Label != "" {
			fmt.Fprintf(&b, "• **Name:** `%s`\n", p.Label)
		}
		if p.Summary != "" {
			fmt.Fprintf(&b, "• **Instruction:** %s\n", p.Summary)
		}
		if p.TargetID != "" {
			fmt.Fprintf(&b, "• **Schedule ID:** `%s`\n", p.TargetID)
		}
		if p.Label != "" {
			fmt.Fprintf(&b, "\nUse the buttons below or reply `/schedule approve %s` or `/schedule deny %s`.", p.Label, p.Label)
		} else {
			b.WriteString("\nUse the buttons below or `/schedule info <name>` to approve or deny.")
		}
	case agentapi.ApprovalKindSandbox:
		b.WriteString("⏳ **Sandbox Approval Requested**\n")
		if p.Summary != "" {
			fmt.Fprintf(&b, "• **Reason:** %s\n", p.Summary)
		}
		b.WriteString("\nUse the buttons below or reply `/sandbox approve` or `/sandbox deny`.")
	}
	return b.String()
}

func buildDeliveryUIPlan(actions []agentapi.Action, suppressMain bool) (fallbackText string, mainUI *models.UIData, followUps []followUpApprovalCard) {
	var prompts []promptItem
	var approvals []approvalItem
	seenApprovals := make(map[string]bool)

	for _, act := range actions {
		switch act.Type {
		case agentapi.ActionInteractivePrompt:
			var p agentapi.InteractivePromptPayload
			if err := json.Unmarshal(act.Payload, &p); err != nil {
				continue
			}
			candidate := ToBesedkaUIData(p.Elements)
			if candidate == nil {
				continue
			}
			sanitized, err := models.SanitizeUIDataForSend(candidate)
			if err != nil {
				continue
			}
			hasReservedID := false
			for _, el := range sanitized.Elements {
				if models.HasReservedUIActionPrefix(el.ID) {
					hasReservedID = true
					break
				}
			}
			if hasReservedID {
				slog.Warn("ignoring interactive prompt containing reserved bob: element ID")
				continue
			}
			msg := strings.TrimSpace(p.Message)
			if msg == "" {
				msg = strings.TrimSpace(act.Content)
			}
			prompts = append(prompts, promptItem{msg: msg, ui: sanitized})
		case agentapi.ActionApprovalRequest:
			var p agentapi.ApprovalRequestPayload
			if err := json.Unmarshal(act.Payload, &p); err != nil {
				continue
			}
			uiData := buildApprovalUIData(p)
			if uiData == nil {
				continue
			}
			sanitized, err := models.SanitizeUIDataForSend(uiData)
			if err != nil {
				continue
			}
			key := string(p.Kind) + ":" + strings.TrimSpace(p.TargetID)
			if seenApprovals[key] {
				continue
			}
			seenApprovals[key] = true
			approvals = append(approvals, approvalItem{payload: p, ui: sanitized})
		}
	}

	if suppressMain {
		for _, p := range prompts {
			followUps = append(followUps, followUpApprovalCard{
				Text: p.msg,
				UI:   p.ui,
			})
		}
		for _, item := range approvals {
			followUps = append(followUps, followUpApprovalCard{
				Text: formatApprovalCardText(item.payload),
				UI:   item.ui,
			})
		}
		return "", nil, followUps
	}

	if len(prompts) > 0 {
		mainUI = prompts[0].ui
		for _, p := range prompts[1:] {
			followUps = append(followUps, followUpApprovalCard{
				Text: p.msg,
				UI:   p.ui,
			})
		}
		for _, item := range approvals {
			followUps = append(followUps, followUpApprovalCard{
				Text: formatApprovalCardText(item.payload),
				UI:   item.ui,
			})
		}
		return prompts[0].msg, mainUI, followUps
	}

	if len(approvals) == 1 {
		mainUI = approvals[0].ui
		return formatApprovalCardText(approvals[0].payload), mainUI, nil
	}

	for _, item := range approvals {
		followUps = append(followUps, followUpApprovalCard{
			Text: formatApprovalCardText(item.payload),
			UI:   item.ui,
		})
	}
	return "", nil, followUps
}

func (g *Gateway) sendCommandReply(chatID, botID, botName, reply string, ui *models.UIData, recordAssistant bool) error {
	if recordAssistant {
		g.contextManager.Push(chatID, chatcontext.Entry{
			Role:       "assistant",
			SenderID:   botID,
			SenderName: botName,
			Content:    FormatOutgoingMessageContent(reply, ui, ""),
			Timestamp:  time.Now().Unix(),
		})
	}
	if ui != nil {
		return g.SendMessageWithUI(chatID, reply, ui)
	}
	return g.SendMessage(chatID, reply)
}

func (g *Gateway) deliverAgentReply(
	ctx context.Context,
	chatID, actorID, botID, botName string,
	isDM bool,
	sourceRun agentapi.RunDescriptor,
	reply string,
	attachments []models.Attachment,
	actions []agentapi.Action,
	allowSandboxPrompt bool,
) error {
	suppressMain := false
	for _, act := range actions {
		if act.Type == agentapi.ActionSuppressReply {
			suppressMain = true
			break
		}
	}

	fallbackText, mainUI, followUps := buildDeliveryUIPlan(actions, suppressMain)
	if suppressMain {
		g.contextManager.Push(chatID, chatcontext.Entry{
			Role:       "assistant",
			SenderID:   botID,
			SenderName: botName,
			Content:    "Sandbox approval requested.",
			Timestamp:  time.Now().Unix(),
		})
	} else {
		if strings.TrimSpace(reply) == "" && fallbackText != "" {
			reply = fallbackText
		}

		reply = g.finalizeImageReply(ctx, sourceRun, reply)
		if reply == "" && len(attachments) == 0 && mainUI == nil && len(followUps) == 0 {
			return nil
		}

		if allowSandboxPrompt && isDM {
			g.mu.Lock()
			sm := g.sandboxManager
			g.mu.Unlock()
			if sm != nil {
				if sbx, ok := sm.GetStatus(actorID); ok && sbx != nil && sbx.Status == sandbox.StatusRunning {
					if !strings.Contains(reply, "/sandbox destroy") {
						rem := time.Until(sbx.ExpiresAt).Round(time.Minute)
						if rem < 0 {
							rem = 0
						}
						reply += fmt.Sprintf("\n\n💡 Would you like to destroy the sandbox (`/sandbox destroy`) or keep it? It will automatically be destroyed in %s.", rem)
					}
					if mainUI == nil {
						mainUI = models.NewSandboxRunningUI(sbx.ID)
					}
				}
			}
		}

		if reply != "" || len(attachments) > 0 || mainUI != nil {
			formattedReply := FormatResponse(reply, isDM, g.cfg.TownhallMaxParagraphs, g.cfg.DMMaxParagraphs)
			if err := g.SendMessageWithAttachmentsAndUI(chatID, formattedReply, attachments, mainUI); err != nil {
				return fmt.Errorf("failed to deliver reply to chat %s: %w", chatID, err)
			}

			extraText := ""
			if len(attachments) > 0 {
				extraText, _ = g.processAttachments(ctx, attachments)
			}
			g.contextManager.Push(chatID, chatcontext.Entry{
				Role:       "assistant",
				SenderID:   botID,
				SenderName: botName,
				Content:    FormatOutgoingMessageContent(formattedReply, mainUI, extraText),
				Timestamp:  time.Now().Unix(),
			})
		}
	}

	var followUpErrs []error
	for i, card := range followUps {
		if err := g.SendMessageWithUI(chatID, card.Text, card.UI); err != nil {
			slog.Warn("failed to send follow-up approval card", "chat_id", chatID, "card_index", i+1, "error", err)
			followUpErrs = append(followUpErrs, fmt.Errorf("failed to deliver follow-up approval card %d to chat %s: %w", i+1, chatID, err))
			continue
		}
		g.contextManager.Push(chatID, chatcontext.Entry{
			Role:       "assistant",
			SenderID:   botID,
			SenderName: botName,
			Content:    FormatOutgoingMessageContent(card.Text, card.UI, ""),
			Timestamp:  time.Now().Unix(),
		})
	}

	return errors.Join(followUpErrs...)
}

func (g *Gateway) handleUIResponseAction(ctx context.Context, msg models.Message, isDM bool, senderName string) (bool, error) {
	if msg.UIResponse == nil || len(msg.UIResponse.SelectedIDs) == 0 {
		return false, nil
	}

	var reservedIDs []string
	for _, id := range msg.UIResponse.SelectedIDs {
		if models.HasReservedUIActionPrefix(id) {
			reservedIDs = append(reservedIDs, id)
		}
	}
	if len(reservedIDs) == 0 {
		return false, nil
	}

	if len(reservedIDs) > 1 || len(msg.UIResponse.SelectedIDs) > 1 {
		return true, g.SendMessage(msg.ChatID, "⚠️ Reserved UI actions must be submitted individually.")
	}

	parsed, ok := models.ParseReservedUIAction(reservedIDs[0])
	if !ok {
		return true, g.SendMessage(msg.ChatID, "⚠️ Unrecognized or invalid UI action.")
	}

	switch parsed.Kind {
	case models.UIActionSandboxApprove, models.UIActionSandboxDeny, models.UIActionSandboxDestroy, models.UIActionSandboxKeep:
		if !isDM {
			return true, g.SendMessage(msg.ChatID, "⚠️ Sandbox actions are only available in direct messages.")
		}
		if err := g.VerifyDMOwner(ctx, msg.ChatID, msg.UserID); err != nil {
			return true, g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ %v", err))
		}
		switch parsed.Kind {
		case models.UIActionSandboxApprove:
			return true, g.handleSandboxCommandWithTarget(ctx, msg, "/sandbox approve", senderName, parsed.TargetID)
		case models.UIActionSandboxDeny:
			return true, g.handleSandboxCommandWithTarget(ctx, msg, "/sandbox deny", senderName, parsed.TargetID)
		case models.UIActionSandboxDestroy:
			return true, g.handleSandboxCommandWithTarget(ctx, msg, "/sandbox destroy", senderName, parsed.TargetID)
		case models.UIActionSandboxKeep:
			g.mu.Lock()
			botID := g.botUserID
			botUser := g.botUser
			if botUser.ID == "" && botID != "" {
				if u, found := g.userCache.Get(botID); found {
					botUser = u
				}
			}
			sm := g.sandboxManager
			g.mu.Unlock()

			reply := "ℹ️ This sandbox prompt is outdated and no longer matches an active sandbox."
			if sm != nil {
				if sbx, exists := sm.GetStatus(msg.UserID); exists && sbx != nil && sbx.ID == parsed.TargetID && sbx.Status == sandbox.StatusRunning {
					rem := time.Until(sbx.ExpiresAt).Round(time.Minute)
					if rem < 0 {
						rem = 0
					}
					reply = fmt.Sprintf("👍 Keeping your sandbox running (auto-destroys in %s). Use `/sandbox destroy` whenever you are done.", rem)
				}
			}
			return true, g.sendCommandReply(msg.ChatID, botID, botUser.GetDisplayName(), reply, nil, true)
		}

	case models.UIActionMemoryApprove:
		return true, g.handleMemoryCommand(ctx, msg, "/memory approve "+parsed.TargetID, senderName, isDM)
	case models.UIActionMemoryReject:
		return true, g.handleMemoryCommand(ctx, msg, "/memory reject "+parsed.TargetID, senderName, isDM)
	case models.UIActionSkillApprove:
		return true, g.handleSkillCommand(ctx, msg, "/skill approve "+parsed.TargetID, senderName, isDM)
	case models.UIActionSkillReject:
		return true, g.handleSkillCommand(ctx, msg, "/skill reject "+parsed.TargetID, senderName, isDM)

	case models.UIActionScheduleApprove, models.UIActionScheduleDeny:
		if isDM {
			if err := g.VerifyDMOwner(ctx, msg.ChatID, msg.UserID); err != nil {
				return true, g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ %v", err))
			}
		}
		schedStore, err := g.getSchedulerStore(ctx, msg.ChatID, isDM)
		if err != nil {
			return true, g.SendMessage(msg.ChatID, "⚠️ Scheduler storage is currently unavailable.")
		}
		sched, err := schedStore.GetSchedule(ctx, parsed.TargetID)
		if err != nil || sched == nil || sched.ChatID != msg.ChatID {
			if err == nil || errors.Is(err, scheduler.ErrScheduleNotFound) || sched == nil || sched.ChatID != msg.ChatID {
				return true, g.SendMessage(msg.ChatID, "⚠️ Schedule not found.")
			}
			return true, g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Failed to retrieve schedule: %v", err))
		}
		subcmd := "approve"
		if parsed.Kind == models.UIActionScheduleDeny {
			subcmd = "deny"
		}
		return true, g.handleScheduleCommand(ctx, msg, "/schedule "+subcmd+" "+sched.Name, senderName, isDM)
	}

	return true, g.SendMessage(msg.ChatID, "⚠️ Unrecognized or invalid UI action.")
}

type sessionToolExecutor struct {
	registry *tools.Registry
	session  tools.ChatSessionContext
}

func (e *sessionToolExecutor) Execute(ctx context.Context, name string, argsJSON string) (string, error) {
	return e.registry.Execute(ctx, name, argsJSON)
}

func (e *sessionToolExecutor) ShouldStopLoop() (string, bool) {
	if e.session.Actions == nil {
		return "", false
	}
	return agentapi.ExtractInteractivePromptMessage(e.session.Actions.All())
}
