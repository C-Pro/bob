package gateway

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"bob/internal/chatcontext"
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
	if len(parts) > 1 {
		subcmd = strings.ToLower(parts[1])
	}

	schedStore, err := g.getSchedulerStore(ctx, msg.ChatID, isDM)
	if err != nil {
		return g.SendMessage(msg.ChatID, "⚠️ Scheduler storage is currently unavailable.")
	}

	switch subcmd {
	case "list":
		return g.handleScheduleList(ctx, msg, schedStore, parts)

	case "info":
		return g.handleScheduleInfo(ctx, msg, schedStore, parts)

	case "approve":
		return g.handleScheduleApprove(ctx, msg, schedStore, parts, botID, botUser, isDM)

	case "deny":
		return g.handleScheduleDeny(ctx, msg, schedStore, parts, botID, botUser, isDM)

	case "cancel":
		return g.handleScheduleCancel(ctx, msg, schedStore, parts, botID, botUser, isDM)

	case "pause":
		return g.handleSchedulePause(ctx, msg, schedStore, parts, botID, botUser, isDM)

	case "resume":
		return g.handleScheduleResume(ctx, msg, schedStore, parts, botID, botUser, isDM)

	default:
		helpText := "📅 **Schedule Commands:**\n" +
			"• `/schedule list` — List all active schedules in this chat\n" +
			"• `/schedule list all` — List all schedules including past/completed\n" +
			"• `/schedule info <name>` — View schedule details, run metrics, and permission grant\n" +
			"• `/schedule approve <name>` — Approve a pending schedule request\n" +
			"• `/schedule deny <name>` — Reject a pending schedule request\n" +
			"• `/schedule cancel <name>` — Cancel an active or pending schedule\n" +
			"• `/schedule pause <name>` — Temporarily pause an active schedule\n" +
			"• `/schedule resume <name>` — Resume a paused schedule"
		return g.SendMessage(msg.ChatID, helpText)
	}
}

func (g *Gateway) handleScheduleList(ctx context.Context, msg models.Message, store *scheduler.Store, parts []string) error {
	activeOnly := len(parts) <= 2 || strings.ToLower(parts[2]) != "all"

	schedules, err := store.ListSchedules(ctx, msg.ChatID, activeOnly)
	if err != nil {
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Failed to list schedules: %v", err))
	}

	if len(schedules) == 0 {
		if activeOnly {
			return g.SendMessage(msg.ChatID, "ℹ️ No active schedules found for this chat. Use `/schedule list all` to see past schedules.")
		}
		return g.SendMessage(msg.ChatID, "ℹ️ No schedules found for this chat.")
	}

	var b strings.Builder
	b.WriteString("📅 **Task Schedules**\n\n")
	b.WriteString("| Name | Type | Next Run | Status | Permissions |\n")
	b.WriteString("| :--- | :--- | :--- | :--- | :--- |\n")

	for _, s := range schedules {
		typeStr := string(s.ScheduleType)
		switch s.ScheduleType {
		case scheduler.ScheduleTypeCron:
			typeStr = fmt.Sprintf("CRON (`%s`)", s.CronExpr)
		case scheduler.ScheduleTypeInterval:
			typeStr = fmt.Sprintf("INTERVAL (%ds)", s.IntervalSeconds)
		}

		nextRunStr := "—"
		if !s.Status.IsTerminal() && s.NextRunAt > 0 {
			nextRunStr = time.Unix(s.NextRunAt, 0).UTC().Format("2006-01-02 15:04:05 MST")
		}

		permStr := "None"
		grant, err := store.GetGrant(ctx, s.ID)
		if err == nil && grant != nil {
			var env scheduler.PermissionEnvelope
			if jsonErr := json.Unmarshal([]byte(grant.PermissionRequestJSON), &env); jsonErr == nil && env.Sandbox != nil {
				driver := env.Sandbox.Driver
				if driver == "" {
					driver = "sandbox"
				}
				permStr = fmt.Sprintf("Sandbox (%s)", driver)
			} else {
				permStr = "Elevated"
			}
		}

		fmt.Fprintf(&b, "| `%s` | %s | %s | %s | %s |\n", s.Name, typeStr, nextRunStr, s.Status, permStr)
	}

	return g.SendMessage(msg.ChatID, b.String())
}

func (g *Gateway) handleScheduleInfo(ctx context.Context, msg models.Message, store *scheduler.Store, parts []string) error {
	if len(parts) < 3 {
		return g.SendMessage(msg.ChatID, "⚠️ Please specify a schedule name: `/schedule info <name>`")
	}
	name := strings.TrimSpace(parts[2])
	sched, err := store.GetScheduleByName(ctx, msg.ChatID, name)
	if err != nil {
		if errors.Is(err, scheduler.ErrScheduleNotFound) {
			return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Schedule %q not found.", name))
		}
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Failed to retrieve schedule: %v", err))
	}

	var b strings.Builder
	fmt.Fprintf(&b, "📋 **Schedule: `%s`**\n\n", sched.Name)
	fmt.Fprintf(&b, "- **Status:** %s\n", sched.Status)
	fmt.Fprintf(&b, "- **ID:** `%s`\n", sched.ID)
	if sched.UserID != "" {
		creatorName := g.userCache.GetDisplayName(sched.UserID)
		if creatorName == "" {
			creatorName = sched.UserID
		}
		fmt.Fprintf(&b, "- **Created By:** %s\n", creatorName)
	}

	switch sched.ScheduleType {
	case scheduler.ScheduleTypeCron:
		fmt.Fprintf(&b, "- **Type:** CRON (`%s`)\n", sched.CronExpr)
	case scheduler.ScheduleTypeInterval:
		fmt.Fprintf(&b, "- **Type:** INTERVAL (every %d seconds)\n", sched.IntervalSeconds)
	case scheduler.ScheduleTypeOnce:
		fmt.Fprintf(&b, "- **Type:** ONCE (delayed)\n")
	default:
		fmt.Fprintf(&b, "- **Type:** %s\n", sched.ScheduleType)
	}

	if sched.NextRunAt > 0 && !sched.Status.IsTerminal() {
		fmt.Fprintf(&b, "- **Next Run:** %s\n", time.Unix(sched.NextRunAt, 0).UTC().Format("2006-01-02 15:04:05 MST"))
	}
	if sched.LastRunAt != nil {
		fmt.Fprintf(&b, "- **Last Run:** %s (Status: %s)\n",
			time.Unix(*sched.LastRunAt, 0).UTC().Format("2006-01-02 15:04:05 MST"),
			sched.LastStatus)
	}
	fmt.Fprintf(&b, "- **Runs:** %d", sched.RunCount)
	if sched.MaxRuns > 0 {
		fmt.Fprintf(&b, " / %d max", sched.MaxRuns)
	}
	if sched.MissedCount > 0 {
		fmt.Fprintf(&b, " (Missed: %d)", sched.MissedCount)
	}
	b.WriteString("\n")
	fmt.Fprintf(&b, "- **Execution Limits:** Timeout %ds, Max Turns %d\n", sched.RunTimeoutSeconds, sched.MaxTurns)
	fmt.Fprintf(&b, "- **Instruction:**\n> %s\n\n", strings.ReplaceAll(sched.Instruction, "\n", "\n> "))

	// Permissions / Grant breakdown
	b.WriteString("🔐 **Permissions & Grant:**\n")
	grant, err := store.GetGrant(ctx, sched.ID)
	if err != nil || grant == nil {
		b.WriteString("- Standard execution (no elevated permissions requested)\n")
	} else {
		hashSnippet := grant.ParamsHash
		if len(hashSnippet) > 16 {
			hashSnippet = hashSnippet[:16] + "..."
		}
		fmt.Fprintf(&b, "- **Params Hash:** `%s`\n", hashSnippet)
		if grant.GrantedBy != "" {
			granterName := g.userCache.GetDisplayName(grant.GrantedBy)
			if granterName == "" {
				granterName = grant.GrantedBy
			}
			fmt.Fprintf(&b, "- **Approved By:** %s at %s\n",
				granterName, time.Unix(grant.GrantedAt, 0).UTC().Format("2006-01-02 15:04:05 MST"))
		} else {
			b.WriteString("- **Approval Status:** ⏳ Pending Approval\n")
		}
		if grant.ValidUntil > 0 {
			fmt.Fprintf(&b, "- **Valid Until:** %s\n", time.Unix(grant.ValidUntil, 0).UTC().Format("2006-01-02 15:04:05 MST"))
		}

		var env scheduler.PermissionEnvelope
		if jsonErr := json.Unmarshal([]byte(grant.PermissionRequestJSON), &env); jsonErr == nil && env.Sandbox != nil {
			sbx := env.Sandbox
			fmt.Fprintf(&b, "- **Sandbox Driver:** %s\n", sbx.Driver)
			if sbx.Image != "" {
				fmt.Fprintf(&b, "- **Sandbox Image:** %s\n", sbx.Image)
			}
			if sbx.Network != "" {
				fmt.Fprintf(&b, "- **Network Mode:** %s\n", sbx.Network)
			}
			if len(sbx.Domains) > 0 {
				fmt.Fprintf(&b, "- **Allowed Domains:** %s\n", strings.Join(sbx.Domains, ", "))
			}
			if len(sbx.Mounts) > 0 {
				b.WriteString("- **Mounts:**\n")
				for _, m := range sbx.Mounts {
					ro := "read-write"
					if m.ReadOnly {
						ro = "read-only"
					}
					fmt.Fprintf(&b, "  • `%s` (%s)\n", m.Path, ro)
				}
			}
		}
	}

	return g.SendMessage(msg.ChatID, b.String())
}

func (g *Gateway) handleScheduleApprove(ctx context.Context, msg models.Message, store *scheduler.Store, parts []string, botID string, botUser models.User, isDM bool) error {
	if botID != "" && msg.UserID == botID {
		return g.SendMessage(msg.ChatID, "⚠️ Agents cannot self-grant or approve schedules.")
	}
	if len(parts) < 3 {
		return g.SendMessage(msg.ChatID, "⚠️ Please specify a schedule name to approve: `/schedule approve <name>`")
	}
	name := strings.TrimSpace(parts[2])

	sched, err := store.GetScheduleByName(ctx, msg.ChatID, name)
	if err != nil {
		if errors.Is(err, scheduler.ErrScheduleNotFound) {
			return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Schedule %q not found.", name))
		}
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Failed to retrieve schedule: %v", err))
	}

	if sched.UserID != msg.UserID {
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Permission denied: schedule %q was created by another user.", name))
	}

	if sched.Status != scheduler.ScheduleStatusPendingApproval {
		return g.SendMessage(msg.ChatID, fmt.Sprintf("ℹ️ Schedule %q is not awaiting approval (current status: %s).", name, sched.Status))
	}

	grant, err := store.GetGrant(ctx, sched.ID)
	if err != nil && !errors.Is(err, scheduler.ErrGrantNotFound) {
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Failed to check schedule grant: %v", err))
	}

	if grant != nil {
		if !grant.Verify() {
			return g.SendMessage(msg.ChatID, "❌ Security alert: Permission request verification failed (hash mismatch). Cannot approve.")
		}
		if !isDM {
			var env scheduler.PermissionEnvelope
			if jsonErr := json.Unmarshal([]byte(grant.PermissionRequestJSON), &env); jsonErr == nil && env.Sandbox != nil {
				return g.SendMessage(msg.ChatID, "❌ Elevated sandbox tasks cannot be approved in Townhall. Sandbox execution is strictly DM-only.")
			}
		}
	}

	approvedSched, _, err := store.ApproveSchedule(ctx, msg.ChatID, name, msg.UserID, 0)
	if err != nil {
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Failed to approve schedule: %v", err))
	}

	nextTime := time.Unix(approvedSched.NextRunAt, 0).UTC()
	var nextRunStr string
	if nextTime.Before(time.Now().UTC()) {
		nextRunStr = "immediately (due now)"
	} else {
		nextRunStr = nextTime.Format("2006-01-02 15:04:05 MST")
	}

	reply := fmt.Sprintf("✅ **Schedule approved and activated!**\nTask `%s` will run next at %s.", name, nextRunStr)
	if grant != nil {
		var env scheduler.PermissionEnvelope
		if jsonErr := json.Unmarshal([]byte(grant.PermissionRequestJSON), &env); jsonErr == nil && env.Sandbox != nil {
			reply += fmt.Sprintf("\n- **Authorized Sandbox:** `%s` driver", env.Sandbox.Driver)
			if env.Sandbox.Image != "" {
				reply += fmt.Sprintf(" (image: `%s`)", env.Sandbox.Image)
			}
			if env.Sandbox.Network != "" {
				reply += fmt.Sprintf(", network: `%s`", env.Sandbox.Network)
			}
			if len(env.Sandbox.Domains) > 0 {
				reply += fmt.Sprintf(", domains: [%s]", strings.Join(env.Sandbox.Domains, ", "))
			}
			if len(env.Sandbox.Mounts) > 0 {
				reply += fmt.Sprintf(", mounts: %d", len(env.Sandbox.Mounts))
			}
		}
	}
	reply += fmt.Sprintf("\n- **Limits:** timeout %ds, max turns %d", approvedSched.RunTimeoutSeconds, approvedSched.MaxTurns)
	reply += fmt.Sprintf("\n- **Instruction:** %s", approvedSched.Instruction)

	g.contextManager.Push(msg.ChatID, chatcontext.Entry{
		Role:       "assistant",
		SenderID:   botID,
		SenderName: botUser.GetDisplayName(),
		Content:    reply,
		Timestamp:  time.Now().Unix(),
	})

	return g.SendMessage(msg.ChatID, reply)
}

func (g *Gateway) handleScheduleDeny(ctx context.Context, msg models.Message, store *scheduler.Store, parts []string, botID string, botUser models.User, isDM bool) error {
	if botID != "" && msg.UserID == botID {
		return g.SendMessage(msg.ChatID, "⚠️ Agents cannot deny schedules.")
	}
	if len(parts) < 3 {
		return g.SendMessage(msg.ChatID, "⚠️ Please specify a schedule name to deny: `/schedule deny <name>`")
	}
	name := strings.TrimSpace(parts[2])

	sched, err := store.GetScheduleByName(ctx, msg.ChatID, name)
	if err != nil {
		if errors.Is(err, scheduler.ErrScheduleNotFound) {
			return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Schedule %q not found.", name))
		}
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Failed to retrieve schedule: %v", err))
	}

	if sched.UserID != msg.UserID {
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Permission denied: schedule %q was created by another user.", name))
	}

	_, err = store.DenySchedule(ctx, msg.ChatID, name)
	if err != nil {
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Failed to deny schedule: %v", err))
	}

	reply := fmt.Sprintf("❌ **Schedule request denied.** Task `%s` has been rejected and associated permissions discarded.", name)
	g.contextManager.Push(msg.ChatID, chatcontext.Entry{
		Role:       "assistant",
		SenderID:   botID,
		SenderName: botUser.GetDisplayName(),
		Content:    reply,
		Timestamp:  time.Now().Unix(),
	})
	return g.SendMessage(msg.ChatID, reply)
}

func (g *Gateway) handleScheduleCancel(ctx context.Context, msg models.Message, store *scheduler.Store, parts []string, botID string, botUser models.User, isDM bool) error {
	if botID != "" && msg.UserID == botID {
		return g.SendMessage(msg.ChatID, "⚠️ Agents cannot cancel schedules directly via slash command.")
	}
	if len(parts) < 3 {
		return g.SendMessage(msg.ChatID, "⚠️ Please specify a schedule name to cancel: `/schedule cancel <name>`")
	}
	name := strings.TrimSpace(parts[2])

	sched, err := store.GetScheduleByName(ctx, msg.ChatID, name)
	if err != nil {
		if errors.Is(err, scheduler.ErrScheduleNotFound) {
			return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Schedule %q not found.", name))
		}
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Failed to retrieve schedule: %v", err))
	}

	if sched.UserID != msg.UserID {
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Permission denied: schedule %q was created by another user.", name))
	}

	_, err = store.CancelSchedule(ctx, msg.ChatID, name)
	if err != nil {
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Failed to cancel schedule: %v", err))
	}

	reply := fmt.Sprintf("🛑 **Schedule cancelled.** Task `%s` has been terminated and any active permissions revoked.", name)
	g.contextManager.Push(msg.ChatID, chatcontext.Entry{
		Role:       "assistant",
		SenderID:   botID,
		SenderName: botUser.GetDisplayName(),
		Content:    reply,
		Timestamp:  time.Now().Unix(),
	})
	return g.SendMessage(msg.ChatID, reply)
}

func (g *Gateway) handleSchedulePause(ctx context.Context, msg models.Message, store *scheduler.Store, parts []string, botID string, botUser models.User, isDM bool) error {
	if botID != "" && msg.UserID == botID {
		return g.SendMessage(msg.ChatID, "⚠️ Agents cannot pause schedules.")
	}
	if len(parts) < 3 {
		return g.SendMessage(msg.ChatID, "⚠️ Please specify a schedule name to pause: `/schedule pause <name>`")
	}
	name := strings.TrimSpace(parts[2])

	sched, err := store.GetScheduleByName(ctx, msg.ChatID, name)
	if err != nil {
		if errors.Is(err, scheduler.ErrScheduleNotFound) {
			return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Schedule %q not found.", name))
		}
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Failed to retrieve schedule: %v", err))
	}

	if sched.UserID != msg.UserID {
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Permission denied: schedule %q was created by another user.", name))
	}

	_, err = store.PauseSchedule(ctx, msg.ChatID, name)
	if err != nil {
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Failed to pause schedule: %v", err))
	}

	reply := fmt.Sprintf("⏸️ **Schedule paused.** Task `%s` is paused and will not run until resumed.", name)
	g.contextManager.Push(msg.ChatID, chatcontext.Entry{
		Role:       "assistant",
		SenderID:   botID,
		SenderName: botUser.GetDisplayName(),
		Content:    reply,
		Timestamp:  time.Now().Unix(),
	})
	return g.SendMessage(msg.ChatID, reply)
}

func (g *Gateway) handleScheduleResume(ctx context.Context, msg models.Message, store *scheduler.Store, parts []string, botID string, botUser models.User, isDM bool) error {
	if botID != "" && msg.UserID == botID {
		return g.SendMessage(msg.ChatID, "⚠️ Agents cannot resume schedules.")
	}
	if len(parts) < 3 {
		return g.SendMessage(msg.ChatID, "⚠️ Please specify a schedule name to resume: `/schedule resume <name>`")
	}
	name := strings.TrimSpace(parts[2])

	sched, err := store.GetScheduleByName(ctx, msg.ChatID, name)
	if err != nil {
		if errors.Is(err, scheduler.ErrScheduleNotFound) {
			return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Schedule %q not found.", name))
		}
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Failed to retrieve schedule: %v", err))
	}

	if sched.UserID != msg.UserID {
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Permission denied: schedule %q was created by another user.", name))
	}

	_, err = store.ResumeSchedule(ctx, msg.ChatID, name)
	if err != nil {
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Failed to resume schedule: %v", err))
	}

	reply := fmt.Sprintf("▶️ **Schedule resumed.** Task `%s` is now active.", name)
	g.contextManager.Push(msg.ChatID, chatcontext.Entry{
		Role:       "assistant",
		SenderID:   botID,
		SenderName: botUser.GetDisplayName(),
		Content:    reply,
		Timestamp:  time.Now().Unix(),
	})
	return g.SendMessage(msg.ChatID, reply)
}
