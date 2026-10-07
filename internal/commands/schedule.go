package commands

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"bob/internal/scheduler"
)

// ScheduleHandler handles /schedule slash commands.
type ScheduleHandler struct {
	Store        *scheduler.Store
	UserResolver UserResolver
	BotID        string
}

// NewScheduleHandler creates a new ScheduleHandler.
func NewScheduleHandler(store *scheduler.Store, resolver UserResolver, botID string) *ScheduleHandler {
	return &ScheduleHandler{
		Store:        store,
		UserResolver: resolver,
		BotID:        botID,
	}
}

// Handle executes the /schedule command specified by req.
func (h *ScheduleHandler) Handle(ctx context.Context, req Request) (Result, error) {
	if h.Store == nil {
		return Result{Reply: "⚠️ Scheduler storage is currently unavailable."}, nil
	}

	subcmd := strings.ToLower(req.Subcommand)
	switch subcmd {
	case "list":
		return h.handleList(ctx, req)
	case "info":
		return h.handleInfo(ctx, req)
	case "approve":
		return h.handleApprove(ctx, req)
	case "deny":
		return h.handleDeny(ctx, req)
	case "cancel":
		return h.handleCancel(ctx, req)
	case "pause":
		return h.handlePause(ctx, req)
	case "resume":
		return h.handleResume(ctx, req)
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
		return Result{Reply: helpText}, nil
	}
}

func (h *ScheduleHandler) handleList(ctx context.Context, req Request) (Result, error) {
	activeOnly := len(req.Args) == 0 || strings.ToLower(req.Args[0]) != "all"

	schedules, err := h.Store.ListSchedules(ctx, req.Session.ScopeID, activeOnly)
	if err != nil {
		return Result{Reply: fmt.Sprintf("⚠️ Failed to list schedules: %v", err)}, nil
	}

	if len(schedules) == 0 {
		if activeOnly {
			return Result{Reply: "ℹ️ No active schedules found for this chat. Use `/schedule list all` to see past schedules."}, nil
		}
		return Result{Reply: "ℹ️ No schedules found for this chat."}, nil
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
		grant, err := h.Store.GetGrant(ctx, s.ID)
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

	return Result{Reply: b.String()}, nil
}

func (h *ScheduleHandler) handleInfo(ctx context.Context, req Request) (Result, error) {
	if len(req.Args) == 0 {
		return Result{Reply: "⚠️ Please specify a schedule name: `/schedule info <name>`"}, nil
	}
	name := strings.TrimSpace(req.Args[0])

	sched, err := h.Store.GetScheduleByName(ctx, req.Session.ScopeID, name)
	if err != nil {
		if errors.Is(err, scheduler.ErrScheduleNotFound) {
			return Result{Reply: fmt.Sprintf("⚠️ Schedule %q not found.", name)}, nil
		}
		return Result{Reply: fmt.Sprintf("⚠️ Failed to retrieve schedule: %v", err)}, nil
	}

	var b strings.Builder
	fmt.Fprintf(&b, "📋 **Schedule: `%s`**\n\n", sched.Name)
	fmt.Fprintf(&b, "- **Status:** %s\n", sched.Status)
	fmt.Fprintf(&b, "- **ID:** `%s`\n", sched.ID)
	if sched.UserID != "" {
		creatorName := ""
		if h.UserResolver != nil {
			creatorName = h.UserResolver.GetDisplayName(ctx, sched.UserID)
		}
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
	grant, err := h.Store.GetGrant(ctx, sched.ID)
	if err != nil || grant == nil {
		b.WriteString("- Standard execution (no elevated permissions requested)\n")
	} else {
		hashSnippet := grant.ParamsHash
		if len(hashSnippet) > 16 {
			hashSnippet = hashSnippet[:16] + "..."
		}
		fmt.Fprintf(&b, "- **Params Hash:** `%s`\n", hashSnippet)
		if grant.GrantedBy != "" {
			granterName := ""
			if h.UserResolver != nil {
				granterName = h.UserResolver.GetDisplayName(ctx, grant.GrantedBy)
			}
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

	return Result{Reply: b.String()}, nil
}

func (h *ScheduleHandler) handleApprove(ctx context.Context, req Request) (Result, error) {
	if h.BotID != "" && req.Actor.ID == h.BotID {
		return Result{Reply: "⚠️ Agents cannot self-grant or approve schedules."}, nil
	}
	if len(req.Args) == 0 {
		return Result{Reply: "⚠️ Please specify a schedule name to approve: `/schedule approve <name>`"}, nil
	}
	name := strings.TrimSpace(req.Args[0])

	sched, err := h.Store.GetScheduleByName(ctx, req.Session.ScopeID, name)
	if err != nil {
		if errors.Is(err, scheduler.ErrScheduleNotFound) {
			return Result{Reply: fmt.Sprintf("⚠️ Schedule %q not found.", name)}, nil
		}
		return Result{Reply: fmt.Sprintf("⚠️ Failed to retrieve schedule: %v", err)}, nil
	}

	if sched.UserID != req.Actor.ID {
		return Result{Reply: fmt.Sprintf("⚠️ Permission denied: schedule %q was created by another user.", name)}, nil
	}

	if sched.Status != scheduler.ScheduleStatusPendingApproval {
		return Result{Reply: fmt.Sprintf("ℹ️ Schedule %q is not awaiting approval (current status: %s).", name, sched.Status)}, nil
	}

	grant, err := h.Store.GetGrant(ctx, sched.ID)
	if err != nil && !errors.Is(err, scheduler.ErrGrantNotFound) {
		return Result{Reply: fmt.Sprintf("⚠️ Failed to check schedule grant: %v", err)}, nil
	}

	if grant != nil {
		if !grant.Verify() {
			return Result{Reply: "❌ Security alert: Permission request verification failed (hash mismatch). Cannot approve."}, nil
		}
		if !req.IsDirect {
			var env scheduler.PermissionEnvelope
			if jsonErr := json.Unmarshal([]byte(grant.PermissionRequestJSON), &env); jsonErr == nil && env.Sandbox != nil {
				return Result{Reply: "❌ Elevated sandbox tasks cannot be approved in Townhall. Sandbox execution is strictly DM-only."}, nil
			}
		}
	}

	approvedSched, _, err := h.Store.ApproveSchedule(ctx, req.Session.ScopeID, name, req.Actor.ID, 0)
	if err != nil {
		return Result{Reply: fmt.Sprintf("⚠️ Failed to approve schedule: %v", err)}, nil
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

	return Result{Reply: reply, RecordAssistantEntry: true}, nil
}

func (h *ScheduleHandler) handleDeny(ctx context.Context, req Request) (Result, error) {
	if h.BotID != "" && req.Actor.ID == h.BotID {
		return Result{Reply: "⚠️ Agents cannot deny schedules."}, nil
	}
	if len(req.Args) == 0 {
		return Result{Reply: "⚠️ Please specify a schedule name to deny: `/schedule deny <name>`"}, nil
	}
	name := strings.TrimSpace(req.Args[0])

	sched, err := h.Store.GetScheduleByName(ctx, req.Session.ScopeID, name)
	if err != nil {
		if errors.Is(err, scheduler.ErrScheduleNotFound) {
			return Result{Reply: fmt.Sprintf("⚠️ Schedule %q not found.", name)}, nil
		}
		return Result{Reply: fmt.Sprintf("⚠️ Failed to retrieve schedule: %v", err)}, nil
	}

	if sched.UserID != req.Actor.ID {
		return Result{Reply: fmt.Sprintf("⚠️ Permission denied: schedule %q was created by another user.", name)}, nil
	}

	_, err = h.Store.DenySchedule(ctx, req.Session.ScopeID, name)
	if err != nil {
		return Result{Reply: fmt.Sprintf("⚠️ Failed to deny schedule: %v", err)}, nil
	}

	reply := fmt.Sprintf("❌ **Schedule request denied.** Task `%s` has been rejected and associated permissions discarded.", name)
	return Result{Reply: reply, RecordAssistantEntry: true}, nil
}

func (h *ScheduleHandler) handleCancel(ctx context.Context, req Request) (Result, error) {
	if h.BotID != "" && req.Actor.ID == h.BotID {
		return Result{Reply: "⚠️ Agents cannot cancel schedules directly via slash command."}, nil
	}
	if len(req.Args) == 0 {
		return Result{Reply: "⚠️ Please specify a schedule name to cancel: `/schedule cancel <name>`"}, nil
	}
	name := strings.TrimSpace(req.Args[0])

	sched, err := h.Store.GetScheduleByName(ctx, req.Session.ScopeID, name)
	if err != nil {
		if errors.Is(err, scheduler.ErrScheduleNotFound) {
			return Result{Reply: fmt.Sprintf("⚠️ Schedule %q not found.", name)}, nil
		}
		return Result{Reply: fmt.Sprintf("⚠️ Failed to retrieve schedule: %v", err)}, nil
	}

	if sched.UserID != req.Actor.ID {
		return Result{Reply: fmt.Sprintf("⚠️ Permission denied: schedule %q was created by another user.", name)}, nil
	}

	_, err = h.Store.CancelSchedule(ctx, req.Session.ScopeID, name)
	if err != nil {
		return Result{Reply: fmt.Sprintf("⚠️ Failed to cancel schedule: %v", err)}, nil
	}

	reply := fmt.Sprintf("🛑 **Schedule cancelled.** Task `%s` has been terminated and any active permissions revoked.", name)
	return Result{Reply: reply, RecordAssistantEntry: true}, nil
}

func (h *ScheduleHandler) handlePause(ctx context.Context, req Request) (Result, error) {
	if h.BotID != "" && req.Actor.ID == h.BotID {
		return Result{Reply: "⚠️ Agents cannot pause schedules."}, nil
	}
	if len(req.Args) == 0 {
		return Result{Reply: "⚠️ Please specify a schedule name to pause: `/schedule pause <name>`"}, nil
	}
	name := strings.TrimSpace(req.Args[0])

	sched, err := h.Store.GetScheduleByName(ctx, req.Session.ScopeID, name)
	if err != nil {
		if errors.Is(err, scheduler.ErrScheduleNotFound) {
			return Result{Reply: fmt.Sprintf("⚠️ Schedule %q not found.", name)}, nil
		}
		return Result{Reply: fmt.Sprintf("⚠️ Failed to retrieve schedule: %v", err)}, nil
	}

	if sched.UserID != req.Actor.ID {
		return Result{Reply: fmt.Sprintf("⚠️ Permission denied: schedule %q was created by another user.", name)}, nil
	}

	_, err = h.Store.PauseSchedule(ctx, req.Session.ScopeID, name)
	if err != nil {
		return Result{Reply: fmt.Sprintf("⚠️ Failed to pause schedule: %v", err)}, nil
	}

	reply := fmt.Sprintf("⏸️ **Schedule paused.** Task `%s` is paused and will not run until resumed.", name)
	return Result{Reply: reply, RecordAssistantEntry: true}, nil
}

func (h *ScheduleHandler) handleResume(ctx context.Context, req Request) (Result, error) {
	if h.BotID != "" && req.Actor.ID == h.BotID {
		return Result{Reply: "⚠️ Agents cannot resume schedules."}, nil
	}
	if len(req.Args) == 0 {
		return Result{Reply: "⚠️ Please specify a schedule name to resume: `/schedule resume <name>`"}, nil
	}
	name := strings.TrimSpace(req.Args[0])

	sched, err := h.Store.GetScheduleByName(ctx, req.Session.ScopeID, name)
	if err != nil {
		if errors.Is(err, scheduler.ErrScheduleNotFound) {
			return Result{Reply: fmt.Sprintf("⚠️ Schedule %q not found.", name)}, nil
		}
		return Result{Reply: fmt.Sprintf("⚠️ Failed to retrieve schedule: %v", err)}, nil
	}

	if sched.UserID != req.Actor.ID {
		return Result{Reply: fmt.Sprintf("⚠️ Permission denied: schedule %q was created by another user.", name)}, nil
	}

	_, err = h.Store.ResumeSchedule(ctx, req.Session.ScopeID, name)
	if err != nil {
		return Result{Reply: fmt.Sprintf("⚠️ Failed to resume schedule: %v", err)}, nil
	}

	reply := fmt.Sprintf("▶️ **Schedule resumed.** Task `%s` is now active.", name)
	return Result{Reply: reply, RecordAssistantEntry: true}, nil
}
