package commands

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"time"

	"bob/internal/knowledge"
)

// SkillHandler handles /skill slash commands.
type SkillHandler struct {
	Store      *knowledge.Store
	Indexer    *knowledge.Indexer
	Authorizer Authorizer
	BotID      string
}

// NewSkillHandler creates a new SkillHandler.
func NewSkillHandler(store *knowledge.Store, indexer *knowledge.Indexer, auth Authorizer, botID string) *SkillHandler {
	return &SkillHandler{
		Store:      store,
		Indexer:    indexer,
		Authorizer: auth,
		BotID:      botID,
	}
}

// Handle executes the /skill command specified by req.
func (h *SkillHandler) Handle(ctx context.Context, req Request) (Result, error) {
	if h.Authorizer == nil {
		return Result{Reply: "⚠️ Authorization failed: authorizer not configured."}, nil
	}
	if err := h.Authorizer.Authorize(ctx, req.Session, req.Actor, "skill", req.Subcommand); err != nil {
		return Result{Reply: fmt.Sprintf("⚠️ %v", err)}, nil
	}

	if h.BotID != "" && req.Actor.ID == h.BotID {
		return Result{IsIgnored: true}, nil
	}

	if h.Store == nil {
		return Result{Reply: "⚠️ Knowledge storage is unavailable: store not initialized."}, nil
	}

	subcmd := strings.ToLower(req.Subcommand)
	switch subcmd {
	case "list":
		return h.handleList(ctx, req)
	case "show":
		return h.handleShow(ctx, req)
	case "approve":
		return h.handleApprove(ctx, req)
	case "reject":
		return h.handleReject(ctx, req)
	case "disable":
		return h.handleDisable(ctx, req)
	case "enable":
		return h.handleEnable(ctx, req)
	case "archive":
		return h.handleArchive(ctx, req)
	case "delete":
		return h.handleDelete(ctx, req)
	default:
		helpText := "🛠️ **Skill Commands:**\n" +
			"• `/skill list [status]` — List skills (active, pending, disabled, rejected, archived, all)\n" +
			"• `/skill show <id>` — View full skill details and instructions\n" +
			"• `/skill approve <version-id>` — Approve a proposed skill version\n" +
			"• `/skill reject <version-id>` — Reject a proposed skill version\n" +
			"• `/skill disable <item-id>` — Temporarily disable an active skill\n" +
			"• `/skill enable <item-id>` — Re-enable a disabled skill\n" +
			"• `/skill archive <item-id>` — Archive a skill from active use\n" +
			"• `/skill delete <item-id>` — Permanently erase a skill"
		return Result{Reply: helpText}, nil
	}
}

func (h *SkillHandler) handleList(ctx context.Context, req Request) (Result, error) {
	statusFilter := "active"
	if len(req.Args) > 0 {
		statusFilter = strings.ToLower(req.Args[0])
	}

	validFilters := map[string]bool{
		"active":   true,
		"pending":  true,
		"disabled": true,
		"rejected": true,
		"archived": true,
		"all":      true,
	}
	if !validFilters[statusFilter] {
		return Result{Reply: fmt.Sprintf("⚠️ Invalid status filter %q. Valid skill filters: pending, active, disabled, rejected, archived, all.", statusFilter)}, nil
	}

	items, err := h.Store.ListItems(ctx, req.Session.ScopeID, knowledge.KindSkill, statusFilter)
	if err != nil {
		return Result{Reply: fmt.Sprintf("⚠️ Failed to list skills: %v", err)}, nil
	}

	if len(items) == 0 {
		return Result{Reply: fmt.Sprintf("ℹ️ No skills found with status %q.", statusFilter)}, nil
	}

	var b strings.Builder
	fmt.Fprintf(&b, "🛠️ **Skills** (%s)\n\n", statusFilter)
	b.WriteString("| ID | Name | Description | Rev | Status |\n")
	b.WriteString("|---|---|---|---|---|\n")

	for _, item := range items {
		vers, err := h.Store.ListVersions(ctx, item.ID)
		var targetVer *knowledge.SkillVersion
		if err == nil && len(vers) > 0 {
		findSkillVer:
			for i := len(vers) - 1; i >= 0; i-- {
				sv, ok := vers[i].(*knowledge.SkillVersion)
				if !ok {
					continue
				}
				switch statusFilter {
				case "pending":
					if sv.Status == knowledge.VersionStatusProposed {
						targetVer = sv
						break findSkillVer
					}
				case "rejected":
					if sv.Status == knowledge.VersionStatusRejected {
						targetVer = sv
						break findSkillVer
					}
				case "archived":
					if sv.Status == knowledge.VersionStatusArchived {
						targetVer = sv
						break findSkillVer
					}
				case "active":
					if sv.Status == knowledge.VersionStatusApproved && sv.ID == item.ActiveVersionID {
						targetVer = sv
						break findSkillVer
					}
				default:
					if item.ActiveVersionID != "" && sv.ID == item.ActiveVersionID {
						targetVer = sv
						break findSkillVer
					}
				}
			}
			if targetVer == nil {
				if item.ActiveVersionID != "" {
					for _, v := range vers {
						if sv, ok := v.(*knowledge.SkillVersion); ok && sv.ID == item.ActiveVersionID {
							targetVer = sv
							break
						}
					}
				}
				if targetVer == nil {
					if sv, ok := vers[len(vers)-1].(*knowledge.SkillVersion); ok {
						targetVer = sv
					}
				}
			}
		}

		name := "-"
		desc := "-"
		revStr := "-"
		statusStr := string(item.Status)

		if targetVer != nil {
			name = strings.ReplaceAll(targetVer.Name, "|", "\\|")
			desc = strings.ReplaceAll(knowledge.TruncateRunes(targetVer.Description, 40), "|", "\\|")
			desc = strings.ReplaceAll(desc, "\n", " ")
			revStr = fmt.Sprintf("@%d", targetVer.Revision)
			if statusFilter == "pending" || statusFilter == "rejected" || item.Status == knowledge.ItemStatusPending {
				statusStr = string(targetVer.Status)
			}
		}

		fmt.Fprintf(&b, "| `%s` | %s | %s | %s | %s |\n", item.ID, name, desc, revStr, statusStr)
	}

	return Result{Reply: b.String()}, nil
}

func (h *SkillHandler) handleShow(ctx context.Context, req Request) (Result, error) {
	if len(req.Args) == 0 {
		return Result{Reply: "⚠️ Please specify a skill ID. Usage: `/skill show <id>` (e.g. `/skill show skill_123` or `/skill show skill_123@1`)"}, nil
	}

	ref := strings.TrimSpace(req.Args[0])
	if !strings.HasPrefix(ref, knowledge.PrefixSkill) {
		return Result{Reply: fmt.Sprintf("⚠️ Invalid skill identifier %q: skill IDs must begin with %q.", ref, knowledge.PrefixSkill)}, nil
	}

	itemID, rev, hasRev, err := knowledge.ParseItemOrVersionRef(ref)
	if err != nil {
		return Result{Reply: fmt.Sprintf("⚠️ Invalid skill reference %q: %v", ref, err)}, nil
	}

	if hasRev {
		verID := knowledge.FormatVersionID(itemID, rev)
		ver, err := h.Store.GetSkillVersion(ctx, verID)
		if err != nil {
			if errors.Is(err, knowledge.ErrNotFound) {
				return Result{Reply: fmt.Sprintf("⚠️ Skill version `%s` not found.", verID)}, nil
			}
			return Result{Reply: fmt.Sprintf("⚠️ Failed to get skill version: %v", err)}, nil
		}
		if ver.Provenance.ChatID != req.Session.ScopeID || ver.Provenance.UserID != req.Actor.ID {
			return Result{Reply: fmt.Sprintf("⚠️ Skill version `%s` not found.", verID)}, nil
		}
		return Result{Reply: h.formatSkillVersionDetails(ver)}, nil
	}

	item, err := h.Store.GetItem(ctx, itemID)
	if err != nil {
		if errors.Is(err, knowledge.ErrNotFound) {
			return Result{Reply: fmt.Sprintf("⚠️ Skill item `%s` not found.", itemID)}, nil
		}
		return Result{Reply: fmt.Sprintf("⚠️ Failed to get skill: %v", err)}, nil
	}

	if item.Kind != knowledge.KindSkill || item.ChatID != req.Session.ScopeID || item.UserID != req.Actor.ID {
		return Result{Reply: fmt.Sprintf("⚠️ Skill item `%s` not found.", itemID)}, nil
	}

	vers, err := h.Store.ListVersions(ctx, itemID)
	if err != nil {
		return Result{Reply: fmt.Sprintf("⚠️ Failed to get skill versions: %v", err)}, nil
	}

	var b strings.Builder
	fmt.Fprintf(&b, "🛠️ **Skill Item: `%s`**\n\n", item.ID)
	fmt.Fprintf(&b, "• **Status:** %s\n", item.Status)
	if item.ActiveVersionID != "" {
		fmt.Fprintf(&b, "• **Active Version:** `%s`\n", item.ActiveVersionID)
	} else {
		b.WriteString("• **Active Version:** *(none)*\n")
	}
	fmt.Fprintf(&b, "• **Created:** %s\n", time.Unix(item.CreatedAt, 0).UTC().Format(time.RFC3339))
	fmt.Fprintf(&b, "• **Updated:** %s\n\n", time.Unix(item.UpdatedAt, 0).UTC().Format(time.RFC3339))

	if len(vers) > 0 {
		b.WriteString("**Revisions:**\n")
		b.WriteString("| Revision | Status | Name | Description |\n")
		b.WriteString("|---|---|---|---|\n")
		for _, v := range vers {
			sv, ok := v.(*knowledge.SkillVersion)
			if ok {
				name := strings.ReplaceAll(sv.Name, "|", "\\|")
				d := strings.ReplaceAll(knowledge.TruncateRunes(sv.Description, 40), "|", "\\|")
				d = strings.ReplaceAll(d, "\n", " ")
				fmt.Fprintf(&b, "| `@%d` | %s | %s | %s |\n", sv.Revision, sv.Status, name, d)
			}
		}
	}

	if item.ActiveVersionID != "" {
		activeVer, err := h.Store.GetSkillVersion(ctx, item.ActiveVersionID)
		if err == nil && activeVer != nil {
			fmt.Fprintf(&b, "\n**Instructions (`%s`):**\n%s\n", activeVer.ID, FormatAdaptiveFence(activeVer.InstructionsMarkdown, "markdown"))
		}
	}

	return Result{Reply: b.String()}, nil
}

func (h *SkillHandler) formatSkillVersionDetails(ver *knowledge.SkillVersion) string {
	var b strings.Builder
	fmt.Fprintf(&b, "🛠️ **Skill Version: `%s`**\n\n", ver.ID)
	fmt.Fprintf(&b, "• **Name:** %s\n", ver.Name)
	fmt.Fprintf(&b, "• **Item ID:** `%s`\n", ver.ItemID)
	fmt.Fprintf(&b, "• **Revision:** %d\n", ver.Revision)
	fmt.Fprintf(&b, "• **Status:** %s\n", ver.Status)
	fmt.Fprintf(&b, "• **Index Status:** %s\n", ver.IndexStatus)
	if ver.Description != "" {
		fmt.Fprintf(&b, "• **Description:** %s\n", ver.Description)
	}
	if len(ver.Triggers) > 0 {
		fmt.Fprintf(&b, "• **Triggers:** %s\n", strings.Join(ver.Triggers, ", "))
	}
	if len(ver.Tags) > 0 {
		fmt.Fprintf(&b, "• **Tags:** %s\n", strings.Join(ver.Tags, ", "))
	}

	b.WriteString("\n**Instructions:**\n" + FormatAdaptiveFence(ver.InstructionsMarkdown, "markdown") + "\n\n")

	b.WriteString("**Provenance:**\n")
	fmt.Fprintf(&b, "• **Proposed By:** `%s`\n", ver.Provenance.UserID)
	fmt.Fprintf(&b, "• **Created At:** %s\n", time.Unix(ver.Provenance.CreatedAt, 0).UTC().Format(time.RFC3339))
	if ver.Provenance.SourceMessageSeq != nil {
		fmt.Fprintf(&b, "• **Source Message Seq:** %d\n", *ver.Provenance.SourceMessageSeq)
	}
	if ver.Provenance.FSMRunID != "" {
		fmt.Fprintf(&b, "• **FSM Run ID:** `%s`\n", ver.Provenance.FSMRunID)
	}
	if ver.Provenance.ReviewedBy != "" {
		fmt.Fprintf(&b, "• **Reviewed By:** `%s`\n", ver.Provenance.ReviewedBy)
	}
	if ver.Provenance.ReviewedAt != nil {
		fmt.Fprintf(&b, "• **Reviewed At:** %s\n", time.Unix(*ver.Provenance.ReviewedAt, 0).UTC().Format(time.RFC3339))
	}
	fmt.Fprintf(&b, "• **Content Hash:** `%s`\n", ver.Provenance.ContentHash)

	return b.String()
}

func (h *SkillHandler) handleApprove(ctx context.Context, req Request) (Result, error) {
	if len(req.Args) == 0 {
		return Result{Reply: "⚠️ Please specify a version ID to approve. Usage: `/skill approve <version-id>` (e.g. `/skill approve skill_123@1`)"}, nil
	}

	versionID := strings.TrimSpace(req.Args[0])
	if !strings.HasPrefix(versionID, knowledge.PrefixSkill) {
		return Result{Reply: fmt.Sprintf("⚠️ Invalid skill version ID %q: skill IDs must begin with %q.", versionID, knowledge.PrefixSkill)}, nil
	}

	itemID, _, err := knowledge.ParseVersionID(versionID)
	if err != nil {
		return Result{Reply: fmt.Sprintf("⚠️ Invalid version ID format %q: %v", versionID, err)}, nil
	}

	item, err := h.Store.GetItem(ctx, itemID)
	if err != nil {
		if errors.Is(err, knowledge.ErrNotFound) {
			return Result{Reply: fmt.Sprintf("⚠️ Skill version `%s` not found.", versionID)}, nil
		}
		return Result{Reply: fmt.Sprintf("⚠️ Failed to get skill: %v", err)}, nil
	}
	if item.Kind != knowledge.KindSkill || item.ChatID != req.Session.ScopeID || item.UserID != req.Actor.ID {
		return Result{Reply: fmt.Sprintf("⚠️ Skill version `%s` not found.", versionID)}, nil
	}

	prevActiveID := item.ActiveVersionID

	if err := h.Store.ApproveVersion(ctx, versionID, req.Actor.ID); err != nil {
		if errors.Is(err, knowledge.ErrNotFound) {
			return Result{Reply: fmt.Sprintf("⚠️ Skill version `%s` not found.", versionID)}, nil
		}
		if errors.Is(err, knowledge.ErrInvalidStatus) {
			return Result{Reply: fmt.Sprintf("⚠️ Cannot approve skill version `%s`: %v", versionID, err)}, nil
		}
		return Result{Reply: fmt.Sprintf("⚠️ Failed to approve skill version: %v", err)}, nil
	}

	if h.Indexer != nil {
		if idxErr := h.Indexer.IndexVersion(ctx, versionID); idxErr != nil {
			slog.Error("failed to index approved skill version", "version_id", versionID, "error", idxErr)
			return Result{Reply: fmt.Sprintf("✅ Skill version `%s` approved, but indexing failed: %v", versionID, idxErr)}, nil
		}
		if prevActiveID != "" && prevActiveID != versionID {
			if _, purgeErr := h.Indexer.PurgePendingDeletion(ctx, prevActiveID); purgeErr != nil {
				slog.Warn("failed to purge superseded skill version from index", "version_id", prevActiveID, "error", purgeErr)
			}
		}
	}

	return Result{Reply: fmt.Sprintf("✅ Skill version `%s` has been approved and activated.", versionID)}, nil
}

func (h *SkillHandler) handleReject(ctx context.Context, req Request) (Result, error) {
	if len(req.Args) == 0 {
		return Result{Reply: "⚠️ Please specify a version ID to reject. Usage: `/skill reject <version-id>` (e.g. `/skill reject skill_123@1`)"}, nil
	}

	versionID := strings.TrimSpace(req.Args[0])
	if !strings.HasPrefix(versionID, knowledge.PrefixSkill) {
		return Result{Reply: fmt.Sprintf("⚠️ Invalid skill version ID %q: skill IDs must begin with %q.", versionID, knowledge.PrefixSkill)}, nil
	}

	itemID, _, err := knowledge.ParseVersionID(versionID)
	if err != nil {
		return Result{Reply: fmt.Sprintf("⚠️ Invalid version ID format %q: %v", versionID, err)}, nil
	}

	item, err := h.Store.GetItem(ctx, itemID)
	if err != nil {
		if errors.Is(err, knowledge.ErrNotFound) {
			return Result{Reply: fmt.Sprintf("⚠️ Skill version `%s` not found.", versionID)}, nil
		}
		return Result{Reply: fmt.Sprintf("⚠️ Failed to get skill: %v", err)}, nil
	}
	if item.Kind != knowledge.KindSkill || item.ChatID != req.Session.ScopeID || item.UserID != req.Actor.ID {
		return Result{Reply: fmt.Sprintf("⚠️ Skill version `%s` not found.", versionID)}, nil
	}

	if err := h.Store.RejectVersion(ctx, versionID, req.Actor.ID); err != nil {
		if errors.Is(err, knowledge.ErrNotFound) {
			return Result{Reply: fmt.Sprintf("⚠️ Skill version `%s` not found.", versionID)}, nil
		}
		if errors.Is(err, knowledge.ErrInvalidStatus) {
			return Result{Reply: fmt.Sprintf("⚠️ Cannot reject skill version `%s`: %v", versionID, err)}, nil
		}
		return Result{Reply: fmt.Sprintf("⚠️ Failed to reject skill version: %v", err)}, nil
	}

	return Result{Reply: fmt.Sprintf("❌ Skill version `%s` has been rejected.", versionID)}, nil
}

func (h *SkillHandler) handleDisable(ctx context.Context, req Request) (Result, error) {
	if len(req.Args) == 0 {
		return Result{Reply: "⚠️ Please specify a skill item ID to disable. Usage: `/skill disable <item-id>` (e.g. `/skill disable skill_123`)"}, nil
	}

	ref := strings.TrimSpace(req.Args[0])
	if !strings.HasPrefix(ref, knowledge.PrefixSkill) {
		return Result{Reply: fmt.Sprintf("⚠️ Invalid skill identifier %q: skill IDs must begin with %q.", ref, knowledge.PrefixSkill)}, nil
	}

	itemID, _, hasRev, err := knowledge.ParseItemOrVersionRef(ref)
	if err != nil {
		return Result{Reply: fmt.Sprintf("⚠️ Invalid skill reference %q: %v", ref, err)}, nil
	}
	if hasRev {
		return Result{Reply: fmt.Sprintf("⚠️ `/skill disable` operates on entire skill items, not specific versions. Please use item ID `%s` without revision.", itemID)}, nil
	}

	item, err := h.Store.GetItem(ctx, itemID)
	if err != nil {
		if errors.Is(err, knowledge.ErrNotFound) {
			return Result{Reply: fmt.Sprintf("⚠️ Skill `%s` not found.", itemID)}, nil
		}
		return Result{Reply: fmt.Sprintf("⚠️ Failed to get skill: %v", err)}, nil
	}
	if item.Kind != knowledge.KindSkill || item.ChatID != req.Session.ScopeID || item.UserID != req.Actor.ID {
		return Result{Reply: fmt.Sprintf("⚠️ Skill `%s` not found.", itemID)}, nil
	}

	activeVerID := item.ActiveVersionID

	if err := h.Store.DisableSkill(ctx, itemID, req.Actor.ID); err != nil {
		if errors.Is(err, knowledge.ErrNotFound) {
			return Result{Reply: fmt.Sprintf("⚠️ Skill `%s` not found.", itemID)}, nil
		}
		return Result{Reply: fmt.Sprintf("⚠️ Failed to disable skill: %v", err)}, nil
	}

	if h.Indexer != nil && activeVerID != "" {
		if _, purgeErr := h.Indexer.PurgePendingDeletion(ctx, activeVerID); purgeErr != nil {
			slog.Warn("failed to purge disabled skill from index", "version_id", activeVerID, "error", purgeErr)
		}
	}

	return Result{Reply: fmt.Sprintf("⏸️ Skill `%s` has been disabled.", itemID)}, nil
}

func (h *SkillHandler) handleEnable(ctx context.Context, req Request) (Result, error) {
	if len(req.Args) == 0 {
		return Result{Reply: "⚠️ Please specify a skill item ID to enable. Usage: `/skill enable <item-id>` (e.g. `/skill enable skill_123`)"}, nil
	}

	ref := strings.TrimSpace(req.Args[0])
	if !strings.HasPrefix(ref, knowledge.PrefixSkill) {
		return Result{Reply: fmt.Sprintf("⚠️ Invalid skill identifier %q: skill IDs must begin with %q.", ref, knowledge.PrefixSkill)}, nil
	}

	itemID, _, hasRev, err := knowledge.ParseItemOrVersionRef(ref)
	if err != nil {
		return Result{Reply: fmt.Sprintf("⚠️ Invalid skill reference %q: %v", ref, err)}, nil
	}
	if hasRev {
		return Result{Reply: fmt.Sprintf("⚠️ `/skill enable` operates on entire skill items, not specific versions. Please use item ID `%s` without revision.", itemID)}, nil
	}

	item, err := h.Store.GetItem(ctx, itemID)
	if err != nil {
		if errors.Is(err, knowledge.ErrNotFound) {
			return Result{Reply: fmt.Sprintf("⚠️ Skill `%s` not found.", itemID)}, nil
		}
		return Result{Reply: fmt.Sprintf("⚠️ Failed to get skill: %v", err)}, nil
	}
	if item.Kind != knowledge.KindSkill || item.ChatID != req.Session.ScopeID || item.UserID != req.Actor.ID {
		return Result{Reply: fmt.Sprintf("⚠️ Skill `%s` not found.", itemID)}, nil
	}

	if err := h.Store.EnableSkill(ctx, itemID, req.Actor.ID); err != nil {
		if errors.Is(err, knowledge.ErrNotFound) {
			return Result{Reply: fmt.Sprintf("⚠️ Skill `%s` not found.", itemID)}, nil
		}
		return Result{Reply: fmt.Sprintf("⚠️ Failed to enable skill: %v", err)}, nil
	}

	if h.Indexer != nil && item.ActiveVersionID != "" {
		if idxErr := h.Indexer.IndexVersion(ctx, item.ActiveVersionID); idxErr != nil {
			slog.Error("failed to index enabled skill version", "version_id", item.ActiveVersionID, "error", idxErr)
		}
	}

	return Result{Reply: fmt.Sprintf("▶️ Skill `%s` has been enabled.", itemID)}, nil
}

func (h *SkillHandler) handleArchive(ctx context.Context, req Request) (Result, error) {
	if len(req.Args) == 0 {
		return Result{Reply: "⚠️ Please specify a skill item ID to archive. Usage: `/skill archive <item-id>` (e.g. `/skill archive skill_123`)"}, nil
	}

	ref := strings.TrimSpace(req.Args[0])
	if !strings.HasPrefix(ref, knowledge.PrefixSkill) {
		return Result{Reply: fmt.Sprintf("⚠️ Invalid skill identifier %q: skill IDs must begin with %q.", ref, knowledge.PrefixSkill)}, nil
	}

	itemID, _, hasRev, err := knowledge.ParseItemOrVersionRef(ref)
	if err != nil {
		return Result{Reply: fmt.Sprintf("⚠️ Invalid skill reference %q: %v", ref, err)}, nil
	}
	if hasRev {
		return Result{Reply: fmt.Sprintf("⚠️ `/skill archive` operates on entire skill items, not specific versions. Please use item ID `%s` without revision.", itemID)}, nil
	}

	item, err := h.Store.GetItem(ctx, itemID)
	if err != nil {
		if errors.Is(err, knowledge.ErrNotFound) {
			return Result{Reply: fmt.Sprintf("⚠️ Skill `%s` not found.", itemID)}, nil
		}
		return Result{Reply: fmt.Sprintf("⚠️ Failed to get skill: %v", err)}, nil
	}
	if item.Kind != knowledge.KindSkill || item.ChatID != req.Session.ScopeID || item.UserID != req.Actor.ID {
		return Result{Reply: fmt.Sprintf("⚠️ Skill `%s` not found.", itemID)}, nil
	}

	activeVerID := item.ActiveVersionID

	if err := h.Store.ArchiveSkill(ctx, itemID, req.Actor.ID); err != nil {
		if errors.Is(err, knowledge.ErrNotFound) {
			return Result{Reply: fmt.Sprintf("⚠️ Skill `%s` not found.", itemID)}, nil
		}
		return Result{Reply: fmt.Sprintf("⚠️ Failed to archive skill: %v", err)}, nil
	}

	if h.Indexer != nil && activeVerID != "" {
		if _, purgeErr := h.Indexer.PurgePendingDeletion(ctx, activeVerID); purgeErr != nil {
			slog.Warn("failed to purge archived skill from index", "version_id", activeVerID, "error", purgeErr)
		}
	}

	return Result{Reply: fmt.Sprintf("📦 Skill `%s` has been archived.", itemID)}, nil
}

func (h *SkillHandler) handleDelete(ctx context.Context, req Request) (Result, error) {
	if len(req.Args) == 0 {
		return Result{Reply: "⚠️ Please specify a skill item ID to delete. Usage: `/skill delete <item-id>` (e.g. `/skill delete skill_123`)"}, nil
	}

	ref := strings.TrimSpace(req.Args[0])
	if !strings.HasPrefix(ref, knowledge.PrefixSkill) {
		return Result{Reply: fmt.Sprintf("⚠️ Invalid skill identifier %q: skill IDs must begin with %q.", ref, knowledge.PrefixSkill)}, nil
	}

	itemID, _, hasRev, err := knowledge.ParseItemOrVersionRef(ref)
	if err != nil {
		return Result{Reply: fmt.Sprintf("⚠️ Invalid skill reference %q: %v", ref, err)}, nil
	}
	if hasRev {
		return Result{Reply: fmt.Sprintf("⚠️ `/skill delete` operates on entire skill items, not specific versions. Please use item ID `%s` without revision.", itemID)}, nil
	}

	item, err := h.Store.GetItem(ctx, itemID)
	if err != nil {
		if errors.Is(err, knowledge.ErrNotFound) {
			return Result{Reply: fmt.Sprintf("⚠️ Skill `%s` not found.", itemID)}, nil
		}
		return Result{Reply: fmt.Sprintf("⚠️ Failed to get skill: %v", err)}, nil
	}
	if item.Kind != knowledge.KindSkill || item.ChatID != req.Session.ScopeID || item.UserID != req.Actor.ID {
		return Result{Reply: fmt.Sprintf("⚠️ Skill `%s` not found.", itemID)}, nil
	}

	vers, _ := h.Store.ListVersions(ctx, itemID)

	if err := h.Store.DeleteSkill(ctx, itemID, req.Actor.ID); err != nil {
		if errors.Is(err, knowledge.ErrNotFound) {
			return Result{Reply: fmt.Sprintf("⚠️ Skill `%s` not found.", itemID)}, nil
		}
		return Result{Reply: fmt.Sprintf("⚠️ Failed to delete skill: %v", err)}, nil
	}

	allPurged := true
	if h.Indexer != nil {
		for _, v := range vers {
			if _, purgeErr := h.Indexer.PurgePendingDeletion(ctx, v.GetID()); purgeErr != nil {
				slog.Warn("failed to purge deleted skill version from index", "version_id", v.GetID(), "error", purgeErr)
				allPurged = false
			}
		}
	} else {
		allPurged = false
	}

	notice := fmt.Sprintf("🗑️ Skill `%s` has been permanently deleted.", itemID)
	if !allPurged {
		notice += " (Background index cleanup is queued)."
	}
	return Result{Reply: notice}, nil
}
