package gateway

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"time"

	"bob/internal/knowledge"
	"bob/internal/models"
)

func (g *Gateway) handleSkillCommand(ctx context.Context, msg models.Message, text, senderName string, isDM bool) error {
	if err := g.VerifyDMOwner(ctx, msg.ChatID, msg.UserID); err != nil {
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ %v", err))
	}

	g.mu.Lock()
	botID := g.botUserID
	g.mu.Unlock()
	if botID != "" && msg.UserID == botID {
		return nil
	}

	parts := strings.Fields(strings.TrimSpace(text))
	subcmd := ""
	if len(parts) > 1 {
		subcmd = strings.ToLower(parts[1])
	}

	kStore, err := g.getKnowledgeStore(ctx, msg.ChatID, isDM)
	if err != nil {
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Knowledge storage is unavailable: %v", err))
	}

	indexer, idxErr := g.getKnowledgeIndexer(ctx, msg.ChatID, isDM)
	if idxErr != nil {
		slog.Warn("knowledge indexer unavailable for skill command", "chat_id", msg.ChatID, "error", idxErr)
	}

	switch subcmd {
	case "list":
		return g.handleSkillList(ctx, msg, kStore, parts)
	case "show":
		return g.handleSkillShow(ctx, msg, kStore, parts)
	case "approve":
		return g.handleSkillApprove(ctx, msg, kStore, indexer, parts)
	case "reject":
		return g.handleSkillReject(ctx, msg, kStore, parts)
	case "disable":
		return g.handleSkillDisable(ctx, msg, kStore, indexer, parts)
	case "enable":
		return g.handleSkillEnable(ctx, msg, kStore, indexer, parts)
	case "archive":
		return g.handleSkillArchive(ctx, msg, kStore, indexer, parts)
	case "delete":
		return g.handleSkillDelete(ctx, msg, kStore, indexer, parts)
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
		return g.SendMessage(msg.ChatID, helpText)
	}
}

func (g *Gateway) handleSkillList(ctx context.Context, msg models.Message, store *knowledge.Store, parts []string) error {
	statusFilter := "active"
	if len(parts) > 2 {
		statusFilter = strings.ToLower(parts[2])
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
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Invalid status filter %q. Valid skill filters: pending, active, disabled, rejected, archived, all.", statusFilter))
	}

	items, err := store.ListItems(ctx, msg.ChatID, knowledge.KindSkill, statusFilter)
	if err != nil {
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Failed to list skills: %v", err))
	}

	if len(items) == 0 {
		return g.SendMessage(msg.ChatID, fmt.Sprintf("ℹ️ No skills found with status %q.", statusFilter))
	}

	var b strings.Builder
	fmt.Fprintf(&b, "🛠️ **Skills** (%s)\n\n", statusFilter)
	b.WriteString("| ID | Name | Description | Tags | Rev | Status |\n")
	b.WriteString("|---|---|---|---|---|---|\n")

	for _, item := range items {
		vers, err := store.ListVersions(ctx, item.ID)
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
				// Fallback to active version
				if item.ActiveVersionID != "" {
					for _, v := range vers {
						if sv, ok := v.(*knowledge.SkillVersion); ok && sv.ID == item.ActiveVersionID {
							targetVer = sv
							break
						}
					}
				}
				// Or newest version
				if targetVer == nil {
					if sv, ok := vers[len(vers)-1].(*knowledge.SkillVersion); ok {
						targetVer = sv
					}
				}
			}
		}

		name := "-"
		desc := "-"
		tagsStr := "-"
		revStr := "-"
		statusStr := string(item.Status)

		if targetVer != nil {
			name = strings.ReplaceAll(targetVer.Name, "|", "\\|")
			desc = knowledge.TruncateRunes(targetVer.Description, 40)
			desc = strings.ReplaceAll(desc, "|", "\\|")
			desc = strings.ReplaceAll(desc, "\n", " ")
			if len(targetVer.Tags) > 0 {
				tagsStr = strings.ReplaceAll(strings.Join(targetVer.Tags, ", "), "|", "\\|")
			}
			revStr = fmt.Sprintf("@%d", targetVer.Revision)
			if statusFilter == "pending" || statusFilter == "rejected" || statusFilter == "archived" || item.Status == knowledge.ItemStatusPending {
				statusStr = string(targetVer.Status)
			}
		}

		fmt.Fprintf(&b, "| `%s` | %s | %s | %s | %s | %s |\n", item.ID, name, desc, tagsStr, revStr, statusStr)
	}

	return g.SendMessage(msg.ChatID, b.String())
}

func (g *Gateway) handleSkillShow(ctx context.Context, msg models.Message, store *knowledge.Store, parts []string) error {
	if len(parts) < 3 {
		return g.SendMessage(msg.ChatID, "⚠️ Please specify a skill ID. Usage: `/skill show <id>` (e.g. `/skill show skill_123` or `/skill show skill_123@1`)")
	}

	ref := strings.TrimSpace(parts[2])
	if !strings.HasPrefix(ref, knowledge.PrefixSkill) {
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Invalid skill identifier %q: skill IDs must begin with %q.", ref, knowledge.PrefixSkill))
	}

	itemID, rev, hasRev, err := knowledge.ParseItemOrVersionRef(ref)
	if err != nil {
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Invalid skill reference %q: %v", ref, err))
	}

	if hasRev {
		verID := knowledge.FormatVersionID(itemID, rev)
		ver, err := store.GetSkillVersion(ctx, verID)
		if err != nil {
			if errors.Is(err, knowledge.ErrNotFound) {
				return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Skill version `%s` not found.", verID))
			}
			return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Failed to get skill version: %v", err))
		}
		if ver.Provenance.ChatID != msg.ChatID || ver.Provenance.UserID != msg.UserID {
			return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Skill version `%s` not found.", verID))
		}
		return g.sendSkillVersionDetails(msg.ChatID, ver)
	}

	item, err := store.GetItem(ctx, itemID)
	if err != nil {
		if errors.Is(err, knowledge.ErrNotFound) {
			return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Skill item `%s` not found.", itemID))
		}
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Failed to get skill: %v", err))
	}

	if item.Kind != knowledge.KindSkill || item.ChatID != msg.ChatID || item.UserID != msg.UserID {
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Skill item `%s` not found.", itemID))
	}

	vers, err := store.ListVersions(ctx, itemID)
	if err != nil {
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Failed to get skill versions: %v", err))
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

	// If active version exists, show its instructions
	if item.ActiveVersionID != "" {
		activeVer, err := store.GetSkillVersion(ctx, item.ActiveVersionID)
		if err == nil && activeVer != nil {
			fmt.Fprintf(&b, "\n**Instructions (`%s`):**\n%s\n", activeVer.ID, formatAdaptiveFence(activeVer.InstructionsMarkdown, "markdown"))
		}
	}

	return g.SendMessage(msg.ChatID, b.String())
}

func (g *Gateway) sendSkillVersionDetails(chatID string, ver *knowledge.SkillVersion) error {
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

	b.WriteString("\n**Instructions:**\n" + formatAdaptiveFence(ver.InstructionsMarkdown, "markdown") + "\n\n")

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

	return g.SendMessage(chatID, b.String())
}

func (g *Gateway) handleSkillApprove(ctx context.Context, msg models.Message, store *knowledge.Store, indexer *knowledge.Indexer, parts []string) error {
	if len(parts) < 3 {
		return g.SendMessage(msg.ChatID, "⚠️ Please specify a version ID to approve. Usage: `/skill approve <version-id>` (e.g. `/skill approve skill_123@1`)")
	}

	versionID := strings.TrimSpace(parts[2])
	if !strings.HasPrefix(versionID, knowledge.PrefixSkill) {
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Invalid skill version ID %q: skill IDs must begin with %q.", versionID, knowledge.PrefixSkill))
	}

	itemID, _, err := knowledge.ParseVersionID(versionID)
	if err != nil {
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Invalid version ID format %q: %v", versionID, err))
	}

	item, err := store.GetItem(ctx, itemID)
	if err != nil {
		if errors.Is(err, knowledge.ErrNotFound) {
			return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Skill version `%s` not found.", versionID))
		}
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Failed to get skill: %v", err))
	}
	if item.Kind != knowledge.KindSkill || item.ChatID != msg.ChatID || item.UserID != msg.UserID {
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Skill version `%s` not found.", versionID))
	}

	prevActiveID := item.ActiveVersionID

	if err := store.ApproveVersion(ctx, versionID, msg.UserID); err != nil {
		if errors.Is(err, knowledge.ErrNotFound) {
			return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Skill version `%s` not found.", versionID))
		}
		if errors.Is(err, knowledge.ErrInvalidStatus) {
			return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Cannot approve skill version `%s`: %v", versionID, err))
		}
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Failed to approve skill version: %v", err))
	}

	// Trigger indexing and purge superseded active index record
	if indexer != nil {
		if idxErr := indexer.IndexVersion(ctx, versionID); idxErr != nil {
			slog.Error("failed to index approved skill version", "version_id", versionID, "error", idxErr)
			return g.SendMessage(msg.ChatID, fmt.Sprintf("✅ Skill version `%s` approved, but indexing failed: %v", versionID, idxErr))
		}
		if prevActiveID != "" && prevActiveID != versionID {
			if _, purgeErr := indexer.PurgePendingDeletion(ctx, prevActiveID); purgeErr != nil {
				slog.Warn("failed to purge superseded skill version from index", "version_id", prevActiveID, "error", purgeErr)
			}
		}
	}

	return g.SendMessage(msg.ChatID, fmt.Sprintf("✅ Skill version `%s` has been approved and activated.", versionID))
}

func (g *Gateway) handleSkillReject(ctx context.Context, msg models.Message, store *knowledge.Store, parts []string) error {
	if len(parts) < 3 {
		return g.SendMessage(msg.ChatID, "⚠️ Please specify a version ID to reject. Usage: `/skill reject <version-id>` (e.g. `/skill reject skill_123@1`)")
	}

	versionID := strings.TrimSpace(parts[2])
	if !strings.HasPrefix(versionID, knowledge.PrefixSkill) {
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Invalid skill version ID %q: skill IDs must begin with %q.", versionID, knowledge.PrefixSkill))
	}

	itemID, _, err := knowledge.ParseVersionID(versionID)
	if err != nil {
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Invalid version ID format %q: %v", versionID, err))
	}

	item, err := store.GetItem(ctx, itemID)
	if err != nil {
		if errors.Is(err, knowledge.ErrNotFound) {
			return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Skill version `%s` not found.", versionID))
		}
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Failed to get skill: %v", err))
	}
	if item.Kind != knowledge.KindSkill || item.ChatID != msg.ChatID || item.UserID != msg.UserID {
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Skill version `%s` not found.", versionID))
	}

	if err := store.RejectVersion(ctx, versionID, msg.UserID); err != nil {
		if errors.Is(err, knowledge.ErrNotFound) {
			return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Skill version `%s` not found.", versionID))
		}
		if errors.Is(err, knowledge.ErrInvalidStatus) {
			return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Cannot reject skill version `%s`: %v", versionID, err))
		}
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Failed to reject skill version: %v", err))
	}

	return g.SendMessage(msg.ChatID, fmt.Sprintf("❌ Skill version `%s` has been rejected.", versionID))
}

func (g *Gateway) handleSkillDisable(ctx context.Context, msg models.Message, store *knowledge.Store, indexer *knowledge.Indexer, parts []string) error {
	if len(parts) < 3 {
		return g.SendMessage(msg.ChatID, "⚠️ Please specify a skill item ID to disable. Usage: `/skill disable <item-id>` (e.g. `/skill disable skill_123`)")
	}

	ref := strings.TrimSpace(parts[2])
	if !strings.HasPrefix(ref, knowledge.PrefixSkill) {
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Invalid skill identifier %q: skill IDs must begin with %q.", ref, knowledge.PrefixSkill))
	}

	itemID, _, hasRev, err := knowledge.ParseItemOrVersionRef(ref)
	if err != nil {
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Invalid skill reference %q: %v", ref, err))
	}
	if hasRev {
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ `/skill disable` operates on entire skill items, not specific versions. Please use item ID `%s` without revision.", itemID))
	}

	item, err := store.GetItem(ctx, itemID)
	if err != nil {
		if errors.Is(err, knowledge.ErrNotFound) {
			return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Skill `%s` not found.", itemID))
		}
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Failed to get skill: %v", err))
	}
	if item.Kind != knowledge.KindSkill || item.ChatID != msg.ChatID || item.UserID != msg.UserID {
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Skill `%s` not found.", itemID))
	}

	activeVerID := item.ActiveVersionID

	if err := store.DisableSkill(ctx, itemID, msg.UserID); err != nil {
		if errors.Is(err, knowledge.ErrNotFound) {
			return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Skill `%s` not found.", itemID))
		}
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Failed to disable skill: %v", err))
	}

	if indexer != nil && activeVerID != "" {
		if _, purgeErr := indexer.PurgePendingDeletion(ctx, activeVerID); purgeErr != nil {
			slog.Warn("failed to purge disabled skill from index", "version_id", activeVerID, "error", purgeErr)
		}
	}

	return g.SendMessage(msg.ChatID, fmt.Sprintf("⏸️ Skill `%s` has been disabled.", itemID))
}

func (g *Gateway) handleSkillEnable(ctx context.Context, msg models.Message, store *knowledge.Store, indexer *knowledge.Indexer, parts []string) error {
	if len(parts) < 3 {
		return g.SendMessage(msg.ChatID, "⚠️ Please specify a skill item ID to enable. Usage: `/skill enable <item-id>` (e.g. `/skill enable skill_123`)")
	}

	ref := strings.TrimSpace(parts[2])
	if !strings.HasPrefix(ref, knowledge.PrefixSkill) {
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Invalid skill identifier %q: skill IDs must begin with %q.", ref, knowledge.PrefixSkill))
	}

	itemID, _, hasRev, err := knowledge.ParseItemOrVersionRef(ref)
	if err != nil {
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Invalid skill reference %q: %v", ref, err))
	}
	if hasRev {
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ `/skill enable` operates on entire skill items, not specific versions. Please use item ID `%s` without revision.", itemID))
	}

	item, err := store.GetItem(ctx, itemID)
	if err != nil {
		if errors.Is(err, knowledge.ErrNotFound) {
			return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Skill `%s` not found.", itemID))
		}
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Failed to get skill: %v", err))
	}
	if item.Kind != knowledge.KindSkill || item.ChatID != msg.ChatID || item.UserID != msg.UserID {
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Skill `%s` not found.", itemID))
	}

	if err := store.EnableSkill(ctx, itemID, msg.UserID); err != nil {
		if errors.Is(err, knowledge.ErrNotFound) {
			return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Skill `%s` not found.", itemID))
		}
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Failed to enable skill: %v", err))
	}

	if indexer != nil && item.ActiveVersionID != "" {
		if idxErr := indexer.IndexVersion(ctx, item.ActiveVersionID); idxErr != nil {
			slog.Error("failed to index enabled skill version", "version_id", item.ActiveVersionID, "error", idxErr)
		}
	}

	return g.SendMessage(msg.ChatID, fmt.Sprintf("▶️ Skill `%s` has been enabled.", itemID))
}

func (g *Gateway) handleSkillArchive(ctx context.Context, msg models.Message, store *knowledge.Store, indexer *knowledge.Indexer, parts []string) error {
	if len(parts) < 3 {
		return g.SendMessage(msg.ChatID, "⚠️ Please specify a skill item ID to archive. Usage: `/skill archive <item-id>` (e.g. `/skill archive skill_123`)")
	}

	ref := strings.TrimSpace(parts[2])
	if !strings.HasPrefix(ref, knowledge.PrefixSkill) {
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Invalid skill identifier %q: skill IDs must begin with %q.", ref, knowledge.PrefixSkill))
	}

	itemID, _, hasRev, err := knowledge.ParseItemOrVersionRef(ref)
	if err != nil {
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Invalid skill reference %q: %v", ref, err))
	}
	if hasRev {
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ `/skill archive` operates on entire skill items, not specific versions. Please use item ID `%s` without revision.", itemID))
	}

	item, err := store.GetItem(ctx, itemID)
	if err != nil {
		if errors.Is(err, knowledge.ErrNotFound) {
			return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Skill `%s` not found.", itemID))
		}
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Failed to get skill: %v", err))
	}
	if item.Kind != knowledge.KindSkill || item.ChatID != msg.ChatID || item.UserID != msg.UserID {
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Skill `%s` not found.", itemID))
	}

	activeVerID := item.ActiveVersionID

	if err := store.ArchiveSkill(ctx, itemID, msg.UserID); err != nil {
		if errors.Is(err, knowledge.ErrNotFound) {
			return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Skill `%s` not found.", itemID))
		}
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Failed to archive skill: %v", err))
	}

	if indexer != nil && activeVerID != "" {
		if _, purgeErr := indexer.PurgePendingDeletion(ctx, activeVerID); purgeErr != nil {
			slog.Warn("failed to purge archived skill from index", "version_id", activeVerID, "error", purgeErr)
		}
	}

	return g.SendMessage(msg.ChatID, fmt.Sprintf("📦 Skill `%s` has been archived.", itemID))
}

func (g *Gateway) handleSkillDelete(ctx context.Context, msg models.Message, store *knowledge.Store, indexer *knowledge.Indexer, parts []string) error {
	if len(parts) < 3 {
		return g.SendMessage(msg.ChatID, "⚠️ Please specify a skill item ID to delete. Usage: `/skill delete <item-id>` (e.g. `/skill delete skill_123`)")
	}

	ref := strings.TrimSpace(parts[2])
	if !strings.HasPrefix(ref, knowledge.PrefixSkill) {
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Invalid skill identifier %q: skill IDs must begin with %q.", ref, knowledge.PrefixSkill))
	}

	itemID, _, hasRev, err := knowledge.ParseItemOrVersionRef(ref)
	if err != nil {
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Invalid skill reference %q: %v", ref, err))
	}
	if hasRev {
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ `/skill delete` operates on entire skill items, not specific versions. Please use item ID `%s` without revision.", itemID))
	}

	item, err := store.GetItem(ctx, itemID)
	if err != nil {
		if errors.Is(err, knowledge.ErrNotFound) {
			return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Skill `%s` not found.", itemID))
		}
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Failed to get skill: %v", err))
	}
	if item.Kind != knowledge.KindSkill || item.ChatID != msg.ChatID || item.UserID != msg.UserID {
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Skill `%s` not found.", itemID))
	}

	// Capture all version IDs before deletion so all indexed records can be purged immediately
	vers, _ := store.ListVersions(ctx, itemID)

	if err := store.DeleteSkill(ctx, itemID, msg.UserID); err != nil {
		if errors.Is(err, knowledge.ErrNotFound) {
			return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Skill `%s` not found.", itemID))
		}
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Failed to delete skill: %v", err))
	}

	allPurged := true
	if indexer != nil {
		for _, v := range vers {
			if _, purgeErr := indexer.PurgePendingDeletion(ctx, v.GetID()); purgeErr != nil {
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
	return g.SendMessage(msg.ChatID, notice)
}
