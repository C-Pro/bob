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

func (g *Gateway) getKnowledgeStore(ctx context.Context, chatID string, isDM bool) (*knowledge.Store, error) {
	g.mu.Lock()
	sp := g.storeProvider
	g.mu.Unlock()
	if sp == nil {
		return nil, errors.New("store provider is not initialized")
	}
	return sp.GetKnowledgeStore(ctx, chatID, isDM)
}

func (g *Gateway) getKnowledgeIndexer(ctx context.Context, chatID string, isDM bool) (*knowledge.Indexer, error) {
	g.mu.Lock()
	sp := g.storeProvider
	g.mu.Unlock()
	if sp == nil {
		return nil, errors.New("store provider is not initialized")
	}
	return sp.GetKnowledgeIndexer(ctx, chatID, isDM)
}

func (g *Gateway) handleMemoryCommand(ctx context.Context, msg models.Message, text, senderName string, isDM bool) error {
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
		slog.Warn("knowledge indexer unavailable for memory command", "chat_id", msg.ChatID, "error", idxErr)
	}

	switch subcmd {
	case "list":
		return g.handleMemoryList(ctx, msg, kStore, parts)
	case "show":
		return g.handleMemoryShow(ctx, msg, kStore, parts)
	case "approve":
		return g.handleMemoryApprove(ctx, msg, kStore, indexer, parts)
	case "reject":
		return g.handleMemoryReject(ctx, msg, kStore, parts)
	case "forget":
		return g.handleMemoryForget(ctx, msg, kStore, indexer, parts)
	default:
		helpText := "🧠 **Memory Commands:**\n" +
			"• `/memory list [status]` — List memories (active, pending, rejected, expired, forgotten, all)\n" +
			"• `/memory show <id>` — View details and provenance of an item or version\n" +
			"• `/memory approve <version-id>` — Approve a proposed memory version\n" +
			"• `/memory reject <version-id>` — Reject a proposed memory version\n" +
			"• `/memory forget <item-id>` — Permanently erase a memory"
		return g.SendMessage(msg.ChatID, helpText)
	}
}

func (g *Gateway) handleMemoryList(ctx context.Context, msg models.Message, store *knowledge.Store, parts []string) error {
	statusFilter := "active"
	if len(parts) > 2 {
		statusFilter = strings.ToLower(parts[2])
	}

	validFilters := map[string]bool{
		"active":    true,
		"pending":   true,
		"rejected":  true,
		"expired":   true,
		"forgotten": true,
		"all":       true,
	}
	if !validFilters[statusFilter] {
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Invalid status filter %q. Valid memory filters: pending, active, rejected, expired, forgotten, all.", statusFilter))
	}

	items, err := store.ListItems(ctx, msg.ChatID, knowledge.KindMemory, statusFilter)
	if err != nil {
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Failed to list memories: %v", err))
	}

	if len(items) == 0 {
		return g.SendMessage(msg.ChatID, fmt.Sprintf("ℹ️ No memories found with status %q.", statusFilter))
	}

	var b strings.Builder
	fmt.Fprintf(&b, "🧠 **Memories** (%s)\n\n", statusFilter)
	b.WriteString("| ID | Type | Snippet | Rev | Status |\n")
	b.WriteString("|---|---|---|---|---|\n")

	for _, item := range items {
		vers, err := store.ListVersions(ctx, item.ID)
		var targetVer *knowledge.MemoryVersion
		if err == nil && len(vers) > 0 {
		findMemVer:
			for i := len(vers) - 1; i >= 0; i-- {
				mv, ok := vers[i].(*knowledge.MemoryVersion)
				if !ok {
					continue
				}
				switch statusFilter {
				case "pending":
					if mv.Status == knowledge.VersionStatusProposed {
						targetVer = mv
						break findMemVer
					}
				case "rejected":
					if mv.Status == knowledge.VersionStatusRejected {
						targetVer = mv
						break findMemVer
					}
				case "expired":
					if mv.Status == knowledge.VersionStatusExpired {
						targetVer = mv
						break findMemVer
					}
				case "active":
					if mv.Status == knowledge.VersionStatusApproved && mv.ID == item.ActiveVersionID {
						targetVer = mv
						break findMemVer
					}
				default:
					if item.ActiveVersionID != "" && mv.ID == item.ActiveVersionID {
						targetVer = mv
						break findMemVer
					}
				}
			}
			if targetVer == nil {
				// Fallback to active version
				if item.ActiveVersionID != "" {
					for _, v := range vers {
						if mv, ok := v.(*knowledge.MemoryVersion); ok && mv.ID == item.ActiveVersionID {
							targetVer = mv
							break
						}
					}
				}
				// Or highest revision
				if targetVer == nil {
					if mv, ok := vers[len(vers)-1].(*knowledge.MemoryVersion); ok {
						targetVer = mv
					}
				}
			}
		}

		memType := "-"
		snippet := "-"
		revStr := "-"
		statusStr := string(item.Status)

		if targetVer != nil {
			memType = string(targetVer.Type)
			snippet = knowledge.TruncateRunes(targetVer.Content, 50)
			snippet = strings.ReplaceAll(snippet, "|", "\\|")
			snippet = strings.ReplaceAll(snippet, "\n", " ")
			revStr = fmt.Sprintf("@%d", targetVer.Revision)
			if statusFilter == "pending" || statusFilter == "rejected" || statusFilter == "expired" || item.Status == knowledge.ItemStatusPending {
				statusStr = string(targetVer.Status)
			}
		}

		fmt.Fprintf(&b, "| `%s` | %s | %s | %s | %s |\n", item.ID, memType, snippet, revStr, statusStr)
	}

	return g.SendMessage(msg.ChatID, b.String())
}

func (g *Gateway) handleMemoryShow(ctx context.Context, msg models.Message, store *knowledge.Store, parts []string) error {
	if len(parts) < 3 {
		return g.SendMessage(msg.ChatID, "⚠️ Please specify a memory ID. Usage: `/memory show <id>` (e.g. `/memory show mem_123` or `/memory show mem_123@1`)")
	}

	ref := strings.TrimSpace(parts[2])
	if !strings.HasPrefix(ref, knowledge.PrefixMemory) {
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Invalid memory identifier %q: memory IDs must begin with %q.", ref, knowledge.PrefixMemory))
	}

	itemID, rev, hasRev, err := knowledge.ParseItemOrVersionRef(ref)
	if err != nil {
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Invalid memory reference %q: %v", ref, err))
	}

	if hasRev {
		verID := knowledge.FormatVersionID(itemID, rev)
		ver, err := store.GetMemoryVersion(ctx, verID)
		if err != nil {
			if errors.Is(err, knowledge.ErrNotFound) {
				return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Memory version `%s` not found.", verID))
			}
			return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Failed to get memory version: %v", err))
		}
		if ver.Provenance.ChatID != msg.ChatID || ver.Provenance.UserID != msg.UserID {
			return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Memory version `%s` not found.", verID))
		}
		return g.sendMemoryVersionDetails(msg.ChatID, ver)
	}

	// Show item overview
	item, err := store.GetItem(ctx, itemID)
	if err != nil {
		if errors.Is(err, knowledge.ErrNotFound) {
			return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Memory item `%s` not found.", itemID))
		}
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Failed to get memory: %v", err))
	}

	if item.Kind != knowledge.KindMemory || item.ChatID != msg.ChatID || item.UserID != msg.UserID {
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Memory item `%s` not found.", itemID))
	}

	vers, err := store.ListVersions(ctx, itemID)
	if err != nil {
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Failed to get memory versions: %v", err))
	}

	var b strings.Builder
	fmt.Fprintf(&b, "🧠 **Memory Item: `%s`**\n\n", item.ID)
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
		b.WriteString("| Revision | Status | Type | Snippet |\n")
		b.WriteString("|---|---|---|---|\n")
		for _, v := range vers {
			mv, ok := v.(*knowledge.MemoryVersion)
			if ok {
				snip := strings.ReplaceAll(knowledge.TruncateRunes(mv.Content, 40), "|", "\\|")
				snip = strings.ReplaceAll(snip, "\n", " ")
				fmt.Fprintf(&b, "| `@%d` | %s | %s | %s |\n", mv.Revision, mv.Status, mv.Type, snip)
			}
		}
	}

	return g.SendMessage(msg.ChatID, b.String())
}

func (g *Gateway) sendMemoryVersionDetails(chatID string, ver *knowledge.MemoryVersion) error {
	var b strings.Builder
	fmt.Fprintf(&b, "🧠 **Memory Version: `%s`**\n\n", ver.ID)
	fmt.Fprintf(&b, "• **Item ID:** `%s`\n", ver.ItemID)
	fmt.Fprintf(&b, "• **Revision:** %d\n", ver.Revision)
	fmt.Fprintf(&b, "• **Type:** %s\n", ver.Type)
	fmt.Fprintf(&b, "• **Status:** %s\n", ver.Status)
	fmt.Fprintf(&b, "• **Index Status:** %s\n", ver.IndexStatus)
	fmt.Fprintf(&b, "• **Confidence:** %.2f\n", ver.Confidence)

	if ver.TTLDays != nil {
		fmt.Fprintf(&b, "• **TTL:** %d days\n", *ver.TTLDays)
	}
	if ver.ExpiresAt != nil {
		fmt.Fprintf(&b, "• **Expires At:** %s\n", time.Unix(*ver.ExpiresAt, 0).UTC().Format(time.RFC3339))
	}

	b.WriteString("\n**Content:**\n" + formatAdaptiveFence(ver.Content, "") + "\n\n")

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

func (g *Gateway) handleMemoryApprove(ctx context.Context, msg models.Message, store *knowledge.Store, indexer *knowledge.Indexer, parts []string) error {
	if len(parts) < 3 {
		return g.SendMessage(msg.ChatID, "⚠️ Please specify a version ID to approve. Usage: `/memory approve <version-id>` (e.g. `/memory approve mem_123@1`)")
	}

	versionID := strings.TrimSpace(parts[2])
	if !strings.HasPrefix(versionID, knowledge.PrefixMemory) {
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Invalid memory version ID %q: memory IDs must begin with %q.", versionID, knowledge.PrefixMemory))
	}

	itemID, _, err := knowledge.ParseVersionID(versionID)
	if err != nil {
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Invalid version ID format %q: %v", versionID, err))
	}

	item, err := store.GetItem(ctx, itemID)
	if err != nil {
		if errors.Is(err, knowledge.ErrNotFound) {
			return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Memory version `%s` not found.", versionID))
		}
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Failed to get memory: %v", err))
	}
	if item.Kind != knowledge.KindMemory || item.ChatID != msg.ChatID || item.UserID != msg.UserID {
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Memory version `%s` not found.", versionID))
	}

	prevActiveID := item.ActiveVersionID

	if err := store.ApproveVersion(ctx, versionID, msg.UserID); err != nil {
		if errors.Is(err, knowledge.ErrNotFound) {
			return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Memory version `%s` not found.", versionID))
		}
		if errors.Is(err, knowledge.ErrInvalidStatus) {
			return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Cannot approve memory version `%s`: %v", versionID, err))
		}
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Failed to approve memory version: %v", err))
	}

	// Trigger indexing and purge superseded index record
	if indexer != nil {
		if idxErr := indexer.IndexVersion(ctx, versionID); idxErr != nil {
			slog.Error("failed to index approved memory version", "version_id", versionID, "error", idxErr)
			return g.SendMessage(msg.ChatID, fmt.Sprintf("✅ Memory version `%s` approved, but indexing failed: %v", versionID, idxErr))
		}
		if prevActiveID != "" && prevActiveID != versionID {
			_ = store.EnqueueIndexDeletion(ctx, prevActiveID, knowledge.NamespaceMemories)
			if remErr := indexer.RemoveFromIndex(ctx, prevActiveID); remErr != nil {
				slog.Error("failed to remove superseded memory version from index", "version_id", prevActiveID, "error", remErr)
			} else {
				_ = store.RemovePendingIndexDeletion(ctx, prevActiveID)
			}
		}
	}

	return g.SendMessage(msg.ChatID, fmt.Sprintf("✅ Memory version `%s` has been approved and activated.", versionID))
}

func (g *Gateway) handleMemoryReject(ctx context.Context, msg models.Message, store *knowledge.Store, parts []string) error {
	if len(parts) < 3 {
		return g.SendMessage(msg.ChatID, "⚠️ Please specify a version ID to reject. Usage: `/memory reject <version-id>` (e.g. `/memory reject mem_123@1`)")
	}

	versionID := strings.TrimSpace(parts[2])
	if !strings.HasPrefix(versionID, knowledge.PrefixMemory) {
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Invalid memory version ID %q: memory IDs must begin with %q.", versionID, knowledge.PrefixMemory))
	}

	itemID, _, err := knowledge.ParseVersionID(versionID)
	if err != nil {
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Invalid version ID format %q: %v", versionID, err))
	}

	item, err := store.GetItem(ctx, itemID)
	if err != nil {
		if errors.Is(err, knowledge.ErrNotFound) {
			return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Memory version `%s` not found.", versionID))
		}
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Failed to get memory: %v", err))
	}
	if item.Kind != knowledge.KindMemory || item.ChatID != msg.ChatID || item.UserID != msg.UserID {
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Memory version `%s` not found.", versionID))
	}

	if err := store.RejectVersion(ctx, versionID, msg.UserID); err != nil {
		if errors.Is(err, knowledge.ErrNotFound) {
			return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Memory version `%s` not found.", versionID))
		}
		if errors.Is(err, knowledge.ErrInvalidStatus) {
			return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Cannot reject memory version `%s`: %v", versionID, err))
		}
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Failed to reject memory version: %v", err))
	}

	return g.SendMessage(msg.ChatID, fmt.Sprintf("❌ Memory version `%s` has been rejected.", versionID))
}

func (g *Gateway) handleMemoryForget(ctx context.Context, msg models.Message, store *knowledge.Store, indexer *knowledge.Indexer, parts []string) error {
	if len(parts) < 3 {
		return g.SendMessage(msg.ChatID, "⚠️ Please specify a memory item ID to forget. Usage: `/memory forget <item-id>` (e.g. `/memory forget mem_123`)")
	}

	ref := strings.TrimSpace(parts[2])
	if !strings.HasPrefix(ref, knowledge.PrefixMemory) {
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Invalid memory identifier %q: memory IDs must begin with %q.", ref, knowledge.PrefixMemory))
	}

	itemID, _, hasRev, err := knowledge.ParseItemOrVersionRef(ref)
	if err != nil {
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Invalid memory reference %q: %v", ref, err))
	}
	if hasRev {
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ `/memory forget` operates on entire memory items, not specific versions. Please use item ID `%s` without revision.", itemID))
	}

	item, err := store.GetItem(ctx, itemID)
	if err != nil {
		if errors.Is(err, knowledge.ErrNotFound) {
			return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Memory `%s` not found.", itemID))
		}
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Failed to get memory: %v", err))
	}
	if item.Kind != knowledge.KindMemory || item.ChatID != msg.ChatID || item.UserID != msg.UserID {
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Memory `%s` not found.", itemID))
	}

	// Capture all version IDs before deletion so all indexed records are purged durably
	vers, _ := store.ListVersions(ctx, itemID)
	for _, v := range vers {
		_ = store.EnqueueIndexDeletion(ctx, v.GetID(), knowledge.NamespaceMemories)
	}

	if err := store.ForgetItem(ctx, itemID, msg.UserID); err != nil {
		if errors.Is(err, knowledge.ErrNotFound) {
			return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Memory `%s` not found.", itemID))
		}
		return g.SendMessage(msg.ChatID, fmt.Sprintf("⚠️ Failed to forget memory: %v", err))
	}

	allPurged := true
	if indexer != nil {
		for _, v := range vers {
			if remErr := indexer.RemoveFromIndex(ctx, v.GetID()); remErr != nil {
				slog.Error("failed to remove forgotten memory version from index", "version_id", v.GetID(), "error", remErr)
				allPurged = false
			} else {
				_ = store.RemovePendingIndexDeletion(ctx, v.GetID())
			}
		}
	} else {
		allPurged = false
	}

	notice := fmt.Sprintf("Memory `%s` has been forgotten and erased from structured memory. *Note: Past conversation transcripts and raw message history remain unchanged.*", itemID)
	if !allPurged {
		notice += " (Background index cleanup is queued)."
	}
	return g.SendMessage(msg.ChatID, notice)
}

func formatAdaptiveFence(content, lang string) string {
	maxRun := 2
	cur := 0
	for _, r := range content {
		if r == '`' {
			cur++
			if cur > maxRun {
				maxRun = cur
			}
		} else {
			cur = 0
		}
	}
	fence := strings.Repeat("`", maxRun+1)
	return fmt.Sprintf("%s%s\n%s\n%s", fence, lang, content, fence)
}
