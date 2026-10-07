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

// MemoryHandler handles /memory slash commands.
type MemoryHandler struct {
	Store      *knowledge.Store
	Indexer    *knowledge.Indexer
	Authorizer Authorizer
	BotID      string
}

// NewMemoryHandler creates a new MemoryHandler.
func NewMemoryHandler(store *knowledge.Store, indexer *knowledge.Indexer, auth Authorizer, botID string) *MemoryHandler {
	return &MemoryHandler{
		Store:      store,
		Indexer:    indexer,
		Authorizer: auth,
		BotID:      botID,
	}
}

// Handle executes the /memory command specified by req.
func (h *MemoryHandler) Handle(ctx context.Context, req Request) (Result, error) {
	if h.Authorizer == nil {
		return Result{Reply: "⚠️ Authorization failed: authorizer not configured."}, nil
	}
	if err := h.Authorizer.Authorize(ctx, req.Session, req.Actor, "memory", req.Subcommand); err != nil {
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
	case "forget":
		return h.handleForget(ctx, req)
	default:
		helpText := "🧠 **Memory Commands:**\n" +
			"• `/memory list [status]` — List memories (active, pending, rejected, expired, forgotten, all)\n" +
			"• `/memory show <id>` — View details and provenance of an item or version\n" +
			"• `/memory approve <version-id>` — Approve a proposed memory version\n" +
			"• `/memory reject <version-id>` — Reject a proposed memory version\n" +
			"• `/memory forget <item-id>` — Permanently erase a memory"
		return Result{Reply: helpText}, nil
	}
}

func (h *MemoryHandler) handleList(ctx context.Context, req Request) (Result, error) {
	statusFilter := "active"
	if len(req.Args) > 0 {
		statusFilter = strings.ToLower(req.Args[0])
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
		return Result{Reply: fmt.Sprintf("⚠️ Invalid status filter %q. Valid memory filters: pending, active, rejected, expired, forgotten, all.", statusFilter)}, nil
	}

	items, err := h.Store.ListItems(ctx, req.Session.ScopeID, knowledge.KindMemory, statusFilter)
	if err != nil {
		return Result{Reply: fmt.Sprintf("⚠️ Failed to list memories: %v", err)}, nil
	}

	if len(items) == 0 {
		return Result{Reply: fmt.Sprintf("ℹ️ No memories found with status %q.", statusFilter)}, nil
	}

	var b strings.Builder
	fmt.Fprintf(&b, "🧠 **Memories** (%s)\n\n", statusFilter)
	b.WriteString("| ID | Type | Snippet | Rev | Status |\n")
	b.WriteString("|---|---|---|---|---|\n")

	now := time.Now().Unix()

	for _, item := range items {
		vers, err := h.Store.ListVersions(ctx, item.ID)
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
					if mv.Status == knowledge.VersionStatusExpired || (mv.ExpiresAt != nil && *mv.ExpiresAt <= now) {
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
			if targetVer == nil && statusFilter != "expired" {
				if item.ActiveVersionID != "" {
					for _, v := range vers {
						if mv, ok := v.(*knowledge.MemoryVersion); ok && mv.ID == item.ActiveVersionID {
							targetVer = mv
							break
						}
					}
				}
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
			if statusFilter == "expired" || (targetVer.ExpiresAt != nil && *targetVer.ExpiresAt <= now) {
				statusStr = "expired"
			} else if statusFilter == "pending" || statusFilter == "rejected" || item.Status == knowledge.ItemStatusPending {
				statusStr = string(targetVer.Status)
			}
		}

		fmt.Fprintf(&b, "| `%s` | %s | %s | %s | %s |\n", item.ID, memType, snippet, revStr, statusStr)
	}

	return Result{Reply: b.String()}, nil
}

func (h *MemoryHandler) handleShow(ctx context.Context, req Request) (Result, error) {
	if len(req.Args) == 0 {
		return Result{Reply: "⚠️ Please specify a memory ID. Usage: `/memory show <id>` (e.g. `/memory show mem_123` or `/memory show mem_123@1`)"}, nil
	}

	ref := strings.TrimSpace(req.Args[0])
	if !strings.HasPrefix(ref, knowledge.PrefixMemory) {
		return Result{Reply: fmt.Sprintf("⚠️ Invalid memory identifier %q: memory IDs must begin with %q.", ref, knowledge.PrefixMemory)}, nil
	}

	itemID, rev, hasRev, err := knowledge.ParseItemOrVersionRef(ref)
	if err != nil {
		return Result{Reply: fmt.Sprintf("⚠️ Invalid memory reference %q: %v", ref, err)}, nil
	}

	if hasRev {
		verID := knowledge.FormatVersionID(itemID, rev)
		ver, err := h.Store.GetMemoryVersion(ctx, verID)
		if err != nil {
			if errors.Is(err, knowledge.ErrNotFound) {
				return Result{Reply: fmt.Sprintf("⚠️ Memory version `%s` not found.", verID)}, nil
			}
			return Result{Reply: fmt.Sprintf("⚠️ Failed to get memory version: %v", err)}, nil
		}
		if ver.Provenance.ChatID != req.Session.ScopeID || ver.Provenance.UserID != req.Actor.ID {
			return Result{Reply: fmt.Sprintf("⚠️ Memory version `%s` not found.", verID)}, nil
		}
		return Result{Reply: h.formatMemoryVersionDetails(ver)}, nil
	}

	item, err := h.Store.GetItem(ctx, itemID)
	if err != nil {
		if errors.Is(err, knowledge.ErrNotFound) {
			return Result{Reply: fmt.Sprintf("⚠️ Memory item `%s` not found.", itemID)}, nil
		}
		return Result{Reply: fmt.Sprintf("⚠️ Failed to get memory: %v", err)}, nil
	}

	if item.Kind != knowledge.KindMemory || item.ChatID != req.Session.ScopeID || item.UserID != req.Actor.ID {
		return Result{Reply: fmt.Sprintf("⚠️ Memory item `%s` not found.", itemID)}, nil
	}

	vers, err := h.Store.ListVersions(ctx, itemID)
	if err != nil {
		return Result{Reply: fmt.Sprintf("⚠️ Failed to get memory versions: %v", err)}, nil
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

	return Result{Reply: b.String()}, nil
}

func (h *MemoryHandler) formatMemoryVersionDetails(ver *knowledge.MemoryVersion) string {
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

	b.WriteString("\n**Content:**\n" + FormatAdaptiveFence(ver.Content, "") + "\n\n")

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

func (h *MemoryHandler) handleApprove(ctx context.Context, req Request) (Result, error) {
	if len(req.Args) == 0 {
		return Result{Reply: "⚠️ Please specify a version ID to approve. Usage: `/memory approve <version-id>` (e.g. `/memory approve mem_123@1`)"}, nil
	}

	versionID := strings.TrimSpace(req.Args[0])
	if !strings.HasPrefix(versionID, knowledge.PrefixMemory) {
		return Result{Reply: fmt.Sprintf("⚠️ Invalid memory version ID %q: memory IDs must begin with %q.", versionID, knowledge.PrefixMemory)}, nil
	}

	itemID, _, err := knowledge.ParseVersionID(versionID)
	if err != nil {
		return Result{Reply: fmt.Sprintf("⚠️ Invalid version ID format %q: %v", versionID, err)}, nil
	}

	item, err := h.Store.GetItem(ctx, itemID)
	if err != nil {
		if errors.Is(err, knowledge.ErrNotFound) {
			return Result{Reply: fmt.Sprintf("⚠️ Memory version `%s` not found.", versionID)}, nil
		}
		return Result{Reply: fmt.Sprintf("⚠️ Failed to get memory: %v", err)}, nil
	}
	if item.Kind != knowledge.KindMemory || item.ChatID != req.Session.ScopeID || item.UserID != req.Actor.ID {
		return Result{Reply: fmt.Sprintf("⚠️ Memory version `%s` not found.", versionID)}, nil
	}

	prevActiveID := item.ActiveVersionID

	if err := h.Store.ApproveVersion(ctx, versionID, req.Actor.ID); err != nil {
		if errors.Is(err, knowledge.ErrNotFound) {
			return Result{Reply: fmt.Sprintf("⚠️ Memory version `%s` not found.", versionID)}, nil
		}
		if errors.Is(err, knowledge.ErrInvalidStatus) {
			return Result{Reply: fmt.Sprintf("⚠️ Cannot approve memory version `%s`: %v", versionID, err)}, nil
		}
		return Result{Reply: fmt.Sprintf("⚠️ Failed to approve memory version: %v", err)}, nil
	}

	if h.Indexer != nil {
		if idxErr := h.Indexer.IndexVersion(ctx, versionID); idxErr != nil {
			slog.Error("failed to index approved memory version", "version_id", versionID, "error", idxErr)
			return Result{Reply: fmt.Sprintf("✅ Memory version `%s` approved, but indexing failed: %v", versionID, idxErr)}, nil
		}
		if prevActiveID != "" && prevActiveID != versionID {
			if _, purgeErr := h.Indexer.PurgePendingDeletion(ctx, prevActiveID); purgeErr != nil {
				slog.Warn("failed to purge superseded memory version from index", "version_id", prevActiveID, "error", purgeErr)
			}
		}
	}

	return Result{Reply: fmt.Sprintf("✅ Memory version `%s` has been approved and activated.", versionID)}, nil
}

func (h *MemoryHandler) handleReject(ctx context.Context, req Request) (Result, error) {
	if len(req.Args) == 0 {
		return Result{Reply: "⚠️ Please specify a version ID to reject. Usage: `/memory reject <version-id>` (e.g. `/memory reject mem_123@1`)"}, nil
	}

	versionID := strings.TrimSpace(req.Args[0])
	if !strings.HasPrefix(versionID, knowledge.PrefixMemory) {
		return Result{Reply: fmt.Sprintf("⚠️ Invalid memory version ID %q: memory IDs must begin with %q.", versionID, knowledge.PrefixMemory)}, nil
	}

	itemID, _, err := knowledge.ParseVersionID(versionID)
	if err != nil {
		return Result{Reply: fmt.Sprintf("⚠️ Invalid version ID format %q: %v", versionID, err)}, nil
	}

	item, err := h.Store.GetItem(ctx, itemID)
	if err != nil {
		if errors.Is(err, knowledge.ErrNotFound) {
			return Result{Reply: fmt.Sprintf("⚠️ Memory version `%s` not found.", versionID)}, nil
		}
		return Result{Reply: fmt.Sprintf("⚠️ Failed to get memory: %v", err)}, nil
	}
	if item.Kind != knowledge.KindMemory || item.ChatID != req.Session.ScopeID || item.UserID != req.Actor.ID {
		return Result{Reply: fmt.Sprintf("⚠️ Memory version `%s` not found.", versionID)}, nil
	}

	if err := h.Store.RejectVersion(ctx, versionID, req.Actor.ID); err != nil {
		if errors.Is(err, knowledge.ErrNotFound) {
			return Result{Reply: fmt.Sprintf("⚠️ Memory version `%s` not found.", versionID)}, nil
		}
		if errors.Is(err, knowledge.ErrInvalidStatus) {
			return Result{Reply: fmt.Sprintf("⚠️ Cannot reject memory version `%s`: %v", versionID, err)}, nil
		}
		return Result{Reply: fmt.Sprintf("⚠️ Failed to reject memory version: %v", err)}, nil
	}

	return Result{Reply: fmt.Sprintf("❌ Memory version `%s` has been rejected.", versionID)}, nil
}

func (h *MemoryHandler) handleForget(ctx context.Context, req Request) (Result, error) {
	if len(req.Args) == 0 {
		return Result{Reply: "⚠️ Please specify a memory item ID to forget. Usage: `/memory forget <item-id>` (e.g. `/memory forget mem_123`)"}, nil
	}

	ref := strings.TrimSpace(req.Args[0])
	if !strings.HasPrefix(ref, knowledge.PrefixMemory) {
		return Result{Reply: fmt.Sprintf("⚠️ Invalid memory identifier %q: memory IDs must begin with %q.", ref, knowledge.PrefixMemory)}, nil
	}

	itemID, _, hasRev, err := knowledge.ParseItemOrVersionRef(ref)
	if err != nil {
		return Result{Reply: fmt.Sprintf("⚠️ Invalid memory reference %q: %v", ref, err)}, nil
	}
	if hasRev {
		return Result{Reply: fmt.Sprintf("⚠️ `/memory forget` operates on entire memory items, not specific versions. Please use item ID `%s` without revision.", itemID)}, nil
	}

	item, err := h.Store.GetItem(ctx, itemID)
	if err != nil {
		if errors.Is(err, knowledge.ErrNotFound) {
			return Result{Reply: fmt.Sprintf("⚠️ Memory `%s` not found.", itemID)}, nil
		}
		return Result{Reply: fmt.Sprintf("⚠️ Failed to get memory: %v", err)}, nil
	}
	if item.Kind != knowledge.KindMemory || item.ChatID != req.Session.ScopeID || item.UserID != req.Actor.ID {
		return Result{Reply: fmt.Sprintf("⚠️ Memory `%s` not found.", itemID)}, nil
	}

	vers, _ := h.Store.ListVersions(ctx, itemID)

	if err := h.Store.ForgetItem(ctx, itemID, req.Actor.ID); err != nil {
		if errors.Is(err, knowledge.ErrNotFound) {
			return Result{Reply: fmt.Sprintf("⚠️ Memory `%s` not found.", itemID)}, nil
		}
		return Result{Reply: fmt.Sprintf("⚠️ Failed to forget memory: %v", err)}, nil
	}

	allPurged := true
	if h.Indexer != nil {
		for _, v := range vers {
			if _, purgeErr := h.Indexer.PurgePendingDeletion(ctx, v.GetID()); purgeErr != nil {
				slog.Warn("failed to purge forgotten memory version from index", "version_id", v.GetID(), "error", purgeErr)
				allPurged = false
			}
		}
	} else {
		allPurged = false
	}

	notice := fmt.Sprintf("Memory `%s` has been forgotten and erased from structured memory. *Note: Past conversation transcripts and raw message history remain unchanged.*", itemID)
	if !allPurged {
		notice += " (Background index cleanup is queued)."
	}
	return Result{Reply: notice}, nil
}
