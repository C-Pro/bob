package knowledge

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"strconv"
	"strings"
)

// ItemKind represents whether a knowledge entity is a memory or a skill.
type ItemKind string

const (
	KindMemory ItemKind = "memory"
	KindSkill  ItemKind = "skill"

	PrefixMemory = "mem_"
	PrefixSkill  = "skill_"
)

// ItemStatus represents the lifecycle state of a knowledge entity.
type ItemStatus string

const (
	ItemStatusPending   ItemStatus = "pending"
	ItemStatusActive    ItemStatus = "active"
	ItemStatusDisabled  ItemStatus = "disabled"
	ItemStatusArchived  ItemStatus = "archived"
	ItemStatusForgotten ItemStatus = "forgotten"
)

// VersionStatus represents the approval and deployment state of a specific revision.
type VersionStatus string

const (
	VersionStatusProposed   VersionStatus = "proposed"
	VersionStatusApproved   VersionStatus = "approved"
	VersionStatusRejected   VersionStatus = "rejected"
	VersionStatusSuperseded VersionStatus = "superseded"
	VersionStatusExpired    VersionStatus = "expired"
	VersionStatusArchived   VersionStatus = "archived"
)

// MemoryType represents the semantic classification of a structured memory.
type MemoryType string

const (
	MemoryTypePreference  MemoryType = "preference"
	MemoryTypeFact        MemoryType = "fact"
	MemoryTypeDecision    MemoryType = "decision"
	MemoryTypeOngoingTask MemoryType = "ongoing_task"
)

// IndexStatus represents the synchronization state in CortexDB vector/FTS indexes.
type IndexStatus string

const (
	IndexStatusPending IndexStatus = "pending"
	IndexStatusReady   IndexStatus = "ready"
	IndexStatusError   IndexStatus = "error"
)

// Provenance captures tamper-resistant origin metadata for revisions.
type Provenance struct {
	ChatID           string `json:"chat_id"`
	UserID           string `json:"user_id"`
	SourceMessageSeq *int64 `json:"source_message_seq,omitempty"`
	FSMRunID         string `json:"fsm_run_id,omitempty"`
	ContentHash      string `json:"content_hash"`
	CreatedAt        int64  `json:"created_at"`
	ReviewedAt       *int64 `json:"reviewed_at,omitempty"`
	ReviewedBy       string `json:"reviewed_by,omitempty"`
}

// KnowledgeItem tracks the stable identity and lifecycle state of a knowledge entity.
type KnowledgeItem struct {
	ID              string     `json:"id"`
	ChatID          string     `json:"chat_id"`
	UserID          string     `json:"user_id"`
	Kind            ItemKind   `json:"kind"`
	Status          ItemStatus `json:"status"`
	ActiveVersionID string     `json:"active_version_id,omitempty"`
	CreatedAt       int64      `json:"created_at"`
	UpdatedAt       int64      `json:"updated_at"`
}

// MemoryVersion represents an immutable revision of a structured memory.
type MemoryVersion struct {
	ID          string        `json:"id"`
	ItemID      string        `json:"item_id"`
	Revision    int           `json:"revision"`
	Type        MemoryType    `json:"type"`
	Content     string        `json:"content"`
	Confidence  float64       `json:"confidence"`
	TTLDays     *int          `json:"ttl_days,omitempty"`
	ExpiresAt   *int64        `json:"expires_at,omitempty"`
	Status      VersionStatus `json:"status"`
	Provenance  Provenance    `json:"provenance"`
	IndexStatus IndexStatus   `json:"index_status"`
}

// SearchableContent builds the searchable representation for a structured memory version.
func (ver *MemoryVersion) SearchableContent() string {
	if ver == nil {
		return ""
	}
	return fmt.Sprintf("[%s] %s", ver.Type, ver.Content)
}

// IndexMetadata builds the metadata map for vector/FTS indexing.
func (ver *MemoryVersion) IndexMetadata() map[string]any {
	if ver == nil {
		return map[string]any{}
	}
	return map[string]any{
		"item_id":        ver.ItemID,
		"version_id":     ver.ID,
		"kind":           string(KindMemory),
		"knowledge_kind": string(KindMemory),
		"chat_id":        ver.Provenance.ChatID,
		"user_id":        ver.Provenance.UserID,
		"type":           string(ver.Type),
		"confidence":     ver.Confidence,
	}
}

// SkillVersion represents an immutable revision of a procedural skill.
type SkillVersion struct {
	ID                   string        `json:"id"`
	ItemID               string        `json:"item_id"`
	Revision             int           `json:"revision"`
	Name                 string        `json:"name"`
	Description          string        `json:"description"`
	Triggers             []string      `json:"triggers"`
	Tags                 []string      `json:"tags"`
	InstructionsMarkdown string        `json:"instructions_markdown"`
	Status               VersionStatus `json:"status"`
	Provenance           Provenance    `json:"provenance"`
	IndexStatus          IndexStatus   `json:"index_status"`
}

// SearchableContent builds the searchable representation for a skill version, combining metadata with procedural instructions.
func (ver *SkillVersion) SearchableContent() string {
	if ver == nil {
		return ""
	}
	var sb strings.Builder
	sb.WriteString("Skill: " + ver.Name + "\n")
	if ver.Description != "" {
		sb.WriteString("Description: " + ver.Description + "\n")
	}
	if len(ver.Triggers) > 0 {
		sb.WriteString("Triggers: " + strings.Join(ver.Triggers, ", ") + "\n")
	}
	if len(ver.Tags) > 0 {
		sb.WriteString("Tags: " + strings.Join(ver.Tags, ", ") + "\n")
	}
	sb.WriteString("\nInstructions:\n" + ver.InstructionsMarkdown)
	return sb.String()
}

// IndexMetadata builds the metadata map for vector/FTS indexing.
func (ver *SkillVersion) IndexMetadata() map[string]any {
	if ver == nil {
		return map[string]any{}
	}
	return map[string]any{
		"item_id":        ver.ItemID,
		"version_id":     ver.ID,
		"kind":           string(KindSkill),
		"knowledge_kind": string(KindSkill),
		"chat_id":        ver.Provenance.ChatID,
		"user_id":        ver.Provenance.UserID,
		"name":           ver.Name,
		"description":    ver.Description,
		"triggers":       ver.Triggers,
		"tags":           ver.Tags,
	}
}

// Tombstone preserves minimal audit metadata after an item's content is forgotten or deleted.
type Tombstone struct {
	ItemID    string   `json:"item_id"`
	Kind      ItemKind `json:"kind"`
	ChatID    string   `json:"chat_id"`
	UserID    string   `json:"user_id"`
	DeletedBy string   `json:"deleted_by"`
	DeletedAt int64    `json:"deleted_at"`
	Reason    string   `json:"reason,omitempty"`
}

// GenerateItemID creates a unique item identifier with the specified kind prefix (e.g. mem_... or skill_...).
func GenerateItemID(kind ItemKind) (string, error) {
	if !IsValidItemKind(kind) {
		return "", fmt.Errorf("invalid item kind: %q", kind)
	}
	b := make([]byte, 8)
	if _, err := rand.Read(b); err != nil {
		return "", fmt.Errorf("failed to generate random bytes for item ID: %w", err)
	}
	return fmt.Sprintf("%s_%x", kindPrefix(kind), b), nil
}

func kindPrefix(kind ItemKind) string {
	if kind == KindSkill {
		return "skill"
	}
	return "mem"
}

// FormatVersionID constructs the version identifier string "<item_id>@<revision>".
func FormatVersionID(itemID string, revision int) string {
	return fmt.Sprintf("%s@%d", itemID, revision)
}

// ParseVersionID strictly parses a version ID formatted as "<item_id>@<revision>".
func ParseVersionID(versionID string) (string, int, error) {
	parts := strings.Split(versionID, "@")
	if len(parts) != 2 || strings.TrimSpace(parts[0]) == "" || strings.TrimSpace(parts[1]) == "" {
		return "", 0, fmt.Errorf("invalid version ID %q: must be formatted as <item_id>@<revision>", versionID)
	}
	itemID := strings.TrimSpace(parts[0])
	rev, err := strconv.Atoi(strings.TrimSpace(parts[1]))
	if err != nil || rev <= 0 {
		return "", 0, fmt.Errorf("invalid revision in version ID %q: must be a positive integer", versionID)
	}
	return itemID, rev, nil
}

// ParseItemOrVersionRef parses either an item ID ("mem_xyz") or a version ID ("mem_xyz@2").
func ParseItemOrVersionRef(ref string) (itemID string, revision int, hasRevision bool, err error) {
	trimmed := strings.TrimSpace(ref)
	if trimmed == "" {
		return "", 0, false, errors.New("empty item or version reference")
	}
	if strings.Contains(trimmed, "@") {
		id, rev, err := ParseVersionID(trimmed)
		if err != nil {
			return "", 0, false, err
		}
		return id, rev, true, nil
	}
	return trimmed, 0, false, nil
}

// ComputeContentHash computes the SHA-256 digest in hex of the given content string.
func ComputeContentHash(content string) string {
	sum := sha256.Sum256([]byte(content))
	return hex.EncodeToString(sum[:])
}

// IsValidItemKind validates if the kind represents a supported ItemKind.
func IsValidItemKind(k ItemKind) bool {
	switch k {
	case KindMemory, KindSkill:
		return true
	default:
		return false
	}
}

// IsValidMemoryType validates if the memory type represents a supported MemoryType.
func IsValidMemoryType(t MemoryType) bool {
	switch t {
	case MemoryTypePreference, MemoryTypeFact, MemoryTypeDecision, MemoryTypeOngoingTask:
		return true
	default:
		return false
	}
}

// IsValidItemStatus validates if the item status represents a supported ItemStatus.
func IsValidItemStatus(s ItemStatus) bool {
	switch s {
	case ItemStatusPending, ItemStatusActive, ItemStatusDisabled, ItemStatusArchived, ItemStatusForgotten:
		return true
	default:
		return false
	}
}

// IsValidVersionStatus validates if the version status represents a supported VersionStatus.
func IsValidVersionStatus(s VersionStatus) bool {
	switch s {
	case VersionStatusProposed, VersionStatusApproved, VersionStatusRejected, VersionStatusSuperseded, VersionStatusExpired, VersionStatusArchived:
		return true
	default:
		return false
	}
}

// IsValidIndexStatus validates if the index status represents a supported IndexStatus.
func IsValidIndexStatus(s IndexStatus) bool {
	switch s {
	case IndexStatusPending, IndexStatusReady, IndexStatusError:
		return true
	default:
		return false
	}
}

// Common knowledge errors.
var (
	ErrNotFound      = errors.New("knowledge item or version not found")
	ErrInvalidStatus = errors.New("invalid status transition")
	ErrUnauthorized  = errors.New("unauthorized action on knowledge item")
)

// Version represents a version record of either a memory or a skill.
type Version interface {
	GetID() string
	GetItemID() string
	GetRevision() int
	GetStatus() VersionStatus
	GetProvenance() Provenance
	GetIndexStatus() IndexStatus
	GetKind() ItemKind
}

func (v *MemoryVersion) GetID() string               { return v.ID }
func (v *MemoryVersion) GetItemID() string           { return v.ItemID }
func (v *MemoryVersion) GetRevision() int            { return v.Revision }
func (v *MemoryVersion) GetStatus() VersionStatus    { return v.Status }
func (v *MemoryVersion) GetProvenance() Provenance   { return v.Provenance }
func (v *MemoryVersion) GetIndexStatus() IndexStatus { return v.IndexStatus }
func (v *MemoryVersion) GetKind() ItemKind           { return KindMemory }

func (v *SkillVersion) GetID() string               { return v.ID }
func (v *SkillVersion) GetItemID() string           { return v.ItemID }
func (v *SkillVersion) GetRevision() int            { return v.Revision }
func (v *SkillVersion) GetStatus() VersionStatus    { return v.Status }
func (v *SkillVersion) GetProvenance() Provenance   { return v.Provenance }
func (v *SkillVersion) GetIndexStatus() IndexStatus { return v.IndexStatus }
func (v *SkillVersion) GetKind() ItemKind           { return KindSkill }

// PendingIndexDeletion represents a version queued for durable index removal.
type PendingIndexDeletion struct {
	VersionID string `json:"version_id"`
	Namespace string `json:"namespace"`
	CreatedAt int64  `json:"created_at"`
}

