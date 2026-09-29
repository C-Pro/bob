package knowledge

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"
)

// Store provides isolated SQLite relational persistence for knowledge items and versions.
type Store struct {
	db *sql.DB
}

// NewStore creates a new knowledge Store backed by db.
func NewStore(db *sql.DB) *Store {
	return &Store{db: db}
}

// DB returns the underlying SQLite connection.
func (s *Store) DB() *sql.DB {
	return s.db
}

func validateActor(userID string) error {
	if strings.TrimSpace(userID) == "" {
		return fmt.Errorf("%w: user id cannot be empty", ErrUnauthorized)
	}
	return nil
}

// ProposeMemory creates a new memory item or appends a new proposed revision to an existing memory item.
func (s *Store) ProposeMemory(ctx context.Context, item *KnowledgeItem, version *MemoryVersion) (*KnowledgeItem, *MemoryVersion, error) {
	if s.db == nil {
		return nil, nil, errors.New("database connection is nil")
	}
	if version == nil {
		return nil, nil, errors.New("memory version cannot be nil")
	}
	if !IsValidMemoryType(version.Type) {
		return nil, nil, fmt.Errorf("%w: invalid memory type %q", ErrInvalidStatus, version.Type)
	}
	if strings.TrimSpace(version.Content) == "" {
		return nil, nil, errors.New("memory content cannot be empty")
	}
	if version.Confidence < 0.0 || version.Confidence > 1.0 {
		return nil, nil, fmt.Errorf("confidence must be between 0.0 and 1.0, got %f", version.Confidence)
	}

	now := time.Now().Unix()
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to begin transaction: %w", err)
	}
	defer func() { _ = tx.Rollback() }()

	isNewItem := false
	targetItemID := version.ItemID
	if targetItemID == "" && item != nil {
		targetItemID = item.ID
	}

	if targetItemID != "" {
		if !strings.HasPrefix(targetItemID, "mem_") {
			return nil, nil, fmt.Errorf("%w: invalid memory item id %q", ErrInvalidStatus, targetItemID)
		}
		// Revision of existing item
		var existing KnowledgeItem
		var activeVer sql.NullString
		err := tx.QueryRowContext(ctx, `
			SELECT id, chat_id, user_id, kind, status, active_version_id, created_at, updated_at
			FROM knowledge_items WHERE id = ?
		`, targetItemID).Scan(
			&existing.ID, &existing.ChatID, &existing.UserID, &existing.Kind,
			&existing.Status, &activeVer, &existing.CreatedAt, &existing.UpdatedAt,
		)
		if err != nil {
			if errors.Is(err, sql.ErrNoRows) {
				return nil, nil, fmt.Errorf("%w: item %s not found", ErrNotFound, targetItemID)
			}
			return nil, nil, fmt.Errorf("failed to query knowledge item: %w", err)
		}
		if existing.Kind != KindMemory {
			return nil, nil, fmt.Errorf("item %s is a %s, not a memory", targetItemID, existing.Kind)
		}
		if existing.Status == ItemStatusForgotten || existing.Status == ItemStatusArchived {
			return nil, nil, fmt.Errorf("%w: cannot propose revision for %s item %s", ErrInvalidStatus, existing.Status, targetItemID)
		}
		actorID := version.Provenance.UserID
		if actorID == "" && item != nil {
			actorID = item.UserID
		}
		if err := validateActor(actorID); err != nil {
			return nil, nil, err
		}
		if actorID != existing.UserID {
			return nil, nil, fmt.Errorf("%w: user %s cannot propose revision for item owned by %s", ErrUnauthorized, actorID, existing.UserID)
		}
		actorChatID := version.Provenance.ChatID
		if actorChatID == "" && item != nil {
			actorChatID = item.ChatID
		}
		if actorChatID != "" && actorChatID != existing.ChatID {
			return nil, nil, fmt.Errorf("%w: chat %s cannot propose revision for item in chat %s", ErrUnauthorized, actorChatID, existing.ChatID)
		}
		if existing.Status == ItemStatusRejected && (!activeVer.Valid || activeVer.String == "") {
			_, err = tx.ExecContext(ctx, "UPDATE knowledge_items SET status = 'pending', updated_at = ? WHERE id = ?", now, targetItemID)
			if err != nil {
				return nil, nil, fmt.Errorf("failed to transition rejected item to pending: %w", err)
			}
			existing.Status = ItemStatusPending
		}
		if activeVer.Valid {
			existing.ActiveVersionID = activeVer.String
		}
		item = &existing
	} else {
		// New item
		isNewItem = true
		if item == nil {
			item = &KnowledgeItem{}
		}
		if err := validateActor(item.UserID); err != nil {
			return nil, nil, err
		}
		if strings.TrimSpace(item.ChatID) == "" {
			return nil, nil, fmt.Errorf("%w: chat id cannot be empty", ErrUnauthorized)
		}
		if item.ID == "" {
			generatedID, err := GenerateItemID(KindMemory)
			if err != nil {
				return nil, nil, err
			}
			item.ID = generatedID
		} else if !strings.HasPrefix(item.ID, "mem_") {
			return nil, nil, fmt.Errorf("%w: memory item id %q must have prefix %q", ErrInvalidStatus, item.ID, "mem_")
		}
		item.Kind = KindMemory
		item.Status = ItemStatusPending
		item.CreatedAt = now
		item.UpdatedAt = now
		item.ActiveVersionID = ""

		_, err := tx.ExecContext(ctx, `
			INSERT INTO knowledge_items (id, chat_id, user_id, kind, status, active_version_id, created_at, updated_at)
			VALUES (?, ?, ?, ?, ?, NULL, ?, ?)
		`, item.ID, item.ChatID, item.UserID, item.Kind, item.Status, item.CreatedAt, item.UpdatedAt)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to insert knowledge item: %w", err)
		}
	}

	// Calculate next revision
	var maxRev int
	err = tx.QueryRowContext(ctx, `
		SELECT COALESCE(MAX(revision), 0) FROM memory_versions WHERE item_id = ?
	`, item.ID).Scan(&maxRev)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to get max revision: %w", err)
	}

	version.ItemID = item.ID
	version.Revision = maxRev + 1
	version.ID = FormatVersionID(item.ID, version.Revision)
	version.Status = VersionStatusProposed
	version.IndexStatus = IndexStatusPending
	version.Provenance.ChatID = item.ChatID
	version.Provenance.UserID = item.UserID
	version.Provenance.CreatedAt = now
	version.Provenance.ContentHash = ComputeContentHash(version.Content)

	if version.TTLDays != nil {
		if *version.TTLDays <= 0 {
			return nil, nil, fmt.Errorf("ttl_days must be positive, got %d", *version.TTLDays)
		}
		exp := now + int64(*version.TTLDays)*86400
		version.ExpiresAt = &exp
	}

	_, err = tx.ExecContext(ctx, `
		INSERT INTO memory_versions (
			id, item_id, revision, type, content, confidence, ttl_days, expires_at,
			status, chat_id, user_id, source_message_seq, fsm_run_id, content_hash,
			created_at, reviewed_at, reviewed_by, index_status
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?)
	`,
		version.ID, version.ItemID, version.Revision, version.Type, version.Content,
		version.Confidence, version.TTLDays, version.ExpiresAt, version.Status,
		version.Provenance.ChatID, version.Provenance.UserID, version.Provenance.SourceMessageSeq,
		nullIfEmpty(version.Provenance.FSMRunID), version.Provenance.ContentHash,
		version.Provenance.CreatedAt, version.IndexStatus,
	)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to insert memory version: %w", err)
	}

	if !isNewItem {
		item.UpdatedAt = now
		_, err = tx.ExecContext(ctx, `
			UPDATE knowledge_items SET updated_at = ? WHERE id = ?
		`, now, item.ID)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to update knowledge item updated_at: %w", err)
		}
	}

	if err := tx.Commit(); err != nil {
		return nil, nil, fmt.Errorf("failed to commit propose memory: %w", err)
	}

	return item, version, nil
}

// ProposeSkill creates a new skill item or appends a new proposed revision to an existing skill item.
func (s *Store) ProposeSkill(ctx context.Context, item *KnowledgeItem, version *SkillVersion) (*KnowledgeItem, *SkillVersion, error) {
	if s.db == nil {
		return nil, nil, errors.New("database connection is nil")
	}
	if version == nil {
		return nil, nil, errors.New("skill version cannot be nil")
	}
	if strings.TrimSpace(version.Name) == "" {
		return nil, nil, errors.New("skill name cannot be empty")
	}
	if strings.TrimSpace(version.InstructionsMarkdown) == "" {
		return nil, nil, errors.New("skill instructions_markdown cannot be empty")
	}

	now := time.Now().Unix()
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to begin transaction: %w", err)
	}
	defer func() { _ = tx.Rollback() }()

	isNewItem := false
	targetItemID := version.ItemID
	if targetItemID == "" && item != nil {
		targetItemID = item.ID
	}

	if targetItemID != "" {
		if !strings.HasPrefix(targetItemID, "skill_") {
			return nil, nil, fmt.Errorf("%w: invalid skill item id %q", ErrInvalidStatus, targetItemID)
		}
		var existing KnowledgeItem
		var activeVer sql.NullString
		err := tx.QueryRowContext(ctx, `
			SELECT id, chat_id, user_id, kind, status, active_version_id, created_at, updated_at
			FROM knowledge_items WHERE id = ?
		`, targetItemID).Scan(
			&existing.ID, &existing.ChatID, &existing.UserID, &existing.Kind,
			&existing.Status, &activeVer, &existing.CreatedAt, &existing.UpdatedAt,
		)
		if err != nil {
			if errors.Is(err, sql.ErrNoRows) {
				return nil, nil, fmt.Errorf("%w: item %s not found", ErrNotFound, targetItemID)
			}
			return nil, nil, fmt.Errorf("failed to query knowledge item: %w", err)
		}
		if existing.Kind != KindSkill {
			return nil, nil, fmt.Errorf("item %s is a %s, not a skill", targetItemID, existing.Kind)
		}
		if existing.Status == ItemStatusForgotten || existing.Status == ItemStatusArchived {
			return nil, nil, fmt.Errorf("%w: cannot propose revision for %s item %s", ErrInvalidStatus, existing.Status, targetItemID)
		}
		actorID := version.Provenance.UserID
		if actorID == "" && item != nil {
			actorID = item.UserID
		}
		if err := validateActor(actorID); err != nil {
			return nil, nil, err
		}
		if actorID != existing.UserID {
			return nil, nil, fmt.Errorf("%w: user %s cannot propose revision for item owned by %s", ErrUnauthorized, actorID, existing.UserID)
		}
		actorChatID := version.Provenance.ChatID
		if actorChatID == "" && item != nil {
			actorChatID = item.ChatID
		}
		if actorChatID != "" && actorChatID != existing.ChatID {
			return nil, nil, fmt.Errorf("%w: chat %s cannot propose revision for item in chat %s", ErrUnauthorized, actorChatID, existing.ChatID)
		}
		if existing.Status == ItemStatusRejected && (!activeVer.Valid || activeVer.String == "") {
			_, err = tx.ExecContext(ctx, "UPDATE knowledge_items SET status = 'pending', updated_at = ? WHERE id = ?", now, targetItemID)
			if err != nil {
				return nil, nil, fmt.Errorf("failed to transition rejected item to pending: %w", err)
			}
			existing.Status = ItemStatusPending
		}
		if activeVer.Valid {
			existing.ActiveVersionID = activeVer.String
		}
		item = &existing
	} else {
		isNewItem = true
		if item == nil {
			item = &KnowledgeItem{}
		}
		if err := validateActor(item.UserID); err != nil {
			return nil, nil, err
		}
		if strings.TrimSpace(item.ChatID) == "" {
			return nil, nil, fmt.Errorf("%w: chat id cannot be empty", ErrUnauthorized)
		}
		if item.ID == "" {
			generatedID, err := GenerateItemID(KindSkill)
			if err != nil {
				return nil, nil, err
			}
			item.ID = generatedID
		} else if !strings.HasPrefix(item.ID, "skill_") {
			return nil, nil, fmt.Errorf("%w: skill item id %q must have prefix %q", ErrInvalidStatus, item.ID, "skill_")
		}
		item.Kind = KindSkill
		item.Status = ItemStatusPending
		item.CreatedAt = now
		item.UpdatedAt = now
		item.ActiveVersionID = ""

		_, err := tx.ExecContext(ctx, `
			INSERT INTO knowledge_items (id, chat_id, user_id, kind, status, active_version_id, created_at, updated_at)
			VALUES (?, ?, ?, ?, ?, NULL, ?, ?)
		`, item.ID, item.ChatID, item.UserID, item.Kind, item.Status, item.CreatedAt, item.UpdatedAt)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to insert knowledge item: %w", err)
		}
	}

	var maxRev int
	err = tx.QueryRowContext(ctx, `
		SELECT COALESCE(MAX(revision), 0) FROM skill_versions WHERE item_id = ?
	`, item.ID).Scan(&maxRev)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to get max revision: %w", err)
	}

	version.ItemID = item.ID
	version.Revision = maxRev + 1
	version.ID = FormatVersionID(item.ID, version.Revision)
	version.Status = VersionStatusProposed
	version.IndexStatus = IndexStatusPending
	if version.Triggers == nil {
		version.Triggers = []string{}
	}
	if version.Tags == nil {
		version.Tags = []string{}
	}
	version.Provenance.ChatID = item.ChatID
	version.Provenance.UserID = item.UserID
	version.Provenance.CreatedAt = now
	version.Provenance.ContentHash = ComputeContentHash(version.InstructionsMarkdown)

	triggersJSON, err := json.Marshal(version.Triggers)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to marshal triggers: %w", err)
	}
	tagsJSON, err := json.Marshal(version.Tags)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to marshal tags: %w", err)
	}

	_, err = tx.ExecContext(ctx, `
		INSERT INTO skill_versions (
			id, item_id, revision, name, description, triggers_json, tags_json,
			instructions_markdown, status, chat_id, user_id, source_message_seq,
			fsm_run_id, content_hash, created_at, reviewed_at, reviewed_by, index_status
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?)
	`,
		version.ID, version.ItemID, version.Revision, version.Name, version.Description,
		string(triggersJSON), string(tagsJSON), version.InstructionsMarkdown,
		version.Status, version.Provenance.ChatID, version.Provenance.UserID,
		version.Provenance.SourceMessageSeq, nullIfEmpty(version.Provenance.FSMRunID),
		version.Provenance.ContentHash, version.Provenance.CreatedAt, version.IndexStatus,
	)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to insert skill version: %w", err)
	}

	if !isNewItem {
		item.UpdatedAt = now
		_, err = tx.ExecContext(ctx, `
			UPDATE knowledge_items SET updated_at = ? WHERE id = ?
		`, now, item.ID)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to update knowledge item updated_at: %w", err)
		}
	}

	if err := tx.Commit(); err != nil {
		return nil, nil, fmt.Errorf("failed to commit propose skill: %w", err)
	}

	return item, version, nil
}

// GetItem retrieves a KnowledgeItem by its ID.
func (s *Store) GetItem(ctx context.Context, id string) (*KnowledgeItem, error) {
	if s.db == nil {
		return nil, errors.New("database connection is nil")
	}

	var item KnowledgeItem
	var activeVer sql.NullString
	err := s.db.QueryRowContext(ctx, `
		SELECT id, chat_id, user_id, kind, status, active_version_id, created_at, updated_at
		FROM knowledge_items WHERE id = ?
	`, id).Scan(
		&item.ID, &item.ChatID, &item.UserID, &item.Kind,
		&item.Status, &activeVer, &item.CreatedAt, &item.UpdatedAt,
	)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return nil, ErrNotFound
		}
		return nil, fmt.Errorf("failed to query knowledge item %s: %w", id, err)
	}
	if activeVer.Valid {
		item.ActiveVersionID = activeVer.String
	}
	return &item, nil
}

// GetMemoryVersion retrieves a specific MemoryVersion by its version ID.
func (s *Store) GetMemoryVersion(ctx context.Context, versionID string) (*MemoryVersion, error) {
	if s.db == nil {
		return nil, errors.New("database connection is nil")
	}

	var v MemoryVersion
	var ttlDays sql.NullInt32
	var expiresAt, srcSeq, reviewedAt sql.NullInt64
	var fsmRunID, reviewedBy sql.NullString

	err := s.db.QueryRowContext(ctx, `
		SELECT id, item_id, revision, type, content, confidence, ttl_days, expires_at,
		       status, chat_id, user_id, source_message_seq, fsm_run_id, content_hash,
		       created_at, reviewed_at, reviewed_by, index_status
		FROM memory_versions WHERE id = ?
	`, versionID).Scan(
		&v.ID, &v.ItemID, &v.Revision, &v.Type, &v.Content, &v.Confidence, &ttlDays, &expiresAt,
		&v.Status, &v.Provenance.ChatID, &v.Provenance.UserID, &srcSeq, &fsmRunID, &v.Provenance.ContentHash,
		&v.Provenance.CreatedAt, &reviewedAt, &reviewedBy, &v.IndexStatus,
	)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return nil, ErrNotFound
		}
		return nil, fmt.Errorf("failed to query memory version %s: %w", versionID, err)
	}

	if ttlDays.Valid {
		d := int(ttlDays.Int32)
		v.TTLDays = &d
	}
	if expiresAt.Valid {
		e := expiresAt.Int64
		v.ExpiresAt = &e
	}
	if srcSeq.Valid {
		s := srcSeq.Int64
		v.Provenance.SourceMessageSeq = &s
	}
	if fsmRunID.Valid {
		v.Provenance.FSMRunID = fsmRunID.String
	}
	if reviewedAt.Valid {
		r := reviewedAt.Int64
		v.Provenance.ReviewedAt = &r
	}
	if reviewedBy.Valid {
		v.Provenance.ReviewedBy = reviewedBy.String
	}

	return &v, nil
}

// GetSkillVersion retrieves a specific SkillVersion by its version ID.
func (s *Store) GetSkillVersion(ctx context.Context, versionID string) (*SkillVersion, error) {
	if s.db == nil {
		return nil, errors.New("database connection is nil")
	}

	var v SkillVersion
	var triggersJSON, tagsJSON string
	var srcSeq, reviewedAt sql.NullInt64
	var fsmRunID, reviewedBy sql.NullString

	err := s.db.QueryRowContext(ctx, `
		SELECT id, item_id, revision, name, description, triggers_json, tags_json,
		       instructions_markdown, status, chat_id, user_id, source_message_seq,
		       fsm_run_id, content_hash, created_at, reviewed_at, reviewed_by, index_status
		FROM skill_versions WHERE id = ?
	`, versionID).Scan(
		&v.ID, &v.ItemID, &v.Revision, &v.Name, &v.Description, &triggersJSON, &tagsJSON,
		&v.InstructionsMarkdown, &v.Status, &v.Provenance.ChatID, &v.Provenance.UserID,
		&srcSeq, &fsmRunID, &v.Provenance.ContentHash, &v.Provenance.CreatedAt,
		&reviewedAt, &reviewedBy, &v.IndexStatus,
	)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return nil, ErrNotFound
		}
		return nil, fmt.Errorf("failed to query skill version %s: %w", versionID, err)
	}

	if err := json.Unmarshal([]byte(triggersJSON), &v.Triggers); err != nil {
		return nil, fmt.Errorf("failed to unmarshal triggers for skill version %s: %w", versionID, err)
	}
	if v.Triggers == nil {
		v.Triggers = []string{}
	}
	if err := json.Unmarshal([]byte(tagsJSON), &v.Tags); err != nil {
		return nil, fmt.Errorf("failed to unmarshal tags for skill version %s: %w", versionID, err)
	}
	if v.Tags == nil {
		v.Tags = []string{}
	}
	if srcSeq.Valid {
		s := srcSeq.Int64
		v.Provenance.SourceMessageSeq = &s
	}
	if fsmRunID.Valid {
		v.Provenance.FSMRunID = fsmRunID.String
	}
	if reviewedAt.Valid {
		r := reviewedAt.Int64
		v.Provenance.ReviewedAt = &r
	}
	if reviewedBy.Valid {
		v.Provenance.ReviewedBy = reviewedBy.String
	}

	return &v, nil
}

// GetVersion retrieves a Version (MemoryVersion or SkillVersion) by its version ID.
func (s *Store) GetVersion(ctx context.Context, versionID string) (Version, error) {
	if strings.HasPrefix(versionID, "skill_") {
		return s.GetSkillVersion(ctx, versionID)
	}
	if strings.HasPrefix(versionID, "mem_") {
		return s.GetMemoryVersion(ctx, versionID)
	}

	// Try memory version first, then skill version
	mem, err := s.GetMemoryVersion(ctx, versionID)
	if err == nil {
		return mem, nil
	}
	if !errors.Is(err, ErrNotFound) {
		return nil, err
	}
	return s.GetSkillVersion(ctx, versionID)
}

// ListItems retrieves knowledge items matching the chat ID, kind, and optional status filter.
func (s *Store) ListItems(ctx context.Context, chatID string, kind ItemKind, statusFilter string) ([]*KnowledgeItem, error) {
	if s.db == nil {
		return nil, errors.New("database connection is nil")
	}

	normFilter := strings.ToLower(strings.TrimSpace(statusFilter))
	if normFilter == "" {
		normFilter = "active"
	}

	var query string
	var args []any
	now := time.Now().Unix()

	switch normFilter {
	case "active":
		if kind == KindMemory {
			query = `
				SELECT id, chat_id, user_id, kind, status, active_version_id, created_at, updated_at
				FROM knowledge_items
				WHERE chat_id = ? AND kind = ? AND status = 'active'
				  AND active_version_id IS NOT NULL AND active_version_id != ''
				  AND active_version_id NOT IN (
					SELECT id FROM memory_versions WHERE expires_at IS NOT NULL AND expires_at <= ?
				  )
				ORDER BY updated_at DESC
			`
			args = []any{chatID, kind, now}
		} else {
			query = `
				SELECT id, chat_id, user_id, kind, status, active_version_id, created_at, updated_at
				FROM knowledge_items
				WHERE chat_id = ? AND kind = ? AND status = 'active' AND active_version_id IS NOT NULL AND active_version_id != ''
				ORDER BY updated_at DESC
			`
			args = []any{chatID, kind}
		}
	case "pending":
		if kind == KindMemory {
			query = `
				SELECT id, chat_id, user_id, kind, status, active_version_id, created_at, updated_at
				FROM knowledge_items
				WHERE chat_id = ? AND kind = ? AND status NOT IN ('forgotten', 'archived')
				  AND id IN (SELECT item_id FROM memory_versions WHERE status = 'proposed')
				ORDER BY updated_at DESC
			`
		} else {
			query = `
				SELECT id, chat_id, user_id, kind, status, active_version_id, created_at, updated_at
				FROM knowledge_items
				WHERE chat_id = ? AND kind = ? AND status NOT IN ('forgotten', 'archived')
				  AND id IN (SELECT item_id FROM skill_versions WHERE status = 'proposed')
				ORDER BY updated_at DESC
			`
		}
		args = []any{chatID, kind}
	case "rejected":
		if kind == KindMemory {
			query = `
				SELECT id, chat_id, user_id, kind, status, active_version_id, created_at, updated_at
				FROM knowledge_items
				WHERE chat_id = ? AND kind = ? AND id IN (SELECT item_id FROM memory_versions WHERE status = 'rejected')
				ORDER BY updated_at DESC
			`
		} else {
			query = `
				SELECT id, chat_id, user_id, kind, status, active_version_id, created_at, updated_at
				FROM knowledge_items
				WHERE chat_id = ? AND kind = ? AND id IN (SELECT item_id FROM skill_versions WHERE status = 'rejected')
				ORDER BY updated_at DESC
			`
		}
		args = []any{chatID, kind}
	case "expired":
		if kind == KindMemory {
			query = `
				SELECT id, chat_id, user_id, kind, status, active_version_id, created_at, updated_at
				FROM knowledge_items
				WHERE chat_id = ? AND kind = 'memory' AND status NOT IN ('forgotten', 'archived') AND (
					id IN (SELECT item_id FROM memory_versions WHERE status = 'expired' OR (expires_at IS NOT NULL AND expires_at <= ?))
				)
				ORDER BY updated_at DESC
			`
			args = []any{chatID, now}
		} else {
			return []*KnowledgeItem{}, nil
		}
	case "disabled":
		query = `
			SELECT id, chat_id, user_id, kind, status, active_version_id, created_at, updated_at
			FROM knowledge_items
			WHERE chat_id = ? AND kind = ? AND status = 'disabled'
			ORDER BY updated_at DESC
		`
		args = []any{chatID, kind}
	case "archived":
		query = `
			SELECT id, chat_id, user_id, kind, status, active_version_id, created_at, updated_at
			FROM knowledge_items
			WHERE chat_id = ? AND kind = ? AND status = 'archived'
			ORDER BY updated_at DESC
		`
		args = []any{chatID, kind}
	case "forgotten":
		query = `
			SELECT id, chat_id, user_id, kind, status, active_version_id, created_at, updated_at
			FROM knowledge_items
			WHERE chat_id = ? AND kind = ? AND status = 'forgotten'
			ORDER BY updated_at DESC
		`
		args = []any{chatID, kind}
	case "all":
		query = `
			SELECT id, chat_id, user_id, kind, status, active_version_id, created_at, updated_at
			FROM knowledge_items
			WHERE chat_id = ? AND kind = ?
			ORDER BY updated_at DESC
		`
		args = []any{chatID, kind}
	default:
		return nil, fmt.Errorf("%w: unknown status filter %q", ErrInvalidStatus, statusFilter)
	}

	rows, err := s.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("failed to list knowledge items: %w", err)
	}
	defer func() { _ = rows.Close() }()

	var items []*KnowledgeItem
	for rows.Next() {
		var item KnowledgeItem
		var activeVer sql.NullString
		if err := rows.Scan(
			&item.ID, &item.ChatID, &item.UserID, &item.Kind,
			&item.Status, &activeVer, &item.CreatedAt, &item.UpdatedAt,
		); err != nil {
			return nil, fmt.Errorf("failed to scan knowledge item: %w", err)
		}
		if activeVer.Valid {
			item.ActiveVersionID = activeVer.String
		}
		items = append(items, &item)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return items, nil
}

// ListMemoryVersions returns all revisions of a memory item in ascending revision order.
func (s *Store) ListMemoryVersions(ctx context.Context, itemID string) ([]*MemoryVersion, error) {
	if s.db == nil {
		return nil, errors.New("database connection is nil")
	}

	rows, err := s.db.QueryContext(ctx, `
		SELECT id, item_id, revision, type, content, confidence, ttl_days, expires_at,
		       status, chat_id, user_id, source_message_seq, fsm_run_id, content_hash,
		       created_at, reviewed_at, reviewed_by, index_status
		FROM memory_versions
		WHERE item_id = ?
		ORDER BY revision ASC
	`, itemID)
	if err != nil {
		return nil, fmt.Errorf("failed to query memory versions: %w", err)
	}
	defer func() { _ = rows.Close() }()

	var versions []*MemoryVersion
	for rows.Next() {
		var v MemoryVersion
		var ttlDays sql.NullInt32
		var expiresAt, srcSeq, reviewedAt sql.NullInt64
		var fsmRunID, reviewedBy sql.NullString

		if err := rows.Scan(
			&v.ID, &v.ItemID, &v.Revision, &v.Type, &v.Content, &v.Confidence, &ttlDays, &expiresAt,
			&v.Status, &v.Provenance.ChatID, &v.Provenance.UserID, &srcSeq, &fsmRunID, &v.Provenance.ContentHash,
			&v.Provenance.CreatedAt, &reviewedAt, &reviewedBy, &v.IndexStatus,
		); err != nil {
			return nil, fmt.Errorf("failed to scan memory version: %w", err)
		}

		if ttlDays.Valid {
			d := int(ttlDays.Int32)
			v.TTLDays = &d
		}
		if expiresAt.Valid {
			e := expiresAt.Int64
			v.ExpiresAt = &e
		}
		if srcSeq.Valid {
			s := srcSeq.Int64
			v.Provenance.SourceMessageSeq = &s
		}
		if fsmRunID.Valid {
			v.Provenance.FSMRunID = fsmRunID.String
		}
		if reviewedAt.Valid {
			r := reviewedAt.Int64
			v.Provenance.ReviewedAt = &r
		}
		if reviewedBy.Valid {
			v.Provenance.ReviewedBy = reviewedBy.String
		}
		versions = append(versions, &v)
	}
	return versions, rows.Err()
}

// ListSkillVersions returns all revisions of a skill item in ascending revision order.
func (s *Store) ListSkillVersions(ctx context.Context, itemID string) ([]*SkillVersion, error) {
	if s.db == nil {
		return nil, errors.New("database connection is nil")
	}

	rows, err := s.db.QueryContext(ctx, `
		SELECT id, item_id, revision, name, description, triggers_json, tags_json,
		       instructions_markdown, status, chat_id, user_id, source_message_seq,
		       fsm_run_id, content_hash, created_at, reviewed_at, reviewed_by, index_status
		FROM skill_versions
		WHERE item_id = ?
		ORDER BY revision ASC
	`, itemID)
	if err != nil {
		return nil, fmt.Errorf("failed to query skill versions: %w", err)
	}
	defer func() { _ = rows.Close() }()

	var versions []*SkillVersion
	for rows.Next() {
		var v SkillVersion
		var triggersJSON, tagsJSON string
		var srcSeq, reviewedAt sql.NullInt64
		var fsmRunID, reviewedBy sql.NullString

		if err := rows.Scan(
			&v.ID, &v.ItemID, &v.Revision, &v.Name, &v.Description, &triggersJSON, &tagsJSON,
			&v.InstructionsMarkdown, &v.Status, &v.Provenance.ChatID, &v.Provenance.UserID,
			&srcSeq, &fsmRunID, &v.Provenance.ContentHash, &v.Provenance.CreatedAt,
			&reviewedAt, &reviewedBy, &v.IndexStatus,
		); err != nil {
			return nil, fmt.Errorf("failed to scan skill version: %w", err)
		}

		if err := json.Unmarshal([]byte(triggersJSON), &v.Triggers); err != nil {
			return nil, fmt.Errorf("failed to unmarshal triggers for skill version %s: %w", v.ID, err)
		}
		if v.Triggers == nil {
			v.Triggers = []string{}
		}
		if err := json.Unmarshal([]byte(tagsJSON), &v.Tags); err != nil {
			return nil, fmt.Errorf("failed to unmarshal tags for skill version %s: %w", v.ID, err)
		}
		if v.Tags == nil {
			v.Tags = []string{}
		}
		if srcSeq.Valid {
			s := srcSeq.Int64
			v.Provenance.SourceMessageSeq = &s
		}
		if fsmRunID.Valid {
			v.Provenance.FSMRunID = fsmRunID.String
		}
		if reviewedAt.Valid {
			r := reviewedAt.Int64
			v.Provenance.ReviewedAt = &r
		}
		if reviewedBy.Valid {
			v.Provenance.ReviewedBy = reviewedBy.String
		}
		versions = append(versions, &v)
	}
	return versions, rows.Err()
}

// ListVersions returns all revisions for an item as a slice of Version.
func (s *Store) ListVersions(ctx context.Context, itemID string) ([]Version, error) {
	item, err := s.GetItem(ctx, itemID)
	if err != nil {
		return nil, err
	}

	if item.Kind == KindMemory {
		mvs, err := s.ListMemoryVersions(ctx, itemID)
		if err != nil {
			return nil, err
		}
		res := make([]Version, len(mvs))
		for i, v := range mvs {
			res[i] = v
		}
		return res, nil
	}

	svs, err := s.ListSkillVersions(ctx, itemID)
	if err != nil {
		return nil, err
	}
	res := make([]Version, len(svs))
	for i, v := range svs {
		res[i] = v
	}
	return res, nil
}

// ApproveVersion transitions a proposed version to approved, supersedes prior approved versions,
// activates the parent item, enqueues superseded version for index removal, and flags the version for index synchronization.
func (s *Store) ApproveVersion(ctx context.Context, versionID string, userID string) error {
	if s.db == nil {
		return errors.New("database connection is nil")
	}
	if err := validateActor(userID); err != nil {
		return err
	}

	itemID, _, err := ParseVersionID(versionID)
	if err != nil {
		return err
	}

	now := time.Now().Unix()
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
	}
	defer func() { _ = tx.Rollback() }()

	var itemStatus, itemUserID, itemKind, prevActiveID string
	var activeVerNull sql.NullString
	err = tx.QueryRowContext(ctx, `
		SELECT status, user_id, kind, active_version_id
		FROM knowledge_items
		WHERE id = ?
	`, itemID).Scan(&itemStatus, &itemUserID, &itemKind, &activeVerNull)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return fmt.Errorf("%w: item %s", ErrNotFound, itemID)
		}
		return fmt.Errorf("failed to query knowledge item: %w", err)
	}
	if activeVerNull.Valid {
		prevActiveID = activeVerNull.String
	}

	if itemUserID != userID {
		return fmt.Errorf("%w: user %s cannot approve item owned by %s", ErrUnauthorized, userID, itemUserID)
	}
	if itemStatus == string(ItemStatusForgotten) || itemStatus == string(ItemStatusArchived) {
		return fmt.Errorf("%w: cannot approve version for %s item %s", ErrInvalidStatus, itemStatus, itemID)
	}

	table := "memory_versions"
	namespace := NamespaceMemories
	if itemKind == string(KindSkill) {
		table = "skill_versions"
		namespace = NamespaceSkills
	}

	var status, verItemID string
	err = tx.QueryRowContext(ctx, fmt.Sprintf("SELECT item_id, status FROM %s WHERE id = ?", table), versionID).Scan(&verItemID, &status)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return fmt.Errorf("%w: version %s", ErrNotFound, versionID)
		}
		return err
	}
	if verItemID != itemID {
		return fmt.Errorf("%w: version %s belongs to item %s, not %s", ErrNotFound, versionID, verItemID, itemID)
	}
	if status != string(VersionStatusProposed) {
		return fmt.Errorf("version %s has status %s, cannot approve: %w", versionID, status, ErrInvalidStatus)
	}

	// Supersede previous approved versions
	_, err = tx.ExecContext(ctx, fmt.Sprintf(`
		UPDATE %s SET status = 'superseded' WHERE item_id = ? AND status = 'approved'
	`, table), itemID)
	if err != nil {
		return fmt.Errorf("failed to supersede prior versions: %w", err)
	}

	// Enqueue previous active version for index deletion if being replaced
	if prevActiveID != "" && prevActiveID != versionID {
		_, err = tx.ExecContext(ctx, `
			INSERT INTO pending_index_deletions (version_id, namespace, created_at)
			VALUES (?, ?, ?)
			ON CONFLICT(version_id) DO NOTHING
		`, prevActiveID, namespace, now)
		if err != nil {
			return fmt.Errorf("failed to enqueue superseded version for deletion: %w", err)
		}
	}

	// Approve target version
	res, err := tx.ExecContext(ctx, fmt.Sprintf(`
		UPDATE %s
		SET status = 'approved', reviewed_at = ?, reviewed_by = ?, index_status = 'pending'
		WHERE id = ? AND status = 'proposed'
	`, table), now, userID, versionID)
	if err != nil {
		return fmt.Errorf("failed to approve version: %w", err)
	}
	n, err := res.RowsAffected()
	if err != nil {
		return err
	}
	if n == 0 {
		return fmt.Errorf("version %s is no longer proposed: %w", versionID, ErrInvalidStatus)
	}

	// Guarded item activation
	res, err = tx.ExecContext(ctx, `
		UPDATE knowledge_items
		SET status = 'active', active_version_id = ?, updated_at = ?
		WHERE id = ? AND user_id = ? AND status NOT IN ('forgotten', 'archived')
	`, versionID, now, itemID, userID)
	if err != nil {
		return fmt.Errorf("failed to update knowledge item to active: %w", err)
	}
	n, err = res.RowsAffected()
	if err != nil {
		return err
	}
	if n == 0 {
		return fmt.Errorf("%w: item %s is not in an approvable state", ErrInvalidStatus, itemID)
	}

	return tx.Commit()
}

// RejectVersion marks a proposed version as rejected.
func (s *Store) RejectVersion(ctx context.Context, versionID string, userID string) error {
	if s.db == nil {
		return errors.New("database connection is nil")
	}
	if err := validateActor(userID); err != nil {
		return err
	}

	itemID, _, err := ParseVersionID(versionID)
	if err != nil {
		return err
	}

	now := time.Now().Unix()
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
	}
	defer func() { _ = tx.Rollback() }()

	var itemStatus, itemUserID, itemKind string
	var activeVerNull sql.NullString
	err = tx.QueryRowContext(ctx, `
		SELECT status, user_id, kind, active_version_id
		FROM knowledge_items
		WHERE id = ?
	`, itemID).Scan(&itemStatus, &itemUserID, &itemKind, &activeVerNull)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return fmt.Errorf("%w: item %s", ErrNotFound, itemID)
		}
		return fmt.Errorf("failed to query knowledge item: %w", err)
	}

	if itemUserID != userID {
		return fmt.Errorf("%w: user %s cannot reject item owned by %s", ErrUnauthorized, userID, itemUserID)
	}
	if itemStatus == string(ItemStatusForgotten) || itemStatus == string(ItemStatusArchived) {
		return fmt.Errorf("%w: cannot reject version for %s item %s", ErrInvalidStatus, itemStatus, itemID)
	}

	table := "memory_versions"
	if itemKind == string(KindSkill) {
		table = "skill_versions"
	}

	var status, verItemID string
	err = tx.QueryRowContext(ctx, fmt.Sprintf("SELECT item_id, status FROM %s WHERE id = ?", table), versionID).Scan(&verItemID, &status)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return fmt.Errorf("%w: version %s", ErrNotFound, versionID)
		}
		return err
	}
	if verItemID != itemID {
		return fmt.Errorf("%w: version %s belongs to item %s, not %s", ErrNotFound, versionID, verItemID, itemID)
	}
	if status != string(VersionStatusProposed) {
		return fmt.Errorf("version %s has status %s, cannot reject: %w", versionID, status, ErrInvalidStatus)
	}

	res, err := tx.ExecContext(ctx, fmt.Sprintf(`
		UPDATE %s
		SET status = 'rejected', reviewed_at = ?, reviewed_by = ?
		WHERE id = ? AND status = 'proposed'
	`, table), now, userID, versionID)
	if err != nil {
		return fmt.Errorf("failed to reject version: %w", err)
	}
	n, err := res.RowsAffected()
	if err != nil {
		return err
	}
	if n == 0 {
		return fmt.Errorf("version %s is no longer proposed: %w", versionID, ErrInvalidStatus)
	}

	// Check remaining proposed versions for itemID
	var remainingProposed int
	err = tx.QueryRowContext(ctx, fmt.Sprintf("SELECT COUNT(*) FROM %s WHERE item_id = ? AND status = 'proposed'", table), itemID).Scan(&remainingProposed)
	if err != nil {
		return err
	}

	hasActive := activeVerNull.Valid && activeVerNull.String != ""
	if !hasActive && remainingProposed == 0 {
		_, err = tx.ExecContext(ctx, `
			UPDATE knowledge_items SET status = 'rejected', updated_at = ? WHERE id = ?
		`, now, itemID)
		if err != nil {
			return fmt.Errorf("failed to update item status to rejected: %w", err)
		}
	} else {
		_, err = tx.ExecContext(ctx, `
			UPDATE knowledge_items SET updated_at = ? WHERE id = ?
		`, now, itemID)
		if err != nil {
			return fmt.Errorf("failed to update item updated_at: %w", err)
		}
	}

	return tx.Commit()
}

// DisableSkill sets an active skill's status to disabled and enqueues its active version for index removal.
func (s *Store) DisableSkill(ctx context.Context, itemID string, userID string) error {
	if s.db == nil {
		return errors.New("database connection is nil")
	}
	if err := validateActor(userID); err != nil {
		return err
	}

	now := time.Now().Unix()
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
	}
	defer func() { _ = tx.Rollback() }()

	var itemStatus, itemUserID, itemKind string
	var activeVerNull sql.NullString
	err = tx.QueryRowContext(ctx, `
		SELECT status, user_id, kind, active_version_id
		FROM knowledge_items
		WHERE id = ?
	`, itemID).Scan(&itemStatus, &itemUserID, &itemKind, &activeVerNull)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return fmt.Errorf("%w: skill %s", ErrNotFound, itemID)
		}
		return fmt.Errorf("failed to query knowledge item: %w", err)
	}
	if itemKind != string(KindSkill) {
		return fmt.Errorf("item %s is not a skill", itemID)
	}
	if itemUserID != userID {
		return fmt.Errorf("%w: user %s cannot disable item owned by %s", ErrUnauthorized, userID, itemUserID)
	}
	if itemStatus != string(ItemStatusActive) {
		return fmt.Errorf("%w: cannot disable skill with status %s", ErrInvalidStatus, itemStatus)
	}

	if activeVerNull.Valid && activeVerNull.String != "" {
		_, err = tx.ExecContext(ctx, `
			INSERT INTO pending_index_deletions (version_id, namespace, created_at)
			VALUES (?, ?, ?)
			ON CONFLICT(version_id) DO NOTHING
		`, activeVerNull.String, NamespaceSkills, now)
		if err != nil {
			return fmt.Errorf("failed to enqueue index deletion: %w", err)
		}
	}

	res, err := tx.ExecContext(ctx, `
		UPDATE knowledge_items SET status = 'disabled', updated_at = ? WHERE id = ? AND status = 'active' AND user_id = ?
	`, now, itemID, userID)
	if err != nil {
		return err
	}
	n, err := res.RowsAffected()
	if err != nil {
		return err
	}
	if n == 0 {
		return fmt.Errorf("%w: skill %s is not active", ErrInvalidStatus, itemID)
	}

	return tx.Commit()
}

// EnableSkill sets a disabled skill's status back to active, cancels pending index deletion, and schedules reindexing.
func (s *Store) EnableSkill(ctx context.Context, itemID string, userID string) error {
	if s.db == nil {
		return errors.New("database connection is nil")
	}
	if err := validateActor(userID); err != nil {
		return err
	}

	now := time.Now().Unix()
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
	}
	defer func() { _ = tx.Rollback() }()

	var itemStatus, itemUserID, itemKind string
	var activeVerNull sql.NullString
	err = tx.QueryRowContext(ctx, `
		SELECT status, user_id, kind, active_version_id
		FROM knowledge_items
		WHERE id = ?
	`, itemID).Scan(&itemStatus, &itemUserID, &itemKind, &activeVerNull)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return fmt.Errorf("%w: skill %s", ErrNotFound, itemID)
		}
		return fmt.Errorf("failed to query knowledge item: %w", err)
	}
	if itemKind != string(KindSkill) {
		return fmt.Errorf("item %s is not a skill", itemID)
	}
	if itemUserID != userID {
		return fmt.Errorf("%w: user %s cannot enable item owned by %s", ErrUnauthorized, userID, itemUserID)
	}
	if itemStatus != string(ItemStatusDisabled) {
		return fmt.Errorf("%w: cannot enable skill with status %s", ErrInvalidStatus, itemStatus)
	}
	if !activeVerNull.Valid || activeVerNull.String == "" {
		return fmt.Errorf("%w: cannot enable skill without active version", ErrInvalidStatus)
	}
	activeVerID := activeVerNull.String

	// Cancel pending index deletion for this active version
	_, err = tx.ExecContext(ctx, "DELETE FROM pending_index_deletions WHERE version_id = ?", activeVerID)
	if err != nil {
		return fmt.Errorf("failed to clear pending index deletion: %w", err)
	}

	// Mark active version index_status as pending so indexer reconciles it
	verRes, err := tx.ExecContext(ctx, `
		UPDATE skill_versions SET index_status = 'pending' WHERE id = ? AND item_id = ? AND status = 'approved'
	`, activeVerID, itemID)
	if err != nil {
		return fmt.Errorf("failed to mark skill version pending for indexing: %w", err)
	}
	verRows, err := verRes.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to check rows affected: %w", err)
	}
	if verRows != 1 {
		return fmt.Errorf("%w: active version %s is not an approved version of skill %s", ErrInvalidStatus, activeVerID, itemID)
	}

	res, err := tx.ExecContext(ctx, `
		UPDATE knowledge_items SET status = 'active', updated_at = ? WHERE id = ? AND status = 'disabled' AND user_id = ?
	`, now, itemID, userID)
	if err != nil {
		return err
	}
	n, err := res.RowsAffected()
	if err != nil {
		return err
	}
	if n == 0 {
		return fmt.Errorf("%w: skill %s is not disabled", ErrInvalidStatus, itemID)
	}

	return tx.Commit()
}

// ArchiveSkill marks a skill item and its active version as archived, and enqueues index deletion.
func (s *Store) ArchiveSkill(ctx context.Context, itemID string, userID string) error {
	if s.db == nil {
		return errors.New("database connection is nil")
	}
	if err := validateActor(userID); err != nil {
		return err
	}

	now := time.Now().Unix()
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
	}
	defer func() { _ = tx.Rollback() }()

	var itemStatus, itemUserID, itemKind string
	var activeVerNull sql.NullString
	err = tx.QueryRowContext(ctx, `
		SELECT status, user_id, kind, active_version_id
		FROM knowledge_items
		WHERE id = ?
	`, itemID).Scan(&itemStatus, &itemUserID, &itemKind, &activeVerNull)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return fmt.Errorf("%w: skill %s", ErrNotFound, itemID)
		}
		return fmt.Errorf("failed to query knowledge item: %w", err)
	}
	if itemKind != string(KindSkill) {
		return fmt.Errorf("item %s is not a skill", itemID)
	}
	if itemUserID != userID {
		return fmt.Errorf("%w: user %s cannot archive item owned by %s", ErrUnauthorized, userID, itemUserID)
	}
	if itemStatus != string(ItemStatusActive) && itemStatus != string(ItemStatusDisabled) {
		return fmt.Errorf("%w: cannot archive skill with status %s", ErrInvalidStatus, itemStatus)
	}

	if activeVerNull.Valid && activeVerNull.String != "" {
		_, err = tx.ExecContext(ctx, `
			INSERT INTO pending_index_deletions (version_id, namespace, created_at)
			VALUES (?, ?, ?)
			ON CONFLICT(version_id) DO NOTHING
		`, activeVerNull.String, NamespaceSkills, now)
		if err != nil {
			return fmt.Errorf("failed to enqueue index deletion: %w", err)
		}
	}

	res, err := tx.ExecContext(ctx, `
		UPDATE knowledge_items SET status = 'archived', updated_at = ? WHERE id = ? AND status IN ('active', 'disabled') AND user_id = ?
	`, now, itemID, userID)
	if err != nil {
		return err
	}
	n, err := res.RowsAffected()
	if err != nil {
		return err
	}
	if n == 0 {
		return fmt.Errorf("%w: skill %s cannot be archived", ErrInvalidStatus, itemID)
	}

	_, err = tx.ExecContext(ctx, `
		UPDATE skill_versions SET status = 'archived' WHERE item_id = ? AND status = 'approved'
	`, itemID)
	if err != nil {
		return err
	}

	return tx.Commit()
}

// ForgetItem permanently erases all versions for a memory item, records a tombstone, queues all versions for index deletion, and updates the item status.
func (s *Store) ForgetItem(ctx context.Context, itemID string, userID string) error {
	if s.db == nil {
		return errors.New("database connection is nil")
	}
	if err := validateActor(userID); err != nil {
		return err
	}

	now := time.Now().Unix()
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
	}
	defer func() { _ = tx.Rollback() }()

	var itemStatus, itemUserID, itemKind, itemChatID string
	err = tx.QueryRowContext(ctx, `
		SELECT status, user_id, kind, chat_id
		FROM knowledge_items
		WHERE id = ?
	`, itemID).Scan(&itemStatus, &itemUserID, &itemKind, &itemChatID)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return fmt.Errorf("%w: memory %s", ErrNotFound, itemID)
		}
		return fmt.Errorf("failed to query knowledge item: %w", err)
	}
	if itemKind != string(KindMemory) {
		return fmt.Errorf("item %s is not a memory", itemID)
	}
	if itemUserID != userID {
		return fmt.Errorf("%w: user %s cannot forget item owned by %s", ErrUnauthorized, userID, itemUserID)
	}
	if itemStatus == string(ItemStatusForgotten) {
		return fmt.Errorf("%w: item %s is already forgotten", ErrInvalidStatus, itemID)
	}

	// Enumerate all version IDs to enqueue for index deletion
	rows, err := tx.QueryContext(ctx, "SELECT id FROM memory_versions WHERE item_id = ?", itemID)
	if err != nil {
		return fmt.Errorf("failed to query memory versions: %w", err)
	}
	var verIDs []string
	for rows.Next() {
		var vid string
		if err := rows.Scan(&vid); err != nil {
			_ = rows.Close()
			return err
		}
		verIDs = append(verIDs, vid)
	}
	if err := rows.Err(); err != nil {
		_ = rows.Close()
		return fmt.Errorf("failed to iterate memory versions: %w", err)
	}
	if err := rows.Close(); err != nil {
		return fmt.Errorf("failed to close memory version rows: %w", err)
	}

	for _, vid := range verIDs {
		_, err = tx.ExecContext(ctx, `
			INSERT INTO pending_index_deletions (version_id, namespace, created_at)
			VALUES (?, ?, ?)
			ON CONFLICT(version_id) DO NOTHING
		`, vid, NamespaceMemories, now)
		if err != nil {
			return fmt.Errorf("failed to enqueue index deletion for version %s: %w", vid, err)
		}
	}

	// Record tombstone
	_, err = tx.ExecContext(ctx, `
		INSERT INTO knowledge_tombstones (item_id, kind, chat_id, user_id, deleted_by, deleted_at, reason)
		VALUES (?, ?, ?, ?, ?, ?, ?)
		ON CONFLICT(item_id) DO UPDATE SET deleted_by=excluded.deleted_by, deleted_at=excluded.deleted_at
	`, itemID, KindMemory, itemChatID, itemUserID, userID, now, "forgotten via slash command")
	if err != nil {
		return fmt.Errorf("failed to insert tombstone: %w", err)
	}

	// Delete all content versions
	_, err = tx.ExecContext(ctx, "DELETE FROM memory_versions WHERE item_id = ?", itemID)
	if err != nil {
		return fmt.Errorf("failed to erase memory versions: %w", err)
	}

	// Mark item as forgotten
	res, err := tx.ExecContext(ctx, `
		UPDATE knowledge_items
		SET status = 'forgotten', active_version_id = NULL, updated_at = ?
		WHERE id = ? AND status != 'forgotten' AND user_id = ?
	`, now, itemID, userID)
	if err != nil {
		return fmt.Errorf("failed to mark item forgotten: %w", err)
	}
	n, err := res.RowsAffected()
	if err != nil {
		return err
	}
	if n == 0 {
		return fmt.Errorf("%w: item %s is already forgotten", ErrInvalidStatus, itemID)
	}

	return tx.Commit()
}

// DeleteSkill permanently erases all versions for a skill item, records a tombstone, queues all versions for index deletion, and updates the item status.
func (s *Store) DeleteSkill(ctx context.Context, itemID string, userID string) error {
	if s.db == nil {
		return errors.New("database connection is nil")
	}
	if err := validateActor(userID); err != nil {
		return err
	}

	now := time.Now().Unix()
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
	}
	defer func() { _ = tx.Rollback() }()

	var itemStatus, itemUserID, itemKind, itemChatID string
	err = tx.QueryRowContext(ctx, `
		SELECT status, user_id, kind, chat_id
		FROM knowledge_items
		WHERE id = ?
	`, itemID).Scan(&itemStatus, &itemUserID, &itemKind, &itemChatID)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return fmt.Errorf("%w: skill %s", ErrNotFound, itemID)
		}
		return fmt.Errorf("failed to query knowledge item: %w", err)
	}
	if itemKind != string(KindSkill) {
		return fmt.Errorf("item %s is not a skill", itemID)
	}
	if itemUserID != userID {
		return fmt.Errorf("%w: user %s cannot delete item owned by %s", ErrUnauthorized, userID, itemUserID)
	}
	if itemStatus == string(ItemStatusForgotten) {
		return fmt.Errorf("%w: skill %s is already deleted", ErrInvalidStatus, itemID)
	}

	// Enumerate all version IDs to enqueue for index deletion
	rows, err := tx.QueryContext(ctx, "SELECT id FROM skill_versions WHERE item_id = ?", itemID)
	if err != nil {
		return fmt.Errorf("failed to query skill versions: %w", err)
	}
	var verIDs []string
	for rows.Next() {
		var vid string
		if err := rows.Scan(&vid); err != nil {
			_ = rows.Close()
			return err
		}
		verIDs = append(verIDs, vid)
	}
	if err := rows.Err(); err != nil {
		_ = rows.Close()
		return fmt.Errorf("failed to iterate skill versions: %w", err)
	}
	if err := rows.Close(); err != nil {
		return fmt.Errorf("failed to close skill version rows: %w", err)
	}

	for _, vid := range verIDs {
		_, err = tx.ExecContext(ctx, `
			INSERT INTO pending_index_deletions (version_id, namespace, created_at)
			VALUES (?, ?, ?)
			ON CONFLICT(version_id) DO NOTHING
		`, vid, NamespaceSkills, now)
		if err != nil {
			return fmt.Errorf("failed to enqueue index deletion for version %s: %w", vid, err)
		}
	}

	// Record tombstone
	_, err = tx.ExecContext(ctx, `
		INSERT INTO knowledge_tombstones (item_id, kind, chat_id, user_id, deleted_by, deleted_at, reason)
		VALUES (?, ?, ?, ?, ?, ?, ?)
		ON CONFLICT(item_id) DO UPDATE SET deleted_by=excluded.deleted_by, deleted_at=excluded.deleted_at
	`, itemID, KindSkill, itemChatID, itemUserID, userID, now, "deleted via slash command")
	if err != nil {
		return fmt.Errorf("failed to insert tombstone: %w", err)
	}

	// Delete all content versions
	_, err = tx.ExecContext(ctx, "DELETE FROM skill_versions WHERE item_id = ?", itemID)
	if err != nil {
		return fmt.Errorf("failed to erase skill versions: %w", err)
	}

	// Mark item as forgotten
	res, err := tx.ExecContext(ctx, `
		UPDATE knowledge_items
		SET status = 'forgotten', active_version_id = NULL, updated_at = ?
		WHERE id = ? AND status != 'forgotten' AND user_id = ?
	`, now, itemID, userID)
	if err != nil {
		return fmt.Errorf("failed to mark skill item forgotten: %w", err)
	}
	n, err := res.RowsAffected()
	if err != nil {
		return err
	}
	if n == 0 {
		return fmt.Errorf("%w: skill %s is already deleted", ErrInvalidStatus, itemID)
	}

	return tx.Commit()
}

// GetTombstone retrieves the tombstone audit record for an item.
func (s *Store) GetTombstone(ctx context.Context, itemID string) (*Tombstone, error) {
	if s.db == nil {
		return nil, errors.New("database connection is nil")
	}

	var t Tombstone
	var reason sql.NullString
	err := s.db.QueryRowContext(ctx, `
		SELECT item_id, kind, chat_id, user_id, deleted_by, deleted_at, reason
		FROM knowledge_tombstones WHERE item_id = ?
	`, itemID).Scan(
		&t.ItemID, &t.Kind, &t.ChatID, &t.UserID, &t.DeletedBy, &t.DeletedAt, &reason,
	)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return nil, ErrNotFound
		}
		return nil, err
	}
	if reason.Valid {
		t.Reason = reason.String
	}
	return &t, nil
}

// IndexCandidate describes a version needing CortexDB indexing or reconciliation.
type IndexCandidate struct {
	VersionID string
	ItemID    string
	Kind      ItemKind
	Revision  int
}

// GetPendingIndexVersions returns approved versions of active items that have pending or error index status.
// For memories, expired versions are excluded. Results are ordered by created_at ascending.
func (s *Store) GetPendingIndexVersions(ctx context.Context, limit int) ([]IndexCandidate, error) {
	if s.db == nil {
		return nil, errors.New("database connection is nil")
	}
	if limit <= 0 {
		limit = 50
	}

	now := time.Now().Unix()
	rows, err := s.db.QueryContext(ctx, `
		SELECT id, item_id, kind, revision FROM (
			SELECT v.id, v.item_id, 'memory' AS kind, v.revision, v.created_at, v.index_status
			FROM memory_versions v
			JOIN knowledge_items i ON v.item_id = i.id
			WHERE i.status = 'active'
			  AND i.active_version_id = v.id
			  AND v.status = 'approved'
			  AND v.index_status IN ('pending', 'error')
			  AND (v.expires_at IS NULL OR v.expires_at > ?)
			UNION ALL
			SELECT v.id, v.item_id, 'skill' AS kind, v.revision, v.created_at, v.index_status
			FROM skill_versions v
			JOIN knowledge_items i ON v.item_id = i.id
			WHERE i.status = 'active'
			  AND i.active_version_id = v.id
			  AND v.status = 'approved'
			  AND v.index_status IN ('pending', 'error')
		) ORDER BY CASE WHEN index_status = 'pending' THEN 0 ELSE 1 END, created_at ASC
		LIMIT ?
	`, now, limit)
	if err != nil {
		return nil, fmt.Errorf("failed to query pending index versions: %w", err)
	}
	defer func() { _ = rows.Close() }()

	var candidates []IndexCandidate
	for rows.Next() {
		var c IndexCandidate
		var kindStr string
		if err := rows.Scan(&c.VersionID, &c.ItemID, &kindStr, &c.Revision); err != nil {
			return nil, fmt.Errorf("failed to scan index candidate: %w", err)
		}
		c.Kind = ItemKind(kindStr)
		candidates = append(candidates, c)
	}
	return candidates, rows.Err()
}

// UpdateIndexStatus updates the indexing status of a version.
func (s *Store) UpdateIndexStatus(ctx context.Context, versionID string, status IndexStatus) error {
	if s.db == nil {
		return errors.New("database connection is nil")
	}
	if !IsValidIndexStatus(status) {
		return fmt.Errorf("%w: invalid index status %q", ErrInvalidStatus, status)
	}

	if strings.HasPrefix(versionID, "mem_") {
		res, err := s.db.ExecContext(ctx, "UPDATE memory_versions SET index_status = ? WHERE id = ?", status, versionID)
		if err != nil {
			return err
		}
		n, err := res.RowsAffected()
		if err != nil {
			return err
		}
		if n == 0 {
			return fmt.Errorf("%w: version %s", ErrNotFound, versionID)
		}
		return nil
	}
	if strings.HasPrefix(versionID, "skill_") {
		res, err := s.db.ExecContext(ctx, "UPDATE skill_versions SET index_status = ? WHERE id = ?", status, versionID)
		if err != nil {
			return err
		}
		n, err := res.RowsAffected()
		if err != nil {
			return err
		}
		if n == 0 {
			return fmt.Errorf("%w: version %s", ErrNotFound, versionID)
		}
		return nil
	}

	res, err := s.db.ExecContext(ctx, "UPDATE memory_versions SET index_status = ? WHERE id = ?", status, versionID)
	if err != nil {
		return err
	}
	n, err := res.RowsAffected()
	if err != nil {
		return err
	}
	if n > 0 {
		return nil
	}

	res, err = s.db.ExecContext(ctx, "UPDATE skill_versions SET index_status = ? WHERE id = ?", status, versionID)
	if err != nil {
		return err
	}
	n, err = res.RowsAffected()
	if err != nil {
		return err
	}
	if n == 0 {
		return fmt.Errorf("%w: version %s", ErrNotFound, versionID)
	}
	return nil
}

// MarkVersionIndexed transitions an approved version of an active item to index_status 'ready'.
// If the item or version is no longer active, approved, or has expired, it returns ErrInvalidStatus without updating.
func (s *Store) MarkVersionIndexed(ctx context.Context, versionID string, optNow ...int64) error {
	if s.db == nil {
		return errors.New("database connection is nil")
	}

	itemID, _, err := ParseVersionID(versionID)
	if err != nil {
		return err
	}

	nowUnix := time.Now().Unix()
	if len(optNow) > 0 && optNow[0] > 0 {
		nowUnix = optNow[0]
	}

	var res sql.Result
	if strings.HasPrefix(versionID, "mem_") {
		res, err = s.db.ExecContext(ctx, `
			UPDATE memory_versions
			SET index_status = 'ready'
			WHERE id = ? AND status = 'approved'
			  AND (expires_at IS NULL OR expires_at > ?)
			  AND item_id IN (
				SELECT id FROM knowledge_items WHERE id = ? AND status = 'active' AND active_version_id = ?
			  )
		`, versionID, nowUnix, itemID, versionID)
	} else if strings.HasPrefix(versionID, "skill_") {
		res, err = s.db.ExecContext(ctx, `
			UPDATE skill_versions
			SET index_status = 'ready'
			WHERE id = ? AND status = 'approved'
			  AND item_id IN (
				SELECT id FROM knowledge_items WHERE id = ? AND status = 'active' AND active_version_id = ?
			  )
		`, versionID, itemID, versionID)
	} else {
		return fmt.Errorf("%w: unknown version kind for %s", ErrInvalidStatus, versionID)
	}

	if err != nil {
		return err
	}
	n, err := res.RowsAffected()
	if err != nil {
		return err
	}
	if n == 0 {
		return fmt.Errorf("%w: version %s is no longer active, approved, and unexpired", ErrInvalidStatus, versionID)
	}
	return nil
}

// SearchCandidate represents a joined item and version record for search revalidation.
type SearchCandidate struct {
	Item   KnowledgeItem
	Memory *MemoryVersion
	Skill  *SkillVersion
}

// GetSearchCandidates batch-loads joined knowledge items and versions by version IDs for revalidation.
func (s *Store) GetSearchCandidates(ctx context.Context, kind ItemKind, versionIDs []string) (map[string]SearchCandidate, error) {
	if s.db == nil {
		return nil, errors.New("database connection is nil")
	}
	if !IsValidItemKind(kind) {
		return nil, fmt.Errorf("%w: invalid item kind %q", ErrInvalidStatus, kind)
	}
	if len(versionIDs) == 0 {
		return make(map[string]SearchCandidate), nil
	}

	uniqueIDs := make([]string, 0, len(versionIDs))
	seen := make(map[string]struct{}, len(versionIDs))
	for _, id := range versionIDs {
		trimmed := strings.TrimSpace(id)
		if trimmed == "" {
			continue
		}
		if _, exists := seen[trimmed]; !exists {
			seen[trimmed] = struct{}{}
			uniqueIDs = append(uniqueIDs, trimmed)
		}
	}
	if len(uniqueIDs) == 0 {
		return make(map[string]SearchCandidate), nil
	}

	placeholders := make([]string, len(uniqueIDs))
	args := make([]any, len(uniqueIDs))
	for i, id := range uniqueIDs {
		placeholders[i] = "?"
		args[i] = id
	}

	results := make(map[string]SearchCandidate, len(uniqueIDs))

	switch kind {
	case KindMemory:
		query := fmt.Sprintf(`
			SELECT
				i.id, i.chat_id, i.user_id, i.kind, i.status, i.active_version_id, i.created_at, i.updated_at,
				v.id, v.item_id, v.revision, v.type, v.content, v.confidence, v.ttl_days, v.expires_at,
				v.status, v.chat_id, v.user_id, v.source_message_seq, v.fsm_run_id, v.content_hash,
				v.created_at, v.reviewed_at, v.reviewed_by, v.index_status
			FROM memory_versions v
			JOIN knowledge_items i ON v.item_id = i.id
			WHERE v.id IN (%s)
		`, strings.Join(placeholders, ","))

		rows, err := s.db.QueryContext(ctx, query, args...)
		if err != nil {
			return nil, fmt.Errorf("failed to query search candidates: %w", err)
		}
		defer func() { _ = rows.Close() }()

		for rows.Next() {
			var cand SearchCandidate
			var itemActiveVer sql.NullString
			var mem MemoryVersion
			var ttlDays sql.NullInt32
			var expiresAt, srcSeq, reviewedAt sql.NullInt64
			var fsmRunID, reviewedBy sql.NullString

			err := rows.Scan(
				&cand.Item.ID, &cand.Item.ChatID, &cand.Item.UserID, &cand.Item.Kind,
				&cand.Item.Status, &itemActiveVer, &cand.Item.CreatedAt, &cand.Item.UpdatedAt,
				&mem.ID, &mem.ItemID, &mem.Revision, &mem.Type, &mem.Content, &mem.Confidence,
				&ttlDays, &expiresAt, &mem.Status, &mem.Provenance.ChatID, &mem.Provenance.UserID,
				&srcSeq, &fsmRunID, &mem.Provenance.ContentHash, &mem.Provenance.CreatedAt,
				&reviewedAt, &reviewedBy, &mem.IndexStatus,
			)
			if err != nil {
				return nil, fmt.Errorf("failed to scan memory search candidate: %w", err)
			}
			if itemActiveVer.Valid {
				cand.Item.ActiveVersionID = itemActiveVer.String
			}
			if ttlDays.Valid {
				d := int(ttlDays.Int32)
				mem.TTLDays = &d
			}
			if expiresAt.Valid {
				e := expiresAt.Int64
				mem.ExpiresAt = &e
			}
			if srcSeq.Valid {
				s := srcSeq.Int64
				mem.Provenance.SourceMessageSeq = &s
			}
			if fsmRunID.Valid {
				mem.Provenance.FSMRunID = fsmRunID.String
			}
			if reviewedAt.Valid {
				r := reviewedAt.Int64
				mem.Provenance.ReviewedAt = &r
			}
			if reviewedBy.Valid {
				mem.Provenance.ReviewedBy = reviewedBy.String
			}
			cand.Memory = &mem
			results[mem.ID] = cand
		}
		if err := rows.Err(); err != nil {
			return nil, err
		}
		return results, nil
	case KindSkill:
		query := fmt.Sprintf(`
			SELECT
				i.id, i.chat_id, i.user_id, i.kind, i.status, i.active_version_id, i.created_at, i.updated_at,
				v.id, v.item_id, v.revision, v.name, v.description, v.triggers_json, v.tags_json,
				v.instructions_markdown, v.status, v.chat_id, v.user_id, v.source_message_seq,
				v.fsm_run_id, v.content_hash, v.created_at, v.reviewed_at, v.reviewed_by, v.index_status
			FROM skill_versions v
			JOIN knowledge_items i ON v.item_id = i.id
			WHERE v.id IN (%s)
		`, strings.Join(placeholders, ","))

		rows, err := s.db.QueryContext(ctx, query, args...)
		if err != nil {
			return nil, fmt.Errorf("failed to query search candidates: %w", err)
		}
		defer func() { _ = rows.Close() }()

		for rows.Next() {
			var cand SearchCandidate
			var itemActiveVer sql.NullString
			var skill SkillVersion
			var triggersJSON, tagsJSON string
			var srcSeq, reviewedAt sql.NullInt64
			var fsmRunID, reviewedBy sql.NullString

			err := rows.Scan(
				&cand.Item.ID, &cand.Item.ChatID, &cand.Item.UserID, &cand.Item.Kind,
				&cand.Item.Status, &itemActiveVer, &cand.Item.CreatedAt, &cand.Item.UpdatedAt,
				&skill.ID, &skill.ItemID, &skill.Revision, &skill.Name, &skill.Description,
				&triggersJSON, &tagsJSON, &skill.InstructionsMarkdown, &skill.Status,
				&skill.Provenance.ChatID, &skill.Provenance.UserID, &srcSeq, &fsmRunID,
				&skill.Provenance.ContentHash, &skill.Provenance.CreatedAt,
				&reviewedAt, &reviewedBy, &skill.IndexStatus,
			)
			if err != nil {
				return nil, fmt.Errorf("failed to scan skill search candidate: %w", err)
			}
			if itemActiveVer.Valid {
				cand.Item.ActiveVersionID = itemActiveVer.String
			}
			if err := json.Unmarshal([]byte(triggersJSON), &skill.Triggers); err != nil {
				return nil, fmt.Errorf("failed to unmarshal triggers for skill candidate %s: %w", skill.ID, err)
			}
			if skill.Triggers == nil {
				skill.Triggers = []string{}
			}
			if err := json.Unmarshal([]byte(tagsJSON), &skill.Tags); err != nil {
				return nil, fmt.Errorf("failed to unmarshal tags for skill candidate %s: %w", skill.ID, err)
			}
			if skill.Tags == nil {
				skill.Tags = []string{}
			}
			if srcSeq.Valid {
				s := srcSeq.Int64
				skill.Provenance.SourceMessageSeq = &s
			}
			if fsmRunID.Valid {
				skill.Provenance.FSMRunID = fsmRunID.String
			}
			if reviewedAt.Valid {
				r := reviewedAt.Int64
				skill.Provenance.ReviewedAt = &r
			}
			if reviewedBy.Valid {
				skill.Provenance.ReviewedBy = reviewedBy.String
			}
			cand.Skill = &skill
			results[skill.ID] = cand
		}
		if err := rows.Err(); err != nil {
			return nil, err
		}
		return results, nil
	default:
		return nil, fmt.Errorf("%w: invalid item kind %q", ErrInvalidStatus, kind)
	}
}

func nullIfEmpty(s string) any {
	if s == "" {
		return nil
	}
	return s
}

// EnqueueIndexDeletion records a version ID to be durably purged from vector/FTS5 indexes.
func (s *Store) EnqueueIndexDeletion(ctx context.Context, versionID, namespace string) error {
	if s.db == nil {
		return errors.New("database connection is nil")
	}
	now := time.Now().Unix()
	_, err := s.db.ExecContext(ctx, `
		INSERT INTO pending_index_deletions (version_id, namespace, created_at)
		VALUES (?, ?, ?)
		ON CONFLICT(version_id) DO NOTHING
	`, versionID, namespace, now)
	return err
}

// RemovePendingIndexDeletion removes a version ID from pending index deletions after successful purge.
func (s *Store) RemovePendingIndexDeletion(ctx context.Context, versionID string) error {
	if s.db == nil {
		return errors.New("database connection is nil")
	}
	_, err := s.db.ExecContext(ctx, "DELETE FROM pending_index_deletions WHERE version_id = ?", versionID)
	return err
}

// GetPendingIndexDeletions returns queued index deletions up to the specified limit.
func (s *Store) GetPendingIndexDeletions(ctx context.Context, limit int) ([]PendingIndexDeletion, error) {
	if s.db == nil {
		return nil, errors.New("database connection is nil")
	}
	if limit <= 0 {
		limit = 50
	}
	rows, err := s.db.QueryContext(ctx, `
		SELECT version_id, namespace, created_at
		FROM pending_index_deletions
		ORDER BY created_at ASC
		LIMIT ?
	`, limit)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()

	var items []PendingIndexDeletion
	for rows.Next() {
		var item PendingIndexDeletion
		if err := rows.Scan(&item.VersionID, &item.Namespace, &item.CreatedAt); err != nil {
			return nil, err
		}
		items = append(items, item)
	}
	return items, rows.Err()
}

// IsVersionActiveAndApproved checks whether a version currently belongs to an active item,
// is the item's active_version_id, and has status 'approved'.
func (s *Store) IsVersionActiveAndApproved(ctx context.Context, versionID string) (bool, error) {
	if s.db == nil {
		return false, errors.New("database connection is nil")
	}
	itemID, _, err := ParseVersionID(versionID)
	if err != nil {
		return false, nil
	}

	var itemStatus, activeVerID, kind string
	err = s.db.QueryRowContext(ctx, `
		SELECT status, COALESCE(active_version_id, ''), kind
		FROM knowledge_items
		WHERE id = ?
	`, itemID).Scan(&itemStatus, &activeVerID, &kind)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return false, nil
		}
		return false, err
	}
	if itemStatus != string(ItemStatusActive) || activeVerID != versionID {
		return false, nil
	}

	table := "memory_versions"
	if kind == string(KindSkill) {
		table = "skill_versions"
	}

	var verStatus string
	err = s.db.QueryRowContext(ctx, fmt.Sprintf("SELECT status FROM %s WHERE id = ?", table), versionID).Scan(&verStatus)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return false, nil
		}
		return false, err
	}

	return verStatus == string(VersionStatusApproved), nil
}

// DiscardStaleDeletionJob checks whether a version is currently the active approved version of an active item.
// If it is, it atomically removes the version from pending_index_deletions and marks its index status as pending,
// returning (true, nil). If the version is not active and approved, the deletion job is retained and it returns (false, nil).
func (s *Store) DiscardStaleDeletionJob(ctx context.Context, versionID string) (bool, error) {
	if s.db == nil {
		return false, errors.New("database connection is nil")
	}

	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return false, fmt.Errorf("failed to begin tx: %w", err)
	}
	defer func() { _ = tx.Rollback() }()

	var updateQuery string
	if strings.HasPrefix(versionID, PrefixMemory) {
		updateQuery = `
			UPDATE memory_versions
			SET index_status = 'pending'
			WHERE id = ?
			  AND status = 'approved'
			  AND EXISTS (
				SELECT 1 FROM knowledge_items
				WHERE id = memory_versions.item_id
				  AND status = 'active'
				  AND active_version_id = ?
			  )
		`
	} else if strings.HasPrefix(versionID, PrefixSkill) {
		updateQuery = `
			UPDATE skill_versions
			SET index_status = 'pending'
			WHERE id = ?
			  AND status = 'approved'
			  AND EXISTS (
				SELECT 1 FROM knowledge_items
				WHERE id = skill_versions.item_id
				  AND status = 'active'
				  AND active_version_id = ?
			  )
		`
	} else {
		return false, fmt.Errorf("unknown version prefix for %s", versionID)
	}

	res, err := tx.ExecContext(ctx, updateQuery, versionID, versionID)
	if err != nil {
		return false, fmt.Errorf("failed to check and update version index status: %w", err)
	}

	n, err := res.RowsAffected()
	if err != nil {
		return false, fmt.Errorf("failed to get rows affected: %w", err)
	}

	if n == 0 {
		return false, nil
	}

	if _, err := tx.ExecContext(ctx, "DELETE FROM pending_index_deletions WHERE version_id = ?", versionID); err != nil {
		return false, fmt.Errorf("failed to delete pending index deletion: %w", err)
	}

	if err := tx.Commit(); err != nil {
		return false, fmt.Errorf("failed to commit tx: %w", err)
	}
	return true, nil
}
