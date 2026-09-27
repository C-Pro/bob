package knowledge

import (
	"context"
	"database/sql"
	"fmt"
)

// EnsureKnowledgeSchema ensures that the knowledge tables and indexes
// exist in the target SQLite database.
func EnsureKnowledgeSchema(ctx context.Context, db *sql.DB) error {
	if db == nil {
		return fmt.Errorf("database connection is nil")
	}

	schemaSQL := `
		CREATE TABLE IF NOT EXISTS knowledge_items (
			id TEXT PRIMARY KEY,
			chat_id TEXT NOT NULL,
			user_id TEXT NOT NULL,
			kind TEXT NOT NULL,
			status TEXT NOT NULL,
			active_version_id TEXT,
			created_at INTEGER NOT NULL,
			updated_at INTEGER NOT NULL
		);

		CREATE INDEX IF NOT EXISTS idx_knowledge_items_lookup
			ON knowledge_items(chat_id, kind, status);

		CREATE TABLE IF NOT EXISTS memory_versions (
			id TEXT PRIMARY KEY,
			item_id TEXT NOT NULL,
			revision INTEGER NOT NULL,
			type TEXT NOT NULL,
			content TEXT NOT NULL,
			confidence REAL NOT NULL DEFAULT 1.0,
			ttl_days INTEGER,
			expires_at INTEGER,
			status TEXT NOT NULL,
			chat_id TEXT NOT NULL,
			user_id TEXT NOT NULL,
			source_message_seq INTEGER,
			fsm_run_id TEXT,
			content_hash TEXT NOT NULL,
			created_at INTEGER NOT NULL,
			reviewed_at INTEGER,
			reviewed_by TEXT,
			index_status TEXT NOT NULL DEFAULT 'pending',
			FOREIGN KEY(item_id) REFERENCES knowledge_items(id) ON DELETE CASCADE,
			CONSTRAINT uq_memory_versions_item_rev UNIQUE(item_id, revision)
		);

		CREATE INDEX IF NOT EXISTS idx_memory_versions_item
			ON memory_versions(item_id, revision);

		CREATE INDEX IF NOT EXISTS idx_memory_versions_index_sync
			ON memory_versions(status, index_status);

		CREATE TABLE IF NOT EXISTS skill_versions (
			id TEXT PRIMARY KEY,
			item_id TEXT NOT NULL,
			revision INTEGER NOT NULL,
			name TEXT NOT NULL,
			description TEXT NOT NULL,
			triggers_json TEXT NOT NULL,
			tags_json TEXT NOT NULL,
			instructions_markdown TEXT NOT NULL,
			status TEXT NOT NULL,
			chat_id TEXT NOT NULL,
			user_id TEXT NOT NULL,
			source_message_seq INTEGER,
			fsm_run_id TEXT,
			content_hash TEXT NOT NULL,
			created_at INTEGER NOT NULL,
			reviewed_at INTEGER,
			reviewed_by TEXT,
			index_status TEXT NOT NULL DEFAULT 'pending',
			FOREIGN KEY(item_id) REFERENCES knowledge_items(id) ON DELETE CASCADE,
			CONSTRAINT uq_skill_versions_item_rev UNIQUE(item_id, revision)
		);

		CREATE INDEX IF NOT EXISTS idx_skill_versions_item
			ON skill_versions(item_id, revision);

		CREATE INDEX IF NOT EXISTS idx_skill_versions_index_sync
			ON skill_versions(status, index_status);

		CREATE TABLE IF NOT EXISTS knowledge_tombstones (
			item_id TEXT PRIMARY KEY,
			kind TEXT NOT NULL,
			chat_id TEXT NOT NULL,
			user_id TEXT NOT NULL,
			deleted_by TEXT NOT NULL,
			deleted_at INTEGER NOT NULL,
			reason TEXT
		);

		CREATE TABLE IF NOT EXISTS knowledge_schema_version (
			version INTEGER PRIMARY KEY,
			description TEXT NOT NULL,
			applied_at INTEGER NOT NULL
		);

		CREATE TABLE IF NOT EXISTS pending_index_deletions (
			version_id TEXT PRIMARY KEY,
			namespace TEXT NOT NULL,
			created_at INTEGER NOT NULL
		);
	`

	if _, err := db.ExecContext(ctx, schemaSQL); err != nil {
		return fmt.Errorf("failed to ensure knowledge schema: %w", err)
	}

	// Record version 1 if not present
	_, err := db.ExecContext(ctx, `
		INSERT OR IGNORE INTO knowledge_schema_version (version, description, applied_at)
		VALUES (1, 'initial knowledge schema', strftime('%s', 'now'))
	`)
	if err != nil {
		return fmt.Errorf("failed to record knowledge schema version: %w", err)
	}

	return nil
}
