package scheduler

import (
	"context"
	"database/sql"
	"fmt"
	"time"
)

const (
	// SchemaVersion is the current schema version for scheduler tables.
	SchemaVersion = 1
	// SchemaDescription describes the current scheduler schema.
	SchemaDescription = "initial scheduler schema with schedules and schedule_grants"
)

// EnsureScheduleSchema ensures that scheduler tables (scheduler_schema_version, schedules, schedule_grants)
// and indexes exist in the target SQLite database.
func EnsureScheduleSchema(ctx context.Context, db *sql.DB) error {
	if db == nil {
		return fmt.Errorf("database connection is nil")
	}

	// Apply core runtime pragmas
	pragmas := []string{
		"PRAGMA journal_mode=WAL;",
		"PRAGMA foreign_keys=ON;",
		"PRAGMA busy_timeout=5000;",
	}
	for _, p := range pragmas {
		if _, err := db.ExecContext(ctx, p); err != nil {
			return fmt.Errorf("failed to configure sqlite pragma %q: %w", p, err)
		}
	}

	// auto_vacuum takes effect on empty databases before any tables exist
	var allTables int
	if err := db.QueryRowContext(ctx, "SELECT count(*) FROM sqlite_master WHERE type='table'").Scan(&allTables); err == nil && allTables == 0 {
		_, _ = db.ExecContext(ctx, "PRAGMA auto_vacuum = INCREMENTAL;")
	}

	// Create scheduler_schema_version table for decoupled version tracking
	_, err := db.ExecContext(ctx, `
		CREATE TABLE IF NOT EXISTS scheduler_schema_version (
			version INTEGER PRIMARY KEY,
			description TEXT NOT NULL,
			applied_at INTEGER NOT NULL
		);
	`)
	if err != nil {
		return fmt.Errorf("failed to create scheduler_schema_version table: %w", err)
	}

	// Create schedules and schedule_grants tables
	schemaSQL := `
		CREATE TABLE IF NOT EXISTS schedules (
			id TEXT PRIMARY KEY,
			name TEXT NOT NULL,
			chat_id TEXT NOT NULL,
			user_id TEXT NOT NULL,
			schedule_type TEXT NOT NULL,
			cron_expr TEXT,
			interval_seconds INTEGER,
			instruction TEXT NOT NULL,
			status TEXT NOT NULL,
			next_run_at INTEGER NOT NULL,
			last_run_at INTEGER,
			expires_at INTEGER,
			max_runs INTEGER NOT NULL DEFAULT 0,
			run_count INTEGER NOT NULL DEFAULT 0,
			missed_count INTEGER NOT NULL DEFAULT 0,
			run_timeout_seconds INTEGER NOT NULL,
			max_turns INTEGER NOT NULL,
			last_status TEXT,
			created_at INTEGER NOT NULL,
			updated_at INTEGER NOT NULL,
			CONSTRAINT uq_schedules_chat_name UNIQUE(chat_id, name)
		);

		CREATE INDEX IF NOT EXISTS idx_schedules_due
			ON schedules(status, next_run_at)
			WHERE status = 'ACTIVE';

		CREATE INDEX IF NOT EXISTS idx_schedules_chat_name
			ON schedules(chat_id, name);

		CREATE INDEX IF NOT EXISTS idx_schedules_chat_status
			ON schedules(chat_id, status, updated_at);

		CREATE TABLE IF NOT EXISTS schedule_grants (
			id TEXT PRIMARY KEY,
			schedule_id TEXT NOT NULL UNIQUE,
			permission_request_json TEXT NOT NULL,
			params_hash TEXT NOT NULL,
			granted_by TEXT NOT NULL,
			granted_at INTEGER NOT NULL,
			valid_until INTEGER NOT NULL,
			FOREIGN KEY(schedule_id) REFERENCES schedules(id) ON DELETE CASCADE
		);

		CREATE INDEX IF NOT EXISTS idx_schedule_grants_schedule_id
			ON schedule_grants(schedule_id);
	`
	if _, err := db.ExecContext(ctx, schemaSQL); err != nil {
		return fmt.Errorf("failed to initialize scheduler tables: %w", err)
	}

	// Record version marker if not present
	var currentVersion int
	err = db.QueryRowContext(ctx, "SELECT COALESCE(MAX(version), 0) FROM scheduler_schema_version").Scan(&currentVersion)
	if err != nil {
		return fmt.Errorf("failed to check scheduler schema version: %w", err)
	}
	if currentVersion < SchemaVersion {
		_, err = db.ExecContext(ctx, "INSERT OR REPLACE INTO scheduler_schema_version (version, description, applied_at) VALUES (?, ?, ?)",
			SchemaVersion, SchemaDescription, time.Now().Unix())
		if err != nil {
			return fmt.Errorf("failed to record scheduler schema version: %w", err)
		}
	}

	return nil
}
