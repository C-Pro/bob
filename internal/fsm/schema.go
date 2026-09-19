package fsm

import (
	"context"
	"database/sql"
	"fmt"
)

const (
	// SchemaVersion is the current schema version for FSM tables.
	SchemaVersion = 1
	// SchemaDescription describes the current FSM schema.
	SchemaDescription = "initial fsm schema"
)

// EnsureDBSchema ensures that the namespaced FSM tables and version markers
// (fsm_schema_version, fsm_runs, fsm_steps) exist and are up to date in the target SQLite database.
func EnsureDBSchema(ctx context.Context, db *sql.DB) error {
	if db == nil {
		return fmt.Errorf("database connection is nil")
	}

	// Apply core runtime pragmas for connections opened without DSN pragmas
	if _, err := db.ExecContext(ctx, "PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON; PRAGMA busy_timeout=5000;"); err != nil {
		return fmt.Errorf("failed to configure sqlite pragmas: %w", err)
	}

	// auto_vacuum only takes effect on empty databases before any tables exist
	var allTables int
	if err := db.QueryRowContext(ctx, "SELECT count(*) FROM sqlite_master WHERE type='table'").Scan(&allTables); err == nil && allTables == 0 {
		_, _ = db.ExecContext(ctx, "PRAGMA auto_vacuum = INCREMENTAL;")
	}

	// Create fsm_schema_version table for decoupled version tracking
	_, err := db.ExecContext(ctx, `
		CREATE TABLE IF NOT EXISTS fsm_schema_version (
			version INTEGER PRIMARY KEY,
			description TEXT NOT NULL,
			applied_at INTEGER NOT NULL
		);
	`)
	if err != nil {
		return fmt.Errorf("failed to create fsm_schema_version table: %w", err)
	}

	// Create fsm_runs and fsm_steps tables
	schemaSQL := `
		CREATE TABLE IF NOT EXISTS fsm_runs (
			id TEXT PRIMARY KEY,
			chat_id TEXT NOT NULL,
			user_id TEXT NOT NULL,
			is_dm INTEGER NOT NULL DEFAULT 0,
			fsm_type TEXT NOT NULL,
			status TEXT NOT NULL,
			current_state TEXT NOT NULL,
			iteration INTEGER NOT NULL DEFAULT 0,
			max_iterations INTEGER NOT NULL DEFAULT 20,
			wait_cycles INTEGER NOT NULL DEFAULT 0,
			context_json TEXT NOT NULL,
			result_json TEXT,
			error_text TEXT,
			resume_at INTEGER,
			version INTEGER NOT NULL DEFAULT 0,
			created_at INTEGER NOT NULL,
			updated_at INTEGER NOT NULL
		);

		CREATE INDEX IF NOT EXISTS idx_fsm_runs_active
			ON fsm_runs(status, resume_at)
			WHERE status IN ('PENDING', 'RUNNING', 'WAITING');

		CREATE INDEX IF NOT EXISTS idx_fsm_runs_chat_status
			ON fsm_runs(chat_id, status, updated_at);

		CREATE TABLE IF NOT EXISTS fsm_steps (
			id TEXT PRIMARY KEY,
			run_id TEXT NOT NULL,
			iteration INTEGER NOT NULL,
			step_index INTEGER NOT NULL,
			tool_name TEXT NOT NULL,
			tool_call_id TEXT NOT NULL,
			args_json TEXT NOT NULL,
			result_json TEXT,
			execution_mode TEXT NOT NULL,
			status TEXT NOT NULL,
			attempt INTEGER NOT NULL DEFAULT 0,
			max_attempts INTEGER NOT NULL DEFAULT 3,
			timeout_seconds INTEGER NOT NULL DEFAULT 30,
			started_at INTEGER,
			completed_at INTEGER,
			error_text TEXT,
			FOREIGN KEY(run_id) REFERENCES fsm_runs(id) ON DELETE CASCADE
		);

		CREATE INDEX IF NOT EXISTS idx_fsm_steps_run_iter
			ON fsm_steps(run_id, iteration, step_index);
	`
	if _, err := db.ExecContext(ctx, schemaSQL); err != nil {
		return fmt.Errorf("failed to initialize fsm tables: %w", err)
	}

	// Ensure any newly added columns exist in fsm_runs for databases created with earlier iterations
	if err := ensureFSMColumns(ctx, db); err != nil {
		return fmt.Errorf("failed to ensure fsm columns: %w", err)
	}

	// Record version in fsm_schema_version
	_, err = db.ExecContext(ctx, `
		INSERT OR IGNORE INTO fsm_schema_version (version, description, applied_at)
		VALUES (?, ?, strftime('%s', 'now'));
	`, SchemaVersion, SchemaDescription)
	if err != nil {
		return fmt.Errorf("failed to record fsm schema version: %w", err)
	}

	return nil
}

// GetSchemaVersion returns the latest applied FSM schema version.
func GetSchemaVersion(ctx context.Context, db *sql.DB) (int, error) {
	if db == nil {
		return 0, fmt.Errorf("database connection is nil")
	}
	var v int
	err := db.QueryRowContext(ctx, "SELECT coalesce(max(version), 0) FROM fsm_schema_version").Scan(&v)
	if err != nil {
		return 0, fmt.Errorf("failed to get fsm schema version: %w", err)
	}
	return v, nil
}

func ensureFSMColumns(ctx context.Context, db *sql.DB) error {
	rows, err := db.QueryContext(ctx, "PRAGMA table_info(fsm_runs)")
	if err != nil {
		return nil
	}
	defer func() { _ = rows.Close() }()

	hasIsDM := false
	hasWaitCycles := false
	hasVersion := false
	for rows.Next() {
		var cid int
		var name, colType string
		var notnull, pk int
		var dfltValue sql.NullString
		if err := rows.Scan(&cid, &name, &colType, &notnull, &dfltValue, &pk); err == nil {
			if name == "is_dm" {
				hasIsDM = true
			}
			if name == "wait_cycles" {
				hasWaitCycles = true
			}
			if name == "version" {
				hasVersion = true
			}
		}
	}
	if !hasIsDM {
		if _, err := db.ExecContext(ctx, "ALTER TABLE fsm_runs ADD COLUMN is_dm INTEGER NOT NULL DEFAULT 0"); err != nil {
			return fmt.Errorf("failed to add is_dm column to fsm_runs: %w", err)
		}
	}
	if !hasWaitCycles {
		if _, err := db.ExecContext(ctx, "ALTER TABLE fsm_runs ADD COLUMN wait_cycles INTEGER NOT NULL DEFAULT 0"); err != nil {
			return fmt.Errorf("failed to add wait_cycles column to fsm_runs: %w", err)
		}
	}
	if !hasVersion {
		if _, err := db.ExecContext(ctx, "ALTER TABLE fsm_runs ADD COLUMN version INTEGER NOT NULL DEFAULT 0"); err != nil {
			return fmt.Errorf("failed to add version column to fsm_runs: %w", err)
		}
	}
	return nil
}
