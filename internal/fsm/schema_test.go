package fsm

import (
	"context"
	"database/sql"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	_ "modernc.org/sqlite"
)

func TestEnsureDBSchema(t *testing.T) {
	ctx := context.Background()

	t.Run("nil db returns error", func(t *testing.T) {
		err := EnsureDBSchema(ctx, nil)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "database connection is nil")
	})

	t.Run("fresh database initializes fsm tables and namespaced version", func(t *testing.T) {
		dbPath := filepath.Join(t.TempDir(), "fresh_fsm.db")
		db, err := sql.Open("sqlite", dbPath)
		require.NoError(t, err)
		defer func() { _ = db.Close() }()

		err = EnsureDBSchema(ctx, db)
		require.NoError(t, err)

		v, err := GetSchemaVersion(ctx, db)
		require.NoError(t, err)
		assert.Equal(t, SchemaVersion, v)

		// Must NOT create store package's schema_version table
		var storeSchemaCount int
		err = db.QueryRowContext(ctx, "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='schema_version'").Scan(&storeSchemaCount)
		require.NoError(t, err)
		assert.Equal(t, 0, storeSchemaCount, "generic schema_version table must not be created in chat DB")

		// Verify fsm_runs, fsm_steps, and fsm_compactions exist
		var fsmRunsCount, fsmStepsCount, fsmCompactionsCount int
		err = db.QueryRowContext(ctx, "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='fsm_runs'").Scan(&fsmRunsCount)
		require.NoError(t, err)
		assert.Equal(t, 1, fsmRunsCount)

		err = db.QueryRowContext(ctx, "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='fsm_steps'").Scan(&fsmStepsCount)
		require.NoError(t, err)
		assert.Equal(t, 1, fsmStepsCount)

		err = db.QueryRowContext(ctx, "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='fsm_compactions'").Scan(&fsmCompactionsCount)
		require.NoError(t, err)
		assert.Equal(t, 1, fsmCompactionsCount)

		// Calling it again on up-to-date db is idempotent
		err = EnsureDBSchema(ctx, db)
		require.NoError(t, err)
	})

	t.Run("auto-migrates older fsm_runs missing columns", func(t *testing.T) {
		dbPath := filepath.Join(t.TempDir(), "older_fsm.db")
		db, err := sql.Open("sqlite", dbPath)
		require.NoError(t, err)
		defer func() { _ = db.Close() }()

		// Create fsm_runs without is_dm, wait_cycles, version
		_, err = db.ExecContext(ctx, `
			CREATE TABLE fsm_runs (
				id TEXT PRIMARY KEY,
				chat_id TEXT NOT NULL,
				user_id TEXT NOT NULL,
				fsm_type TEXT NOT NULL,
				status TEXT NOT NULL,
				current_state TEXT NOT NULL,
				iteration INTEGER NOT NULL DEFAULT 0,
				max_iterations INTEGER NOT NULL DEFAULT 20,
				context_json TEXT NOT NULL,
				result_json TEXT,
				error_text TEXT,
				resume_at INTEGER,
				created_at INTEGER NOT NULL,
				updated_at INTEGER NOT NULL
			);
		`)
		require.NoError(t, err)

		err = EnsureDBSchema(ctx, db)
		require.NoError(t, err)

		// Verify columns were added
		rows, err := db.QueryContext(ctx, "PRAGMA table_info(fsm_runs)")
		require.NoError(t, err)
		defer func() { _ = rows.Close() }()

		hasIsDM, hasWaitCycles, hasVersion, hasSourceMessageSeq := false, false, false, false
		hasFrontendID, hasScopeID, hasModel, hasExecutionKind := false, false, false, false
		for rows.Next() {
			var cid int
			var name, colType string
			var notnull, pk int
			var dflt sql.NullString
			require.NoError(t, rows.Scan(&cid, &name, &colType, &notnull, &dflt, &pk))
			if name == "is_dm" {
				hasIsDM = true
			}
			if name == "wait_cycles" {
				hasWaitCycles = true
			}
			if name == "version" {
				hasVersion = true
			}
			if name == "source_message_seq" {
				hasSourceMessageSeq = true
			}
			if name == "frontend_id" {
				hasFrontendID = true
			}
			if name == "scope_id" {
				hasScopeID = true
			}
			if name == "model" {
				hasModel = true
			}
			if name == "execution_kind" {
				hasExecutionKind = true
			}
		}
		assert.True(t, hasIsDM)
		assert.True(t, hasWaitCycles)
		assert.True(t, hasVersion)
		assert.True(t, hasSourceMessageSeq)
		assert.True(t, hasFrontendID)
		assert.True(t, hasScopeID)
		assert.True(t, hasModel)
		assert.True(t, hasExecutionKind)
	})

	t.Run("migrates populated version-3 database to version 4 preserving runs and steps", func(t *testing.T) {
		dbPath := filepath.Join(t.TempDir(), "v3_populated_fsm.db")
		db, err := sql.Open("sqlite", dbPath)
		require.NoError(t, err)
		defer func() { _ = db.Close() }()

		_, err = db.ExecContext(ctx, `
			CREATE TABLE fsm_schema_version (
				version INTEGER PRIMARY KEY,
				description TEXT NOT NULL,
				applied_at INTEGER NOT NULL
			);
			INSERT INTO fsm_schema_version (version, description, applied_at)
			VALUES (3, 'add frontend_id, scope_id, model, execution_kind to fsm_runs', 1700000000);

			CREATE TABLE fsm_runs (
				id TEXT PRIMARY KEY,
				chat_id TEXT NOT NULL,
				user_id TEXT NOT NULL,
				is_dm INTEGER NOT NULL DEFAULT 0,
				frontend_id TEXT NOT NULL DEFAULT '',
				scope_id TEXT NOT NULL DEFAULT '',
				model TEXT NOT NULL DEFAULT '',
				execution_kind TEXT NOT NULL DEFAULT '',
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
				source_message_seq INTEGER,
				created_at INTEGER NOT NULL,
				updated_at INTEGER NOT NULL
			);

			CREATE TABLE fsm_steps (
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

			INSERT INTO fsm_runs (id, chat_id, user_id, fsm_type, status, current_state, context_json, created_at, updated_at)
			VALUES ('run_v3_1', 'chat_1', 'user_1', 'tool_loop', 'RUNNING', 'EXECUTE_STEPS', '[]', 1700000000, 1700000000);

			INSERT INTO fsm_steps (id, run_id, iteration, step_index, tool_name, tool_call_id, args_json, execution_mode, status)
			VALUES ('step_v3_1', 'run_v3_1', 1, 0, 'web_search', 'call_1', '{}', 'parallel', 'COMPLETED');
		`)
		require.NoError(t, err)

		require.NoError(t, EnsureDBSchema(ctx, db))

		v, err := GetSchemaVersion(ctx, db)
		require.NoError(t, err)
		assert.Equal(t, 4, v)

		store := NewStore(db)
		run, err := store.GetRun(ctx, "run_v3_1")
		require.NoError(t, err)
		assert.Equal(t, "chat_1", run.ChatID)

		step, err := store.GetStep(ctx, "step_v3_1")
		require.NoError(t, err)
		assert.Equal(t, "web_search", step.ToolName)

		var fsmCompactionsCount int
		err = db.QueryRowContext(ctx, "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='fsm_compactions'").Scan(&fsmCompactionsCount)
		require.NoError(t, err)
		assert.Equal(t, 1, fsmCompactionsCount)
	})
}
