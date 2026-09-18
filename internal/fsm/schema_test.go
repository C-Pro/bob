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

		// Verify fsm_runs and fsm_steps exist
		var fsmRunsCount, fsmStepsCount int
		err = db.QueryRowContext(ctx, "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='fsm_runs'").Scan(&fsmRunsCount)
		require.NoError(t, err)
		assert.Equal(t, 1, fsmRunsCount)

		err = db.QueryRowContext(ctx, "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='fsm_steps'").Scan(&fsmStepsCount)
		require.NoError(t, err)
		assert.Equal(t, 1, fsmStepsCount)

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

		hasIsDM, hasWaitCycles, hasVersion := false, false, false
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
		}
		assert.True(t, hasIsDM)
		assert.True(t, hasWaitCycles)
		assert.True(t, hasVersion)
	})
}
