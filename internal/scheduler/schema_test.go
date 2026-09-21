package scheduler

import (
	"context"
	"database/sql"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	_ "modernc.org/sqlite"
)

func TestEnsureScheduleSchema(t *testing.T) {
	ctx := context.Background()

	t.Run("creates tables and version marker on fresh database", func(t *testing.T) {
		dbPath := filepath.Join(t.TempDir(), "fresh.db")
		db, err := sql.Open("sqlite", "file:"+dbPath+"?_pragma=foreign_keys(ON)")
		require.NoError(t, err)
		defer func() { _ = db.Close() }()

		err = EnsureScheduleSchema(ctx, db)
		require.NoError(t, err)

		// Verify tables exist
		tables := []string{"scheduler_schema_version", "schedules", "schedule_grants"}
		for _, tbl := range tables {
			var exists int
			err = db.QueryRowContext(ctx, "SELECT count(*) FROM sqlite_master WHERE type='table' AND name=?", tbl).Scan(&exists)
			require.NoError(t, err)
			assert.Equal(t, 1, exists, "table %s should exist", tbl)
		}

		// Verify version marker
		var version int
		var desc string
		err = db.QueryRowContext(ctx, "SELECT version, description FROM scheduler_schema_version WHERE version = ?", SchemaVersion).Scan(&version, &desc)
		require.NoError(t, err)
		assert.Equal(t, SchemaVersion, version)
		assert.Equal(t, SchemaDescription, desc)

		// Idempotency: calling EnsureScheduleSchema again succeeds
		err = EnsureScheduleSchema(ctx, db)
		require.NoError(t, err)
	})

	t.Run("enforces UNIQUE(chat_id, name)", func(t *testing.T) {
		dbPath := filepath.Join(t.TempDir(), "unique_name.db")
		db, err := sql.Open("sqlite", "file:"+dbPath+"?_pragma=foreign_keys(ON)")
		require.NoError(t, err)
		defer func() { _ = db.Close() }()

		require.NoError(t, EnsureScheduleSchema(ctx, db))

		insertSQL := `INSERT INTO schedules (id, name, chat_id, user_id, schedule_type, instruction, status, next_run_at, run_timeout_seconds, max_turns, created_at, updated_at)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
		now := time.Now().Unix()

		// First insert succeeds
		_, err = db.ExecContext(ctx, insertSQL, "s1", "daily_brief", "chat1", "u1", "CRON", "prompt", "ACTIVE", now, 300, 10, now, now)
		require.NoError(t, err)

		// Same chat_id, same name fails
		_, err = db.ExecContext(ctx, insertSQL, "s2", "daily_brief", "chat1", "u2", "CRON", "prompt", "ACTIVE", now, 300, 10, now, now)
		assert.Error(t, err, "duplicate name in same chat must fail")

		// Different chat_id, same name succeeds
		_, err = db.ExecContext(ctx, insertSQL, "s3", "daily_brief", "chat2", "u1", "CRON", "prompt", "ACTIVE", now, 300, 10, now, now)
		assert.NoError(t, err, "same name in different chat must succeed")
	})

	t.Run("cascades deletion of schedule_grants on schedule deletion", func(t *testing.T) {
		dbPath := filepath.Join(t.TempDir(), "cascade.db")
		db, err := sql.Open("sqlite", "file:"+dbPath+"?_pragma=foreign_keys(ON)")
		require.NoError(t, err)
		defer func() { _ = db.Close() }()

		require.NoError(t, EnsureScheduleSchema(ctx, db))

		now := time.Now().Unix()
		_, err = db.ExecContext(ctx, `INSERT INTO schedules (id, name, chat_id, user_id, schedule_type, instruction, status, next_run_at, run_timeout_seconds, max_turns, created_at, updated_at)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
			"sched_cascade", "my_task", "chat1", "u1", "CRON", "prompt", "ACTIVE", now, 300, 10, now, now)
		require.NoError(t, err)

		_, err = db.ExecContext(ctx, `INSERT INTO schedule_grants (id, schedule_id, permission_request_json, params_hash, granted_by, granted_at, valid_until)
			VALUES (?, ?, ?, ?, ?, ?, ?)`,
			"grant_cascade", "sched_cascade", `{"sandbox":{}}`, "hash123", "u1", now, now+3600)
		require.NoError(t, err)

		// Verify grant exists
		var grantCount int
		err = db.QueryRowContext(ctx, "SELECT count(*) FROM schedule_grants WHERE schedule_id = 'sched_cascade'").Scan(&grantCount)
		require.NoError(t, err)
		assert.Equal(t, 1, grantCount)

		// Delete schedule
		_, err = db.ExecContext(ctx, "DELETE FROM schedules WHERE id = 'sched_cascade'")
		require.NoError(t, err)

		// Grant must be cascade deleted
		err = db.QueryRowContext(ctx, "SELECT count(*) FROM schedule_grants WHERE schedule_id = 'sched_cascade'").Scan(&grantCount)
		require.NoError(t, err)
		assert.Equal(t, 0, grantCount, "grant must be cascade deleted")
	})
}
