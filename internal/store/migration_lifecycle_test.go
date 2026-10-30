package store

import (
	"context"
	"database/sql"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestFullLifecycleMigration verifies:
// 1. Starting with no DB: creates DB at v1.
// 2. Starting with DB at version N-1: auto-migrates to current version N.
// 3. Starting with DB at version N-2: fails with explicit mismatch error.
func TestFullLifecycleMigration(t *testing.T) {
	t.Run("No DB creates fresh at current version", func(t *testing.T) {
		dbPath := filepath.Join(t.TempDir(), "new.db")
		s, err := OpenOrCreate(dbPath)
		require.NoError(t, err)
		defer func() { _ = s.Close() }()

		v, err := s.GetSchemaVersion(context.Background())
		require.NoError(t, err)
		assert.Equal(t, CurrentVersion(), v)
		assert.FileExists(t, dbPath)
	})

	t.Run("DB at previous version (v-1) migrates cleanly to current version", func(t *testing.T) {
		dbPath := filepath.Join(t.TempDir(), "migrating.db")

		// 1. Create DB at Version 1
		db, err := sql.Open("sqlite", dbPath)
		require.NoError(t, err)
		setupV1 := `
create table schema_version(
  version integer primary key,
  description text not null,
  is_current boolean default 0 check (is_current in (0, 1))
);
create unique index schema_version_uk on schema_version(is_current) where is_current = 1;
insert into schema_version(version, description, is_current) values(1, 'initial schema', 1);`
		_, err = db.Exec(setupV1)
		require.NoError(t, err)
		require.NoError(t, db.Close())

		// 2. Open with NewSQLiteStore (init=false) - should auto-migrate from v1 to current version
		s, err := NewSQLiteStore(dbPath, false)
		require.NoError(t, err)
		defer func() { _ = s.Close() }()

		vCurrent, err := s.GetSchemaVersion(context.Background())
		require.NoError(t, err)
		assert.Equal(t, CurrentVersion(), vCurrent)

		// Verify trajectory tables exist
		var count int
		err = s.DB().QueryRow("SELECT count(*) FROM sqlite_master WHERE type='table' AND name='agent_trajectories'").Scan(&count)
		require.NoError(t, err)
		assert.Equal(t, 1, count)
	})

	t.Run("DB at v-2 fails on open", func(t *testing.T) {
		dbPath := filepath.Join(t.TempDir(), "unsupported.db")
		require.NoError(t, os.MkdirAll(filepath.Dir(dbPath), 0o755))

		db, err := sql.Open("sqlite", dbPath)
		require.NoError(t, err)

		setup := `
create table schema_version(
  version integer primary key,
  description text not null,
  is_current boolean default 0 check (is_current in (0, 1))
);
create unique index schema_version_uk on schema_version(is_current) where is_current = 1;
insert into schema_version(version, description, is_current) values(0, 'ancient', 1);`
		_, err = db.Exec(setup)
		require.NoError(t, err)
		require.NoError(t, db.Close())

		_, err = NewSQLiteStore(dbPath, false)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "database version mismatch")
	})
}
