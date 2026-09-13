package store

import (
	"context"
	"database/sql"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestFullLifecycleMigration verifies:
// 1. Starting with no DB: creates DB at current version.
// 2. Starting with DB at version N-1: auto-migrates to current version N.
// 3. Starting with DB at version N-2: fails with explicit mismatch error.
func TestFullLifecycleMigration(t *testing.T) {
	t.Run(fmt.Sprintf("No DB creates fresh at v%d", version.Version), func(t *testing.T) {
		dbPath := filepath.Join(t.TempDir(), "new.db")
		s, err := OpenOrCreate(dbPath)
		require.NoError(t, err)
		defer func() { _ = s.Close() }()

		v, err := s.GetSchemaVersion(context.Background())
		require.NoError(t, err)
		assert.Equal(t, version.Version, v)
		assert.FileExists(t, dbPath)
	})

	t.Run(fmt.Sprintf("DB at previous version (v%d) migrates cleanly to v%d", version.Version-1, version.Version), func(t *testing.T) {
		dbPath := filepath.Join(t.TempDir(), "migrating.db")

		// 1. Create DB at Version N-1
		db, err := sql.Open("sqlite", dbPath)
		require.NoError(t, err)
		setup := fmt.Sprintf(`
create table schema_version(
  version integer primary key,
  description text not null,
  is_current boolean default 0 check (is_current in (0, 1))
);
create unique index schema_version_uk on schema_version(is_current) where is_current = 1;
insert into schema_version(version, description, is_current) values(%d, 'previous schema', 1);`,
			version.Version-1)
		_, err = db.Exec(setup)
		require.NoError(t, err)
		require.NoError(t, db.Close())

		// 2. Open existing db (init=false) - triggers auto-migration
		s, err := NewSQLiteStore(dbPath, false)
		require.NoError(t, err)
		defer func() { _ = s.Close() }()

		vCurrent, err := s.GetSchemaVersion(context.Background())
		require.NoError(t, err)
		assert.Equal(t, version.Version, vCurrent)

		// 3. Verify migrated tables exist and are functional
		var runCount int
		err = s.DB().QueryRow("select count(*) from fsm_runs").Scan(&runCount)
		require.NoError(t, err)
		assert.Equal(t, 0, runCount)
	})

	t.Run("DB at v-2 fails on open", func(t *testing.T) {
		dbPath := filepath.Join(t.TempDir(), "unsupported.db")
		require.NoError(t, os.MkdirAll(filepath.Dir(dbPath), 0o755))

		db, err := sql.Open("sqlite", dbPath)
		require.NoError(t, err)

		setup := fmt.Sprintf(`
create table schema_version(
  version integer primary key,
  description text not null,
  is_current boolean default 0 check (is_current in (0, 1))
);
create unique index schema_version_uk on schema_version(is_current) where is_current = 1;
insert into schema_version(version, description, is_current) values(%d, 'ancient', 1);`,
			version.Version-2)
		_, err = db.Exec(setup)
		require.NoError(t, err)
		require.NoError(t, db.Close())

		_, err = NewSQLiteStore(dbPath, false)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "database version mismatch")
	})
}

func TestEnsureDBSchema(t *testing.T) {
	ctx := context.Background()

	t.Run("nil db returns error", func(t *testing.T) {
		err := EnsureDBSchema(ctx, nil)
		require.Error(t, err)
	})

	t.Run("in-memory fresh db", func(t *testing.T) {
		db, err := sql.Open("sqlite", ":memory:")
		require.NoError(t, err)
		defer func() { _ = db.Close() }()

		err = EnsureDBSchema(ctx, db)
		require.NoError(t, err)

		var v int
		err = db.QueryRowContext(ctx, "select version from schema_version where is_current=1").Scan(&v)
		require.NoError(t, err)
		assert.Equal(t, version.Version, v)

		// Calling it again on up-to-date db is idempotent
		err = EnsureDBSchema(ctx, db)
		require.NoError(t, err)
	})

	t.Run("in-memory db migration from N-1", func(t *testing.T) {
		db, err := sql.Open("sqlite", ":memory:")
		require.NoError(t, err)
		defer func() { _ = db.Close() }()

		setup := fmt.Sprintf(`
create table schema_version(
  version integer primary key,
  description text not null,
  is_current boolean default 0 check (is_current in (0, 1))
);
create unique index schema_version_uk on schema_version(is_current) where is_current = 1;
insert into schema_version(version, description, is_current) values(%d, 'prev', 1);`,
			version.Version-1)
		_, err = db.ExecContext(ctx, setup)
		require.NoError(t, err)

		err = EnsureDBSchema(ctx, db)
		require.NoError(t, err)

		var v int
		err = db.QueryRowContext(ctx, "select version from schema_version where is_current=1").Scan(&v)
		require.NoError(t, err)
		assert.Equal(t, version.Version, v)
	})
}
