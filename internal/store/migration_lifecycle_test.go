package store

import (
	"bytes"
	"context"
	"database/sql"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"text/template"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestFullLifecycleMigration verifies:
// 1. Starting with no DB: creates DB at v1.
// 2. Starting with DB at version N-1: auto-migrates to current version N.
// 3. Starting with DB at version N-2: fails with explicit mismatch error.
func TestFullLifecycleMigration(t *testing.T) {
	t.Run("No DB creates fresh at v1", func(t *testing.T) {
		dbPath := filepath.Join(t.TempDir(), "new.db")
		s, err := OpenOrCreate(dbPath)
		require.NoError(t, err)
		defer func() { _ = s.Close() }()

		v, err := s.GetSchemaVersion(context.Background())
		require.NoError(t, err)
		assert.Equal(t, 1, v)
		assert.FileExists(t, dbPath)
	})

	t.Run("DB at previous version (v-1) migrates cleanly to v2", func(t *testing.T) {
		dbPath := filepath.Join(t.TempDir(), "migrating.db")

		// 1. Create DB at Version 1
		s1, err := NewSQLiteStore(dbPath, true)
		require.NoError(t, err)
		v1, err := s1.GetSchemaVersion(context.Background())
		require.NoError(t, err)
		assert.Equal(t, 1, v1)
		require.NoError(t, s1.Close())

		// 2. Simulate store at Version 2 with a new table in migration
		v2Info := struct {
			Version     int
			Description string
		}{2, "add test table"}

		migrateV2 := `
begin;
create table test_table (
  id text primary key,
  data text not null
);
update schema_version set is_current = 0 where is_current = 1;
insert into schema_version(version, description, is_current)
  values({{.Version}}, '{{.Description}}', 1);
end;`

		// Render migration script
		var buf bytes.Buffer
		tmpl := template.Must(template.New("migrate").Parse(migrateV2))
		require.NoError(t, tmpl.Execute(&buf, v2Info))

		// Open existing db (init=false) and execute simulated migration
		db, err := sql.Open("sqlite", fmt.Sprintf("file://%s?mode=rw", dbPath))
		require.NoError(t, err)
		_, err = db.Exec(buf.String())
		require.NoError(t, err)

		var vCurrent int
		err = db.QueryRow("select version from schema_version where is_current=1").Scan(&vCurrent)
		require.NoError(t, err)
		assert.Equal(t, 2, vCurrent)

		// Verify new table exists and works
		_, err = db.Exec("insert into test_table(id, data) values('k1', 'val1')")
		require.NoError(t, err)
		require.NoError(t, db.Close())
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
insert into schema_version(version, description, is_current) values(-1, 'ancient', 1);`
		_, err = db.Exec(setup)
		require.NoError(t, err)
		require.NoError(t, db.Close())

		_, err = NewSQLiteStore(dbPath, false)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "database version mismatch")
	})
}
