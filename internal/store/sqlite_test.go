package store

import (
	"bytes"
	"context"
	"database/sql"
	"fmt"
	"os/exec"
	"path/filepath"
	"testing"
	"text/template"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCreateSchema(t *testing.T) {
	// Without "init" flag we expect to fail when db file does not exist
	nonexistent := filepath.Join(t.TempDir(), "nonexistent.db")
	s, err := NewSQLiteStore(nonexistent, false)
	require.Error(t, err)
	assert.Nil(t, s)

	// With init flag set new db file should be created in nested directories if needed
	nestedDB := filepath.Join(t.TempDir(), "sub", "dir", "nested.db")
	s, err = NewSQLiteStore(nestedDB, true)
	require.NoError(t, err)
	require.NotNil(t, s)
	defer func() { _ = s.Close() }()

	v, err := s.GetSchemaVersion(context.Background())
	require.NoError(t, err)
	assert.Equal(t, version.Version, v)
	assert.NotNil(t, s.DB())
}

func TestOpenOrCreate(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "open_or_create.db")

	// First call should initialize schema
	s1, err := OpenOrCreate(dbPath)
	require.NoError(t, err)
	require.NotNil(t, s1)

	v1, err := s1.GetSchemaVersion(context.Background())
	require.NoError(t, err)
	assert.Equal(t, version.Version, v1)
	require.NoError(t, s1.Close())

	// Second call on existing file should open without re-initializing
	s2, err := OpenOrCreate(dbPath)
	require.NoError(t, err)
	require.NotNil(t, s2)

	v2, err := s2.GetSchemaVersion(context.Background())
	require.NoError(t, err)
	assert.Equal(t, version.Version, v2)
	require.NoError(t, s2.Close())
}

func TestOpenUnsupportedVersion(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "ancient.db")
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
		version.Version-5)
	_, err = db.Exec(setup)
	require.NoError(t, err)
	require.NoError(t, db.Close())

	s, err := NewSQLiteStore(dbPath, false)
	require.Error(t, err)
	assert.Nil(t, s)
	assert.Contains(t, err.Error(), "database version mismatch")
}

func TestMigrateFromPrevVersion(t *testing.T) {
	if version.Version <= 1 {
		// Verify migrate.tmpl and schema.tmpl render valid SQL templates
		var buf bytes.Buffer
		tmpl := template.Must(template.New("migrate").Parse(migrateTmpl))
		err := tmpl.Execute(&buf, version)
		require.NoError(t, err)
		return
	}

	prevTag := fmt.Sprintf("db-release-%d", version.Version-1)

	prevSchema, err := exec.Command("git", "show", prevTag+":internal/store/schema.tmpl").Output()
	if err != nil {
		t.Fatalf("could not read %s:internal/store/schema.tmpl - is tag present? error: %v", prevTag, err)
	}

	prevPath := filepath.Join(t.TempDir(), "prev.db")
	prevDB, err := sql.Open("sqlite", prevPath)
	require.NoError(t, err)

	var buf bytes.Buffer
	tmpl := template.Must(template.New("prev").Parse(string(prevSchema)))
	prevVersion := struct {
		Version     int
		Description string
	}{version.Version - 1, "previous schema"}
	require.NoError(t, tmpl.Execute(&buf, prevVersion))
	_, err = prevDB.Exec(buf.String())
	require.NoError(t, err)
	require.NoError(t, prevDB.Close())

	migrated, err := NewSQLiteStore(prevPath, false)
	require.NoError(t, err)
	defer func() { _ = migrated.Close() }()

	v, err := migrated.GetSchemaVersion(context.Background())
	require.NoError(t, err)
	assert.Equal(t, version.Version, v)

	fresh, err := NewSQLiteStore(filepath.Join(t.TempDir(), "fresh.db"), true)
	require.NoError(t, err)
	defer func() { _ = fresh.Close() }()

	got := dumpSchema(t, migrated.db)
	want := dumpSchema(t, fresh.db)
	assert.Equal(t, want, got)
}

func dumpSchema(t *testing.T, db *sql.DB) []string {
	t.Helper()

	rows, err := db.Query(`select type, name, coalesce(sql, '') from sqlite_master ` +
		`where name not like 'sqlite_%' order by type, name`)
	require.NoError(t, err)
	defer func() { _ = rows.Close() }()

	var objs []string
	for rows.Next() {
		var typ, name, ddl string
		require.NoError(t, rows.Scan(&typ, &name, &ddl))
		objs = append(objs, fmt.Sprintf("%s %s: %s", typ, name, ddl))
	}
	require.NoError(t, rows.Err())
	return objs
}
