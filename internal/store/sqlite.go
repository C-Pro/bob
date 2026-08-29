// Package store contains the storage layer implemented as a SQLite database.
package store

import (
	"bytes"
	"context"
	"database/sql"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"text/template" // nosemgrep: go.lang.security.audit.xss.import-text-template.import-text-template

	_ "embed"

	_ "modernc.org/sqlite"
)

var (
	//go:embed schema.tmpl
	schemaTmpl string
	//go:embed migrate.tmpl
	migrateTmpl string
)

// SQLiteStorage wraps an active SQLite database connection.
type SQLiteStorage struct {
	db *sql.DB
}

// NewSQLiteStore opens or initializes a SQLite database.
// When init is true, parent directories are created if necessary, the database file
// is initialized if missing, and initial schema.tmpl is executed.
// When init is false, the database is opened in read-write mode (mode=rw) and
// auto-migrated if its current version is version.Version - 1.
func NewSQLiteStore(fname string, init bool) (*SQLiteStorage, error) {
	dsn := fname
	if init {
		dir := filepath.Dir(fname)
		if dir != "" && dir != "." {
			if err := os.MkdirAll(dir, 0o755); err != nil {
				return nil, fmt.Errorf("failed to create database directory %q: %w", dir, err)
			}
		}
	} else {
		// mode=rw will fail if fname does not exist as opposed to default mode=rwc
		dsn = fmt.Sprintf("file:%s?mode=rw", fname)
	}

	db, err := sql.Open("sqlite", dsn)
	if err != nil {
		return nil, fmt.Errorf("failed to open sqlite database %q: %w", fname, err)
	}

	// Open doesn't check if the db file exists; Ping validates connection and mode=rw existence
	if err := db.Ping(); err != nil {
		_ = db.Close()
		return nil, fmt.Errorf("failed to Ping the database %q: %w", fname, err)
	}

	// Configure SQLite pragmas for concurrency and durability
	if _, err := db.Exec("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON; PRAGMA busy_timeout=5000;"); err != nil {
		_ = db.Close()
		return nil, fmt.Errorf("failed to configure sqlite pragmas: %w", err)
	}

	s := &SQLiteStorage{
		db: db,
	}

	if init {
		if err := s.initSchema(); err != nil {
			_ = db.Close()
			return nil, err
		}
	} else {
		v, err := s.getSchemaVersion()
		if err != nil {
			_ = db.Close()
			return nil, err
		}

		switch v {
		case version.Version:
			// nothing to do
		case version.Version - 1:
			// run migration
			if err := s.migrate(); err != nil {
				_ = db.Close()
				return nil, fmt.Errorf("migration failed: %w", err)
			}
		default:
			_ = db.Close()
			return nil, fmt.Errorf(
				"database version mismatch: expected %d or %d, but got %d",
				version.Version-1,
				version.Version,
				v)
		}
	}

	return s, nil
}

// OpenOrCreate checks if the SQLite file exists. If it exists, it opens and migrates it (init=false).
// If it does not exist, it creates the database and initializes its schema (init=true).
func OpenOrCreate(fname string) (*SQLiteStorage, error) {
	if _, err := os.Stat(fname); errors.Is(err, os.ErrNotExist) {
		return NewSQLiteStore(fname, true)
	}
	return NewSQLiteStore(fname, false)
}

// DB returns the underlying sql.DB connection pool.
func (s *SQLiteStorage) DB() *sql.DB {
	return s.db
}

// Close closes the underlying database connection.
func (s *SQLiteStorage) Close() error {
	if s.db != nil {
		return s.db.Close()
	}
	return nil
}

// GetSchemaVersion returns the currently active schema version.
func (s *SQLiteStorage) GetSchemaVersion(ctx context.Context) (int, error) {
	var v int
	if err := s.db.QueryRowContext(ctx, "select version from schema_version where is_current=1").Scan(&v); err != nil {
		return 0, fmt.Errorf("failed to get schema version: %w", err)
	}
	return v, nil
}

func (s *SQLiteStorage) initSchema() error {
	var buf bytes.Buffer
	t := template.Must(template.New("schema").Parse(schemaTmpl))
	if err := t.Execute(&buf, version); err != nil {
		return fmt.Errorf("failed to render schema template: %w", err)
	}
	if _, err := s.db.Exec(buf.String()); err != nil {
		return fmt.Errorf("schema creation failed: %w", err)
	}

	return nil
}

func (s *SQLiteStorage) migrate() error {
	var buf bytes.Buffer
	t := template.Must(template.New("migrate").Parse(migrateTmpl))
	if err := t.Execute(&buf, version); err != nil {
		return fmt.Errorf("migration template render failed: %w", err)
	}
	if _, err := s.db.Exec(buf.String()); err != nil {
		return fmt.Errorf("schema migration failed: %w", err)
	}

	return nil
}

func (s *SQLiteStorage) getSchemaVersion() (int, error) {
	return s.GetSchemaVersion(context.Background())
}
