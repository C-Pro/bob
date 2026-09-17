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

	// Configure SQLite pragmas for concurrency, durability, and incremental space reclamation
	if _, err := db.Exec("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON; PRAGMA busy_timeout=5000; PRAGMA auto_vacuum = INCREMENTAL;"); err != nil {
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
			if err := ensureFSMColumns(context.Background(), s.db); err != nil {
				_ = db.Close()
				return nil, err
			}
		case version.Version - 1:
			// run migration
			if err := s.migrate(); err != nil {
				_ = db.Close()
				return nil, fmt.Errorf("migration failed: %w", err)
			}
			if err := ensureFSMColumns(context.Background(), s.db); err != nil {
				_ = db.Close()
				return nil, err
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

// EnsureDBSchema ensures SQLite pragmas are set and either initializes the schema or migrates it to the current version.
func EnsureDBSchema(ctx context.Context, db *sql.DB) error {
	if db == nil {
		return fmt.Errorf("database connection is nil")
	}

	if _, err := db.ExecContext(ctx, "PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON; PRAGMA busy_timeout=5000; PRAGMA auto_vacuum = INCREMENTAL;"); err != nil {
		return fmt.Errorf("failed to configure sqlite pragmas: %w", err)
	}

	var tableCount int
	err := db.QueryRowContext(ctx, "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='schema_version'").Scan(&tableCount)
	if err != nil {
		return fmt.Errorf("failed to check schema_version table: %w", err)
	}

	if tableCount == 0 {
		return executeSchema(ctx, db)
	}

	var v int
	if err := db.QueryRowContext(ctx, "SELECT version FROM schema_version WHERE is_current=1").Scan(&v); err != nil {
		return fmt.Errorf("failed to get schema version: %w", err)
	}

	switch v {
	case version.Version:
		return ensureFSMColumns(ctx, db)
	case version.Version - 1:
		if err := executeMigrate(ctx, db); err != nil {
			return fmt.Errorf("migration failed: %w", err)
		}
		return ensureFSMColumns(ctx, db)
	default:
		return fmt.Errorf(
			"database version mismatch: expected %d or %d, but got %d",
			version.Version-1,
			version.Version,
			v)
	}
}

func ensureFSMColumns(ctx context.Context, db *sql.DB) error {
	var fsmRunsCount int
	err := db.QueryRowContext(ctx, "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='fsm_runs'").Scan(&fsmRunsCount)
	if err != nil || fsmRunsCount == 0 {
		return nil
	}

	rows, err := db.QueryContext(ctx, "PRAGMA table_info(fsm_runs)")
	if err != nil {
		return nil
	}
	defer func() { _ = rows.Close() }()

	hasIsDM := false
	for rows.Next() {
		var cid int
		var name, colType string
		var notnull, pk int
		var dfltValue sql.NullString
		if err := rows.Scan(&cid, &name, &colType, &notnull, &dfltValue, &pk); err == nil {
			if name == "is_dm" {
				hasIsDM = true
				break
			}
		}
	}
	if !hasIsDM {
		if _, err := db.ExecContext(ctx, "ALTER TABLE fsm_runs ADD COLUMN is_dm INTEGER NOT NULL DEFAULT 0"); err != nil {
			return fmt.Errorf("failed to add is_dm column to fsm_runs: %w", err)
		}
	}
	return nil
}

func executeSchema(ctx context.Context, db *sql.DB) error {
	var buf bytes.Buffer
	t := template.Must(template.New("schema").Parse(schemaTmpl))
	if err := t.Execute(&buf, version); err != nil {
		return fmt.Errorf("failed to render schema template: %w", err)
	}
	if _, err := db.ExecContext(ctx, buf.String()); err != nil {
		return fmt.Errorf("schema creation failed: %w", err)
	}
	return nil
}

func executeMigrate(ctx context.Context, db *sql.DB) error {
	var buf bytes.Buffer
	t := template.Must(template.New("migrate").Parse(migrateTmpl))
	if err := t.Execute(&buf, version); err != nil {
		return fmt.Errorf("migration template render failed: %w", err)
	}
	if _, err := db.ExecContext(ctx, buf.String()); err != nil {
		return fmt.Errorf("schema migration failed: %w", err)
	}
	return nil
}

func (s *SQLiteStorage) initSchema() error {
	return executeSchema(context.Background(), s.db)
}

func (s *SQLiteStorage) migrate() error {
	return executeMigrate(context.Background(), s.db)
}

func (s *SQLiteStorage) getSchemaVersion() (int, error) {
	return s.GetSchemaVersion(context.Background())
}
