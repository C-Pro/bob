# Specification: SQLite Database Scaffolding & Single-Step Migration

## 1. Overview
Introduce SQLite database storage scaffolding into Bob in `internal/store` with automatic schema initialization and single-step version migration logic adopted from `degen-bot`. This track establishes pure Go SQLite connectivity, automated schema creation, single-step templated migrations, data directory configuration, service startup integration, and a post-commit git hook to tag `db-release-N`.

## 2. Functional Requirements

### 2.1 Package `internal/store`
- **SQLite Storage Driver**: Pure Go `modernc.org/sqlite` (no CGO required).
- **Core Store Struct & Operations**:
  - `SQLiteStorage` wrapping `*sql.DB`.
  - `NewSQLiteStore(fname string, init bool) (*SQLiteStorage, error)`:
    - If `init == true`: creates database file (ensuring parent directory exists) and applies `schema.tmpl`.
    - If `init == false`: opens database in read-write mode (`file://<path>?mode=rw`), verifies file exists via `Ping()`, inspects current `schema_version`, and handles version states:
      - `version.Version`: No migration needed.
      - `version.Version - 1`: Executes `migrate.tmpl` inside a transaction to advance schema by exactly one step.
      - Any other version: Returns an explicit mismatch error (`database version mismatch: expected <N-1> or <N>, but got <V>`).
  - `OpenOrCreate(fname string) (*SQLiteStorage, error)`: Helper to check if file exists; if not, calls `NewSQLiteStore(fname, true)`, else calls `NewSQLiteStore(fname, false)`.
  - `Close() error`: Closes the underlying database connection cleanly.
  - `DB() *sql.DB`: Accessor for the underlying `*sql.DB` connection pool.

### 2.2 Schema & Migration Templates
- **Version Tracking (`internal/store/version.go`)**:
  - Defines `version = struct { Version int; Description string }{1, "initial schema"}`.
- **Embedded Schema Template (`internal/store/schema.tmpl`)**:
  - Creates `schema_version` table:
    ```sql
    create table schema_version(
      version integer primary key,
      description text not null,
      is_current boolean default 0 check (is_current in (0, 1))
    );
    create unique index schema_version_uk on schema_version(is_current) where is_current = 1;
    insert into schema_version(version, description, is_current)
      values({{.Version}}, '{{.Description}}', 1);
    ```
- **Embedded Single-Step Migration Template (`internal/store/migrate.tmpl`)**:
  - Runs in a transaction:
    ```sql
    begin;
    -- START migration code
    -- (Schema changes for current version go here)
    -- END migration code
    update schema_version set is_current = 0 where is_current = 1;
    insert into schema_version(version, description, is_current)
      values({{.Version}}, '{{.Description}}', 1);
    end;
    ```

### 2.3 Configuration & Service Integration (`internal/config`, `cmd/agent`)
- Add `DATA_DIR` environment variable support (defaulting to `./data`), accessible via `cfg.DataDir`.
- Add helper method `cfg.DBPath(filename string) string` returning `filepath.Join(cfg.DataDir, filename)` (e.g. `cfg.DBPath("bob.db")`).
- Update `cmd/agent/main.go` to open/create the database on startup and close it on shutdown.
- Update `.gitignore` to ignore SQLite database files and write-ahead / journal artifacts (`data/*.db`, `*.db`, `*.db-journal`, `*.db-wal`, `*.db-shm`).

### 2.4 Git Post-Commit Hook for DB Release Tagging
- Add a post-commit git hook script (`.githooks/post-commit`):
  - Detects if `internal/store/version.go` was modified in the latest commit.
  - Extracts current schema version `N`.
  - Checks if tag `db-release-N` exists; if not, automatically tags the commit with `db-release-N`.

## 3. Non-Functional Requirements
- **Driver**: Pure Go `modernc.org/sqlite` with zero CGO dependencies for cross-platform portability.
- **Concurrency & Robustness**: Safe connection configuration (WAL mode, busy timeout, foreign keys enabled).
- **Vendor & Dependencies**: Run `go mod tidy` and `go mod vendor` to keep vendor directory consistent.
- **Verification**: 100% clean check with `make check` (golangci-lint, `go test -race`, semgrep, osv-scanner).

## 4. Acceptance Criteria
- Unit tests in `internal/store/sqlite_test.go`:
  - `TestCreateSchema`: Verifies fresh database creation, `schema_version` table initialization, and version verification.
  - `TestOpenNonExistentWithoutInit`: Verifies opening non-existent DB with `init=false` fails as expected.
  - `TestOpenUnsupportedVersion`: Verifies database with invalid/out-of-range versions triggers explicit mismatch errors.
  - `TestMigrateFromPrevVersion`: Validates migration from `db-release-<N-1>` when tag exists, as well as template rendering.
- Unit tests in `internal/config/config_test.go` verifying `DATA_DIR` default and `DBPath` resolution.
- Post-commit hook script tested and verified for version detection and tag creation.
- **End-to-End Service Run Verification**:
  - Full service run test with no DB: creates DB at current version.
  - Full service run test with DB of previous version (v-1): migrates cleanly to current version.
  - Full service run test with DB of v-2: service fails with version mismatch error on startup.
  - Roll back any test schema bump back to v1 clean state after test completion.
- `make check` passes cleanly with 0 errors.

## 5. Out of Scope
- Domain entity tables (FSM state, cron jobs, etc.) - deferred to future tracks.
