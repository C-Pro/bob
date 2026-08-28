# Implementation Plan: SQLite Database Scaffolding & Single-Step Migration

## Phase 1: Dependencies, Data Path Configuration & Git Ignore
- [ ] Task: Add `modernc.org/sqlite` dependency and configure `DATA_DIR` in `internal/config`
    - [ ] Add `modernc.org/sqlite` to `go.mod`, run `go mod tidy` and `go mod vendor`
    - [ ] Write unit tests in `internal/config/config_test.go` for `DATA_DIR` defaults and `cfg.DBPath()` helper
    - [ ] Implement `DataDir` field in `Config`, `LoadFromEnv()`, and `DBPath(filename string)` helper in `internal/config/config.go`
    - [ ] Update `.gitignore` to ignore `data/*.db`, `*.db`, `*.db-journal`, `*.db-wal`, `*.db-shm`
    - [ ] Run `go test -race ./internal/config/...` and verify tests pass

## Phase 2: Store Package Implementation & Schema/Migration Templates
- [ ] Task: Implement `internal/store` with SQLite storage, schema initialization, and single-step migration
    - [ ] Create `internal/store/version.go` defining schema version struct (`Version: 1, Description: "initial schema"`)
    - [ ] Create `internal/store/schema.tmpl` with `schema_version` table and unique index
    - [ ] Create `internal/store/migrate.tmpl` with transactional single-step migration scaffold
    - [ ] Write comprehensive unit tests in `internal/store/sqlite_test.go`:
        - `TestCreateSchema`: fresh db initialization and version validation
        - `TestOpenNonExistentWithoutInit`: opening missing DB with `init=false` returns error
        - `TestOpenOrCreate`: helper creating DB if missing, or opening/migrating existing
        - `TestOpenUnsupportedVersion`: version mismatch handling (version behind by >1 or ahead)
        - `TestMigrateFromPrevVersion`: migration execution from previous schema
    - [ ] Implement `SQLiteStorage`, `NewSQLiteStore`, `OpenOrCreate`, `initSchema`, `migrate`, `getSchemaVersion`, `Close`, and `DB` in `internal/store/sqlite.go` with WAL mode and foreign keys enabled
    - [ ] Run `go test -race ./internal/store/...` and verify all tests pass

## Phase 3: Git Post-Commit Hook for DB Release Tagging
- [ ] Task: Implement and verify `.githooks/post-commit` hook for tagging `db-release-N`
    - [ ] Create `.githooks/post-commit` script detecting `internal/store/version.go` changes and applying `db-release-N` tag if missing
    - [ ] Add test verification for post-commit hook logic
    - [ ] Update `Makefile` or documentation to configure `git config core.hooksPath .githooks`

## Phase 4: Service Startup Integration & End-to-End Startup Lifecycle Tests
- [ ] Task: Integrate store initialization into service entrypoint and execute end-to-end lifecycle verification
    - [ ] Update `cmd/agent/main.go` to open/create database via `store.OpenOrCreate(cfg.DBPath("bob.db"))` and close it on graceful shutdown
    - [ ] Write end-to-end service test covering:
        1. Starting with no DB: automatically creates DB at v1
        2. Starting with DB at version N-1: automatically migrates to current version
        3. Starting with DB at version N-2: fails on startup with explicit version mismatch error
    - [ ] Execute full service run tests and roll back test schema bump back to v1 clean state

## Phase 5: Verification & Quality Checks
- [ ] Task: Run full test suite and quality enforcement
    - [ ] Run `go test -race ./...` across all packages
    - [ ] Run `make check` (`golangci-lint`, `go test -race`, `semgrep`, `osv-scanner`) ensuring 0 errors
