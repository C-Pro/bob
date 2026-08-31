# Implementation Plan: S3 Encrypted Database Backup & Restore

## Phase 1: Configuration, Crypto & S3 Object Store Client
- [x] Task: Implement S3 & Backup Configuration & Validation
    - [x] Add `SECRET` (with `AUTH_SECRET` fallback), `S3_ENDPOINT`, `S3_REGION`, `S3_BUCKET`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`, `S3_PATH_STYLE`, `S3_BACKUP_INTERVAL`, `S3_BACKUP_KEEP`, `S3_BACKUP_PREFIX` to `internal/config/config.go`
    - [x] Add config validation rules for S3 settings and secrets
    - [x] Write unit tests in `internal/config/config_test.go`
- [x] Task: Implement Cryptographic Engine & Header Codec (`BOBB` magic)
    - [x] Create `internal/backup/crypto.go` implementing Argon2id key derivation and AES-256-GCM encryption/decryption
    - [x] Create `internal/backup/header.go` for writing and parsing `BOBB` magic byte headers with version and salt
    - [x] Write unit tests for crypto and header codec in `internal/backup/crypto_test.go` and `internal/backup/header_test.go`
- [x] Task: Implement Minimal S3 Object Storage Client
    - [x] Create `internal/objectstore` package with AWS SigV4 authorization and REST methods (`Put`, `Get`, `List`, `Delete`)
    - [x] Write unit tests with HTTP mock servers in `internal/objectstore/client_test.go` and `internal/objectstore/sigv4_test.go`

## Phase 2: Snapshot Pipeline (VACUUM INTO, Gzip Compression, Encryption)
- [x] Task: Implement SQLite Vacuum & Snapshot Exporter
    - [x] Create `internal/backup/snapshot.go` with sequential SQLite `VACUUM INTO` execution on temp files for `bob.db`, `townhall.db`, and active `dm_*.db` databases
    - [x] Write unit tests verifying snapshot creation and cleanup in `internal/backup/snapshot_test.go`
- [x] Task: Implement Streaming Gzip + AES Encryption Pipeline
    - [x] Implement writer chain: snapshot file -> gzip compression -> AES-256-GCM encryption -> S3 Put upload stream
    - [x] Write unit tests verifying streaming compression and encryption in `internal/backup/pipeline_test.go`

## Phase 3: Manifest Management & Retention Pruner
- [x] Task: Implement Backup Manifest Manager
    - [x] Create `internal/backup/manifest.go` with JSON serialization/deserialization for database backup entries (`bob_agent/manifest.json`)
    - [x] Write unit tests in `internal/backup/manifest_test.go`
- [x] Task: Implement Retention Policy & Pruning Logic
    - [x] Create `internal/backup/prune.go` to prune objects older than `S3_BACKUP_KEEP` per database
    - [x] Write unit tests in `internal/backup/prune_test.go`
- [x] Task: Implement Periodic Backup Scheduler
    - [x] Create `internal/backup/scheduler.go` with hourly ticker, sequential execution, manifest updates, and pruning
    - [x] Write unit tests in `internal/backup/scheduler_test.go`

## Phase 4: Startup Recovery & CLI Integration
- [x] Task: Implement Database Recovery on Missing Data
    - [x] Create `internal/backup/recover.go` to fetch manifest, download snapshots, decrypt, decompress, and atomically place `.db` files into `DATA_DIR`
    - [x] Write unit tests for recovery success, corrupt artifacts, and wrong secrets in `internal/backup/recover_test.go`
- [x] Task: Integrate CLI Flags & Graceful Shutdown in `cmd/agent/main.go`
    - [x] Add `--init-db` flag to bypass recovery and initialize fresh empty databases
    - [x] Add `--backup` flag to trigger immediate backup and exit
    - [x] Wire automatic recovery before database initialization in `main.go`
    - [x] Wire background scheduler and final backup during graceful shutdown (SIGTERM/SIGINT)

## Phase 5: MinIO Integration Tests & CI Pipeline
- [x] Task: Implement MinIO Integration Tests
    - [x] Create `internal/backup/integration_test.go` (build tag `integration`) testing end-to-end backup, pruning, wrong secret rejection, and recovery against MinIO
- [x] Task: Update CI Workflow and Local Test Tooling
    - [x] Update `.github/workflows/pipeline.yml` to include MinIO service container for integration tests
    - [x] Verify full test suite passes locally with `make check`
