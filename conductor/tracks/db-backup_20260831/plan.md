# Implementation Plan: S3 Encrypted Database Backup & Restore

## Phase 1: Configuration, Crypto & S3 Object Store Client
- [ ] Task: Implement S3 & Backup Configuration & Validation
    - [ ] Add `SECRET` (with `AUTH_SECRET` fallback), `S3_ENDPOINT`, `S3_REGION`, `S3_BUCKET`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`, `S3_PATH_STYLE`, `S3_BACKUP_INTERVAL`, `S3_BACKUP_KEEP`, `S3_BACKUP_PREFIX` to `internal/config/config.go`
    - [ ] Add config validation rules for S3 settings and secrets
    - [ ] Write unit tests in `internal/config/config_test.go`
- [ ] Task: Implement Cryptographic Engine & Header Codec (`BOBB` magic)
    - [ ] Create `internal/backup/crypto.go` implementing Argon2id key derivation and AES-256-GCM encryption/decryption
    - [ ] Create `internal/backup/header.go` for writing and parsing `BOBB` magic byte headers with version and salt
    - [ ] Write unit tests for crypto and header codec in `internal/backup/crypto_test.go` and `internal/backup/header_test.go`
- [ ] Task: Implement Minimal S3 Object Storage Client
    - [ ] Create `internal/objectstore` package with AWS SigV4 authorization and REST methods (`Put`, `Get`, `List`, `Delete`)
    - [ ] Write unit tests with HTTP mock servers in `internal/objectstore/client_test.go` and `internal/objectstore/sigv4_test.go`

## Phase 2: Snapshot Pipeline (VACUUM INTO, Gzip Compression, Encryption)
- [ ] Task: Implement SQLite Vacuum & Snapshot Exporter
    - [ ] Create `internal/backup/snapshot.go` with sequential SQLite `VACUUM INTO` execution on temp files for `bob.db`, `townhall.db`, and active `dm_*.db` databases
    - [ ] Write unit tests verifying snapshot creation and cleanup in `internal/backup/snapshot_test.go`
- [ ] Task: Implement Streaming Gzip + AES Encryption Pipeline
    - [ ] Implement writer chain: snapshot file -> gzip compression -> AES-256-GCM encryption -> S3 Put upload stream
    - [ ] Write unit tests verifying streaming compression and encryption in `internal/backup/pipeline_test.go`

## Phase 3: Manifest Management & Retention Pruner
- [ ] Task: Implement Backup Manifest Manager
    - [ ] Create `internal/backup/manifest.go` with JSON serialization/deserialization for database backup entries (`bob_agent/manifest.json`)
    - [ ] Write unit tests in `internal/backup/manifest_test.go`
- [ ] Task: Implement Retention Policy & Pruning Logic
    - [ ] Create `internal/backup/prune.go` to prune objects older than `S3_BACKUP_KEEP` per database
    - [ ] Write unit tests in `internal/backup/prune_test.go`
- [ ] Task: Implement Periodic Backup Scheduler
    - [ ] Create `internal/backup/scheduler.go` with hourly ticker, sequential execution, manifest updates, and pruning
    - [ ] Write unit tests in `internal/backup/scheduler_test.go`

## Phase 4: Startup Recovery & CLI Integration
- [ ] Task: Implement Database Recovery on Missing Data
    - [ ] Create `internal/backup/recover.go` to fetch manifest, download snapshots, decrypt, decompress, and atomically place `.db` files into `DATA_DIR`
    - [ ] Write unit tests for recovery success, corrupt artifacts, and wrong secrets in `internal/backup/recover_test.go`
- [ ] Task: Integrate CLI Flags & Graceful Shutdown in `cmd/agent/main.go`
    - [ ] Add `--init-db` flag to bypass recovery and initialize fresh empty databases
    - [ ] Add `--backup` flag to trigger immediate backup and exit
    - [ ] Wire automatic recovery before database initialization in `main.go`
    - [ ] Wire background scheduler and final backup during graceful shutdown (SIGTERM/SIGINT)

## Phase 5: MinIO Integration Tests & CI Pipeline
- [ ] Task: Implement MinIO Integration Tests
    - [ ] Create `internal/backup/integration_test.go` (build tag `integration`) testing end-to-end backup, pruning, wrong secret rejection, and recovery against MinIO
- [ ] Task: Update CI Workflow and Local Test Tooling
    - [ ] Update `.github/workflows/pipeline.yml` to include MinIO service container for integration tests
    - [ ] Verify full test suite passes locally with `make check`
