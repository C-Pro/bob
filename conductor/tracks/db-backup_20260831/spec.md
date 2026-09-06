# Specification: S3 Encrypted Database Backup & Restore

## Overview
Implement scheduled snapshot database backups and automatic startup recovery for Bob, mirroring Besedka's object storage backup architecture. Backups are encrypted with AES-256-GCM using keys derived via Argon2id from a `SECRET` environment variable, compressed with gzip, and uploaded to S3-compatible object storage under the `bob_agent/` prefix. A central JSON manifest tracks snapshot metadata per database, and old snapshots are pruned according to a retention policy. On startup, missing databases are automatically restored from S3 unless bypassed by `--init-db`. Local and CI integration testing is supported using MinIO.

## Functional Requirements

### 1. Configuration & Credentials
- **SECRET / AUTH_SECRET:** Support `SECRET` (with fallback to `AUTH_SECRET`) for deriving the AES-256-GCM encryption key using Argon2id with random salt.
- **S3 Configuration:** Support standard S3 environment variables:
  - `S3_ENDPOINT`: Endpoint URL (e.g. `http://localhost:9000` or AWS/Cloudflare R2/GCS endpoint).
  - `S3_REGION`: S3 region (default: `us-east-1`).
  - `S3_BUCKET`: Target bucket name.
  - `S3_ACCESS_KEY` & `S3_SECRET_KEY`: Credentials for S3 API operations.
  - `S3_PATH_STYLE`: Boolean flag (default `true`) for path-style vs virtual-host addressing.
  - `S3_BACKUP_INTERVAL`: Duration between full snapshot runs (default: `1h`).
  - `S3_BACKUP_KEEP`: Retention count for snapshots per database (default: `7`).
  - `S3_BACKUP_PREFIX`: Storage prefix inside bucket (default: `bob_agent/`).
- **Validation:** Object storage is disabled if `S3_BUCKET` or `S3_ENDPOINT` is empty. If enabled, `SECRET`, `S3_ACCESS_KEY`, and `S3_SECRET_KEY` are strictly validated.

### 2. S3 Object Store Client
- Dependency-free AWS SigV4 REST client (in `internal/objectstore`), supporting `Put`, `Get`, `List`, and `Delete`.

### 3. Snapshot Creation & Pipeline
- **Sequential Snapshots:** Iterate one-by-one through known databases (`bob.db`, `townhall.db`, and active `dm_*.db` databases) to conserve memory and CPU on weak instances.
- **SQLite VACUUM INTO:** Execute `VACUUM INTO 'tmp_file'` on each database connection to produce a consistent SQLite snapshot file in a temporary directory.
- **Writer Chain (Gzip + AES-256-GCM):**
  - Generate a unique 16-byte cryptographic salt per artifact.
  - Stream the SQLite snapshot through gzip compression and encrypt with AES-256-GCM.
  - Write standard self-describing header (`BOBB` magic bytes prefix, version 1, salt length, salt bytes) followed by ciphertext.
- **Object Key Naming:** Upload artifacts to `bob_agent/<dbname>/<timestamp>.bk` (e.g., `bob_agent/bob.db/20260831T103500Z.bk`).
- **Clean Up Temporary Files:** Delete temporary SQLite snapshot files immediately after streaming to S3.

### 4. Manifest & Retention Management
- **Manifest File:** Maintain `bob_agent/manifest.json` in the bucket.
  - JSON schema records: `version`, `updated_at`, and a map of database names (`bob.db`, `townhall.db`, etc.) to their latest backups with object key, timestamp, and byte size.
  - Atomically update `manifest.json` after successful database uploads.
- **Retention Pruning:** For each database, list existing `.bk` objects under `bob_agent/<dbname>/`, retain the newest `S3_BACKUP_KEEP` snapshots, and delete older objects.

### 5. Recovery on Startup & CLI Flags
- **Automatic Recovery:**
  - On startup before opening databases, check if databases exist in `DATA_DIR`.
  - If any database is missing and S3 is enabled:
    - Fetch `bob_agent/manifest.json`.
    - If manifest exists, download each referenced database backup, decrypt, decompress, and atomically place in `DATA_DIR`.
    - If no backups exist in bucket (fresh install), proceed with fresh initialization.
  - Decryption error or corrupt payload fails startup with a fatal error.
- **`--init-db` Flag:** When passed, bypass S3 recovery and create fresh empty databases.
- **`--backup` CLI Flag:** Trigger an immediate out-of-schedule full backup and exit.
- **Graceful Shutdown Backup:** Trigger a final full backup during graceful SIGTERM/SIGINT shutdown before process termination.

### 6. Testing & CI Pipeline
- **Unit Tests:** Encryption/decryption roundtrips, header serialization (`BOBB`), manifest parsing, retention pruning logic, and mock objectstore tests.
- **Integration Tests:** End-to-end backup, corruption handling, wrong secret rejection, retention pruning, and recovery tests against MinIO (build tag `integration`).
- **CI Workflow:** Update `.github/workflows/pipeline.yml` to run MinIO service and integration tests.

## Non-Functional Requirements
- **Memory Efficiency:** Sequential snapshotting and streaming to avoid memory spikes on low-resource VMs.
- **Zero Data Loss & Safe Failures:** Decryption errors fail loudly instead of overwriting databases.
- **Code Standards:** 100% compliance with `make check` (golangci-lint, go test -race, semgrep, osv-scanner).

## Out of Scope
- Incremental / delta WAL streaming backups (deferred to a future track).
- Multi-region cross-bucket replication.
