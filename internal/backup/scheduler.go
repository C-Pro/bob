package backup

import (
	"bytes"
	"context"
	"database/sql"
	"fmt"
	"log/slog"
	"os"
	"sync"
	"time"

	"bob/internal/config"
	"bob/internal/objectstore"
)

// Scheduler manages scheduled and on-demand database backups to object storage.
type Scheduler struct {
	cfg       *config.Config
	obj       *objectstore.Client
	activeDBs func() map[string]*sql.DB
	interval  time.Duration
	keep      int
	prefix    string
	now       func() time.Time

	mu sync.Mutex
}

// NewScheduler creates an initialized Scheduler.
func NewScheduler(cfg *config.Config, obj *objectstore.Client, activeDBs func() map[string]*sql.DB) *Scheduler {
	prefix := cfg.S3BackupPrefix
	if prefix == "" {
		prefix = "bob_agent/"
	}
	keep := cfg.S3BackupKeep
	if keep < 1 {
		keep = 7
	}
	interval := cfg.S3BackupInterval
	if interval <= 0 {
		interval = time.Hour
	}
	return &Scheduler{
		cfg:       cfg,
		obj:       obj,
		activeDBs: activeDBs,
		interval:  interval,
		keep:      keep,
		prefix:    prefix,
		now:       time.Now,
	}
}

// Run executes periodic backups on the configured interval until ctx is canceled.
func (s *Scheduler) Run(ctx context.Context) error {
	ticker := time.NewTicker(s.interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			if err := s.DoBackup(ctx); err != nil {
				slog.Error("scheduled database backup failed", "error", err)
			}
		}
	}
}

// DoBackup takes a full snapshot of all databases sequentially, encrypts and uploads them,
// updates the manifest, and prunes old backups.
func (s *Scheduler) DoBackup(ctx context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.obj == nil {
		return fmt.Errorf("object storage is not configured")
	}

	var dbs map[string]*sql.DB
	if s.activeDBs != nil {
		dbs = s.activeDBs()
	}

	targets, err := DiscoverDBTargets(s.cfg.DataDir, dbs)
	if err != nil {
		return fmt.Errorf("failed to discover databases: %w", err)
	}
	if len(targets) == 0 {
		slog.Info("no databases found to backup")
		return nil
	}

	tmpDir, err := os.MkdirTemp("", "bob-backup-*")
	if err != nil {
		return fmt.Errorf("failed to create temporary backup directory: %w", err)
	}
	defer func() { _ = os.RemoveAll(tmpDir) }()

	manifest, err := FetchManifest(ctx, s.obj, s.prefix)
	if err != nil {
		return fmt.Errorf("failed to fetch backup manifest: %w", err)
	}

	for _, target := range targets {
		// 1. Snapshot database using VACUUM INTO
		snapFile, err := SnapshotDatabase(ctx, target, tmpDir)
		if err != nil {
			return fmt.Errorf("failed to snapshot %s: %w", target.Name, err)
		}

		// 2. Compress and encrypt snapshot
		artifactBytes, err := EncodeSnapshot(snapFile, s.cfg.Secret)
		_ = os.Remove(snapFile)
		if err != nil {
			return fmt.Errorf("failed to encode snapshot for %s: %w", target.Name, err)
		}

		// 3. Upload to S3
		ts := s.now().UTC()
		key := fmt.Sprintf("%s%s/%s.bk", s.prefix, target.Name, ts.Format("20060102T150405Z"))
		if err := s.obj.Put(ctx, key, bytes.NewReader(artifactBytes), int64(len(artifactBytes))); err != nil {
			return fmt.Errorf("failed to upload backup for %s: %w", target.Name, err)
		}
		slog.Info("database backup uploaded", "db", target.Name, "key", key, "bytes", len(artifactBytes))

		// 4. Update manifest entry
		manifest.Databases[target.Name] = DBBackupEntry{
			Key:       key,
			Timestamp: ts,
			Size:      int64(len(artifactBytes)),
		}

		// 5. Retention prune for this database
		if err := PruneDatabaseBackups(ctx, s.obj, s.prefix, target.Name, s.keep); err != nil {
			slog.Warn("retention pruning failed", "db", target.Name, "error", err)
		}
	}

	// 6. Upload updated manifest
	manifest.UpdatedAt = s.now().UTC()
	if err := PutManifest(ctx, s.obj, s.prefix, manifest); err != nil {
		return fmt.Errorf("failed to update backup manifest: %w", err)
	}

	slog.Info("database backup process completed successfully", "databases", len(targets))
	return nil
}
