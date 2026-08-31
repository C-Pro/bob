package backup

import (
	"context"
	"database/sql"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
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

	first := true
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			if first {
				first = false
				continue
			}
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

	if err := os.MkdirAll(s.cfg.DataDir, 0755); err != nil {
		return fmt.Errorf("failed to create data directory: %w", err)
	}

	tmpDir, err := os.MkdirTemp(s.cfg.DataDir, "bob-backup-*")
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

		// 2. Compress and encrypt snapshot to an artifact file on disk
		artifactFile := filepath.Join(tmpDir, fmt.Sprintf("%s.artifact.bk", filepath.Base(snapFile)))
		artF, err := os.Create(artifactFile)
		if err != nil {
			_ = os.Remove(snapFile)
			return fmt.Errorf("failed to create artifact file: %w", err)
		}

		err = EncodeSnapshot(snapFile, s.cfg.Secret, artF)
		_ = artF.Close()
		_ = os.Remove(snapFile)
		if err != nil {
			_ = os.Remove(artifactFile)
			return fmt.Errorf("failed to encode snapshot for %s: %w", target.Name, err)
		}

		artStat, err := os.Stat(artifactFile)
		if err != nil {
			_ = os.Remove(artifactFile)
			return fmt.Errorf("failed to stat artifact file: %w", err)
		}

		uploadF, err := os.Open(artifactFile)
		if err != nil {
			_ = os.Remove(artifactFile)
			return fmt.Errorf("failed to open artifact file for upload: %w", err)
		}

		// 3. Upload to S3
		ts := s.now().UTC()
		key := fmt.Sprintf("%s%s/%s.bk", s.prefix, target.Name, ts.Format("20060102T150405Z"))
		err = s.obj.Put(ctx, key, uploadF, artStat.Size())
		_ = uploadF.Close()
		_ = os.Remove(artifactFile)
		if err != nil {
			return fmt.Errorf("failed to upload backup for %s: %w", target.Name, err)
		}
		slog.Info("database backup uploaded", "db", target.Name, "key", key, "bytes", artStat.Size())

		// 4. Update manifest entry
		manifest.Databases[target.Name] = DBBackupEntry{
			Key:       key,
			Timestamp: ts,
			Size:      artStat.Size(),
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
