package backup

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"

	"bob/internal/config"
	"bob/internal/objectstore"
)

// HasExistingDatabases checks if any SQLite databases (.db) exist in dataDir.
func HasExistingDatabases(dataDir string) (bool, error) {
	entries, err := os.ReadDir(dataDir)
	if err != nil {
		if os.IsNotExist(err) {
			return false, nil
		}
		return false, fmt.Errorf("failed to read data directory: %w", err)
	}

	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		if strings.HasSuffix(entry.Name(), ".db") {
			return true, nil
		}
	}
	return false, nil
}

// RecoverDBsIfMissing restores all databases recorded in the S3 manifest into dataDir
// if no local databases exist. If initDB is true, recovery is bypassed.
func RecoverDBsIfMissing(ctx context.Context, cfg *config.Config, obj *objectstore.Client, initDB bool) (bool, error) {
	if initDB {
		slog.Info("database initialization flag set (--init-db); skipping S3 recovery")
		return false, nil
	}
	if obj == nil {
		return false, nil
	}

	exists, err := HasExistingDatabases(cfg.DataDir)
	if err != nil {
		return false, err
	}
	if exists {
		// Never overwrite existing local databases
		return false, nil
	}

	prefix := cfg.S3BackupPrefix
	if prefix == "" {
		prefix = "bob_agent/"
	}

	manifest, err := FetchManifest(ctx, obj, prefix)
	if err != nil {
		return false, fmt.Errorf("failed to fetch manifest for recovery: %w", err)
	}
	if len(manifest.Databases) == 0 {
		slog.Info("no existing backup manifest found in object storage; starting fresh")
		return false, nil
	}

	if err := os.MkdirAll(cfg.DataDir, 0755); err != nil {
		return false, fmt.Errorf("failed to create data directory %s: %w", cfg.DataDir, err)
	}

	for dbName, entry := range manifest.Databases {
		rc, err := obj.Get(ctx, entry.Key)
		if err != nil {
			return false, fmt.Errorf("failed to download backup for %s (%s): %w", dbName, entry.Key, err)
		}

		destPath := filepath.Join(cfg.DataDir, dbName)
		err = DecodeSnapshot(rc, cfg.Secret, destPath)
		_ = rc.Close()
		if err != nil {
			return false, fmt.Errorf("failed to decode/restore database %s: %w", dbName, err)
		}
		slog.Info("recovered database from object storage", "db", dbName, "key", entry.Key, "dest", destPath)
	}

	return true, nil
}
