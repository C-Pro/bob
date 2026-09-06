package backup

import (
	"context"
	"fmt"
	"log/slog"
	"sort"
	"strings"

	"bob/internal/objectstore"
)

// PruneDatabaseBackups removes backups older than keep for the specified database.
func PruneDatabaseBackups(ctx context.Context, obj *objectstore.Client, prefix string, dbName string, keep int) error {
	if obj == nil {
		return fmt.Errorf("objectstore client is nil")
	}
	if keep < 1 {
		keep = 1
	}

	dbPrefix := prefix + dbName + "/"
	objs, err := obj.List(ctx, dbPrefix)
	if err != nil {
		return fmt.Errorf("failed to list database backups for %s: %w", dbName, err)
	}

	var backupKeys []string
	for _, o := range objs {
		if strings.HasSuffix(o.Key, ".bk") {
			backupKeys = append(backupKeys, o.Key)
		}
	}
	// Sort chronologically (timestamped keys format 20060102T150405Z.bk sorts alphabetically)
	sort.Strings(backupKeys)

	if len(backupKeys) <= keep {
		return nil
	}

	numToDelete := len(backupKeys) - keep
	for i := 0; i < numToDelete; i++ {
		key := backupKeys[i]
		if err := obj.Delete(ctx, key); err != nil {
			return fmt.Errorf("failed to delete old backup %s: %w", key, err)
		}
		slog.Info("pruned old backup", "db", dbName, "key", key)
	}

	return nil
}
