package backup

import (
	"context"
	"database/sql"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	_ "modernc.org/sqlite"
)

// DBTarget represents a database target to snapshot.
type DBTarget struct {
	Name string // e.g. "bob.db", "townhall.db", "dm_user1.db"
	Path string // full file path
	DB   *sql.DB
}

// isSQLiteDB checks if a file begins with the standard SQLite header or is an active db.
func isSQLiteDB(path string) bool {
	f, err := os.Open(path)
	if err != nil {
		return false
	}
	defer func() { _ = f.Close() }()

	var header [16]byte
	n, err := f.Read(header[:])
	if err != nil || n < 16 {
		return false
	}
	return string(header[:]) == "SQLite format 3\x00"
}

// DiscoverDBTargets lists all known and existing SQLite databases in dataDir.
func DiscoverDBTargets(dataDir string, activeDBs map[string]*sql.DB) ([]DBTarget, error) {
	if activeDBs == nil {
		activeDBs = make(map[string]*sql.DB)
	}

	targetsMap := make(map[string]DBTarget)

	// Always check for standard databases
	stdDBs := []string{"bob.db", "townhall.db"}
	for _, name := range stdDBs {
		p := filepath.Join(dataDir, name)
		if db, ok := activeDBs[name]; ok {
			targetsMap[name] = DBTarget{Name: name, Path: p, DB: db}
		} else if _, err := os.Stat(p); err == nil {
			targetsMap[name] = DBTarget{Name: name, Path: p}
		}
	}

	// Scan dataDir for all other SQLite *.db files (like dm_*.db)
	entries, err := os.ReadDir(dataDir)
	if err == nil {
		for _, entry := range entries {
			if entry.IsDir() {
				continue
			}
			name := entry.Name()
			if name == "besedka.db" {
				continue
			}
			if strings.HasSuffix(name, ".db") {
				p := filepath.Join(dataDir, name)
				if _, ok := activeDBs[name]; !ok && !isSQLiteDB(p) {
					continue
				}
				if _, exists := targetsMap[name]; !exists {
					target := DBTarget{Name: name, Path: p}
					if db, ok := activeDBs[name]; ok {
						target.DB = db
					}
					targetsMap[name] = target
				}
			}
		}
	}

	// Also add any active DBs not yet on disk
	for name, db := range activeDBs {
		if _, exists := targetsMap[name]; !exists {
			targetsMap[name] = DBTarget{
				Name: name,
				Path: filepath.Join(dataDir, name),
				DB:   db,
			}
		}
	}

	targets := make([]DBTarget, 0, len(targetsMap))
	for _, t := range targetsMap {
		targets = append(targets, t)
	}
	sort.Slice(targets, func(i, j int) bool {
		return targets[i].Name < targets[j].Name
	})

	return targets, nil
}

// SnapshotDatabase takes a consistent snapshot of the target database into destDir using VACUUM INTO.
// Returns the absolute path of the created snapshot file.
func SnapshotDatabase(ctx context.Context, target DBTarget, destDir string) (string, error) {
	if err := os.MkdirAll(destDir, 0755); err != nil {
		return "", fmt.Errorf("failed to create snapshot dest dir: %w", err)
	}

	tmpFile := filepath.Join(destDir, fmt.Sprintf("%s.%d.tmp", target.Name, time.Now().UnixNano()))

	db := target.DB
	shouldClose := false
	if db == nil {
		// Open connection to database on disk
		dsn := fmt.Sprintf("file:%s?mode=ro", target.Path)
		opened, err := sql.Open("sqlite", dsn)
		if err != nil {
			return "", fmt.Errorf("failed to open database %s: %w", target.Path, err)
		}
		db = opened
		shouldClose = true
	}
	if shouldClose {
		defer func() { _ = db.Close() }()
	}

	// Execute VACUUM INTO
	escapedPath := strings.ReplaceAll(tmpFile, "'", "''")
	query := fmt.Sprintf("VACUUM INTO '%s';", escapedPath) // nosemgrep: go.lang.security.audit.database.string-formatted-query.string-formatted-query
	if _, err := db.ExecContext(ctx, query); err != nil {
		_ = os.Remove(tmpFile)
		return "", fmt.Errorf("VACUUM INTO failed for %s: %w", target.Name, err)
	}

	return tmpFile, nil
}
