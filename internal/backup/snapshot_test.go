package backup

import (
	"context"
	"database/sql"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	_ "modernc.org/sqlite"
)

func TestDiscoverDBTargets(t *testing.T) {
	tempDir := t.TempDir()

	// Create test database files on disk
	f1, err := os.Create(filepath.Join(tempDir, "bob.db"))
	require.NoError(t, err)
	_ = f1.Close()

	f2, err := os.Create(filepath.Join(tempDir, "dm_user123.db"))
	require.NoError(t, err)
	_ = f2.Close()

	// Also create a non-db file to test filtering
	f3, err := os.Create(filepath.Join(tempDir, "ignore.txt"))
	require.NoError(t, err)
	_ = f3.Close()

	targets, err := DiscoverDBTargets(tempDir, nil)
	require.NoError(t, err)
	require.Len(t, targets, 2)
	assert.Equal(t, "bob.db", targets[0].Name)
	assert.Equal(t, "dm_user123.db", targets[1].Name)
}

func TestSnapshotDatabase(t *testing.T) {
	tempDir := t.TempDir()
	dbPath := filepath.Join(tempDir, "test.db")

	db, err := sql.Open("sqlite", dbPath)
	require.NoError(t, err)
	defer func() { _ = db.Close() }()

	_, err = db.Exec("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT);")
	require.NoError(t, err)
	_, err = db.Exec("INSERT INTO users (name) VALUES ('Alice'), ('Bob');")
	require.NoError(t, err)

	destDir := filepath.Join(tempDir, "snapshots")
	target := DBTarget{Name: "test.db", Path: dbPath, DB: db}

	snapFile, err := SnapshotDatabase(context.Background(), target, destDir)
	require.NoError(t, err)
	defer func() { _ = os.Remove(snapFile) }()

	// Verify snapshot file exists and can be opened
	snapDB, err := sql.Open("sqlite", snapFile)
	require.NoError(t, err)
	defer func() { _ = snapDB.Close() }()

	var count int
	err = snapDB.QueryRow("SELECT COUNT(*) FROM users;").Scan(&count)
	require.NoError(t, err)
	assert.Equal(t, 2, count)
}
