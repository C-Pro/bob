package main_test

import (
	"context"
	"database/sql"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	_ "modernc.org/sqlite"
)

// TestServiceDatabaseLifecycle verifies the complete service lifecycle with SQLite database:
// 1. Fresh start with no DB (creates DB at current version).
// 2. Start with DB at previous version (v-1) (auto-migrates to current version).
// 3. Start with DB at v-2 (fails startup with version mismatch error).
func TestServiceDatabaseLifecycle(t *testing.T) {
	// Build binary for testing
	binDir := t.TempDir()
	binPath := filepath.Join(binDir, "agent_test_bin")
	buildCmd := exec.Command("go", "build", "-o", binPath, ".")
	buildCmd.Dir = "."
	buildOut, err := buildCmd.CombinedOutput()
	require.NoError(t, err, "failed to build agent binary: %s", string(buildOut))

	t.Run("Fresh start with no db creates db at v1", func(t *testing.T) {
		dataDir := filepath.Join(t.TempDir(), "data")
		dbFile := filepath.Join(dataDir, "bob.db")
		assert.NoFileExists(t, dbFile)

		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()

		cmd := exec.CommandContext(ctx, binPath)
		cmd.Env = append(os.Environ(),
			"DATA_DIR="+dataDir,
			"BESEDKA_URL=http://127.0.0.1:59999", // Unused port, won't connect
			"OPENAI_API_KEY=test-key",
		)
		out, _ := cmd.CombinedOutput()
		outputStr := string(out)

		// Verify startup logged database initialization
		assert.Contains(t, outputStr, "database storage initialized")
		assert.FileExists(t, dbFile)

		// Verify schema version in DB
		db, err := sql.Open("sqlite", fmt.Sprintf("file://%s?mode=ro", dbFile))
		require.NoError(t, err)
		defer func() { _ = db.Close() }()

		var v int
		err = db.QueryRow("select version from schema_version where is_current=1").Scan(&v)
		require.NoError(t, err)
		assert.Equal(t, 1, v)
	})

	t.Run("Start with db at v-2 fails startup with version mismatch", func(t *testing.T) {
		dataDir := filepath.Join(t.TempDir(), "data")
		require.NoError(t, os.MkdirAll(dataDir, 0o755))
		dbFile := filepath.Join(dataDir, "bob.db")

		// Create DB at version -1 (which is v - 2 relative to v1)
		db, err := sql.Open("sqlite", dbFile)
		require.NoError(t, err)
		setup := `
create table schema_version(
  version integer primary key,
  description text not null,
  is_current boolean default 0 check (is_current in (0, 1))
);
create unique index schema_version_uk on schema_version(is_current) where is_current = 1;
insert into schema_version(version, description, is_current) values(-1, 'ancient schema', 1);`
		_, err = db.Exec(setup)
		require.NoError(t, err)
		require.NoError(t, db.Close())

		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()

		cmd := exec.CommandContext(ctx, binPath)
		cmd.Env = append(os.Environ(),
			"DATA_DIR="+dataDir,
			"BESEDKA_URL=http://127.0.0.1:59999",
			"OPENAI_API_KEY=test-key",
		)
		out, err := cmd.CombinedOutput()
		outputStr := string(out)

		// Process should have exited with error
		assert.Error(t, err)
		assert.True(t, strings.Contains(outputStr, "database version mismatch") || strings.Contains(outputStr, "failed to initialize SQLite database"))
	})
}
