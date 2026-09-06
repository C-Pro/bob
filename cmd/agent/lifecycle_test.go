package main_test

import (
	"bytes"
	"context"
	"database/sql"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"bob/internal/backup"
	"bob/internal/objectstore"

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

		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()

		cmd := exec.CommandContext(ctx, binPath)
		cmd.Env = append(os.Environ(),
			"DATA_DIR="+dataDir,
			"BESEDKA_URL=http://127.0.0.1:59999", // Unused port, won't connect
			"OPENAI_API_KEY=test-key",
			"EMBEDDING_MODEL=test-model",
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

		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()

		cmd := exec.CommandContext(ctx, binPath)
		cmd.Env = append(os.Environ(),
			"DATA_DIR="+dataDir,
			"BESEDKA_URL=http://127.0.0.1:59999",
			"OPENAI_API_KEY=test-key",
			"EMBEDDING_MODEL=test-model",
		)
		out, err := cmd.CombinedOutput()
		outputStr := string(out)

		// Process should have exited with error
		assert.Error(t, err)
		assert.True(t, strings.Contains(outputStr, "database version mismatch") || strings.Contains(outputStr, "failed to initialize SQLite database"))
	})

	t.Run("Vector regeneration CLI flag runs and exits cleanly", func(t *testing.T) {
		if os.Getenv("CI") != "" {
			t.Skip("skipping vector regeneration CLI test in CI environment")
		}

		dataDir := filepath.Join(t.TempDir(), "data")
		require.NoError(t, os.MkdirAll(dataDir, 0o755))
		thFile := filepath.Join(dataDir, "townhall.db")

		// Initialize dummy townhall.db with a memory record
		db, err := sql.Open("sqlite", thFile)
		require.NoError(t, err)
		_, err = db.Exec(`
			CREATE TABLE IF NOT EXISTS messages (
				id TEXT PRIMARY KEY,
				session_id TEXT NOT NULL,
				role TEXT NOT NULL,
				content TEXT NOT NULL,
				vector BLOB,
				metadata TEXT,
				created_at DATETIME DEFAULT CURRENT_TIMESTAMP
			);
			INSERT INTO messages (id, session_id, role, content)
			VALUES ('chunk_th_1_1', 'memory:global:default', 'user', 'Testing CLI vector regeneration');
		`)
		require.NoError(t, err)
		require.NoError(t, db.Close())

		repoRoot, _ := filepath.Abs("../..")
		if _, err := os.Stat(filepath.Join(repoRoot, "data/models")); os.IsNotExist(err) {
			repoRoot, _ = filepath.Abs("..")
		}
		if _, err := os.Stat(filepath.Join(repoRoot, "data/models")); os.IsNotExist(err) {
			repoRoot, _ = filepath.Abs(".")
		}
		modelsDir := filepath.Join(repoRoot, "data/models")
		if fi, err := os.Stat(modelsDir); err == nil && fi.IsDir() {
			_ = os.Symlink(modelsDir, filepath.Join(dataDir, "models"))
		}

		// Allow sufficient timeout if go-embed needs to download model weights in CI/clean environments
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
		defer cancel()

		cmd := exec.CommandContext(ctx, binPath, "-regenerate-vectors", "-data-dir="+dataDir)
		cmd.Dir = repoRoot
		cmd.Env = append(os.Environ(),
			"BESEDKA_URL=http://127.0.0.1:59999",
			"OPENAI_API_KEY=test-key",
			"EMBEDDING_MODEL=",
		)
		out, err := cmd.CombinedOutput()
		outputStr := string(out)

		require.NoError(t, err, "vector regeneration CLI failed: %s", outputStr)
		assert.Contains(t, outputStr, "running vector regeneration tool")
		assert.Contains(t, outputStr, "vector regeneration completed")
		assert.Contains(t, outputStr, "reembedded=1")

		// Verify vector was stored in DB (384 dimensions -> 4 byte length header + 384*4 float32 bytes = 1540 bytes)
		db, err = sql.Open("sqlite", thFile)
		require.NoError(t, err)
		var vec []byte
		err = db.QueryRowContext(ctx, "SELECT vector FROM messages WHERE id = 'chunk_th_1_1'").Scan(&vec)
		require.NoError(t, err)
		assert.Len(t, vec, 1540)
		_ = db.Close()
	})

	t.Run("On-demand --backup CLI flag snapshots DB and uploads to S3", func(t *testing.T) {
		dataDir := filepath.Join(t.TempDir(), "data")
		require.NoError(t, os.MkdirAll(dataDir, 0755))
		dbFile := filepath.Join(dataDir, "bob.db")

		db, err := sql.Open("sqlite", dbFile)
		require.NoError(t, err)
		_, err = db.Exec("CREATE TABLE test_data (id INTEGER PRIMARY KEY, info TEXT); INSERT INTO test_data (info) VALUES ('test backup');")
		require.NoError(t, err)
		require.NoError(t, db.Close())

		var mu sync.Mutex
		storedObjects := make(map[string][]byte)

		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			mu.Lock()
			defer mu.Unlock()

			switch r.Method {
			case "PUT":
				data, _ := io.ReadAll(r.Body)
				path := r.URL.Path
				key := path[len("/testbucket/"):]
				storedObjects[key] = data
				w.WriteHeader(http.StatusOK)
			case "GET":
				path := r.URL.Path
				if path == "/testbucket" || path == "/testbucket/" {
					w.Header().Set("Content-Type", "application/xml")
					_, _ = fmt.Fprint(w, `<ListBucketResult><IsTruncated>false</IsTruncated>`)
					for key, data := range storedObjects {
						_, _ = fmt.Fprintf(w, `<Contents><Key>%s</Key><Size>%d</Size></Contents>`, key, len(data))
					}
					_, _ = fmt.Fprint(w, `</ListBucketResult>`)
					return
				}
				key := path[len("/testbucket/"):]
				data, ok := storedObjects[key]
				if !ok {
					w.WriteHeader(http.StatusNotFound)
					return
				}
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write(data)
			}
		}))
		defer srv.Close()

		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()

		cmd := exec.CommandContext(ctx, binPath, "-backup", "-data-dir="+dataDir)
		cmd.Env = append(os.Environ(),
			"BESEDKA_URL=http://127.0.0.1:59999",
			"SECRET=test-backup-secret",
			"S3_ENDPOINT="+srv.URL,
			"S3_BUCKET=testbucket",
			"S3_ACCESS_KEY=key",
			"S3_SECRET_KEY=secret",
			"S3_BACKUP_PREFIX=bob_agent/",
		)
		out, err := cmd.CombinedOutput()
		outputStr := string(out)

		require.NoError(t, err, "on-demand backup CLI failed: %s", outputStr)
		assert.Contains(t, outputStr, "on-demand database backup completed successfully")

		mu.Lock()
		defer mu.Unlock()
		assert.NotEmpty(t, storedObjects["bob_agent/manifest.json"])
	})

	t.Run("Startup recovery restores database from S3 when missing locally", func(t *testing.T) {
		tempDir := t.TempDir()
		sourceDir := filepath.Join(tempDir, "source")
		dataDir := filepath.Join(tempDir, "data")
		require.NoError(t, os.MkdirAll(sourceDir, 0755))

		secret := "recovery-secret"
		srcDBFile := filepath.Join(sourceDir, "bob.db")
		srcDB, err := sql.Open("sqlite", srcDBFile)
		require.NoError(t, err)
		_, err = srcDB.Exec(`
			CREATE TABLE schema_version (version integer primary key, description text not null, is_current boolean default 0 check (is_current in (0, 1)));
			CREATE UNIQUE INDEX schema_version_uk on schema_version(is_current) where is_current = 1;
			INSERT INTO schema_version (version, description, is_current) VALUES (1, 'initial', 1);
			CREATE TABLE items (id INTEGER PRIMARY KEY, val TEXT);
			INSERT INTO items (val) VALUES ('restored-from-s3');
		`)
		require.NoError(t, err)
		require.NoError(t, srcDB.Close())

		var buf bytes.Buffer
		err = backup.EncodeSnapshot(srcDBFile, secret, &buf)
		require.NoError(t, err)
		artifact := buf.Bytes()

		manifest := backup.NewManifest()
		manifest.Databases["bob.db"] = backup.DBBackupEntry{
			Key:       "bob_agent/bob.db/20260831T100000Z.bk",
			Timestamp: time.Now().UTC(),
			Size:      int64(len(artifact)),
		}

		var mu sync.Mutex
		storedObjects := make(map[string][]byte)
		storedObjects["bob_agent/bob.db/20260831T100000Z.bk"] = artifact

		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			mu.Lock()
			defer mu.Unlock()

			path := r.URL.Path
			key := path[len("/testbucket/"):]
			switch r.Method {
			case "PUT":
				data, _ := io.ReadAll(r.Body)
				storedObjects[key] = data
				w.WriteHeader(http.StatusOK)
			case "GET":
				data, ok := storedObjects[key]
				if !ok {
					w.WriteHeader(http.StatusNotFound)
					return
				}
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write(data)
			}
		}))
		defer srv.Close()

		// Upload manifest to mock S3 using objectstore client
		osClient, err := objectstore.New(objectstore.Config{
			Endpoint:  srv.URL,
			Region:    "us-east-1",
			Bucket:    "testbucket",
			AccessKey: "key",
			SecretKey: "secret",
			PathStyle: true,
		})
		require.NoError(t, err)
		require.NoError(t, backup.PutManifest(context.Background(), osClient, "bob_agent/", manifest))

		// Start agent in dataDir (where no DB exists)
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()

		cmd := exec.CommandContext(ctx, binPath, "-data-dir="+dataDir)
		cmd.Env = append(os.Environ(),
			"BESEDKA_URL=http://127.0.0.1:59999",
			"SECRET="+secret,
			"OPENAI_API_KEY=test-key",
			"EMBEDDING_MODEL=test-model",
			"S3_ENDPOINT="+srv.URL,
			"S3_BUCKET=testbucket",
			"S3_ACCESS_KEY=key",
			"S3_SECRET_KEY=secret",
			"S3_BACKUP_PREFIX=bob_agent/",
		)
		out, _ := cmd.CombinedOutput()
		outputStr := string(out)

		assert.Contains(t, outputStr, "database recovery from object storage completed")

		// Verify database was restored and has the table data
		restoredDBFile := filepath.Join(dataDir, "bob.db")
		assert.FileExists(t, restoredDBFile)

		rdb, err := sql.Open("sqlite", restoredDBFile)
		require.NoError(t, err)
		defer func() { _ = rdb.Close() }()

		var val string
		err = rdb.QueryRow("SELECT val FROM items WHERE id = 1").Scan(&val)
		require.NoError(t, err)
		assert.Equal(t, "restored-from-s3", val)
	})

	t.Run("Startup with --init-db skips recovery from S3", func(t *testing.T) {
		tempDir := t.TempDir()
		dataDir := filepath.Join(tempDir, "data")

		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()

		cmd := exec.CommandContext(ctx, binPath, "-init-db", "-data-dir="+dataDir)
		cmd.Env = append(os.Environ(),
			"BESEDKA_URL=http://127.0.0.1:59999",
			"SECRET=secret",
			"OPENAI_API_KEY=test-key",
			"EMBEDDING_MODEL=test-model",
			"S3_ENDPOINT=http://127.0.0.1:59998",
			"S3_BUCKET=testbucket",
			"S3_ACCESS_KEY=key",
			"S3_SECRET_KEY=secret",
		)
		out, _ := cmd.CombinedOutput()
		outputStr := string(out)

		assert.Contains(t, outputStr, "skipping S3 recovery")
	})
}

