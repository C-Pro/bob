package backup

import (
	"context"
	"database/sql"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"bob/internal/config"
	"bob/internal/objectstore"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	_ "modernc.org/sqlite"
)

func TestRecoverDBsIfMissing(t *testing.T) {
	tempDir := t.TempDir()
	sourceDir := filepath.Join(tempDir, "source")
	destDir := filepath.Join(tempDir, "dest")

	require.NoError(t, os.MkdirAll(sourceDir, 0755))
	require.NoError(t, os.MkdirAll(destDir, 0755))

	secret := "recovery-secret-test"
	dbPath := filepath.Join(sourceDir, "bob.db")
	db, err := sql.Open("sqlite", dbPath)
	require.NoError(t, err)

	_, err = db.Exec("CREATE TABLE records (id INTEGER PRIMARY KEY, note TEXT);")
	require.NoError(t, err)
	_, err = db.Exec("INSERT INTO records (note) VALUES ('persisted in s3');")
	require.NoError(t, err)
	_ = db.Close()

	artifactBytes, err := EncodeSnapshot(dbPath, secret)
	require.NoError(t, err)

	manifest := NewManifest()
	manifest.Databases["bob.db"] = DBBackupEntry{
		Key:       "bob_agent/bob.db/20260831T100000Z.bk",
		Timestamp: time.Now().UTC(),
		Size:      int64(len(artifactBytes)),
	}

	var mu sync.Mutex
	stored := make(map[string][]byte)
	stored["bob_agent/bob.db/20260831T100000Z.bk"] = artifactBytes

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()

		switch r.Method {
		case "PUT":
			data, _ := io.ReadAll(r.Body)
			path := r.URL.Path
			key := path[len("/testbucket/"):]
			stored[key] = data
			w.WriteHeader(http.StatusOK)
		case "GET":
			path := r.URL.Path
			key := path[len("/testbucket/"):]
			data, ok := stored[key]
			if !ok {
				w.WriteHeader(http.StatusNotFound)
				return
			}
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write(data)
		}
	}))
	defer srv.Close()

	client, err := objectstore.New(objectstore.Config{
		Endpoint:  srv.URL,
		Region:    "us-east-1",
		Bucket:    "testbucket",
		AccessKey: "AK",
		SecretKey: "SK",
		PathStyle: true,
	})
	require.NoError(t, err)

	// Put manifest to server
	require.NoError(t, PutManifest(context.Background(), client, "bob_agent/", manifest))

	cfg := &config.Config{
		DataDir:        destDir,
		Secret:         secret,
		S3BackupPrefix: "bob_agent/",
	}

	// 1. First recovery attempt (destDir is empty)
	recovered, err := RecoverDBsIfMissing(context.Background(), cfg, client, false)
	require.NoError(t, err)
	assert.True(t, recovered)

	// Verify restored db content
	restoredDBPath := filepath.Join(destDir, "bob.db")
	assert.FileExists(t, restoredDBPath)

	rdb, err := sql.Open("sqlite", restoredDBPath)
	require.NoError(t, err)
	defer func() { _ = rdb.Close() }()

	var note string
	err = rdb.QueryRow("SELECT note FROM records WHERE id = 1;").Scan(&note)
	require.NoError(t, err)
	assert.Equal(t, "persisted in s3", note)

	// 2. Second recovery attempt with existing database should be a no-op
	recovered2, err := RecoverDBsIfMissing(context.Background(), cfg, client, false)
	require.NoError(t, err)
	assert.False(t, recovered2)

	// 3. Recovery with initDB=true should skip recovery
	emptyDir := filepath.Join(tempDir, "empty")
	cfgEmpty := &config.Config{
		DataDir:        emptyDir,
		Secret:         secret,
		S3BackupPrefix: "bob_agent/",
	}
	recovered3, err := RecoverDBsIfMissing(context.Background(), cfgEmpty, client, true)
	require.NoError(t, err)
	assert.False(t, recovered3)
	assert.NoFileExists(t, filepath.Join(emptyDir, "bob.db"))
}

func TestRecoverDBsCorruptOrWrongSecret(t *testing.T) {
	tempDir := t.TempDir()
	destDir := filepath.Join(tempDir, "dest")

	secret := "correct-secret"
	manifest := NewManifest()
	manifest.Databases["bob.db"] = DBBackupEntry{
		Key:       "bob_agent/bob.db/20260831T100000Z.bk",
		Timestamp: time.Now().UTC(),
		Size:      100,
	}

	stored := make(map[string][]byte)
	stored["bob_agent/bob.db/20260831T100000Z.bk"] = []byte("corrupt-not-a-valid-backup")

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path := r.URL.Path
		key := path[len("/testbucket/"):]
		switch r.Method {
		case "PUT":
			data, _ := io.ReadAll(r.Body)
			stored[key] = data
			w.WriteHeader(http.StatusOK)
		case "GET":
			data, ok := stored[key]
			if !ok {
				w.WriteHeader(http.StatusNotFound)
				return
			}
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write(data)
		}
	}))
	defer srv.Close()

	client, err := objectstore.New(objectstore.Config{
		Endpoint:  srv.URL,
		Region:    "us-east-1",
		Bucket:    "testbucket",
		AccessKey: "AK",
		SecretKey: "SK",
		PathStyle: true,
	})
	require.NoError(t, err)
	require.NoError(t, PutManifest(context.Background(), client, "bob_agent/", manifest))

	cfg := &config.Config{
		DataDir:        destDir,
		Secret:         secret,
		S3BackupPrefix: "bob_agent/",
	}

	_, err = RecoverDBsIfMissing(context.Background(), cfg, client, false)
	assert.Error(t, err)
}
