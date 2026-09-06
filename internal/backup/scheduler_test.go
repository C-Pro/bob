package backup

import (
	"context"
	"database/sql"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
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

func TestSchedulerDoBackup(t *testing.T) {
	tempDir := t.TempDir()
	dbPath := filepath.Join(tempDir, "bob.db")

	db, err := sql.Open("sqlite", dbPath)
	require.NoError(t, err)
	defer func() { _ = db.Close() }()

	_, err = db.Exec("CREATE TABLE test (id INTEGER PRIMARY KEY);")
	require.NoError(t, err)

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

	client, err := objectstore.New(objectstore.Config{
		Endpoint:  srv.URL,
		Region:    "us-east-1",
		Bucket:    "testbucket",
		AccessKey: "AK",
		SecretKey: "SK",
		PathStyle: true,
	})
	require.NoError(t, err)

	cfg := &config.Config{
		DataDir:          tempDir,
		Secret:           "backup-secret-key",
		S3BackupPrefix:   "bob_agent/",
		S3BackupKeep:     5,
		S3BackupInterval: time.Hour,
	}

	scheduler := NewScheduler(cfg, client, func() map[string]*sql.DB {
		return map[string]*sql.DB{"bob.db": db}
	})

	err = scheduler.DoBackup(context.Background())
	require.NoError(t, err)

	mu.Lock()
	defer mu.Unlock()
	assert.NotEmpty(t, storedObjects["bob_agent/manifest.json"])

	var foundBackup bool
	for k := range storedObjects {
		if filepath.Dir(k) == "bob_agent/bob.db" {
			foundBackup = true
			break
		}
	}
	assert.True(t, foundBackup, "expected backup file in S3 for bob.db")
}
