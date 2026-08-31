package backup

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"bob/internal/objectstore"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestManifestFetchAndPut(t *testing.T) {
	var storedManifest []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case "PUT":
			buf := make([]byte, r.ContentLength)
			_, _ = r.Body.Read(buf)
			storedManifest = buf
			w.WriteHeader(http.StatusOK)
		case "GET":
			if storedManifest == nil {
				w.WriteHeader(http.StatusNotFound)
				return
			}
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write(storedManifest)
		}
	}))
	defer srv.Close()

	client, err := objectstore.New(objectstore.Config{
		Endpoint:  srv.URL,
		Region:    "us-east-1",
		Bucket:    "bob-bucket",
		AccessKey: "KEY",
		SecretKey: "SECRET",
		PathStyle: true,
	})
	require.NoError(t, err)

	ctx := context.Background()
	prefix := "bob_agent/"

	// Initial fetch on empty store returns empty manifest
	m, err := FetchManifest(ctx, client, prefix)
	require.NoError(t, err)
	assert.Equal(t, ManifestVersion, m.Version)
	assert.Empty(t, m.Databases)

	// Update and Put
	now := time.Date(2026, 8, 31, 10, 0, 0, 0, time.UTC)
	m.UpdatedAt = now
	m.Databases["bob.db"] = DBBackupEntry{
		Key:       "bob_agent/bob.db/20260831T100000Z.bk",
		Timestamp: now,
		Size:      1024,
	}

	err = PutManifest(ctx, client, prefix, m)
	require.NoError(t, err)

	// Fetch again and verify
	fetched, err := FetchManifest(ctx, client, prefix)
	require.NoError(t, err)
	assert.Equal(t, ManifestVersion, fetched.Version)
	assert.Len(t, fetched.Databases, 1)
	assert.Equal(t, "bob_agent/bob.db/20260831T100000Z.bk", fetched.Databases["bob.db"].Key)
	assert.Equal(t, int64(1024), fetched.Databases["bob.db"].Size)
}
