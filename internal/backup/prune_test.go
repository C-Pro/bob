package backup

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"bob/internal/objectstore"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestPruneDatabaseBackups(t *testing.T) {
	var mu sync.Mutex
	objects := make(map[string]bool)

	// Pre-populate 5 backups
	for i := 1; i <= 5; i++ {
		key := fmt.Sprintf("bob_agent/bob.db/20260831T100%d00Z.bk", i)
		objects[key] = true
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()

		switch r.Method {
		case "GET":
			w.Header().Set("Content-Type", "application/xml")
			_, _ = fmt.Fprint(w, `<ListBucketResult><IsTruncated>false</IsTruncated>`)
			for key := range objects {
				_, _ = fmt.Fprintf(w, `<Contents><Key>%s</Key><Size>100</Size></Contents>`, key)
			}
			_, _ = fmt.Fprint(w, `</ListBucketResult>`)
		case "DELETE":
			// URL path is /bucket/key
			path := r.URL.Path
			key := path[len("/bob-bucket/"):]
			delete(objects, key)
			w.WriteHeader(http.StatusOK)
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
	// Keep 3 backups out of 5
	err = PruneDatabaseBackups(ctx, client, "bob_agent/", "bob.db", 3)
	require.NoError(t, err)

	mu.Lock()
	defer mu.Unlock()
	assert.Len(t, objects, 3)
	// Older backups (1 and 2) should be deleted
	assert.False(t, objects["bob_agent/bob.db/20260831T100100Z.bk"])
	assert.False(t, objects["bob_agent/bob.db/20260831T100200Z.bk"])
	// Newer backups (3, 4, 5) should remain
	assert.True(t, objects["bob_agent/bob.db/20260831T100300Z.bk"])
	assert.True(t, objects["bob_agent/bob.db/20260831T100400Z.bk"])
	assert.True(t, objects["bob_agent/bob.db/20260831T100500Z.bk"])
}
