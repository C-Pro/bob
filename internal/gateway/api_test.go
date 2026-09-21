package gateway

import (
	"context"
	"encoding/json"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"bob/internal/config"
	"bob/internal/models"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestUserCache(t *testing.T) {
	cache := NewUserCache()

	// Initial empty check - returns empty fallback for unknown user, never raw UUID
	_, ok := cache.Get("u1")
	assert.False(t, ok)
	assert.Equal(t, "", cache.GetDisplayName("u1"))
	assert.Equal(t, "", cache.GetUserName("u1"))

	// Set single
	cache.Set(models.User{
		ID:          "u1",
		UserName:    "alice",
		DisplayName: "Alice Smith",
	})
	u, ok := cache.Get("u1")
	require.True(t, ok)
	assert.Equal(t, "Alice Smith", u.DisplayName)
	assert.Equal(t, "Alice Smith", cache.GetDisplayName("u1"))
	assert.Equal(t, "alice", cache.GetUserName("u1"))

	// SetAll
	cache.SetAll([]models.User{
		{ID: "u2", UserName: "bob", Name: "Bob Jones"},
		{ID: "u3", UserName: "charlie"},
	})
	assert.Equal(t, "Bob Jones", cache.GetDisplayName("u2"))
	assert.Equal(t, "charlie", cache.GetDisplayName("u3"))
	assert.Len(t, cache.All(), 3)
}

func TestFetchBotUser(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/api/me", r.URL.Path)
		assert.Equal(t, "Bearer test-key", r.Header.Get("Authorization"))

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(models.User{
			ID:          "bot-999",
			UserName:    "bot",
			DisplayName: "AI Assistant",
		})
	}))
	defer server.Close()

	cfg := &config.Config{
		BesedkaURL:        server.URL,
		BesedkaAPIKey:     "test-key",
		MsgRingBufferSize: 50,
	}
	gw := NewGateway(cfg, nil)
	gw.httpClient = server.Client()

	botUser, err := gw.FetchBotUser(context.Background())
	require.NoError(t, err)
	assert.Equal(t, "bot-999", botUser.ID)
	assert.Equal(t, "AI Assistant", botUser.DisplayName)
	assert.Equal(t, "bot-999", gw.botUserID)
	assert.Equal(t, "AI Assistant", gw.userCache.GetDisplayName("bot-999"))
}

func TestFetchUsers(t *testing.T) {
	// Array format
	server1 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/api/users", r.URL.Path)
		_ = json.NewEncoder(w).Encode([]models.User{
			{ID: "u1", DisplayName: "Alice"},
			{ID: "u2", DisplayName: "Bob"},
		})
	}))
	defer server1.Close()

	cfg1 := &config.Config{BesedkaURL: server1.URL, MsgRingBufferSize: 50}
	gw1 := NewGateway(cfg1, nil)
	gw1.httpClient = server1.Client()

	users, err := gw1.FetchUsers(context.Background())
	require.NoError(t, err)
	assert.Len(t, users, 2)
	assert.Equal(t, "Alice", gw1.userCache.GetDisplayName("u1"))
	assert.Equal(t, "Bob", gw1.userCache.GetDisplayName("u2"))

	// Wrapped object format
	server2 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"users": []models.User{
				{ID: "u3", DisplayName: "Charlie"},
			},
		})
	}))
	defer server2.Close()

	cfg2 := &config.Config{BesedkaURL: server2.URL, MsgRingBufferSize: 50}
	gw2 := NewGateway(cfg2, nil)
	gw2.httpClient = server2.Client()

	users2, err := gw2.FetchUsers(context.Background())
	require.NoError(t, err)
	assert.Len(t, users2, 1)
	assert.Equal(t, "Charlie", gw2.userCache.GetDisplayName("u3"))
}

func TestFetchChats(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/api/chats", r.URL.Path)
		_ = json.NewEncoder(w).Encode([]models.Chat{
			{ID: "townhall", Name: "Townhall", Type: "townhall", LastSeq: 15},
			{ID: "dm_user1", Name: "DM with User 1", Type: "dm", LastSeq: 5},
		})
	}))
	defer server.Close()

	cfg := &config.Config{BesedkaURL: server.URL, MsgRingBufferSize: 50}
	gw := NewGateway(cfg, nil)
	gw.httpClient = server.Client()

	chats, err := gw.FetchChats(context.Background())
	require.NoError(t, err)
	assert.Len(t, chats, 2)
	assert.Equal(t, "townhall", chats[0].ID)
	assert.Equal(t, 15, chats[0].LastSeq)
	assert.Equal(t, "dm_user1", chats[1].ID)
	assert.Equal(t, 5, chats[1].LastSeq)
}

func TestFetchChatMessages(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/api/chats/townhall/messages", r.URL.Path)
		assert.Equal(t, "1", r.URL.Query().Get("fromSeq"))
		assert.Equal(t, "10", r.URL.Query().Get("toSeq"))
		_ = json.NewEncoder(w).Encode([]models.Message{
			{Seq: 1, ChatID: "townhall", UserID: "u1", Content: "First msg", Timestamp: time.Now().Unix()},
			{Seq: 2, ChatID: "townhall", UserID: "u2", Content: "Second msg", Timestamp: time.Now().Unix()},
		})
	}))
	defer server.Close()

	cfg := &config.Config{BesedkaURL: server.URL, MsgRingBufferSize: 50}
	gw := NewGateway(cfg, nil)
	gw.httpClient = server.Client()

	msgs, err := gw.FetchChatMessages(context.Background(), "townhall", 1, 10)
	require.NoError(t, err)
	assert.Len(t, msgs, 2)
	assert.Equal(t, "First msg", msgs[0].Content)
	assert.Equal(t, "Second msg", msgs[1].Content)
}

func TestFetchImageThumbnail(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/api/images/img-123", r.URL.Path)
		assert.Equal(t, "1", r.URL.Query().Get("thumb"))
		assert.Equal(t, "Bearer secret-token", r.Header.Get("Authorization"))

		w.Header().Set("Content-Type", "image/png; charset=utf-8")
		_, _ = w.Write([]byte("fake-png-data"))
	}))
	defer server.Close()

	cfg := &config.Config{
		BesedkaURL:    server.URL,
		BesedkaAPIKey: "secret-token",
	}
	gw := NewGateway(cfg, nil)
	gw.httpClient = server.Client()

	data, mime, err := gw.FetchImageThumbnail(context.Background(), "img-123")
	require.NoError(t, err)
	assert.Equal(t, []byte("fake-png-data"), data)
	assert.Equal(t, "image/png", mime)

	// Error path: empty fileID
	_, _, err = gw.FetchImageThumbnail(context.Background(), "")
	assert.Error(t, err)
}

func TestFetchFileContent(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/api/files/file-456", r.URL.Path)
		assert.Equal(t, "Bearer secret-token", r.Header.Get("Authorization"))

		w.Header().Set("Content-Type", "text/plain")
		_, _ = w.Write([]byte("hello world config file content"))
	}))
	defer server.Close()

	cfg := &config.Config{
		BesedkaURL:    server.URL,
		BesedkaAPIKey: "secret-token",
	}
	gw := NewGateway(cfg, nil)
	gw.httpClient = server.Client()

	data, mime, err := gw.FetchFileContent(context.Background(), "file-456", 100)
	require.NoError(t, err)
	assert.Equal(t, "hello world config file content", string(data))
	assert.Equal(t, "text/plain", mime)

	// Truncation test
	dataTrunc, _, err := gw.FetchFileContent(context.Background(), "file-456", 5)
	require.NoError(t, err)
	assert.Equal(t, "hello", string(dataTrunc))

	// Error path: empty fileID
	_, _, err = gw.FetchFileContent(context.Background(), "", 100)
	assert.Error(t, err)
}

func TestDownloadAttachment(t *testing.T) {
	var imageCalled atomic.Bool
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "Bearer auth-token", r.Header.Get("Authorization"))

		switch r.URL.Path {
		case "/api/files/file-ok":
			w.Header().Set("Content-Type", "application/pdf; charset=utf-8")
			_, _ = w.Write([]byte("pdf-content-bytes"))
		case "/api/files/img-fallback":
			http.NotFound(w, r)
		case "/api/images/img-fallback":
			imageCalled.Store(true)
			w.Header().Set("Content-Type", "image/png")
			_, _ = w.Write([]byte("png-content-bytes"))
		case "/api/files/server-error":
			http.Error(w, "internal database failure", http.StatusInternalServerError)
		case "/api/images/server-error":
			t.Fatal("fallback to /api/images must NOT happen on non-404 status")
		case "/api/files/both-404":
			http.NotFound(w, r)
		case "/api/images/both-404":
			http.NotFound(w, r)
		case "/api/files/over-limit":
			w.Header().Set("Content-Type", "application/octet-stream")
			_, _ = w.Write(make([]byte, 1024*1024+10))
		case "/api/files/exact-limit":
			w.Header().Set("Content-Type", "text/plain")
			_, _ = w.Write(make([]byte, 1024*1024))
		case "/api/files/invalid-mime":
			w.Header().Set("Content-Type", "invalid/type; extra/;;;")
			_, _ = w.Write([]byte{0x00, 0x01, 0x02, 0x03})
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	cfg := &config.Config{
		BesedkaURL:             server.URL,
		BesedkaAPIKey:          "auth-token",
		MaxAttachmentSizeBytes: 1024 * 1024,
	}
	gw := NewGateway(cfg, nil)
	gw.httpClient = server.Client()

	t.Run("file success without fallback", func(t *testing.T) {
		data, mimeType, err := gw.DownloadAttachment(context.Background(), "file-ok")
		require.NoError(t, err)
		assert.Equal(t, []byte("pdf-content-bytes"), data)
		assert.Equal(t, "application/pdf", mimeType)
	})

	t.Run("file 404 falls back to image", func(t *testing.T) {
		imageCalled.Store(false)
		data, mimeType, err := gw.DownloadAttachment(context.Background(), "img-fallback")
		require.NoError(t, err)
		assert.True(t, imageCalled.Load())
		assert.Equal(t, []byte("png-content-bytes"), data)
		assert.Equal(t, "image/png", mimeType)
	})

	t.Run("non-404 status does not fallback", func(t *testing.T) {
		_, _, err := gw.DownloadAttachment(context.Background(), "server-error")
		require.Error(t, err)
		assert.Contains(t, err.Error(), "500")
		assert.Contains(t, err.Error(), "internal database failure")
	})

	t.Run("both 404 returns error", func(t *testing.T) {
		_, _, err := gw.DownloadAttachment(context.Background(), "both-404")
		require.Error(t, err)
		assert.Contains(t, err.Error(), "404")
	})

	t.Run("empty ID returns error", func(t *testing.T) {
		_, _, err := gw.DownloadAttachment(context.Background(), "   ")
		require.Error(t, err)
		assert.Contains(t, err.Error(), "empty fileID")
	})

	t.Run("over-limit returns error", func(t *testing.T) {
		_, _, err := gw.DownloadAttachment(context.Background(), "over-limit")
		require.Error(t, err)
		assert.Contains(t, err.Error(), "exceeds limit")
	})

	t.Run("exact-limit succeeds", func(t *testing.T) {
		data, _, err := gw.DownloadAttachment(context.Background(), "exact-limit")
		require.NoError(t, err)
		assert.Len(t, data, 1024*1024)
	})

	t.Run("math.MaxInt64 limit does not overflow", func(t *testing.T) {
		gwMax := NewGateway(&config.Config{
			BesedkaURL:             server.URL,
			BesedkaAPIKey:          "auth-token",
			MaxAttachmentSizeBytes: math.MaxInt64,
		}, nil)
		gwMax.httpClient = server.Client()

		data, _, err := gwMax.DownloadAttachment(context.Background(), "file-ok")
		require.NoError(t, err)
		assert.Equal(t, []byte("pdf-content-bytes"), data)
	})

	t.Run("invalid mime falls back to octet-stream", func(t *testing.T) {
		_, mimeType, err := gw.DownloadAttachment(context.Background(), "invalid-mime")
		require.NoError(t, err)
		assert.Equal(t, "application/octet-stream", mimeType)
	})

	t.Run("cancelled context returns error", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		_, _, err := gw.DownloadAttachment(ctx, "file-ok")
		require.Error(t, err)
	})
}

func TestUploadFile(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "Bearer auth-token", r.Header.Get("Authorization"))

		switch r.URL.Path {
		case "/api/upload/file":
			body, _ := io.ReadAll(r.Body)
			if len(body) == 0 {
				w.Header().Set("Content-Type", "application/json")
				_ = json.NewEncoder(w).Encode(map[string]string{"id": "empty-file-id"})
				return
			}
			assert.Equal(t, "text/csv", r.Header.Get("Content-Type"))
			assert.Equal(t, "a,b,c\n1,2,3", string(body))

			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]string{"id": "file-12345"})
		case "/api/upload/error":
			http.Error(w, "payload too large", http.StatusRequestEntityTooLarge)
		case "/api/upload/bad-json":
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"id": ""}`))
		case "/api/upload/malformed-json":
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"id": `))
		case "/api/upload/fallback-mime":
			assert.Equal(t, "application/octet-stream", r.Header.Get("Content-Type"))
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]string{"id": "mime-id"})
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	cfg := &config.Config{
		BesedkaURL:    server.URL,
		BesedkaAPIKey: "auth-token",
	}
	gw := NewGateway(cfg, nil)
	gw.httpClient = server.Client()

	t.Run("success", func(t *testing.T) {
		id, err := gw.UploadFile(context.Background(), []byte("a,b,c\n1,2,3"), "data.csv", "text/csv")
		require.NoError(t, err)
		assert.Equal(t, "file-12345", id)
	})

	t.Run("empty data succeeds for generic file", func(t *testing.T) {
		id, err := gw.UploadFile(context.Background(), []byte{}, "empty.txt", "text/plain")
		require.NoError(t, err)
		assert.Equal(t, "empty-file-id", id)
	})

	t.Run("server error bounded text", func(t *testing.T) {
		_, err := gw.uploadPayload(context.Background(), "/api/upload/error", []byte("xyz"), "f.txt", "text/plain")
		require.Error(t, err)
		assert.Contains(t, err.Error(), "413")
		assert.Contains(t, err.Error(), "payload too large")
	})

	t.Run("empty response id", func(t *testing.T) {
		_, err := gw.uploadPayload(context.Background(), "/api/upload/bad-json", []byte("xyz"), "f.txt", "text/plain")
		require.Error(t, err)
		assert.Contains(t, err.Error(), "empty file ID")
	})

	t.Run("malformed json response", func(t *testing.T) {
		_, err := gw.uploadPayload(context.Background(), "/api/upload/malformed-json", []byte("xyz"), "f.txt", "text/plain")
		require.Error(t, err)
		assert.Contains(t, err.Error(), "failed to decode upload response")
	})

	t.Run("invalid mime falls back to octet-stream", func(t *testing.T) {
		id, err := gw.uploadPayload(context.Background(), "/api/upload/fallback-mime", []byte("xyz"), "f.bin", "invalid-mime-;;;")
		require.NoError(t, err)
		assert.Equal(t, "mime-id", id)
	})

	t.Run("cancelled context returns error", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		_, err := gw.UploadFile(ctx, []byte("test"), "test.txt", "text/plain")
		require.Error(t, err)
	})
}

func TestUploadImage(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/api/upload/image", r.URL.Path)
		assert.Equal(t, "image/png", r.Header.Get("Content-Type"))
		assert.Equal(t, "Bearer auth-token", r.Header.Get("Authorization"))

		body, _ := io.ReadAll(r.Body)
		assert.Equal(t, []byte("raw-png-data"), body)

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]string{"id": "img-67890"})
	}))
	defer server.Close()

	cfg := &config.Config{
		BesedkaURL:    server.URL,
		BesedkaAPIKey: "auth-token",
	}
	gw := NewGateway(cfg, nil)
	gw.httpClient = server.Client()

	id, err := gw.UploadImage(context.Background(), []byte("raw-png-data"), "plot.png", "image/png")
	require.NoError(t, err)
	assert.Equal(t, "img-67890", id)
}
