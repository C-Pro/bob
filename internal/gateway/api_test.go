package gateway

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
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

