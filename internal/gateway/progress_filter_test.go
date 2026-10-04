package gateway

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"bob/internal/chatcontext"
	"bob/internal/config"
	"bob/internal/memory"
	"bob/internal/models"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestProcessMessage_ProgressFiltering(t *testing.T) {
	gw := &Gateway{
		contextManager: chatcontext.NewManager(10),
		userCache:      NewUserCache(),
		botUserID:      "bot-1",
		startTime:      time.Now().Add(-1 * time.Hour),
	}

	ctx := context.Background()

	// 1. Typed root progress message
	err := gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_1",
		UserID:    "bot-1",
		Type:      models.MessageTypeProgress,
		Progress:  &models.ProgressData{Title: "Task running"},
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)
	assert.Equal(t, 0, gw.contextManager.GetOrCreate("dm_1").Len())

	// 2. Typed child step message from another bot
	err = gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_1",
		UserID:    "other-bot",
		Type:      models.MessageTypeProgress,
		Progress:  &models.ProgressData{ParentSeq: 10, CardStatus: models.ProgressStatusCompleted},
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)
	assert.Equal(t, 0, gw.contextManager.GetOrCreate("dm_1").Len())

	// 3. Legacy progress prefix
	err = gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_1",
		UserID:    "bot-1",
		Content:   "⏳ Running tool command...",
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)
	assert.Equal(t, 0, gw.contextManager.GetOrCreate("dm_1").Len())

	// 4. Sandbox request awaiting approval should NOT be filtered as progress
	assert.False(t, isLegacyProgressMessage("⏳ **Sandbox Request Awaiting Your Approval**\nDetails..."))

	// 5. Human message starting with ⏳ must NOT be dropped as a progress message
	err = gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_1",
		UserID:    "user-alice",
		Content:   "⏳ Still waiting for your response!",
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)
	assert.Equal(t, 1, gw.contextManager.GetOrCreate("dm_1").Len())
	assert.Equal(t, "⏳ Still waiting for your response!", gw.contextManager.GetOrCreate("dm_1").Entries()[0].Content)
}

func TestWarmupChat_ProgressFiltering(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		msgs := []models.Message{
			{Seq: 1, UserID: "u1", Content: "Hello", Timestamp: 100},
			{Seq: 2, UserID: "bot-1", Type: models.MessageTypeProgress, Progress: &models.ProgressData{Title: "Working"}, Timestamp: 101},
			{Seq: 3, UserID: "bot-1", Content: "⏳ Tool running", Timestamp: 102},
			{Seq: 4, UserID: "u1", Content: "Thanks", Timestamp: 103},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(msgs)
	}))
	defer server.Close()

	cfg := &config.Config{
		BesedkaURL: server.URL,
	}
	gw := NewGateway(cfg, nil)
	gw.httpClient = server.Client()
	gw.botUserID = "bot-1"

	gw.WarmupChat(context.Background(), "chat_w", 4)
	rb := gw.contextManager.GetOrCreate("chat_w")
	entries := rb.Entries()
	require.Len(t, entries, 2)
	assert.Equal(t, "Hello", entries[0].Content)
	assert.Equal(t, "Thanks", entries[1].Content)
}

func TestCatchupChatMemory_ProgressFilteringAndWatermark(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		msgs := []models.Message{
			{Seq: 10, UserID: "u1", Content: "First message", Timestamp: 100},
			{Seq: 11, UserID: "bot-1", Type: models.MessageTypeProgress, Progress: &models.ProgressData{Title: "Root card"}, Timestamp: 101},
			{Seq: 12, UserID: "bot-1", Type: models.MessageTypeProgress, Progress: &models.ProgressData{ParentSeq: 11, CardStatus: models.ProgressStatusCompleted}, Timestamp: 102},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(msgs)
	}))
	defer server.Close()

	tempDir := t.TempDir()
	cfg := &config.Config{
		BesedkaURL: server.URL,
		DataDir:    tempDir,
	}
	gw := NewGateway(cfg, nil)
	gw.httpClient = server.Client()
	gw.botUserID = "bot-1"

	// Mock memory manager
	memMgr := memory.NewManager(cfg, nil)
	gw.memoryManager = memMgr

	ctx := context.Background()
	gw.CatchupChatMemory(ctx, "chat_catchup", false, 12)

	// Watermark must have advanced to 12 despite progress messages 11 and 12 being skipped
	watermark, err := memMgr.GetWatermark(ctx, "chat_catchup", false)
	require.NoError(t, err)
	assert.Equal(t, int64(12), watermark)
}
