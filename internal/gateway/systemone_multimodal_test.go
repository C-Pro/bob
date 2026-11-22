package gateway

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"bob/internal/chatcontext"
	"bob/internal/config"
	"bob/internal/llm"
	"bob/internal/memory"
	"bob/internal/models"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestGateway_MultimodalEvictionAndCatchup(t *testing.T) {
	ctx := context.Background()
	pngBytes := []byte{0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a}
	expectedURI := "data:image/png;base64," + base64.StdEncoding.EncodeToString(pngBytes)

	var mmCalls atomic.Int32
	s1Srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req map[string]any
		require.NoError(t, json.NewDecoder(r.Body).Decode(&req))
		inputs := req["input"].([]any)
		data := make([]map[string]any, len(inputs))
		for i, raw := range inputs {
			if m, ok := raw.(map[string]any); ok {
				parts := m["content"].([]any)
				require.GreaterOrEqual(t, len(parts), 2)
				imgPart := parts[0].(map[string]any)
				assert.Equal(t, "image_url", imgPart["type"])
				assert.Equal(t, expectedURI, imgPart["image_url"].(map[string]any)["url"])
				mmCalls.Add(1)
			}
			// Valid 128-dim base64 float32 zero vector
			rawVec := make([]byte, 128*4)
			data[i] = map[string]any{
				"object":    "embedding",
				"index":     i,
				"embedding": base64.StdEncoding.EncodeToString(rawVec),
			}
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"object": "list", "data": data})
	}))
	defer s1Srv.Close()

	s1Cfg := &config.Config{
		SystemOneBaseURL:       s1Srv.URL,
		SystemOneModel:         "bob-s1-0.8b",
		SystemOneEmbeddingDims: 128,
	}
	s1Client, err := llm.NewSystemOneClient(s1Cfg, s1Srv.Client())
	require.NoError(t, err)
	s1Emb, err := llm.NewSystemOneEmbedder(s1Client, 128, 0, false)
	require.NoError(t, err)

	besedkaSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/images/img-1":
			w.Header().Set("Content-Type", "image/png")
			_, _ = w.Write(pngBytes)
		default:
			msgs := []models.Message{
				{
					Seq:       5,
					UserID:    "u1",
					Timestamp: 1700000005,
					Content:   "",
					Attachments: []models.Attachment{
						{FileID: "img-1", Name: "screen.png", Type: models.AttachmentTypeImage, MimeType: "image/png"},
					},
				},
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(msgs)
		}
	}))
	defer besedkaSrv.Close()

	cfg := &config.Config{
		BesedkaURL:               besedkaSrv.URL,
		DataDir:                  t.TempDir(),
		SystemOneBaseURL:         s1Srv.URL,
		SystemOneFallbackOnError: false,
	}
	memMgr := memory.NewManager(cfg, s1Emb)
	defer func() { _ = memMgr.Close() }()

	gw := NewGateway(cfg, nil)
	gw.httpClient = besedkaSrv.Client()
	gw.memoryManager = memMgr

	// 1. handleEvictedBatch forwards ImageAttachment URLs.
	gw.handleEvictedBatch("townhall", []chatcontext.Entry{
		{
			Seq:        1,
			Role:       "user",
			SenderID:   "u1",
			SenderName: "Alice",
			Content:    "See error screenshot",
			Images:     []chatcontext.ImageAttachment{{URL: expectedURI}},
			Timestamp:  time.Now().Unix(),
		},
	})
	gw.indexingWg.Wait()

	// 2. CatchupChatMemory forwards processAttachments image URLs.
	gw.CatchupChatMemory(ctx, "chat_img", false, 5)
	wm, err := memMgr.GetWatermark(ctx, "chat_img", false)
	require.NoError(t, err)
	assert.Equal(t, int64(5), wm)

	assert.Equal(t, int32(2), mmCalls.Load(), "both evicted batch and catchup should trigger multimodal embedding")
}
