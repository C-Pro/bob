package memory

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"bob/internal/config"
	"bob/internal/llm"

	openai "github.com/sashabaranov/go-openai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSanitizeChatID(t *testing.T) {
	assert.Equal(t, "townhall", SanitizeChatID("townhall"))
	assert.Equal(t, "dm_user_123", SanitizeChatID("dm/user:123"))
	assert.Equal(t, "user_chat-99_abc", SanitizeChatID("user:chat-99@abc"))
	assert.Equal(t, "unknown", SanitizeChatID(""))
}

func TestWatermarkTracking(t *testing.T) {
	tempDir := t.TempDir()

	cfg := &config.Config{
		DataDir:        tempDir,
		EmbeddingModel: "none",
	}
	manager := NewManager(cfg, nil)
	defer func() { _ = manager.Close() }()

	ctx := context.Background()

	// Initial watermark should be 0
	wm, err := manager.GetWatermark(ctx, "townhall", false)
	require.NoError(t, err)
	assert.Equal(t, int64(0), wm)

	// Set and get watermark
	err = manager.SetWatermark(ctx, "townhall", false, 42)
	require.NoError(t, err)

	wm, err = manager.GetWatermark(ctx, "townhall", false)
	require.NoError(t, err)
	assert.Equal(t, int64(42), wm)

	// Test DM watermark isolation
	dmWM, err := manager.GetWatermark(ctx, "user123", true)
	require.NoError(t, err)
	assert.Equal(t, int64(0), dmWM)

	err = manager.SetWatermark(ctx, "user123", true, 99)
	require.NoError(t, err)

	dmWM, err = manager.GetWatermark(ctx, "user123", true)
	require.NoError(t, err)
	assert.Equal(t, int64(99), dmWM)

	// Townhall watermark remains 42
	thWM, err := manager.GetWatermark(ctx, "townhall", false)
	require.NoError(t, err)
	assert.Equal(t, int64(42), thWM)
}

func TestFTS5FallbackSearchAndIsolation(t *testing.T) {
	tempDir := t.TempDir()

	cfg := &config.Config{
		DataDir:        tempDir,
		EmbeddingModel: "none",
	}
	// No embedder configured -> tests FTS5 keyword fallback
	manager := NewManager(cfg, nil)
	defer func() { _ = manager.Close() }()

	ctx := context.Background()
	now := time.Now().Unix()

	// Index messages in Townhall
	thMsgs := []MessageToStore{
		{
			Seq:        1,
			Timestamp:  now,
			ChatID:     "townhall",
			UserID:     "user1",
			SenderName: "Alice",
			Role:       "user",
			Content:    "We are launching the project quantum falcon next Tuesday in townhall.",
		},
		{
			Seq:        2,
			Timestamp:  now + 1,
			ChatID:     "townhall",
			UserID:     "bot1",
			SenderName: "Bob",
			Role:       "assistant",
			Content:    "Acknowledged, quantum falcon is scheduled.",
		},
	}
	err := manager.IndexMessages(ctx, "townhall", false, thMsgs)
	require.NoError(t, err)

	// Index messages in DM 1
	dm1Msgs := []MessageToStore{
		{
			Seq:        10,
			Timestamp:  now + 2,
			ChatID:     "dm_user1",
			UserID:     "user1",
			SenderName: "Alice",
			Role:       "user",
			Content:    "My secret key for falcon is 998877 in private DM.",
		},
	}
	err = manager.IndexMessages(ctx, "dm_user1", true, dm1Msgs)
	require.NoError(t, err)

	// Index messages in DM 2
	dm2Msgs := []MessageToStore{
		{
			Seq:        20,
			Timestamp:  now + 3,
			ChatID:     "dm_user2",
			UserID:     "user2",
			SenderName: "Charlie",
			Role:       "user",
			Content:    "Charlie's secret code is 112233 in private DM.",
		},
	}
	err = manager.IndexMessages(ctx, "dm_user2", true, dm2Msgs)
	require.NoError(t, err)

	// Should find "quantum falcon" from Townhall, with "[Townhall]" label
	// MUST NOT find User 1's secret key or User 2's secret code
	thHits, err := manager.Search(ctx, "falcon", "townhall", false, 5)
	require.NoError(t, err)
	require.Len(t, thHits, 1)
	assert.Equal(t, "[Townhall]", thHits[0].Source)
	assert.Contains(t, thHits[0].Content, "quantum falcon")

	// Verify Townhall search NEVER returns DM secrets
	thSecretHits, err := manager.Search(ctx, "998877", "townhall", false, 5)
	require.NoError(t, err)
	assert.Empty(t, thSecretHits)

	// 2. Search in User 1 DM Context:
	// Should find User 1's private secret labeled "[Direct Message]"
	// AND should also find public Townhall "quantum falcon" labeled "[Townhall]"
	// MUST NOT find User 2's private secret
	u1Hits, err := manager.Search(ctx, "falcon", "dm_user1", true, 5)
	require.NoError(t, err)
	require.Len(t, u1Hits, 2) // 1 from DM1 + 1 from Townhall

	var foundDM, foundTH bool
	for _, hit := range u1Hits {
		if hit.Source == "[Direct Message]" && hit.ChatID == "dm_user1" {
			foundDM = true
			assert.Contains(t, hit.Content, "998877")
		}
		if hit.Source == "[Townhall]" && hit.ChatID == "townhall" {
			foundTH = true
			assert.Contains(t, hit.Content, "quantum falcon")
		}
	}
	assert.True(t, foundDM, "User 1 should find their own DM memory")
	assert.True(t, foundTH, "User 1 should find public Townhall memory")

	// Verify User 1 cannot search User 2's secret
	u1SecretHits, err := manager.Search(ctx, "112233", "dm_user1", true, 5)
	require.NoError(t, err)
	assert.Empty(t, u1SecretHits, "User 1 must not access User 2 DM memories")

	// 3. Search in User 2 DM Context:
	// Should find User 2's private secret
	// MUST NOT find User 1's secret key
	u2Hits, err := manager.Search(ctx, "112233", "dm_user2", true, 5)
	require.NoError(t, err)
	require.Len(t, u2Hits, 1)
	assert.Equal(t, "[Direct Message]", u2Hits[0].Source)
	assert.Contains(t, u2Hits[0].Content, "112233")

	u2SecretHits, err := manager.Search(ctx, "998877", "dm_user2", true, 5)
	require.NoError(t, err)
	assert.Empty(t, u2SecretHits, "User 2 must not access User 1 DM memories")
}

func TestSemanticVectorSearchWithEmbedder(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req openai.EmbeddingRequest
		_ = json.NewDecoder(r.Body).Decode(&req)

		inputs, _ := req.Input.([]interface{})
		data := make([]openai.Embedding, len(inputs))
		for i, inp := range inputs {
			str := inp.(string)
			// Return distinct vectors for Golang vs Python
			vec := []float32{0.1, 0.2, 0.3}
			if strings.Contains(str, "Golang") {
				vec = []float32{0.9, 0.8, 0.7}
			} else if strings.Contains(str, "Python") {
				vec = []float32{0.1, 0.9, 0.1}
			}
			data[i] = openai.Embedding{
				Index:     i,
				Embedding: vec,
			}
		}

		resp := openai.EmbeddingResponse{
			Object: "list",
			Data:   data,
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	tempDir := t.TempDir()

	cfg := &config.Config{
		OpenAIAPIKey:   "test-key",
		OpenAIBaseURL:  server.URL,
		EmbeddingModel: "gemini-embedding-2",
		DataDir:        tempDir,
	}

	client := llm.NewClient(cfg, server.Client())
	embedder := llm.NewEmbedder(client, "gemini-embedding-2")
	manager := NewManager(cfg, embedder)
	defer func() { _ = manager.Close() }()

	ctx := context.Background()
	now := time.Now().Unix()

	// Index messages
	msgs := []MessageToStore{
		{
			Seq:        100,
			Timestamp:  now,
			ChatID:     "townhall",
			UserID:     "cpro",
			SenderName: "C-Pro",
			Role:       "user",
			Content:    "Python is an interpreted high-level programming language.",
		},
		{
			Seq:        101,
			Timestamp:  now + 1,
			ChatID:     "townhall",
			UserID:     "cpro",
			SenderName: "C-Pro",
			Role:       "user",
			Content:    "Golang is statically typed and compiles to native machine code.",
		},
	}

	err := manager.IndexMessages(ctx, "townhall", false, msgs)
	require.NoError(t, err)

	// Verify database file exists on disk
	thDBPath := filepath.Join(tempDir, "townhall.db")
	_, err = os.Stat(thDBPath)
	assert.NoError(t, err)

	// Verify search
	hits, err := manager.Search(ctx, "Golang", "townhall", false, 5)
	require.NoError(t, err)
	require.NotEmpty(t, hits)
	assert.Equal(t, "[Townhall]", hits[0].Source)
	assert.Contains(t, hits[0].Content, "Golang")
}

type mockEmbedder struct {
	dim   int
	model string
	fn    func(text string) []float32
}

func (m *mockEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	if m.fn != nil {
		return m.fn(text), nil
	}
	vec := make([]float32, m.dim)
	for i := range vec {
		vec[i] = 0.1
	}
	return vec, nil
}

func (m *mockEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	results := make([][]float32, len(texts))
	for i, t := range texts {
		v, _ := m.Embed(ctx, t)
		results[i] = v
	}
	return results, nil
}

func (m *mockEmbedder) Dim() int {
	return m.dim
}

func (m *mockEmbedder) Model() string {
	return m.model
}

func TestDimensionMismatchGracefulFallback(t *testing.T) {
	tempDir := t.TempDir()
	cfg := &config.Config{
		DataDir: tempDir,
	}

	ctx := context.Background()
	now := time.Now().Unix()

	// 1. Index messages with a 3-dimensional embedder
	emb3 := &mockEmbedder{dim: 3, model: "model-3d"}
	mgr1 := NewManager(cfg, emb3)
	msgs := []MessageToStore{
		{
			Seq:        1,
			Timestamp:  now,
			ChatID:     "townhall",
			UserID:     "user1",
			SenderName: "Alice",
			Role:       "user",
			Content:    "Kubernetes cluster deployment on bare metal servers.",
		},
	}
	err := mgr1.IndexMessages(ctx, "townhall", false, msgs)
	require.NoError(t, err)
	require.NoError(t, mgr1.Close())

	// 2. Open new manager with a 5-dimensional embedder (simulating model switch)
	emb5 := &mockEmbedder{dim: 5, model: "model-5d"}
	mgr2 := NewManager(cfg, emb5)
	defer func() { _ = mgr2.Close() }()

	// Search query - vector search across mismatched dimensions falls back gracefully to lexical search
	hits, err := mgr2.Search(ctx, "Kubernetes", "townhall", false, 5)
	require.NoError(t, err)
	require.NotEmpty(t, hits, "search should return results via graceful fallback")
	assert.Contains(t, hits[0].Content, "Kubernetes")
}

