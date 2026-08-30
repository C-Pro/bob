package llm

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"bob/internal/config"

	openai "github.com/sashabaranov/go-openai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCreateEmbeddings_Success(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/embeddings", r.URL.Path)
		assert.Equal(t, "Bearer test-key", r.Header.Get("Authorization"))

		var req openai.EmbeddingRequest
		err := json.NewDecoder(r.Body).Decode(&req)
		require.NoError(t, err)
		assert.Equal(t, openai.EmbeddingModel("gemini-embedding-2"), req.Model)

		resp := openai.EmbeddingResponse{
			Object: "list",
			Data: []openai.Embedding{
				{
					Index:     0,
					Embedding: []float32{0.1, 0.2, 0.3},
				},
				{
					Index:     1,
					Embedding: []float32{0.4, 0.5, 0.6},
				},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	cfg := &config.Config{
		OpenAIAPIKey:   "test-key",
		OpenAIBaseURL:  server.URL,
		EmbeddingModel: "gemini-embedding-2",
	}

	client := NewClient(cfg, server.Client())
	vecs, err := client.CreateEmbeddings(context.Background(), []string{"hello", "world"}, "")
	require.NoError(t, err)
	require.Len(t, vecs, 2)
	assert.Equal(t, []float32{0.1, 0.2, 0.3}, vecs[0])
	assert.Equal(t, []float32{0.4, 0.5, 0.6}, vecs[1])
}

func TestCreateEmbeddings_RetrySuccess(t *testing.T) {
	var attempts int32

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		count := atomic.AddInt32(&attempts, 1)
		if count < 2 {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusTooManyRequests)
			_ = json.NewEncoder(w).Encode(openai.APIError{
				Message: "Rate limit exceeded",
				Code:    "429",
			})
			return
		}

		resp := openai.EmbeddingResponse{
			Object: "list",
			Data: []openai.Embedding{
				{
					Index:     0,
					Embedding: []float32{0.7, 0.8, 0.9},
				},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	cfg := &config.Config{
		OpenAIAPIKey:   "test-key",
		OpenAIBaseURL:  server.URL,
		EmbeddingModel: "gemini-embedding-2",
	}

	client := NewClient(cfg, server.Client())
	vecs, err := client.CreateEmbeddings(context.Background(), []string{"test"}, "")
	require.NoError(t, err)
	require.Len(t, vecs, 1)
	assert.Equal(t, []float32{0.7, 0.8, 0.9}, vecs[0])
	assert.Equal(t, int32(2), atomic.LoadInt32(&attempts))
}

func TestCreateEmbeddings_NoModelConfigured(t *testing.T) {
	cfg := &config.Config{
		OpenAIAPIKey: "test-key",
	}
	client := NewClient(cfg, nil)
	vecs, err := client.CreateEmbeddings(context.Background(), []string{"test"}, "")
	assert.ErrorContains(t, err, "embedding model is not configured")
	assert.Nil(t, vecs)
}

func TestEmbedder_EmbedAndDim(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := openai.EmbeddingResponse{
			Object: "list",
			Data: []openai.Embedding{
				{
					Index:     0,
					Embedding: []float32{0.1, 0.2, 0.3, 0.4},
				},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	cfg := &config.Config{
		OpenAIAPIKey:   "test-key",
		OpenAIBaseURL:  server.URL,
		EmbeddingModel: "gemini-embedding-2",
	}

	client := NewClient(cfg, server.Client())
	embedder := NewEmbedder(client, "")
	assert.Equal(t, "gemini-embedding-2", embedder.Model())
	assert.Equal(t, 0, embedder.Dim())

	// Test empty text
	_, err := embedder.Embed(context.Background(), "")
	assert.ErrorContains(t, err, "empty text provided")

	// Test single embed
	vec, err := embedder.Embed(context.Background(), "sample query")
	require.NoError(t, err)
	assert.Equal(t, []float32{0.1, 0.2, 0.3, 0.4}, vec)
	assert.Equal(t, 4, embedder.Dim())
}
