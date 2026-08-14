package llm

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"bob/internal/config"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestGenerateResponseSuccess(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/chat/completions", r.URL.Path)
		assert.Equal(t, "Bearer test-api-key", r.Header.Get("Authorization"))
		assert.Equal(t, "application/json", r.Header.Get("Content-Type"))

		var req CompletionRequest
		err := json.NewDecoder(r.Body).Decode(&req)
		require.NoError(t, err)
		assert.Equal(t, "gemini-3.7-flash", req.Model)
		require.Len(t, req.Messages, 2)
		assert.Equal(t, "system", req.Messages[0].Role)
		assert.Equal(t, "You are a helpful assistant.", req.Messages[0].Content)
		assert.Equal(t, "user", req.Messages[1].Role)
		assert.Equal(t, "Hello", req.Messages[1].Content)

		resp := CompletionResponse{
			ID:      "chatcmpl-test",
			Model:   req.Model,
			Choices: []Choice{{Index: 0, Message: Message{Role: "assistant", Content: "Hello from Gemini!"}}},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	cfg := &config.Config{
		GeminiAPIKey:  "test-api-key",
		GeminiModel:   "gemini-3.7-flash",
		GeminiBaseURL: server.URL,
	}

	client := NewClient(cfg, server.Client())
	reply, err := client.GenerateResponse(context.Background(), "You are a helpful assistant.", "Hello")
	require.NoError(t, err)
	assert.Equal(t, "Hello from Gemini!", reply)
}

func TestGenerateResponseRetrySuccess(t *testing.T) {
	var attempts int32

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		count := atomic.AddInt32(&attempts, 1)
		if count < 3 {
			http.Error(w, "internal server error", http.StatusInternalServerError)
			return
		}

		resp := CompletionResponse{
			Choices: []Choice{{Index: 0, Message: Message{Role: "assistant", Content: "Recovered reply"}}},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	cfg := &config.Config{
		GeminiAPIKey:  "test-key",
		GeminiModel:   "gemini-3.7-flash",
		GeminiBaseURL: server.URL,
	}

	client := NewClient(cfg, server.Client())
	reply, err := client.GenerateResponse(context.Background(), "", "Hello")
	require.NoError(t, err)
	assert.Equal(t, "Recovered reply", reply)
	assert.Equal(t, int32(3), atomic.LoadInt32(&attempts))
}

func TestGenerateResponseRetryExhausted(t *testing.T) {
	var attempts int32

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&attempts, 1)
		http.Error(w, "service unavailable", http.StatusServiceUnavailable)
	}))
	defer server.Close()

	cfg := &config.Config{
		GeminiAPIKey:  "test-key",
		GeminiModel:   "gemini-3.7-flash",
		GeminiBaseURL: server.URL,
	}

	client := NewClient(cfg, server.Client())
	_, err := client.GenerateResponse(context.Background(), "", "Hello")
	assert.ErrorContains(t, err, "request failed after 3 retries")
	assert.Equal(t, int32(4), atomic.LoadInt32(&attempts))
}
