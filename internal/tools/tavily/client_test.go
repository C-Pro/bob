package tavily

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSearchSuccess(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "POST", r.Method)
		assert.Equal(t, "/search", r.URL.Path)
		assert.Equal(t, "Bearer test-api-key", r.Header.Get("Authorization"))
		assert.Equal(t, "application/json", r.Header.Get("Content-Type"))

		var req SearchRequest
		err := json.NewDecoder(r.Body).Decode(&req)
		require.NoError(t, err)
		assert.Equal(t, "golang concurrency", req.Query)
		assert.Equal(t, "basic", req.SearchDepth)
		assert.Equal(t, 5, req.MaxResults)
		assert.True(t, req.IncludeAnswer)

		resp := SearchResponse{
			Query:  "golang concurrency",
			Answer: "Go provides goroutines and channels for concurrency.",
			Results: []SearchResult{
				{
					Title:   "Effective Go: Concurrency",
					URL:     "https://go.dev/doc/effective_go#concurrency",
					Content: "Go provides goroutines and channels for concurrent programming.",
					Score:   0.95,
				},
			},
			ResponseTime: 0.12,
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer ts.Close()

	client := NewClient("test-api-key", ts.URL, ts.Client())
	resp, err := client.Search(context.Background(), SearchRequest{
		Query: "golang concurrency",
	})

	require.NoError(t, err)
	require.NotNil(t, resp)
	assert.Equal(t, "golang concurrency", resp.Query)
	assert.Equal(t, "Go provides goroutines and channels for concurrency.", resp.Answer)
	require.Len(t, resp.Results, 1)
	assert.Equal(t, "Effective Go: Concurrency", resp.Results[0].Title)
	assert.Equal(t, "https://go.dev/doc/effective_go#concurrency", resp.Results[0].URL)
}

func TestSearchCustomParameters(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req SearchRequest
		err := json.NewDecoder(r.Body).Decode(&req)
		require.NoError(t, err)
		assert.Equal(t, "weather in tokyo", req.Query)
		assert.Equal(t, "advanced", req.SearchDepth)
		assert.Equal(t, 3, req.MaxResults)
		assert.Equal(t, "news", req.Topic)

		resp := SearchResponse{
			Query: "weather in tokyo",
			Results: []SearchResult{
				{Title: "Tokyo Weather", URL: "https://weather.example.com", Content: "Sunny, 22C"},
			},
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer ts.Close()

	client := NewClient("test-api-key", ts.URL, ts.Client())
	resp, err := client.Search(context.Background(), SearchRequest{
		Query:       "weather in tokyo",
		SearchDepth: "advanced",
		Topic:       "news",
		MaxResults:  3,
	})

	require.NoError(t, err)
	require.NotNil(t, resp)
	require.Len(t, resp.Results, 1)
}

func TestSearchValidation(t *testing.T) {
	client := NewClient("", "https://api.tavily.com", nil)

	// Missing API key
	_, err := client.Search(context.Background(), SearchRequest{Query: "test"})
	assert.ErrorContains(t, err, "tavily API key is required")

	// Empty query
	client = NewClient("some-key", "https://api.tavily.com", nil)
	_, err = client.Search(context.Background(), SearchRequest{Query: "   "})
	assert.ErrorContains(t, err, "search query cannot be empty")
}

func TestSearchRetryOn5xxAnd429(t *testing.T) {
	var attempts int32
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		curr := atomic.AddInt32(&attempts, 1)
		if curr == 1 {
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = w.Write([]byte(`{"error": "rate limit exceeded"}`))
			return
		}
		if curr == 2 {
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = w.Write([]byte(`{"error": "internal server error"}`))
			return
		}

		resp := SearchResponse{
			Query:   "resilient query",
			Results: []SearchResult{{Title: "Success", URL: "https://example.com"}},
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer ts.Close()

	client := NewClient("test-api-key", ts.URL, ts.Client())
	client.baseDelay = 1 * time.Millisecond

	resp, err := client.Search(context.Background(), SearchRequest{Query: "resilient query"})
	require.NoError(t, err)
	require.NotNil(t, resp)
	assert.Equal(t, int32(3), atomic.LoadInt32(&attempts))
	require.Len(t, resp.Results, 1)
}

func TestSearchFailureAfterRetries(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte(`{"error": "service unavailable"}`))
	}))
	defer ts.Close()

	client := NewClient("test-api-key", ts.URL, ts.Client())
	client.baseDelay = 1 * time.Millisecond
	client.maxRetries = 2

	_, err := client.Search(context.Background(), SearchRequest{Query: "fail query"})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "tavily API request failed after 2 retries")
}

func TestSearchContextCancellation(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(200 * time.Millisecond)
		w.WriteHeader(http.StatusOK)
	}))
	defer ts.Close()

	client := NewClient("test-api-key", ts.URL, ts.Client())
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()

	_, err := client.Search(ctx, SearchRequest{Query: "timeout query"})
	require.Error(t, err)
}
