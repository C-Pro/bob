package webfetch

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"bob/internal/config"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestFetchSuccess(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "GET", r.Method)
		assert.Equal(t, config.DefaultUserAgent, r.Header.Get("User-Agent"))

		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		_, _ = w.Write([]byte("<html><head><title>Test Page</title></head><body><p>Hello world!</p></body></html>"))
	}))
	defer ts.Close()

	res, err := Fetch(context.Background(), ts.URL, ts.Client())
	require.NoError(t, err)
	require.NotNil(t, res)
	assert.Equal(t, ts.URL, res.URL)
	assert.Equal(t, http.StatusOK, res.StatusCode)
	assert.Equal(t, "text/html", res.ContentType)
	assert.True(t, IsHTML(res.ContentType))
	assert.False(t, res.Truncated)
	assert.Contains(t, string(res.RawBody), "Hello world!")
}

func TestFetchTruncationAt16KB(t *testing.T) {
	largeData := strings.Repeat("A", 20*1024) // 20KB

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain")
		_, _ = w.Write([]byte(largeData))
	}))
	defer ts.Close()

	res, err := Fetch(context.Background(), ts.URL, ts.Client())
	require.NoError(t, err)
	require.NotNil(t, res)
	assert.Equal(t, "text/plain", res.ContentType)
	assert.True(t, res.Truncated)
	assert.Equal(t, MaxContentSize, len(res.RawBody))

	truncatedText := TruncateText(string(res.RawBody), res.Truncated)
	assert.Contains(t, truncatedText, TruncationNotice)
}

func TestTypeDetectors(t *testing.T) {
	assert.True(t, IsHTML("text/html"))
	assert.True(t, IsHTML("application/xhtml+xml"))
	assert.False(t, IsHTML("text/plain"))
	assert.False(t, IsHTML("application/json"))

	assert.True(t, IsBinary("application/pdf"))
	assert.True(t, IsBinary("image/png"))
	assert.True(t, IsBinary("video/mp4"))
	assert.True(t, IsBinary("application/octet-stream"))
	assert.False(t, IsBinary("text/plain"))
	assert.False(t, IsBinary("text/html"))
	assert.False(t, IsBinary("application/json"))
}

func TestFetchValidation(t *testing.T) {
	_, err := Fetch(context.Background(), "", nil)
	assert.ErrorContains(t, err, "url cannot be empty")

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err = Fetch(ctx, "http://127.0.0.1:9999", nil)
	assert.Error(t, err)
}

