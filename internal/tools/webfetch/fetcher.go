package webfetch

import (
	"context"
	"errors"
	"fmt"
	"io"
	"mime"
	"net/http"
	"strings"
	"time"
)

// MaxContentSize is the maximum size (16KB) for fetched and returned text.
const MaxContentSize = 16 * 1024

// TruncationNotice is appended when content exceeds MaxContentSize.
const TruncationNotice = "\n\n[Content truncated to 16KB. Full download support will be added in a future update.]"

// FetchResult holds the result of a direct HTTP fetch.
type FetchResult struct {
	URL         string
	StatusCode  int
	ContentType string
	RawBody     []byte
	Truncated   bool
}

// Fetch executes a direct HTTP GET request to the target URL.
// It limits reading to MaxContentSize + 1 bytes to detect truncation without unbounded reading.
func Fetch(ctx context.Context, targetURL string, client *http.Client) (*FetchResult, error) {
	targetURL = strings.TrimSpace(targetURL)
	if targetURL == "" {
		return nil, errors.New("url cannot be empty")
	}

	if client == nil {
		client = &http.Client{
			Timeout: 15 * time.Second,
		}
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, targetURL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create http request: %w", err)
	}

	req.Header.Set("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")
	req.Header.Set("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,text/plain;q=0.8,*/*;q=0.5")

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("fetch error: %w", err)
	}
	defer func() {
		_ = resp.Body.Close()
	}()

	contentType := resp.Header.Get("Content-Type")
	mediaType, _, err := mime.ParseMediaType(contentType)
	if err != nil || mediaType == "" {
		mediaType = "text/plain"
	}

	limitReader := io.LimitReader(resp.Body, MaxContentSize+1)
	buf, err := io.ReadAll(limitReader)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	truncated := false
	if len(buf) > MaxContentSize {
		buf = buf[:MaxContentSize]
		truncated = true
	}

	finalURL := targetURL
	if resp.Request != nil && resp.Request.URL != nil {
		finalURL = resp.Request.URL.String()
	}

	return &FetchResult{
		URL:         finalURL,
		StatusCode:  resp.StatusCode,
		ContentType: mediaType,
		RawBody:     buf,
		Truncated:   truncated,
	}, nil
}

// IsHTML checks if the media type represents HTML content.
func IsHTML(mediaType string) bool {
	mt := strings.ToLower(strings.TrimSpace(mediaType))
	return mt == "text/html" || mt == "application/xhtml+xml"
}

// IsBinary checks if the media type is likely a binary file.
func IsBinary(mediaType string) bool {
	mt := strings.ToLower(strings.TrimSpace(mediaType))
	if strings.HasPrefix(mt, "image/") ||
		strings.HasPrefix(mt, "video/") ||
		strings.HasPrefix(mt, "audio/") ||
		mt == "application/pdf" ||
		mt == "application/zip" ||
		mt == "application/gzip" ||
		mt == "application/octet-stream" {
		return true
	}
	return false
}

// TruncateText truncates a string to MaxContentSize bytes and appends TruncationNotice if truncated.
func TruncateText(text string, wasTruncated bool) string {
	if len(text) > MaxContentSize {
		return text[:MaxContentSize] + TruncationNotice
	}
	if wasTruncated {
		return text + TruncationNotice
	}
	return text
}
