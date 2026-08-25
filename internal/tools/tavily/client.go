package tavily

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// Client is a lightweight HTTP client for the Tavily Search API.
type Client struct {
	apiKey     string
	baseURL    string
	httpClient *http.Client
	maxRetries int
	baseDelay  time.Duration
}

// NewClient creates a new Tavily API client.
func NewClient(apiKey, baseURL string, httpClient *http.Client) *Client {
	if baseURL == "" {
		baseURL = "https://api.tavily.com"
	}
	baseURL = strings.TrimSuffix(baseURL, "/")

	if httpClient == nil {
		httpClient = &http.Client{Timeout: 15 * time.Second}
	}

	return &Client{
		apiKey:     apiKey,
		baseURL:    baseURL,
		httpClient: httpClient,
		maxRetries: 3,
		baseDelay:  10 * time.Millisecond,
	}
}

// isRetryableStatus checks if an HTTP response status code indicates a temporary failure.
func isRetryableStatus(statusCode int) bool {
	return statusCode == http.StatusTooManyRequests || statusCode >= 500
}

// Search executes a web search request against the Tavily API.
func (c *Client) Search(ctx context.Context, req SearchRequest) (*SearchResponse, error) {
	apiKey := req.APIKey
	if apiKey == "" {
		apiKey = c.apiKey
	}
	if strings.TrimSpace(apiKey) == "" {
		return nil, errors.New("tavily API key is required")
	}

	req.Query = strings.TrimSpace(req.Query)
	if req.Query == "" {
		return nil, errors.New("search query cannot be empty")
	}

	// Apply default values
	if req.SearchDepth == "" {
		req.SearchDepth = "basic"
	}
	if req.Topic == "" {
		req.Topic = "general"
	}
	if req.MaxResults <= 0 {
		req.MaxResults = 5
	} else if req.MaxResults > 20 {
		req.MaxResults = 20
	}
	if !req.IncludeAnswer {
		// Include direct answer by default unless explicitly disabled
		req.IncludeAnswer = true
	}

	// Ensure API key is passed in the request body for compatibility
	req.APIKey = apiKey

	bodyBytes, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to encode search request: %w", err)
	}

	searchURL := c.baseURL + "/search"

	var lastErr error
	for attempt := 0; attempt <= c.maxRetries; attempt++ {
		if attempt > 0 {
			delay := c.baseDelay * (1 << (attempt - 1))
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(delay):
			}
		}

		httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, searchURL, bytes.NewReader(bodyBytes))
		if err != nil {
			return nil, fmt.Errorf("failed to create http request: %w", err)
		}

		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("Authorization", "Bearer "+apiKey)

		resp, err := c.httpClient.Do(httpReq)
		if err != nil {
			if errors.Is(ctx.Err(), context.Canceled) || errors.Is(ctx.Err(), context.DeadlineExceeded) {
				return nil, ctx.Err()
			}
			lastErr = err
			continue
		}

		respBody, readErr := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		if readErr != nil {
			lastErr = fmt.Errorf("failed to read response body: %w", readErr)
			continue
		}

		if resp.StatusCode == http.StatusOK {
			var searchResp SearchResponse
			if err := json.Unmarshal(respBody, &searchResp); err != nil {
				return nil, fmt.Errorf("failed to decode search response: %w", err)
			}
			return &searchResp, nil
		}

		lastErr = fmt.Errorf("tavily API returned status %d: %s", resp.StatusCode, strings.TrimSpace(string(respBody)))
		if !isRetryableStatus(resp.StatusCode) {
			return nil, lastErr
		}
	}

	return nil, fmt.Errorf("tavily API request failed after %d retries; last error: %w", c.maxRetries, lastErr)
}

// Extract executes a URL content extraction request against the Tavily API (/extract).
func (c *Client) Extract(ctx context.Context, urls ...string) (*ExtractResponse, error) {
	if strings.TrimSpace(c.apiKey) == "" {
		return nil, errors.New("tavily API key is required")
	}

	var validURLs []string
	for _, u := range urls {
		trimmed := strings.TrimSpace(u)
		if trimmed != "" {
			validURLs = append(validURLs, trimmed)
		}
	}
	if len(validURLs) == 0 {
		return nil, errors.New("urls cannot be empty")
	}

	req := ExtractRequest{
		APIKey: c.apiKey,
		URLs:   validURLs,
	}

	bodyBytes, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to encode extract request: %w", err)
	}

	extractURL := c.baseURL + "/extract"

	var lastErr error
	for attempt := 0; attempt <= c.maxRetries; attempt++ {
		if attempt > 0 {
			delay := c.baseDelay * (1 << (attempt - 1))
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(delay):
			}
		}

		httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, extractURL, bytes.NewReader(bodyBytes))
		if err != nil {
			return nil, fmt.Errorf("failed to create http request: %w", err)
		}

		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)

		resp, err := c.httpClient.Do(httpReq)
		if err != nil {
			if errors.Is(ctx.Err(), context.Canceled) || errors.Is(ctx.Err(), context.DeadlineExceeded) {
				return nil, ctx.Err()
			}
			lastErr = err
			continue
		}

		respBody, readErr := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		if readErr != nil {
			lastErr = fmt.Errorf("failed to read response body: %w", readErr)
			continue
		}

		if resp.StatusCode == http.StatusOK {
			var extractResp ExtractResponse
			if err := json.Unmarshal(respBody, &extractResp); err != nil {
				return nil, fmt.Errorf("failed to decode extract response: %w", err)
			}
			return &extractResp, nil
		}

		lastErr = fmt.Errorf("tavily API returned status %d: %s", resp.StatusCode, strings.TrimSpace(string(respBody)))
		if !isRetryableStatus(resp.StatusCode) {
			return nil, lastErr
		}
	}

	return nil, fmt.Errorf("tavily API request failed after %d retries; last error: %w", c.maxRetries, lastErr)
}

