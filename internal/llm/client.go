package llm

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

	"bob/internal/config"
)

// Message represents a chat completion message role and content.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// CompletionRequest is the request payload sent to the OpenAI-compatible endpoint.
type CompletionRequest struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
}

// Choice represents a completion response choice.
type Choice struct {
	Index        int     `json:"index"`
	Message      Message `json:"message"`
	FinishReason string  `json:"finish_reason"`
}

// CompletionResponse is the response structure returned by the endpoint.
type CompletionResponse struct {
	ID      string    `json:"id"`
	Object  string    `json:"object"`
	Created int64     `json:"created"`
	Model   string    `json:"model"`
	Choices []Choice  `json:"choices"`
	Error   *APIError `json:"error,omitempty"`
}

// APIError represents an error returned in an OpenAI API response.
type APIError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code"`
}

// Client wraps an HTTP client and configuration for LLM requests.
type Client struct {
	cfg        *config.Config
	httpClient *http.Client
	maxRetries int
	baseDelay  time.Duration
}

// NewClient creates a new LLM client.
func NewClient(cfg *config.Config, httpClient *http.Client) *Client {
	if httpClient == nil {
		httpClient = &http.Client{
			Timeout: 30 * time.Second,
		}
	}
	return &Client{
		cfg:        cfg,
		httpClient: httpClient,
		maxRetries: 3,
		baseDelay:  10 * time.Millisecond, // fast delay for unit tests/retries
	}
}

// GenerateResponse sends a chat completion request to Gemini API and returns the assistant's reply.
func (c *Client) GenerateResponse(ctx context.Context, systemPrompt, userMessage string) (string, error) {
	messages := make([]Message, 0, 2)
	if systemPrompt != "" {
		messages = append(messages, Message{Role: "system", Content: systemPrompt})
	}
	messages = append(messages, Message{Role: "user", Content: userMessage})

	reqPayload := CompletionRequest{
		Model:    c.cfg.GeminiModel,
		Messages: messages,
	}

	bodyBytes, err := json.Marshal(reqPayload)
	if err != nil {
		return "", fmt.Errorf("failed to marshal completion request: %w", err)
	}

	url := strings.TrimSuffix(c.cfg.GeminiBaseURL, "/") + "/chat/completions"

	var lastErr error
	for attempt := 0; attempt <= c.maxRetries; attempt++ {
		if attempt > 0 {
			delay := c.baseDelay * (1 << (attempt - 1))
			select {
			case <-ctx.Done():
				return "", ctx.Err()
			case <-time.After(delay):
			}
		}

		req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(bodyBytes))
		if err != nil {
			return "", fmt.Errorf("failed to create request: %w", err)
		}

		req.Header.Set("Content-Type", "application/json")
		if c.cfg.GeminiAPIKey != "" {
			req.Header.Set("Authorization", "Bearer "+c.cfg.GeminiAPIKey)
		}

		resp, err := c.httpClient.Do(req)
		if err != nil {
			lastErr = fmt.Errorf("network error on attempt %d: %w", attempt+1, err)
			continue
		}

		respBody, readErr := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		if readErr != nil {
			lastErr = fmt.Errorf("failed to read response body on attempt %d: %w", attempt+1, readErr)
			continue
		}

		if resp.StatusCode >= 500 || resp.StatusCode == http.StatusTooManyRequests {
			lastErr = fmt.Errorf("HTTP %d error on attempt %d: %s", resp.StatusCode, attempt+1, string(respBody))
			continue
		}

		if resp.StatusCode != http.StatusOK {
			return "", fmt.Errorf("API error (status %d): %s", resp.StatusCode, string(respBody))
		}

		var compResp CompletionResponse
		if err := json.Unmarshal(respBody, &compResp); err != nil {
			return "", fmt.Errorf("failed to parse completion response: %w", err)
		}

		if compResp.Error != nil {
			return "", fmt.Errorf("API error returned: %s", compResp.Error.Message)
		}

		if len(compResp.Choices) == 0 {
			return "", errors.New("no completion choices returned by API")
		}

		return compResp.Choices[0].Message.Content, nil
	}

	return "", fmt.Errorf("request failed after %d retries; last error: %w", c.maxRetries, lastErr)
}
