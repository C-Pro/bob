package llm

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"bob/internal/config"

	openai "github.com/sashabaranov/go-openai"
)

// Client wraps the official go-openai client and configuration for LLM requests.
type Client struct {
	cfg        *config.Config
	apiClient  *openai.Client
	maxRetries int
	baseDelay  time.Duration
}

// NewClient creates a new LLM client configured with standard OpenAI settings.
func NewClient(cfg *config.Config, httpClient *http.Client) *Client {
	clientConfig := openai.DefaultConfig(cfg.OpenAIAPIKey)
	if cfg.OpenAIBaseURL != "" {
		clientConfig.BaseURL = strings.TrimSuffix(cfg.OpenAIBaseURL, "/")
	}
	if httpClient != nil {
		clientConfig.HTTPClient = httpClient
	}

	return &Client{
		cfg:        cfg,
		apiClient:  openai.NewClientWithConfig(clientConfig),
		maxRetries: 3,
		baseDelay:  10 * time.Millisecond,
	}
}

// OpenAIClient returns the underlying *openai.Client instance.
func (c *Client) OpenAIClient() *openai.Client {
	return c.apiClient
}

// isRetryableError determines if an error from the OpenAI API is transient.
func isRetryableError(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return false
	}

	var apiErr *openai.APIError
	if errors.As(err, &apiErr) {
		if apiErr.HTTPStatusCode >= 500 || apiErr.HTTPStatusCode == http.StatusTooManyRequests {
			return true
		}
		// 4xx errors (except 429) are not retryable
		return false
	}

	var reqErr *openai.RequestError
	if errors.As(err, &reqErr) {
		if reqErr.HTTPStatusCode >= 500 || reqErr.HTTPStatusCode == http.StatusTooManyRequests {
			return true
		}
		return false
	}

	// Network / transport errors are retryable
	return true
}

// CreateChatCompletion sends a chat completion request with optional tool definitions and returns the full response.
func (c *Client) CreateChatCompletion(ctx context.Context, req openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error) {
	if req.Model == "" {
		req.Model = c.cfg.OpenAIModel
	}

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

		resp, err := c.apiClient.CreateChatCompletion(ctx, req)
		if err == nil {
			if len(resp.Choices) == 0 {
				return nil, errors.New("no completion choices returned by API")
			}
			return &resp, nil
		}

		lastErr = err
		if !isRetryableError(err) {
			return nil, fmt.Errorf("API error: %w", err)
		}
	}

	return nil, fmt.Errorf("request failed after %d retries; last error: %w", c.maxRetries, lastErr)
}

// GenerateChatResponse sends a multi-turn chat completion request and returns the assistant's reply text.
func (c *Client) GenerateChatResponse(ctx context.Context, messages []openai.ChatCompletionMessage) (string, error) {
	req := openai.ChatCompletionRequest{
		Model:    c.cfg.OpenAIModel,
		Messages: messages,
	}

	resp, err := c.CreateChatCompletion(ctx, req)
	if err != nil {
		return "", err
	}

	return resp.Choices[0].Message.Content, nil
}

// GenerateChatResponseWithTools sends a chat completion request with tools definitions support.
func (c *Client) GenerateChatResponseWithTools(ctx context.Context, messages []openai.ChatCompletionMessage, tools []openai.Tool) (*openai.ChatCompletionResponse, error) {
	req := openai.ChatCompletionRequest{
		Model:    c.cfg.OpenAIModel,
		Messages: messages,
		Tools:    tools,
	}

	return c.CreateChatCompletion(ctx, req)
}

// GenerateResponse sends a single-turn chat completion request and returns the assistant's reply.
func (c *Client) GenerateResponse(ctx context.Context, systemPrompt, userMessage string) (string, error) {
	messages := make([]openai.ChatCompletionMessage, 0, 2)
	if systemPrompt != "" {
		messages = append(messages, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleSystem,
			Content: systemPrompt,
		})
	}
	messages = append(messages, openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleUser,
		Content: userMessage,
	})
	return c.GenerateChatResponse(ctx, messages)
}
