package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strconv"
	"strings"
	"sync"
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

// geminiTransport preserves thought_signatures and extra_content metadata required by Google Gemini models during multi-turn tool calling.
type geminiTransport struct {
	base       http.RoundTripper
	mu         sync.RWMutex
	signatures map[string]string
}

func newGeminiTransport(base http.RoundTripper) *geminiTransport {
	if base == nil {
		base = http.DefaultTransport
	}
	return &geminiTransport{
		base:       base,
		signatures: make(map[string]string),
	}
}

func (g *geminiTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	if req.Body != nil && strings.HasSuffix(req.URL.Path, "/chat/completions") {
		bodyBytes, err := io.ReadAll(req.Body)
		_ = req.Body.Close()
		if err == nil {
			var bodyMap map[string]interface{}
			if err := json.Unmarshal(bodyBytes, &bodyMap); err == nil {
				if g.injectThoughtSignatures(bodyMap) {
					if newBytes, err := json.Marshal(bodyMap); err == nil {
						bodyBytes = newBytes
					}
				}
			}
			req.Body = io.NopCloser(bytes.NewReader(bodyBytes))
			req.ContentLength = int64(len(bodyBytes))
			req.Header.Set("Content-Length", strconv.Itoa(len(bodyBytes)))
		}
	}

	resp, err := g.base.RoundTrip(req)
	if err != nil {
		return nil, err
	}

	if resp.Body != nil && resp.StatusCode == http.StatusOK && strings.HasSuffix(req.URL.Path, "/chat/completions") {
		respBytes, err := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		if err == nil {
			g.extractThoughtSignatures(respBytes)
			resp.Body = io.NopCloser(bytes.NewReader(respBytes))
		}
	}

	return resp, nil
}

func (g *geminiTransport) extractThoughtSignatures(data []byte) {
	var respMap map[string]interface{}
	if err := json.Unmarshal(data, &respMap); err != nil {
		return
	}

	choices, ok := respMap["choices"].([]interface{})
	if !ok {
		return
	}

	g.mu.Lock()
	defer g.mu.Unlock()

	for _, choiceVal := range choices {
		choice, ok := choiceVal.(map[string]interface{})
		if !ok {
			continue
		}
		msg, ok := choice["message"].(map[string]interface{})
		if !ok {
			continue
		}

		var msgSig string
		if sig, ok := msg["thought_signature"].(string); ok && sig != "" {
			msgSig = sig
		}
		if extra, ok := msg["extra_content"].(map[string]interface{}); ok {
			if google, ok := extra["google"].(map[string]interface{}); ok {
				if sig, ok := google["thought_signature"].(string); ok && sig != "" {
					msgSig = sig
				}
			}
		}

		toolCalls, ok := msg["tool_calls"].([]interface{})
		if !ok {
			continue
		}

		for _, tcVal := range toolCalls {
			tc, ok := tcVal.(map[string]interface{})
			if !ok {
				continue
			}
			tcID, _ := tc["id"].(string)
			fn, _ := tc["function"].(map[string]interface{})
			fnName, _ := fn["name"].(string)

			tcSig := msgSig
			if sig, ok := tc["thought_signature"].(string); ok && sig != "" {
				tcSig = sig
			}
			if extra, ok := tc["extra_content"].(map[string]interface{}); ok {
				if google, ok := extra["google"].(map[string]interface{}); ok {
					if sig, ok := google["thought_signature"].(string); ok && sig != "" {
						tcSig = sig
					}
				}
			}

			if tcSig != "" {
				if tcID != "" {
					g.signatures[tcID] = tcSig
				}
				if fnName != "" {
					g.signatures[fnName] = tcSig
				}
			}
		}
	}
}

func (g *geminiTransport) injectThoughtSignatures(bodyMap map[string]interface{}) bool {
	messages, ok := bodyMap["messages"].([]interface{})
	if !ok {
		return false
	}

	g.mu.RLock()
	defer g.mu.RUnlock()

	modified := false
	for _, msgVal := range messages {
		msg, ok := msgVal.(map[string]interface{})
		if !ok {
			continue
		}
		role, _ := msg["role"].(string)
		if role != "assistant" {
			continue
		}

		toolCalls, ok := msg["tool_calls"].([]interface{})
		if !ok || len(toolCalls) == 0 {
			continue
		}

		var lastSig string
		for _, tcVal := range toolCalls {
			tc, ok := tcVal.(map[string]interface{})
			if !ok {
				continue
			}

			tcID, _ := tc["id"].(string)
			fn, _ := tc["function"].(map[string]interface{})
			fnName, _ := fn["name"].(string)

			if _, hasSig := tc["thought_signature"]; hasSig {
				continue
			}

			sig := g.signatures[tcID]
			if sig == "" {
				sig = g.signatures[fnName]
			}
			if sig == "" {
				sig = "skip_thought_signature_validator"
			}

			tc["thought_signature"] = sig
			tc["extra_content"] = map[string]interface{}{
				"google": map[string]interface{}{
					"thought_signature": sig,
				},
			}
			lastSig = sig
			modified = true
		}

		if lastSig != "" {
			if _, hasExtra := msg["extra_content"]; !hasExtra {
				msg["extra_content"] = map[string]interface{}{
					"google": map[string]interface{}{
						"thought_signature": lastSig,
					},
				}
				modified = true
			}
		}
	}

	return modified
}

// NewClient creates a new LLM client configured with standard OpenAI settings.
func NewClient(cfg *config.Config, httpClient *http.Client) *Client {
	clientConfig := openai.DefaultConfig(cfg.OpenAIAPIKey)
	if cfg.OpenAIBaseURL != "" {
		clientConfig.BaseURL = strings.TrimSuffix(cfg.OpenAIBaseURL, "/")
	}

	var baseTransport http.RoundTripper
	clientTimeout := 120 * time.Second
	if httpClient != nil {
		baseTransport = httpClient.Transport
		if httpClient.Timeout > 0 {
			clientTimeout = httpClient.Timeout
		}
	}
	if baseTransport == nil {
		baseTransport = http.DefaultTransport
	}

	geminiWrappedClient := &http.Client{
		Transport: newGeminiTransport(baseTransport),
		Timeout:   clientTimeout,
	}
	clientConfig.HTTPClient = geminiWrappedClient

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

// ToolExecutor defines an interface for executing function/tool calls returned by the LLM.
type ToolExecutor interface {
	Execute(ctx context.Context, name string, argsJSON string) (string, error)
}

// GenerateChatResponseWithToolLoop sends chat completion requests iteratively, executing any requested tool calls until a final text response is returned.
func (c *Client) GenerateChatResponseWithToolLoop(
	ctx context.Context,
	messages []openai.ChatCompletionMessage,
	tools []openai.Tool,
	executor ToolExecutor,
	maxIterations int,
) (string, error) {
	if maxIterations <= 0 {
		maxIterations = 5
	}

	currentMessages := make([]openai.ChatCompletionMessage, len(messages))
	copy(currentMessages, messages)

	for iter := 0; iter < maxIterations; iter++ {
		req := openai.ChatCompletionRequest{
			Model:    c.cfg.OpenAIModel,
			Messages: currentMessages,
			Tools:    tools,
		}

		resp, err := c.CreateChatCompletion(ctx, req)
		if err != nil {
			return "", err
		}

		choice := resp.Choices[0]
		assistantMsg := choice.Message

		// If no tool calls were requested, return the assistant's content
		if len(assistantMsg.ToolCalls) == 0 {
			return assistantMsg.Content, nil
		}

		// Append assistant message with tool calls to conversation history
		currentMessages = append(currentMessages, assistantMsg)

		// Execute each tool call and append the result as a tool role message
		for _, toolCall := range assistantMsg.ToolCalls {
			slog.Info("LLM requested tool execution", "tool", toolCall.Function.Name, "args", toolCall.Function.Arguments)
			var toolResult string
			if executor == nil {
				toolResult = `{"error": "tool execution is not configured"}`
			} else {
				res, execErr := executor.Execute(ctx, toolCall.Function.Name, toolCall.Function.Arguments)
				if execErr != nil {
					toolResult = fmt.Sprintf(`{"error": %q}`, execErr.Error())
				} else {
					toolResult = res
				}
			}

			currentMessages = append(currentMessages, openai.ChatCompletionMessage{
				Role:       openai.ChatMessageRoleTool,
				Content:    toolResult,
				ToolCallID: toolCall.ID,
			})
		}
	}

	// Graceful synthesis step: when maxIterations is reached, request a final completion with tools disabled
	currentMessages = append(currentMessages, openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleUser,
		Content: "You have reached the tool execution limit. Please synthesize and provide the best possible response based on all information gathered so far, including original markdown links to sources found in the search results, without calling any more tools.",
	})

	finalReq := openai.ChatCompletionRequest{
		Model:    c.cfg.OpenAIModel,
		Messages: currentMessages,
		Tools:    nil,
	}

	finalResp, err := c.CreateChatCompletion(ctx, finalReq)
	if err != nil {
		return "", fmt.Errorf("failed to generate final synthesis response after tool iterations: %w", err)
	}

	return finalResp.Choices[0].Message.Content, nil
}

