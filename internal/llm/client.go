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

// TimeoutKind identifies whether a timeout occurred waiting for headers or the entire request.
type TimeoutKind int

const (
	// TimeoutKindFirstByte indicates the server did not return response headers within LLMFirstByteTimeout.
	TimeoutKindFirstByte TimeoutKind = iota
	// TimeoutKindRequest indicates the request exceeded LLMRequestTimeout.
	TimeoutKindRequest
)

// TimeoutError represents an LLM API request timeout with human-readable formatting.
type TimeoutError struct {
	Kind  TimeoutKind
	Limit time.Duration
	Cause error
}

func formatDuration(d time.Duration) string {
	if d <= 0 {
		return "0s"
	}
	if d%time.Hour == 0 {
		return fmt.Sprintf("%dh", d/time.Hour)
	}
	if d%time.Minute == 0 {
		return fmt.Sprintf("%dm", d/time.Minute)
	}
	if d%time.Second == 0 {
		return fmt.Sprintf("%ds", d/time.Second)
	}
	return d.String()
}

func (e *TimeoutError) Error() string {
	limitStr := formatDuration(e.Limit)
	switch e.Kind {
	case TimeoutKindFirstByte:
		return fmt.Sprintf("LLM API first-byte timeout (%s): server did not return response headers within timeout", limitStr)
	case TimeoutKindRequest:
		return fmt.Sprintf("LLM API timeout (%s): request exceeded deadline", limitStr)
	default:
		return fmt.Sprintf("LLM API timeout (%s)", limitStr)
	}
}

func (e *TimeoutError) Unwrap() error {
	return e.Cause
}

// Client wraps the official go-openai client and configuration for LLM requests.
type Client struct {
	cfg              *config.Config
	apiClient        *openai.Client
	maxRetries       int
	baseDelay        time.Duration
	requestTimeout   time.Duration
	firstByteTimeout time.Duration
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
		if err != nil {
			return nil, err
		}
		respBytes = g.extractThoughtSignaturesAndNormalize(respBytes)
		resp.Body = io.NopCloser(bytes.NewReader(respBytes))
	}

	return resp, nil
}

func (g *geminiTransport) extractThoughtSignaturesAndNormalize(data []byte) []byte {
	var respMap map[string]interface{}
	if err := json.Unmarshal(data, &respMap); err != nil {
		return data
	}

	choices, ok := respMap["choices"].([]interface{})
	if !ok {
		return data
	}

	modified := false
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

		// Normalize reasoning if reasoning_content is absent or empty, but thought/reasoning is present
		curReasoning, _ := msg["reasoning_content"].(string)
		if strings.TrimSpace(curReasoning) == "" {
			if thoughtStr, ok := msg["thought"].(string); ok && strings.TrimSpace(thoughtStr) != "" {
				msg["reasoning_content"] = thoughtStr
				modified = true
			} else if reasonStr, ok := msg["reasoning"].(string); ok && strings.TrimSpace(reasonStr) != "" {
				msg["reasoning_content"] = reasonStr
				modified = true
			}
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

	if modified {
		if newBytes, err := json.Marshal(respMap); err == nil {
			return newBytes
		}
	}
	return data
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

	requestTimeout := 10 * time.Minute
	if cfg != nil {
		if cfg.LLMRequestTimeout > 0 {
			requestTimeout = cfg.LLMRequestTimeout
		}
	} else if httpClient != nil && httpClient.Timeout > 0 {
		requestTimeout = httpClient.Timeout
	}

	firstByteTimeout := 1 * time.Minute
	if cfg != nil {
		if cfg.LLMFirstByteTimeout > 0 {
			firstByteTimeout = cfg.LLMFirstByteTimeout
		}
	}

	var transportToWrap http.RoundTripper
	if httpClient != nil && httpClient.Transport != nil {
		if t, ok := httpClient.Transport.(*http.Transport); ok {
			cloned := t.Clone()
			cloned.ResponseHeaderTimeout = firstByteTimeout
			transportToWrap = cloned
		} else {
			transportToWrap = httpClient.Transport
		}
	} else {
		defaultTransport, ok := http.DefaultTransport.(*http.Transport)
		if ok {
			cloned := defaultTransport.Clone()
			cloned.ResponseHeaderTimeout = firstByteTimeout
			transportToWrap = cloned
		} else {
			transportToWrap = http.DefaultTransport
		}
	}

	var geminiWrappedClient *http.Client
	if httpClient != nil {
		cCopy := *httpClient
		cCopy.Transport = newGeminiTransport(transportToWrap)
		cCopy.Timeout = requestTimeout
		geminiWrappedClient = &cCopy
	} else {
		geminiWrappedClient = &http.Client{
			Transport: newGeminiTransport(transportToWrap),
			Timeout:   requestTimeout,
		}
	}
	clientConfig.HTTPClient = geminiWrappedClient

	return &Client{
		cfg:              cfg,
		apiClient:        openai.NewClientWithConfig(clientConfig),
		maxRetries:       3,
		baseDelay:        10 * time.Millisecond,
		requestTimeout:   requestTimeout,
		firstByteTimeout: firstByteTimeout,
	}
}

// OpenAIClient returns the underlying *openai.Client instance.
func (c *Client) OpenAIClient() *openai.Client {
	return c.apiClient
}

// classifyTimeout checks if an error represents an LLM API timeout and returns a *TimeoutError.
// If the caller's context was canceled or exceeded its deadline, it returns ctx.Err() directly.
func (c *Client) classifyTimeout(ctx context.Context, err error) error {
	if err == nil {
		return nil
	}

	if ctx.Err() != nil {
		return ctx.Err()
	}

	var te *TimeoutError
	if errors.As(err, &te) {
		return err
	}

	// Provider HTTP response errors (4xx, 5xx) must not be classified as local client timeouts,
	// even if the provider's response body contains "context deadline exceeded".
	var apiErr *openai.APIError
	if errors.As(err, &apiErr) {
		return nil
	}
	var reqErr *openai.RequestError
	if errors.As(err, &reqErr) && reqErr.HTTPStatusCode > 0 {
		return nil
	}

	errStr := err.Error()

	if strings.Contains(errStr, "timeout awaiting response headers") {
		limit := c.firstByteTimeout
		if limit <= 0 && c.cfg != nil {
			limit = c.cfg.LLMFirstByteTimeout
		}
		if limit <= 0 {
			limit = 1 * time.Minute
		}
		return &TimeoutError{
			Kind:  TimeoutKindFirstByte,
			Limit: limit,
			Cause: err,
		}
	}

	if errors.Is(err, context.DeadlineExceeded) ||
		strings.Contains(errStr, "Client.Timeout exceeded") ||
		strings.Contains(errStr, "context deadline exceeded") {
		limit := c.requestTimeout
		if limit <= 0 && c.cfg != nil {
			limit = c.cfg.LLMRequestTimeout
		}
		if limit <= 0 {
			limit = 10 * time.Minute
		}
		return &TimeoutError{
			Kind:  TimeoutKindRequest,
			Limit: limit,
			Cause: err,
		}
	}

	return nil
}

// formatAPIError transforms low-level or opaque errors into clear human-readable messages.
func (c *Client) formatAPIError(ctx context.Context, err error) error {
	if err == nil {
		return nil
	}
	if ctx.Err() != nil {
		return ctx.Err()
	}
	if tErr := c.classifyTimeout(ctx, err); tErr != nil {
		return tErr
	}
	var te *TimeoutError
	if errors.As(err, &te) {
		return err
	}
	errStr := err.Error()
	if strings.HasPrefix(errStr, "LLM API") {
		return err
	}
	return fmt.Errorf("LLM API error: %w", err)
}

// isRetryableError determines if an error from the OpenAI API is transient.
func isRetryableError(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, context.Canceled) {
		return false
	}

	var te *TimeoutError
	if errors.As(err, &te) {
		if te.Kind == TimeoutKindRequest {
			return false
		}
		if te.Kind == TimeoutKindFirstByte {
			return true
		}
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

	// Network / transport errors (including request timeouts when parent context is still active) are retryable
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
			slog.Warn("llm request failed, retrying", "attempt", attempt, "max_retries", c.maxRetries, "delay", delay, "error", lastErr)
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

		if ctx.Err() != nil {
			return nil, ctx.Err()
		}

		if tErr := c.classifyTimeout(ctx, err); tErr != nil {
			err = tErr
		}

		var timeoutErr *TimeoutError
		if errors.As(err, &timeoutErr) {
			if timeoutErr.Kind == TimeoutKindRequest {
				return nil, timeoutErr
			}
			if timeoutErr.Kind == TimeoutKindFirstByte && attempt >= 1 {
				return nil, timeoutErr
			}
		}

		lastErr = err
		if !isRetryableError(err) {
			return nil, c.formatAPIError(ctx, err)
		}
	}

	var timeoutErr *TimeoutError
	if errors.As(lastErr, &timeoutErr) {
		return nil, timeoutErr
	}
	return nil, fmt.Errorf("request failed after %d retries; last error: %w", c.maxRetries, c.formatAPIError(ctx, lastErr))
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
		Content: "You have reached the tool execution limit. Please synthesize and provide the best possible response based on all information gathered so far, including original markdown links to sources found in the search results, without calling any more tools. If any requested actions, scripts, or files could not be completed or executed due to the tool limit, state clearly what was accomplished and what remains to be run; do not claim files were created if they were not.",
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
