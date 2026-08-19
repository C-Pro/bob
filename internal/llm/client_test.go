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

func TestGenerateResponseSuccess(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/chat/completions", r.URL.Path)
		assert.Equal(t, "Bearer test-api-key", r.Header.Get("Authorization"))
		assert.Equal(t, "application/json", r.Header.Get("Content-Type"))

		var req openai.ChatCompletionRequest
		err := json.NewDecoder(r.Body).Decode(&req)
		require.NoError(t, err)
		assert.Equal(t, "gpt-4o-mini", req.Model)
		require.Len(t, req.Messages, 2)
		assert.Equal(t, openai.ChatMessageRoleSystem, req.Messages[0].Role)
		assert.Equal(t, "You are a helpful assistant.", req.Messages[0].Content)
		assert.Equal(t, openai.ChatMessageRoleUser, req.Messages[1].Role)
		assert.Equal(t, "Hello", req.Messages[1].Content)

		resp := openai.ChatCompletionResponse{
			ID:    "chatcmpl-test",
			Model: req.Model,
			Choices: []openai.ChatCompletionChoice{
				{
					Index: 0,
					Message: openai.ChatCompletionMessage{
						Role:    openai.ChatMessageRoleAssistant,
						Content: "Hello from OpenAI!",
					},
				},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	cfg := &config.Config{
		OpenAIAPIKey:  "test-api-key",
		OpenAIModel:   "gpt-4o-mini",
		OpenAIBaseURL: server.URL,
	}

	client := NewClient(cfg, server.Client())
	reply, err := client.GenerateResponse(context.Background(), "You are a helpful assistant.", "Hello")
	require.NoError(t, err)
	assert.Equal(t, "Hello from OpenAI!", reply)
}

func TestGenerateResponseRetrySuccess(t *testing.T) {
	var attempts int32

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		count := atomic.AddInt32(&attempts, 1)
		if count < 3 {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusInternalServerError)
			_ = json.NewEncoder(w).Encode(openai.APIError{
				Message: "Internal server error",
				Code:    "500",
			})
			return
		}

		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{
					Index: 0,
					Message: openai.ChatCompletionMessage{
						Role:    openai.ChatMessageRoleAssistant,
						Content: "Recovered reply",
					},
				},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	cfg := &config.Config{
		OpenAIAPIKey:  "test-key",
		OpenAIModel:   "gpt-4o-mini",
		OpenAIBaseURL: server.URL,
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
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusServiceUnavailable)
		_ = json.NewEncoder(w).Encode(openai.APIError{
			Message: "Service Unavailable",
			Code:    "503",
		})
	}))
	defer server.Close()

	cfg := &config.Config{
		OpenAIAPIKey:  "test-key",
		OpenAIModel:   "gpt-4o-mini",
		OpenAIBaseURL: server.URL,
	}

	client := NewClient(cfg, server.Client())
	_, err := client.GenerateResponse(context.Background(), "", "Hello")
	assert.ErrorContains(t, err, "request failed after 3 retries")
	assert.Equal(t, int32(4), atomic.LoadInt32(&attempts))
}

func TestGenerateChatResponseMultiTurn(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req openai.ChatCompletionRequest
		err := json.NewDecoder(r.Body).Decode(&req)
		require.NoError(t, err)
		require.Len(t, req.Messages, 3)
		assert.Equal(t, openai.ChatMessageRoleSystem, req.Messages[0].Role)
		assert.Equal(t, openai.ChatMessageRoleUser, req.Messages[1].Role)
		assert.Equal(t, openai.ChatMessageRoleAssistant, req.Messages[2].Role)

		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{
					Index: 0,
					Message: openai.ChatCompletionMessage{
						Role:    openai.ChatMessageRoleAssistant,
						Content: "Turn 3 reply",
					},
				},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	cfg := &config.Config{
		OpenAIAPIKey:  "test-api-key",
		OpenAIModel:   "gpt-4o-mini",
		OpenAIBaseURL: server.URL,
	}

	client := NewClient(cfg, server.Client())
	reply, err := client.GenerateChatResponse(context.Background(), []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleSystem, Content: "sys"},
		{Role: openai.ChatMessageRoleUser, Content: "u1"},
		{Role: openai.ChatMessageRoleAssistant, Content: "a1"},
	})
	require.NoError(t, err)
	assert.Equal(t, "Turn 3 reply", reply)
}

func TestGenerateChatResponseWithTools(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req openai.ChatCompletionRequest
		err := json.NewDecoder(r.Body).Decode(&req)
		require.NoError(t, err)

		require.Len(t, req.Tools, 1)
		assert.Equal(t, openai.ToolTypeFunction, req.Tools[0].Type)
		assert.Equal(t, "get_weather", req.Tools[0].Function.Name)

		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{
					Index: 0,
					Message: openai.ChatCompletionMessage{
						Role: openai.ChatMessageRoleAssistant,
						ToolCalls: []openai.ToolCall{
							{
								ID:   "call_123",
								Type: openai.ToolTypeFunction,
								Function: openai.FunctionCall{
									Name:      "get_weather",
									Arguments: `{"location":"Tokyo"}`,
								},
							},
						},
					},
				},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	cfg := &config.Config{
		OpenAIAPIKey:  "test-key",
		OpenAIModel:   "gpt-4o-mini",
		OpenAIBaseURL: server.URL,
	}

	client := NewClient(cfg, server.Client())
	tools := []openai.Tool{
		{
			Type: openai.ToolTypeFunction,
			Function: &openai.FunctionDefinition{
				Name:        "get_weather",
				Description: "Get weather for a location",
				Parameters:  json.RawMessage(`{"type":"object","properties":{"location":{"type":"string"}}}`),
			},
		},
	}

	resp, err := client.GenerateChatResponseWithTools(context.Background(), []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleUser, Content: "What is the weather in Tokyo?"},
	}, tools)

	require.NoError(t, err)
	require.Len(t, resp.Choices, 1)
	require.Len(t, resp.Choices[0].Message.ToolCalls, 1)
	assert.Equal(t, "get_weather", resp.Choices[0].Message.ToolCalls[0].Function.Name)
	assert.Equal(t, `{"location":"Tokyo"}`, resp.Choices[0].Message.ToolCalls[0].Function.Arguments)
}
