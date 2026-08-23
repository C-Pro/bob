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

type mockToolExecutor struct {
	executeFunc func(ctx context.Context, name, argsJSON string) (string, error)
}

func (m *mockToolExecutor) Execute(ctx context.Context, name, argsJSON string) (string, error) {
	return m.executeFunc(ctx, name, argsJSON)
}

func TestGenerateChatResponseWithToolLoop_Success(t *testing.T) {
	var callCount int32

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req openai.ChatCompletionRequest
		err := json.NewDecoder(r.Body).Decode(&req)
		require.NoError(t, err)

		count := atomic.AddInt32(&callCount, 1)
		w.Header().Set("Content-Type", "application/json")

		if count == 1 {
			// First turn: LLM calls web_search
			resp := openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Index: 0,
						Message: openai.ChatCompletionMessage{
							Role: openai.ChatMessageRoleAssistant,
							ToolCalls: []openai.ToolCall{
								{
									ID:   "call_search_1",
									Type: openai.ToolTypeFunction,
									Function: openai.FunctionCall{
										Name:      "web_search",
										Arguments: `{"query":"golang 1.26"}`,
									},
								},
							},
						},
					},
				},
			}
			_ = json.NewEncoder(w).Encode(resp)
			return
		}

		// Second turn: Verify tool response message is included in request
		require.GreaterOrEqual(t, len(req.Messages), 3)
		lastMsg := req.Messages[len(req.Messages)-1]
		assert.Equal(t, openai.ChatMessageRoleTool, lastMsg.Role)
		assert.Equal(t, "call_search_1", lastMsg.ToolCallID)
		assert.Contains(t, lastMsg.Content, "Go 1.26 release notes")

		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{
					Index: 0,
					Message: openai.ChatCompletionMessage{
						Role:    openai.ChatMessageRoleAssistant,
						Content: "Go 1.26 has been released with many improvements.",
					},
				},
			},
		}
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
				Name: "web_search",
			},
		},
	}

	executor := &mockToolExecutor{
		executeFunc: func(ctx context.Context, name, argsJSON string) (string, error) {
			assert.Equal(t, "web_search", name)
			return `{"results": [{"title": "Go 1.26", "content": "Go 1.26 release notes"}]}`, nil
		},
	}

	reply, err := client.GenerateChatResponseWithToolLoop(
		context.Background(),
		[]openai.ChatCompletionMessage{{Role: openai.ChatMessageRoleUser, Content: "What is in Go 1.26?"}},
		tools,
		executor,
		5,
	)

	require.NoError(t, err)
	assert.Equal(t, "Go 1.26 has been released with many improvements.", reply)
	assert.Equal(t, int32(2), atomic.LoadInt32(&callCount))
}

func TestGenerateChatResponseWithToolLoop_MaxIterationsExceeded(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{
					Index: 0,
					Message: openai.ChatCompletionMessage{
						Role: openai.ChatMessageRoleAssistant,
						ToolCalls: []openai.ToolCall{
							{
								ID:   "call_loop",
								Type: openai.ToolTypeFunction,
								Function: openai.FunctionCall{
									Name:      "web_search",
									Arguments: `{"query":"loop"}`,
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
	executor := &mockToolExecutor{
		executeFunc: func(ctx context.Context, name, argsJSON string) (string, error) {
			return `{"results": []}`, nil
		},
	}

	_, err := client.GenerateChatResponseWithToolLoop(
		context.Background(),
		[]openai.ChatCompletionMessage{{Role: openai.ChatMessageRoleUser, Content: "Loop test"}},
		nil,
		executor,
		2,
	)

	require.Error(t, err)
	assert.Contains(t, err.Error(), "tool execution loop exceeded max iterations (2)")
}

func TestGenerateChatResponseWithToolLoop_GeminiThoughtSignaturePreservation(t *testing.T) {
	var turn int32

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		currentTurn := atomic.AddInt32(&turn, 1)
		w.Header().Set("Content-Type", "application/json")

		if currentTurn == 1 {
			// Turn 1: Return tool call with Gemini thought_signature in JSON
			rawResp := `{
				"id": "chatcmpl-gemini",
				"choices": [
					{
						"index": 0,
						"message": {
							"role": "assistant",
							"tool_calls": [
								{
									"id": "call_gemini_search",
									"type": "function",
									"function": {
										"name": "web_search",
										"arguments": "{\"query\":\"lombok fire\"}"
									},
									"thought_signature": "gemini_thought_sig_secret_xyz"
								}
							],
							"extra_content": {
								"google": {
									"thought_signature": "gemini_thought_sig_secret_xyz"
								}
							}
						}
					}
				]
			}`
			_, _ = w.Write([]byte(rawResp))
			return
		}

		// Turn 2: Verify outgoing JSON contains thought_signature
		var bodyMap map[string]interface{}
		err := json.NewDecoder(r.Body).Decode(&bodyMap)
		require.NoError(t, err)

		messages, ok := bodyMap["messages"].([]interface{})
		require.True(t, ok)
		require.GreaterOrEqual(t, len(messages), 3)

		// Find assistant message with tool calls
		var foundThoughtSig bool
		for _, mVal := range messages {
			m, ok := mVal.(map[string]interface{})
			if !ok || m["role"] != "assistant" {
				continue
			}
			toolCalls, ok := m["tool_calls"].([]interface{})
			if !ok || len(toolCalls) == 0 {
				continue
			}
			tc := toolCalls[0].(map[string]interface{})
			if tc["thought_signature"] == "gemini_thought_sig_secret_xyz" {
				foundThoughtSig = true
			}
		}
		assert.True(t, foundThoughtSig, "expected thought_signature to be re-injected into outgoing request")

		finalResp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{
					Index: 0,
					Message: openai.ChatCompletionMessage{
						Role:    openai.ChatMessageRoleAssistant,
						Content: "Here is the information about the fire in Lombok.",
					},
				},
			},
		}
		_ = json.NewEncoder(w).Encode(finalResp)
	}))
	defer server.Close()

	cfg := &config.Config{
		OpenAIAPIKey:  "test-key",
		OpenAIModel:   "gemini-3.7-flash",
		OpenAIBaseURL: server.URL,
	}

	client := NewClient(cfg, server.Client())
	executor := &mockToolExecutor{
		executeFunc: func(ctx context.Context, name, argsJSON string) (string, error) {
			return `{"results":[{"title":"Lombok Fire","content":"Details about the fire..."}]}`, nil
		},
	}

	reply, err := client.GenerateChatResponseWithToolLoop(
		context.Background(),
		[]openai.ChatCompletionMessage{{Role: openai.ChatMessageRoleUser, Content: "What happened with the fire in Lombok?"}},
		[]openai.Tool{{Type: openai.ToolTypeFunction, Function: &openai.FunctionDefinition{Name: "web_search"}}},
		executor,
		5,
	)

	require.NoError(t, err)
	assert.Equal(t, "Here is the information about the fire in Lombok.", reply)
	assert.Equal(t, int32(2), atomic.LoadInt32(&turn))
}


