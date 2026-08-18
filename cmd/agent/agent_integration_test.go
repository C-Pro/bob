package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"bob/internal/config"
	"bob/internal/gateway"
	"bob/internal/llm"
	"bob/internal/models"

	"github.com/fasthttp/websocket"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestAgentIntegrationLoop(t *testing.T) {
	// Mock Gemini OpenAI completion server
	llmServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := llm.CompletionResponse{
			Choices: []llm.Choice{
				{Index: 0, Message: llm.Message{Role: "assistant", Content: "I am ready to help!"}},
			},
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer llmServer.Close()

	// Mock Besedka WebSocket server
	upgrader := websocket.Upgrader{}
	botReplies := make(chan models.ClientMessage, 10)

	besedkaServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/me" {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(models.User{ID: "bot-user-id", UserName: "bot", DisplayName: "Bot"})
			return
		}
		if r.URL.Path == "/api/users" {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode([]models.User{
				{ID: "human-user", DisplayName: "Human User", UserName: "human"},
			})
			return
		}
		if r.URL.Path == "/api/chats" {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode([]models.Chat{
				{ID: "townhall", Type: "townhall"},
			})
			return
		}
		if strings.HasPrefix(r.URL.Path, "/api/chats/") && strings.HasSuffix(r.URL.Path, "/messages") {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode([]models.Message{})
			return
		}

		conn, err := upgrader.Upgrade(w, r, nil)
		require.NoError(t, err)
		defer func() { _ = conn.Close() }()

		// Send mock mention message to agent
		_ = conn.WriteJSON(models.ServerMessage{
			Type:   models.ServerMessageTypeMessages,
			ChatID: "townhall",
			Messages: []models.Message{
				{
					Seq:       1,
					Timestamp: time.Now().Unix(),
					ChatID:    "townhall",
					UserID:    "human-user",
					Content:   "Hello @bot, show status",
				},
			},
		})

		// Read reply from agent
		for {
			var clientMsg models.ClientMessage
			if err := conn.ReadJSON(&clientMsg); err != nil {
				break
			}
			botReplies <- clientMsg
		}
	}))
	defer besedkaServer.Close()

	cfg := &config.Config{
		BotHandle:             "@bot",
		BesedkaURL:            besedkaServer.URL,
		BesedkaAPIKey:         "test-key",
		GeminiAPIKey:          "test-gemini-key",
		GeminiModel:           "gemini-3.7-flash",
		GeminiBaseURL:         llmServer.URL,
		TownhallMaxParagraphs: 2,
		DMMaxParagraphs:       10,
	}

	llmClient := llm.NewClient(cfg, llmServer.Client())
	gw := gateway.NewGateway(cfg, llmClient)

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	go func() {
		_ = gw.Start(ctx)
	}()

	select {
	case reply := <-botReplies:
		assert.Equal(t, models.ClientMessageTypeSend, reply.Type)
		assert.Equal(t, "townhall", reply.ChatID)
		assert.Equal(t, "I am ready to help!", reply.Content)
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for agent response in integration loop")
	}
}

func TestAgentIntegration_MultiTurnContextFlow(t *testing.T) {
	var capturedRequests []llm.CompletionRequest
	llmServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req llm.CompletionRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		capturedRequests = append(capturedRequests, req)

		resp := llm.CompletionResponse{
			Choices: []llm.Choice{
				{Index: 0, Message: llm.Message{Role: "assistant", Content: "Multi-turn reply"}},
			},
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer llmServer.Close()

	upgrader := websocket.Upgrader{}
	besedkaServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/me" {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(models.User{ID: "bot-id", UserName: "bot", DisplayName: "AI Bot"})
			return
		}
		if r.URL.Path == "/api/users" {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode([]models.User{
				{ID: "user-alice", DisplayName: "Alice", UserName: "alice"},
				{ID: "user-bob", DisplayName: "Bob", UserName: "bob"},
			})
			return
		}
		if r.URL.Path == "/api/chats" {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode([]models.Chat{
				{ID: "townhall", Type: "townhall"},
			})
			return
		}
		if strings.HasPrefix(r.URL.Path, "/api/chats/") && strings.HasSuffix(r.URL.Path, "/messages") {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode([]models.Message{
				{Seq: 1, ChatID: "townhall", UserID: "user-alice", Content: "Initial historical message", Timestamp: time.Now().Unix() - 100},
			})
			return
		}

		conn, err := upgrader.Upgrade(w, r, nil)
		require.NoError(t, err)
		defer func() { _ = conn.Close() }()

		for {
			var clientMsg models.ClientMessage
			if err := conn.ReadJSON(&clientMsg); err != nil {
				break
			}
		}
	}))
	defer besedkaServer.Close()

	cfg := &config.Config{
		BotHandle:             "@bot",
		BesedkaURL:            besedkaServer.URL,
		GeminiBaseURL:         llmServer.URL,
		TownhallMaxParagraphs: 2,
		DMMaxParagraphs:       10,
		MsgRingBufferSize:     100,
	}

	llmClient := llm.NewClient(cfg, llmServer.Client())
	gw := gateway.NewGateway(cfg, llmClient)

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	err := gw.DialWebSocket(ctx)
	require.NoError(t, err)

	err = gw.WarmupContext(ctx)
	require.NoError(t, err)

	// Turn 1: Bob speaks in Townhall (no mention)
	err = gw.ProcessMessage(ctx, models.Message{
		ChatID:    "townhall",
		UserID:    "user-bob",
		Content:   "Should we deploy to prod today?",
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)
	assert.Empty(t, capturedRequests)

	// Turn 2: Alice mentions bot
	err = gw.ProcessMessage(ctx, models.Message{
		ChatID:    "townhall",
		UserID:    "user-alice",
		Content:   "@bot what do you think?",
		Timestamp: time.Now().Unix() + 1,
	})
	require.NoError(t, err)
	require.Len(t, capturedRequests, 1)

	// Context includes: System Prompt, Historical Alice msg from warmup, Bob msg, Alice @bot msg
	messages := capturedRequests[0].Messages
	require.Len(t, messages, 4)
	assert.Equal(t, "system", messages[0].Role)
	assert.Contains(t, messages[0].Content, "AI Bot")
	assert.Equal(t, "Alice: Initial historical message", messages[1].Content)
	assert.Equal(t, "Bob: Should we deploy to prod today?", messages[2].Content)
	assert.Equal(t, "Alice: @bot what do you think?", messages[3].Content)
}

