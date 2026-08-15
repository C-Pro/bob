package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
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
			_ = json.NewEncoder(w).Encode(map[string]string{"id": "bot-user-id", "name": "bot"})
			return
		}
		conn, err := upgrader.Upgrade(w, r, nil)
		require.NoError(t, err)
		defer conn.Close()

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
