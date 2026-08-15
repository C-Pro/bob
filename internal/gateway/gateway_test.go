package gateway

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"bob/internal/config"
	"bob/internal/llm"
	"bob/internal/models"

	"github.com/fasthttp/websocket"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestIsMentionedOrDM(t *testing.T) {
	tests := []struct {
		name          string
		handle        string
		chatID        string
		content       string
		wantProcessed bool
		wantPrompt    string
	}{
		{
			name:          "Townhall direct mention @bot",
			handle:        "@bot",
			chatID:        "townhall",
			content:       "Hello @bot, how are you?",
			wantProcessed: true,
			wantPrompt:    "Hello  how are you?",
		},
		{
			name:          "Townhall mention with colon",
			handle:        "@assistant",
			chatID:        "townhall",
			content:       "@assistant: explain quantum physics",
			wantProcessed: true,
			wantPrompt:    "explain quantum physics",
		},
		{
			name:          "Townhall without mention",
			handle:        "@bot",
			chatID:        "townhall",
			content:       "Hey everyone in townhall",
			wantProcessed: false,
			wantPrompt:    "",
		},
		{
			name:          "DM chat without mention",
			handle:        "@bot",
			chatID:        "dm_user1_user2",
			content:       "What is the weather?",
			wantProcessed: true,
			wantPrompt:    "What is the weather?",
		},
		{
			name:          "DM chat with mention",
			handle:        "@bot",
			chatID:        "dm_user1_user2",
			content:       "@bot tell me a joke",
			wantProcessed: true,
			wantPrompt:    "tell me a joke",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			processed, prompt := IsMentionedOrDM(tt.handle, tt.chatID, tt.content)
			assert.Equal(t, tt.wantProcessed, processed)
			assert.Equal(t, tt.wantPrompt, prompt)
		})
	}
}

func TestFormatResponse(t *testing.T) {
	longText := "Paragraph 1: Introduction.\n\nParagraph 2: Second section.\n\nParagraph 3: Third section.\n\nParagraph 4: Conclusion."

	// Townhall (limit 2)
	townhallRes := FormatResponse(longText, false, 2, 10)
	assert.Equal(t, "Paragraph 1: Introduction.\n\nParagraph 2: Second section.", townhallRes)

	// DM (limit 10)
	dmRes := FormatResponse(longText, true, 2, 10)
	assert.Equal(t, longText, dmRes)

	// Single paragraph
	shortText := "Single paragraph response."
	assert.Equal(t, shortText, FormatResponse(shortText, false, 2, 10))
}

func TestGatewayWebSocketIntegration(t *testing.T) {
	// 1. Mock LLM server
	llmServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := llm.CompletionResponse{
			Choices: []llm.Choice{
				{Index: 0, Message: llm.Message{Role: "assistant", Content: "Hello from LLM!\n\nParagraph 2 response."}},
			},
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer llmServer.Close()

	// 2. Mock Besedka WebSocket server
	upgrader := websocket.Upgrader{}
	receivedMsgs := make(chan models.ClientMessage, 10)

	besedkaServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "Bearer test-besedka-key", r.Header.Get("Authorization"))
		conn, err := upgrader.Upgrade(w, r, nil)
		require.NoError(t, err)
		defer func() { _ = conn.Close() }()

		// Send a test message mentioning @bot
		testServerMsg := models.ServerMessage{
			Type:   models.ServerMessageTypeMessages,
			ChatID: "townhall",
			Messages: []models.Message{
				{
					Seq:       1,
					Timestamp: time.Now().Unix(),
					ChatID:    "townhall",
					UserID:    "user-123",
					Content:   "Hello @bot test",
				},
			},
		}
		_ = conn.WriteJSON(testServerMsg)

		// Read bot's response
		for {
			var clientMsg models.ClientMessage
			if err := conn.ReadJSON(&clientMsg); err != nil {
				break
			}
			receivedMsgs <- clientMsg
		}
	}))
	defer besedkaServer.Close()

	cfg := &config.Config{
		BotHandle:             "@bot",
		BesedkaURL:            besedkaServer.URL,
		BesedkaAPIKey:         "test-besedka-key",
		GeminiAPIKey:          "test-gemini-key",
		GeminiModel:           "gemini-3.7-flash",
		GeminiBaseURL:         llmServer.URL,
		TownhallMaxParagraphs: 2,
		DMMaxParagraphs:       10,
	}

	llmClient := llm.NewClient(cfg, llmServer.Client())
	gw := NewGateway(cfg, llmClient)

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	err := gw.DialWebSocket(ctx)
	require.NoError(t, err)

	// Process message directly
	testMsg := models.Message{
		Seq:       1,
		Timestamp: time.Now().Unix(),
		ChatID:    "townhall",
		UserID:    "user-123",
		Content:   "Hello @bot test",
	}

	err = gw.ProcessMessage(ctx, testMsg)
	require.NoError(t, err)

	select {
	case msg := <-receivedMsgs:
		assert.Equal(t, models.ClientMessageTypeSend, msg.Type)
		assert.Equal(t, "townhall", msg.ChatID)
		assert.Equal(t, "Hello from LLM!\n\nParagraph 2 response.", msg.Content)
	case <-time.After(1 * time.Second):
		t.Fatal("timed out waiting for bot reply")
	}
}

func TestIgnoreSelfMessages(t *testing.T) {
	cfg := &config.Config{
		BotHandle: "@bot",
	}
	gw := NewGateway(cfg, nil)
	gw.botUserID = "bot-123"

	// Self-message should be ignored
	selfMsg := models.Message{
		UserID:    "bot-123",
		ChatID:    "dm_user_bot",
		Content:   "Hello user",
		Timestamp: time.Now().Unix(),
	}

	err := gw.ProcessMessage(context.Background(), selfMsg)
	require.NoError(t, err)
}
