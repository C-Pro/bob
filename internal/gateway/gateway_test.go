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
		var req llm.CompletionRequest
		_ = json.NewDecoder(r.Body).Decode(&req)

		// Verify multi-turn request structure
		assert.GreaterOrEqual(t, len(req.Messages), 2)
		assert.Equal(t, "system", req.Messages[0].Role)
		assert.Contains(t, req.Messages[0].Content, "Townhall")

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
		if r.URL.Path == "/api/me" {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(models.User{
				ID:          "bot-id-123",
				UserName:    "bot",
				DisplayName: "Bot Assistant",
			})
			return
		}
		if r.URL.Path == "/api/users" {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode([]models.User{
				{ID: "user-123", UserName: "alice", DisplayName: "Alice"},
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
		if r.URL.Path == "/api/chats/townhall/messages" {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode([]models.Message{})
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
		MsgRingBufferSize:     100,
	}

	llmClient := llm.NewClient(cfg, llmServer.Client())
	gw := NewGateway(cfg, llmClient)
	gw.httpClient = besedkaServer.Client()

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	err := gw.DialWebSocket(ctx)
	require.NoError(t, err)

	_ = gw.WarmupContext(ctx)

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

	// Verify ring buffer contains user message and assistant reply
	entries := gw.contextManager.GetOrCreate("townhall").Entries()
	require.Len(t, entries, 2)
	assert.Equal(t, "user", entries[0].Role)
	assert.Equal(t, "Alice", entries[0].SenderName)
	assert.Equal(t, "assistant", entries[1].Role)
	assert.Equal(t, "Hello from LLM!\n\nParagraph 2 response.", entries[1].Content)
}

func TestProcessMessage_SelfMessageHandling(t *testing.T) {
	cfg := &config.Config{
		BotHandle:         "@bot",
		MsgRingBufferSize: 10,
	}
	gw := NewGateway(cfg, nil)
	gw.botUserID = "bot-123"
	gw.botUser = models.User{ID: "bot-123", DisplayName: "Bot"}

	// 1. Self message in townhall
	selfTownhallMsg := models.Message{
		UserID:    "bot-123",
		ChatID:    "townhall",
		Content:   "Hello everyone from bot",
		Timestamp: time.Now().Unix(),
	}
	err := gw.ProcessMessage(context.Background(), selfTownhallMsg)
	require.NoError(t, err)

	entries := gw.contextManager.GetOrCreate("townhall").Entries()
	require.Len(t, entries, 1)
	assert.Equal(t, "assistant", entries[0].Role)
	assert.Equal(t, "Hello everyone from bot", entries[0].Content)

	// Duplicate self message should be ignored
	err = gw.ProcessMessage(context.Background(), selfTownhallMsg)
	require.NoError(t, err)
	assert.Equal(t, 1, gw.contextManager.GetOrCreate("townhall").Len())

	// 2. Self message in DM
	selfDMMsg := models.Message{
		UserID:    "bot-123",
		ChatID:    "dm_user_bot",
		Content:   "Hello from bot in DM",
		Timestamp: time.Now().Unix(),
	}
	err = gw.ProcessMessage(context.Background(), selfDMMsg)
	require.NoError(t, err)

	dmEntries := gw.contextManager.GetOrCreate("dm_user_bot").Entries()
	require.Len(t, dmEntries, 1)
	assert.Equal(t, "assistant", dmEntries[0].Role)
	assert.Equal(t, "Hello from bot in DM", dmEntries[0].Content)
}

func TestProcessMessage_TownhallContextAccumulation(t *testing.T) {
	var capturedMessages []llm.Message
	llmServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req llm.CompletionRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		capturedMessages = req.Messages

		resp := llm.CompletionResponse{
			Choices: []llm.Choice{
				{Index: 0, Message: llm.Message{Role: "assistant", Content: "I see what you are discussing!"}},
			},
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer llmServer.Close()

	// Mock WebSocket for egress
	upgrader := websocket.Upgrader{}
	wsServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		require.NoError(t, err)
		defer func() { _ = conn.Close() }()
		for {
			var m models.ClientMessage
			if err := conn.ReadJSON(&m); err != nil {
				break
			}
		}
	}))
	defer wsServer.Close()

	cfg := &config.Config{
		BotHandle:             "@bot",
		BesedkaURL:            wsServer.URL,
		GeminiBaseURL:         llmServer.URL,
		TownhallMaxParagraphs: 2,
		MsgRingBufferSize:     100,
	}

	llmClient := llm.NewClient(cfg, llmServer.Client())
	gw := NewGateway(cfg, llmClient)
	gw.botUserID = "bot-123"
	gw.botUser = models.User{ID: "bot-123", DisplayName: "Bot"}
	gw.userCache.Set(models.User{ID: "user-1", DisplayName: "Alice"})
	gw.userCache.Set(models.User{ID: "user-2", DisplayName: "Bob"})

	ctx := context.Background()
	err := gw.DialWebSocket(ctx)
	require.NoError(t, err)

	now := time.Now().Unix()

	// Message 1 from Alice (no mention -> should NOT trigger LLM)
	err = gw.ProcessMessage(ctx, models.Message{
		UserID:    "user-1",
		ChatID:    "townhall",
		Content:   "Hey Bob, did you check the new deployment?",
		Timestamp: now,
	})
	require.NoError(t, err)
	assert.Nil(t, capturedMessages)

	// Message 2 from Bob (no mention -> should NOT trigger LLM)
	err = gw.ProcessMessage(ctx, models.Message{
		UserID:    "user-2",
		ChatID:    "townhall",
		Content:   "Yes, it looks good!",
		Timestamp: now + 1,
	})
	require.NoError(t, err)
	assert.Nil(t, capturedMessages)

	// Message 3 from Alice (mentions @bot -> SHOULD trigger LLM with prior context!)
	err = gw.ProcessMessage(ctx, models.Message{
		UserID:    "user-1",
		ChatID:    "townhall",
		Content:   "@bot what are we talking about?",
		Timestamp: now + 2,
	})
	require.NoError(t, err)
	require.NotNil(t, capturedMessages)

	// Messages should be: System prompt, Alice msg, Bob msg, Alice @bot msg
	require.Len(t, capturedMessages, 4)
	assert.Equal(t, "system", capturedMessages[0].Role)
	assert.Contains(t, capturedMessages[0].Content, "Townhall")

	assert.Equal(t, "user", capturedMessages[1].Role)
	assert.Equal(t, "Alice: Hey Bob, did you check the new deployment?", capturedMessages[1].Content)

	assert.Equal(t, "user", capturedMessages[2].Role)
	assert.Equal(t, "Bob: Yes, it looks good!", capturedMessages[2].Content)

	assert.Equal(t, "user", capturedMessages[3].Role)
	assert.Equal(t, "Alice: @bot what are we talking about?", capturedMessages[3].Content)
}

func TestProcessMessage_DMChat(t *testing.T) {
	var capturedMessages []llm.Message
	llmServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req llm.CompletionRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		capturedMessages = req.Messages

		resp := llm.CompletionResponse{
			Choices: []llm.Choice{
				{Index: 0, Message: llm.Message{Role: "assistant", Content: "Hello Charlie in DM"}},
			},
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer llmServer.Close()

	upgrader := websocket.Upgrader{}
	wsServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		require.NoError(t, err)
		defer func() { _ = conn.Close() }()
		for {
			var m models.ClientMessage
			if err := conn.ReadJSON(&m); err != nil {
				break
			}
		}
	}))
	defer wsServer.Close()

	cfg := &config.Config{
		BotHandle:         "@bot",
		BesedkaURL:        wsServer.URL,
		GeminiBaseURL:     llmServer.URL,
		DMMaxParagraphs:   10,
		MsgRingBufferSize: 100,
	}

	llmClient := llm.NewClient(cfg, llmServer.Client())
	gw := NewGateway(cfg, llmClient)
	gw.botUserID = "bot-123"
	gw.botUser = models.User{ID: "bot-123", DisplayName: "Bot Assistant", UserName: "bot"}
	gw.userCache.Set(models.User{ID: "user-3", DisplayName: "Charlie Brown"})

	ctx := context.Background()
	err := gw.DialWebSocket(ctx)
	require.NoError(t, err)

	// Send message in DM (no mention required)
	err = gw.ProcessMessage(ctx, models.Message{
		UserID:    "user-3",
		ChatID:    "dm_user3_bot",
		Content:   "Can you help me with a Go question?",
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)
	require.NotNil(t, capturedMessages)

	// System prompt should contain Charlie Brown
	assert.Equal(t, "system", capturedMessages[0].Role)
	assert.Contains(t, capturedMessages[0].Content, "Charlie Brown")
	assert.Contains(t, capturedMessages[0].Content, "Bot Assistant")

	assert.Equal(t, "user", capturedMessages[1].Role)
	assert.Equal(t, "Charlie Brown: Can you help me with a Go question?", capturedMessages[1].Content)
}

func TestWarmupContext(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/me":
			_ = json.NewEncoder(w).Encode(models.User{ID: "bot-1", UserName: "bot", DisplayName: "Bot"})
		case "/api/users":
			_ = json.NewEncoder(w).Encode([]models.User{
				{ID: "u1", DisplayName: "Alice"},
			})
		case "/api/chats":
			_ = json.NewEncoder(w).Encode([]models.Chat{
				{ID: "townhall"},
				{ID: "dm_u1"},
			})
		case "/api/chats/townhall/messages":
			_ = json.NewEncoder(w).Encode([]models.Message{
				{Seq: 1, ChatID: "townhall", UserID: "u1", Content: "Hello world", Timestamp: 10},
				{Seq: 2, ChatID: "townhall", UserID: "bot-1", Content: "Hello Alice", Timestamp: 11},
			})
		case "/api/chats/dm_u1/messages":
			_ = json.NewEncoder(w).Encode([]models.Message{
				{Seq: 1, ChatID: "dm_u1", UserID: "u1", Content: "DM test", Timestamp: 12},
			})
		}
	}))
	defer server.Close()

	cfg := &config.Config{
		BesedkaURL:        server.URL,
		MsgRingBufferSize: 100,
	}
	gw := NewGateway(cfg, nil)
	gw.httpClient = server.Client()

	err := gw.WarmupContext(context.Background())
	require.NoError(t, err)

	thEntries := gw.contextManager.GetOrCreate("townhall").Entries()
	require.Len(t, thEntries, 2)
	assert.Equal(t, "user", thEntries[0].Role)
	assert.Equal(t, "Alice", thEntries[0].SenderName)
	assert.Equal(t, "Hello world", thEntries[0].Content)

	assert.Equal(t, "assistant", thEntries[1].Role)
	assert.Equal(t, "Hello Alice", thEntries[1].Content)

	dmEntries := gw.contextManager.GetOrCreate("dm_u1").Entries()
	require.Len(t, dmEntries, 1)
	assert.Equal(t, "user", dmEntries[0].Role)
	assert.Equal(t, "Alice", dmEntries[0].SenderName)
}
