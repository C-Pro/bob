package gateway

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"bob/internal/chatcontext"
	"bob/internal/config"
	"bob/internal/llm"
	"bob/internal/memory"
	"bob/internal/models"
	"bob/internal/tools/tavily"

	"github.com/fasthttp/websocket"
	openai "github.com/sashabaranov/go-openai"
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
	assert.Equal(t, "Paragraph 1: Introduction.\n\nParagraph 2: Second section.\n\nParagraph 3: Third section.\n\nParagraph 4: Conclusion.", dmRes)

	// Single paragraph
	shortText := "Single paragraph response."
	assert.Equal(t, shortText, FormatResponse(shortText, false, 2, 10))
}

func TestGatewayWebSocketIntegration(t *testing.T) {
	// 1. Mock LLM server
	llmServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req openai.ChatCompletionRequest
		_ = json.NewDecoder(r.Body).Decode(&req)

		// Verify multi-turn request structure
		assert.GreaterOrEqual(t, len(req.Messages), 2)
		assert.Equal(t, openai.ChatMessageRoleSystem, req.Messages[0].Role)
		assert.Contains(t, req.Messages[0].Content, "Townhall")

		resp := openai.ChatCompletionResponse{
			ID:    "chatcmpl-test",
			Model: req.Model,
			Choices: []openai.ChatCompletionChoice{
				{
					Index: 0,
					Message: openai.ChatCompletionMessage{
						Role:    openai.ChatMessageRoleAssistant,
						Content: "Hello from LLM!\n\nParagraph 2 response.",
					},
				},
			},
		}
		w.Header().Set("Content-Type", "application/json")
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
		OpenAIAPIKey:          "test-openai-key",
		OpenAIModel:           "gemini-3.7-flash",
		OpenAIBaseURL:         llmServer.URL,
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

	// Duplicate self message in HTML format (e.g. broadcast from server) should be deduplicated
	echoMsg := models.Message{
		UserID:    "bot-123",
		ChatID:    "townhall",
		Content:   "<p>Hello everyone from bot</p>",
		Timestamp: time.Now().Unix(),
	}
	err = gw.ProcessMessage(context.Background(), echoMsg)
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

func TestExtractMessageText(t *testing.T) {
	// RawContent priority
	msg1 := models.Message{
		Content:    "<p>Paragraph 1</p><p>Paragraph 2</p>",
		RawContent: "Paragraph 1\n\nParagraph 2",
	}
	assert.Equal(t, "Paragraph 1\n\nParagraph 2", ExtractMessageText(msg1))

	// HTML parsing with line breaks and entities
	msg2 := models.Message{
		Content: "<p>Hello &amp; welcome!<br>Next line &lt;3</p><p>Second paragraph &#39;test&#39; &quot;quote&quot;.</p>",
	}
	expected := "Hello & welcome!\nNext line <3\n\nSecond paragraph 'test' \"quote\"."
	assert.Equal(t, expected, ExtractMessageText(msg2))
}

func TestProcessMessage_TownhallContextAccumulation(t *testing.T) {
	var capturedMessages []openai.ChatCompletionMessage
	llmServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req openai.ChatCompletionRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		capturedMessages = req.Messages

		resp := openai.ChatCompletionResponse{
			ID:    "chatcmpl-test",
			Model: req.Model,
			Choices: []openai.ChatCompletionChoice{
				{
					Index: 0,
					Message: openai.ChatCompletionMessage{
						Role:    openai.ChatMessageRoleAssistant,
						Content: "I see what you are discussing!",
					},
				},
			},
		}
		w.Header().Set("Content-Type", "application/json")
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
		OpenAIBaseURL:         llmServer.URL,
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
	assert.Equal(t, openai.ChatMessageRoleSystem, capturedMessages[0].Role)
	assert.Contains(t, capturedMessages[0].Content, "Townhall")

	assert.Equal(t, openai.ChatMessageRoleUser, capturedMessages[1].Role)
	assert.Equal(t, "Alice: Hey Bob, did you check the new deployment?", capturedMessages[1].Content)

	assert.Equal(t, openai.ChatMessageRoleUser, capturedMessages[2].Role)
	assert.Equal(t, "Bob: Yes, it looks good!", capturedMessages[2].Content)

	assert.Equal(t, openai.ChatMessageRoleUser, capturedMessages[3].Role)
	assert.Equal(t, "Alice: @bot what are we talking about?", capturedMessages[3].Content)
}

func TestProcessMessage_DMChat(t *testing.T) {
	var capturedMessages []openai.ChatCompletionMessage
	llmServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req openai.ChatCompletionRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		capturedMessages = req.Messages

		resp := openai.ChatCompletionResponse{
			ID:    "chatcmpl-test",
			Model: req.Model,
			Choices: []openai.ChatCompletionChoice{
				{
					Index: 0,
					Message: openai.ChatCompletionMessage{
						Role:    openai.ChatMessageRoleAssistant,
						Content: "Hello Charlie in DM",
					},
				},
			},
		}
		w.Header().Set("Content-Type", "application/json")
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
		OpenAIBaseURL:     llmServer.URL,
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
	assert.Equal(t, openai.ChatMessageRoleSystem, capturedMessages[0].Role)
	assert.Contains(t, capturedMessages[0].Content, "Charlie Brown")
	assert.Contains(t, capturedMessages[0].Content, "Bot Assistant")

	assert.Equal(t, openai.ChatMessageRoleUser, capturedMessages[1].Role)
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
				{ID: "townhall", LastSeq: 2},
				{ID: "dm_u1", LastSeq: 1},
			})
		case "/api/chats/townhall/messages":
			assert.Equal(t, "1", r.URL.Query().Get("fromSeq"))
			assert.Equal(t, "2", r.URL.Query().Get("toSeq"))
			_ = json.NewEncoder(w).Encode([]models.Message{
				{Seq: 1, ChatID: "townhall", UserID: "u1", Content: "<p>Hello world</p>", Timestamp: 10},
				{Seq: 2, ChatID: "townhall", UserID: "bot-1", Content: "<p>Hello Alice</p>", Timestamp: 11},
			})
		case "/api/chats/dm_u1/messages":
			assert.Equal(t, "1", r.URL.Query().Get("fromSeq"))
			assert.Equal(t, "1", r.URL.Query().Get("toSeq"))
			_ = json.NewEncoder(w).Encode([]models.Message{
				{Seq: 1, ChatID: "dm_u1", UserID: "u1", Content: "<p>DM test</p>", Timestamp: 12},
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

func TestProcessMessage_OnDemandBackfill(t *testing.T) {
	var capturedMessages []openai.ChatCompletionMessage
	llmServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req openai.ChatCompletionRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		capturedMessages = req.Messages

		resp := openai.ChatCompletionResponse{
			ID:    "chatcmpl-test",
			Model: req.Model,
			Choices: []openai.ChatCompletionChoice{
				{
					Index: 0,
					Message: openai.ChatCompletionMessage{
						Role:    openai.ChatMessageRoleAssistant,
						Content: "I recall our previous discussion about fins!",
					},
				},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer llmServer.Close()

	upgrader := websocket.Upgrader{}
	besedkaServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/chats/dm_user_bot/messages" {
			assert.Equal(t, "1", r.URL.Query().Get("fromSeq"))
			assert.Equal(t, "12", r.URL.Query().Get("toSeq"))
			_ = json.NewEncoder(w).Encode([]models.Message{
				{Seq: 1, ChatID: "dm_user_bot", UserID: "user-1", Content: "Where to buy fins in Kuta?", Timestamp: 100},
				{Seq: 2, ChatID: "dm_user_bot", UserID: "bot-1", Content: "Aquamaster on Sunset Road", Timestamp: 101},
			})
			return
		}

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
	defer besedkaServer.Close()

	cfg := &config.Config{
		BotHandle:         "@bot",
		BesedkaURL:        besedkaServer.URL,
		OpenAIBaseURL:     llmServer.URL,
		DMMaxParagraphs:   10,
		MsgRingBufferSize: 100,
	}

	llmClient := llm.NewClient(cfg, llmServer.Client())
	gw := NewGateway(cfg, llmClient)
	gw.httpClient = besedkaServer.Client()
	gw.botUserID = "bot-1"
	gw.botUser = models.User{ID: "bot-1", DisplayName: "Agent Bob", UserName: "bob"}
	gw.userCache.Set(models.User{ID: "user-1", DisplayName: "C-Pro", UserName: "cpro"})

	ctx := context.Background()
	err := gw.DialWebSocket(ctx)
	require.NoError(t, err)

	// Note: We deliberately DO NOT call WarmupContext to test on-demand backfill!
	// Message arriving with Seq 13 should trigger on-demand backfill for seq 1..12!
	err = gw.ProcessMessage(ctx, models.Message{
		Seq:       13,
		ChatID:    "dm_user_bot",
		UserID:    "user-1",
		Content:   "Do you remember what we talked about?",
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)
	require.NotNil(t, capturedMessages)

	// capturedMessages should contain: System prompt, Seq 1, Seq 2, Seq 13!
	require.Len(t, capturedMessages, 4)
	assert.Equal(t, openai.ChatMessageRoleSystem, capturedMessages[0].Role)
	assert.Equal(t, openai.ChatMessageRoleUser, capturedMessages[1].Role)
	assert.Equal(t, "C-Pro: Where to buy fins in Kuta?", capturedMessages[1].Content)
	assert.Equal(t, openai.ChatMessageRoleAssistant, capturedMessages[2].Role)
	assert.Equal(t, "Aquamaster on Sunset Road", capturedMessages[2].Content)
	assert.Equal(t, openai.ChatMessageRoleUser, capturedMessages[3].Role)
	assert.Equal(t, "C-Pro: Do you remember what we talked about?", capturedMessages[3].Content)
}

func TestSendLocation(t *testing.T) {
	upgrader := websocket.Upgrader{}
	receivedMsgs := make(chan models.ClientMessage, 10)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		require.NoError(t, err)
		defer func() { _ = conn.Close() }()

		for {
			var msg models.ClientMessage
			if err := conn.ReadJSON(&msg); err != nil {
				break
			}
			receivedMsgs <- msg
		}
	}))
	defer server.Close()

	cfg := &config.Config{
		BesedkaURL: server.URL,
	}
	gw := NewGateway(cfg, nil)

	// Sending before connection should fail
	err := gw.SendLocation(&models.Location{Lat: 10, Lng: 20})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "not established")

	// Sending nil location should error
	err = gw.SendLocation(nil)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "location is nil")

	// Dial WebSocket
	err = gw.DialWebSocket(context.Background())
	require.NoError(t, err)
	defer gw.Stop()

	// Send valid location
	targetLoc := &models.Location{
		Lat: 37.7749,
		Lng: -122.4194,
	}
	err = gw.SendLocation(targetLoc)
	require.NoError(t, err)

	select {
	case msg := <-receivedMsgs:
		assert.Equal(t, models.ClientMessageTypeLocation, msg.Type)
		require.NotNil(t, msg.Location)
		assert.Equal(t, 37.7749, msg.Location.Lat)
		assert.Equal(t, -122.4194, msg.Location.Lng)
	case <-time.After(1 * time.Second):
		t.Fatal("timed out waiting for location message")
	}
}

func TestGateway_LocationFrameReporting_InitialAndPeriodic(t *testing.T) {
	upgrader := websocket.Upgrader{}
	receivedFrames := make(chan models.ClientMessage, 10)

	besedkaServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/me" {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(models.User{ID: "bot-1", UserName: "bot"})
			return
		}
		if r.URL.Path == "/api/users" {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode([]models.User{})
			return
		}
		if r.URL.Path == "/api/chats" {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode([]models.Chat{})
			return
		}
		if strings.HasPrefix(r.URL.Path, "/api/chats/") {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode([]models.Message{})
			return
		}

		if r.URL.Path == "/api/chat" {
			conn, err := upgrader.Upgrade(w, r, nil)
			require.NoError(t, err)
			defer func() { _ = conn.Close() }()

			for {
				var msg models.ClientMessage
				if err := conn.ReadJSON(&msg); err != nil {
					break
				}
				receivedFrames <- msg
			}
		}
	}))
	defer besedkaServer.Close()

	cfg := &config.Config{
		BesedkaURL: besedkaServer.URL,
	}
	gw := NewGateway(cfg, nil)
	gw.httpClient = besedkaServer.Client()

	loc := &models.Location{
		Lat: -8.4095,
		Lng: 115.1889,
	}
	gw.SetLocation(loc)
	assert.Equal(t, loc, gw.Location())

	// Set short delay and interval for rapid testing
	gw.SetInitialLocationDelay(20 * time.Millisecond)
	gw.SetLocationInterval(60 * time.Millisecond)

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	go func() {
		_ = gw.Start(ctx)
	}()

	// 1. Initial location frame soon after connect
	select {
	case frame := <-receivedFrames:
		assert.Equal(t, models.ClientMessageTypeLocation, frame.Type)
		require.NotNil(t, frame.Location)
		assert.Equal(t, -8.4095, frame.Location.Lat)
		assert.Equal(t, 115.1889, frame.Location.Lng)
	case <-time.After(500 * time.Millisecond):
		t.Fatal("timed out waiting for initial location frame")
	}

	// 2. Subsequent periodic location frame
	select {
	case frame := <-receivedFrames:
		assert.Equal(t, models.ClientMessageTypeLocation, frame.Type)
		require.NotNil(t, frame.Location)
		assert.Equal(t, -8.4095, frame.Location.Lat)
		assert.Equal(t, 115.1889, frame.Location.Lng)
	case <-time.After(500 * time.Millisecond):
		t.Fatal("timed out waiting for periodic location frame")
	}

	gw.Stop()
}

func TestGateway_LocationFrameReporting_NilLocation(t *testing.T) {
	upgrader := websocket.Upgrader{}
	receivedFrames := make(chan models.ClientMessage, 10)

	besedkaServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/me" {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(models.User{ID: "bot-1", UserName: "bot"})
			return
		}
		if r.URL.Path == "/api/users" {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode([]models.User{})
			return
		}
		if r.URL.Path == "/api/chats" {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode([]models.Chat{})
			return
		}
		if strings.HasPrefix(r.URL.Path, "/api/chats/") {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode([]models.Message{})
			return
		}

		if r.URL.Path == "/api/chat" {
			conn, err := upgrader.Upgrade(w, r, nil)
			require.NoError(t, err)
			defer func() { _ = conn.Close() }()

			for {
				var msg models.ClientMessage
				if err := conn.ReadJSON(&msg); err != nil {
					break
				}
				receivedFrames <- msg
			}
		}
	}))
	defer besedkaServer.Close()

	cfg := &config.Config{
		BesedkaURL: besedkaServer.URL,
	}
	gw := NewGateway(cfg, nil)
	gw.httpClient = besedkaServer.Client()
	gw.SetLocation(nil)
	gw.SetInitialLocationDelay(20 * time.Millisecond)
	gw.SetLocationInterval(50 * time.Millisecond)

	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()

	go func() {
		_ = gw.Start(ctx)
	}()

	select {
	case frame := <-receivedFrames:
		t.Fatalf("unexpected frame received when location is nil: %+v", frame)
	case <-time.After(150 * time.Millisecond):
		// Expected: no frames sent
	}

	gw.Stop()
}

func TestProcessMessageWithWebSearchTool(t *testing.T) {
	upgrader := websocket.Upgrader{}
	sentMessages := make(chan models.ClientMessage, 10)

	// 1. Mock Besedka server
	besedkaServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/me" {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(models.User{ID: "bot-id", UserName: "bot", DisplayName: "BotAssistant"})
			return
		}
		if r.URL.Path == "/api/users" {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode([]models.User{
				{ID: "user-1", UserName: "alice", DisplayName: "Alice"},
			})
			return
		}
		if r.URL.Path == "/api/chats" {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode([]models.Chat{})
			return
		}
		if r.URL.Path == "/api/chat" {
			conn, err := upgrader.Upgrade(w, r, nil)
			require.NoError(t, err)
			defer func() { _ = conn.Close() }()

			for {
				var msg models.ClientMessage
				if err := conn.ReadJSON(&msg); err != nil {
					break
				}
				sentMessages <- msg
			}
		}
	}))
	defer besedkaServer.Close()

	// 2. Mock Tavily Search Server
	tavilyServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/search", r.URL.Path)
		resp := tavily.SearchResponse{
			Query:  "golang 1.26 features",
			Answer: "Go 1.26 adds enhanced tool calling support.",
			Results: []tavily.SearchResult{
				{
					Title:   "Go 1.26 Release",
					URL:     "https://go.dev/blog/go1.26",
					Content: "Go 1.26 improves compiler and runtime performance.",
				},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer tavilyServer.Close()

	// 3. Mock OpenAI LLM Server
	var llmCallCount int32
	llmServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		count := atomic.AddInt32(&llmCallCount, 1)
		w.Header().Set("Content-Type", "application/json")

		if count == 1 {
			var req openai.ChatCompletionRequest
			_ = json.NewDecoder(r.Body).Decode(&req)
			require.NotEmpty(t, req.Tools)
			assert.Equal(t, "web_search", req.Tools[0].Function.Name)

			// LLM decides to call web_search
			resp := openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Index: 0,
						Message: openai.ChatCompletionMessage{
							Role: openai.ChatMessageRoleAssistant,
							ToolCalls: []openai.ToolCall{
								{
									ID:   "call_tavily_1",
									Type: openai.ToolTypeFunction,
									Function: openai.FunctionCall{
										Name:      "web_search",
										Arguments: `{"query":"golang 1.26 features"}`,
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

		// Second turn: returns final text
		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{
					Index: 0,
					Message: openai.ChatCompletionMessage{
						Role:    openai.ChatMessageRoleAssistant,
						Content: "Go 1.26 adds enhanced tool calling support.",
					},
				},
			},
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer llmServer.Close()

	cfg := &config.Config{
		BotHandle:             "@bot",
		BesedkaURL:            besedkaServer.URL,
		OpenAIAPIKey:          "test-key",
		OpenAIModel:           "gpt-4o-mini",
		OpenAIBaseURL:         llmServer.URL,
		TavilyAPIKey:          "tavily-test-key",
		TavilyBaseURL:         tavilyServer.URL,
		TownhallMaxParagraphs: 2,
		DMMaxParagraphs:       10,
		MsgRingBufferSize:     10,
	}

	llmClient := llm.NewClient(cfg, llmServer.Client())
	gw := NewGateway(cfg, llmClient)
	gw.httpClient = besedkaServer.Client()

	// Connect websocket
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	err := gw.DialWebSocket(ctx)
	require.NoError(t, err)

	// Pre-fetch bot user metadata
	_, err = gw.FetchBotUser(ctx)
	require.NoError(t, err)

	incomingMsg := models.Message{
		Seq:       1,
		ChatID:    "townhall",
		UserID:    "user-1",
		Content:   "@bot what are the golang 1.26 features?",
		Timestamp: time.Now().Unix(),
	}

	err = gw.ProcessMessage(ctx, incomingMsg)
	require.NoError(t, err)

	// Verify reply sent to Besedka
	select {
	case reply := <-sentMessages:
		assert.Equal(t, models.ClientMessageTypeSend, reply.Type)
		assert.Equal(t, "townhall", reply.ChatID)
		assert.Equal(t, "Go 1.26 adds enhanced tool calling support.", reply.Content)
	case <-time.After(1 * time.Second):
		t.Fatal("timed out waiting for bot response")
	}

	// Verify ring buffer contains user turn and final assistant turn
	entries := gw.contextManager.GetOrCreate("townhall").Entries()
	require.Len(t, entries, 2)
	assert.Equal(t, "user", entries[0].Role)
	assert.Equal(t, "@bot what are the golang 1.26 features?", entries[0].Content)
	assert.Equal(t, "assistant", entries[1].Role)
	assert.Equal(t, "Go 1.26 adds enhanced tool calling support.", entries[1].Content)

	gw.Stop()
}

func TestProcessMessage_WebFetchToolIntegration(t *testing.T) {
	// 1. Mock Target Website
	targetServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		html := `<!DOCTYPE html><html><head><title>Go 1.26 Release Notes</title></head><body><article><h1>Go 1.26 Release Notes</h1><p>Go 1.26 includes major improvements to compiler optimization and tool call integration.</p></article></body></html>`
		_, _ = w.Write([]byte(html))
	}))
	defer targetServer.Close()

	// 2. Mock Besedka Server
	upgrader := websocket.Upgrader{}
	sentMessages := make(chan models.ClientMessage, 10)
	besedkaServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/me":
			_ = json.NewEncoder(w).Encode(models.User{
				ID:          "bot-1",
				DisplayName: "Bob",
				UserName:    "bot",
			})
		case "/api/users":
			_ = json.NewEncoder(w).Encode([]models.User{
				{ID: "bot-1", DisplayName: "Bob", UserName: "bot"},
				{ID: "user-1", DisplayName: "Alice", UserName: "alice"},
			})
		case "/api/chats":
			_ = json.NewEncoder(w).Encode([]models.Chat{
				{ID: "townhall", Name: "Townhall"},
			})
		case "/api/chats/townhall/messages":
			_ = json.NewEncoder(w).Encode([]models.Message{})
		case "/api/chat":
			conn, err := upgrader.Upgrade(w, r, nil)
			require.NoError(t, err)
			defer func() { _ = conn.Close() }()
			for {
				var msg models.ClientMessage
				if err := conn.ReadJSON(&msg); err != nil {
					break
				}
				sentMessages <- msg
			}
		}
	}))
	defer besedkaServer.Close()

	// 3. Mock OpenAI LLM Server
	var llmCallCount int32
	llmServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		count := atomic.AddInt32(&llmCallCount, 1)
		w.Header().Set("Content-Type", "application/json")

		if count == 1 {
			var req openai.ChatCompletionRequest
			_ = json.NewDecoder(r.Body).Decode(&req)
			require.NotEmpty(t, req.Tools)

			// LLM decides to call web_fetch
			resp := openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Index: 0,
						Message: openai.ChatCompletionMessage{
							Role: openai.ChatMessageRoleAssistant,
							ToolCalls: []openai.ToolCall{
								{
									ID:   "call_fetch_1",
									Type: openai.ToolTypeFunction,
									Function: openai.FunctionCall{
										Name:      "web_fetch",
										Arguments: fmt.Sprintf(`{"url":%q,"mode":"auto"}`, targetServer.URL),
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

		// Second turn: returns final text
		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{
					Index: 0,
					Message: openai.ChatCompletionMessage{
						Role:    openai.ChatMessageRoleAssistant,
						Content: "According to the release notes, Go 1.26 includes major compiler improvements.",
					},
				},
			},
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer llmServer.Close()

	cfg := &config.Config{
		BotHandle:             "@bot",
		BesedkaURL:            besedkaServer.URL,
		OpenAIAPIKey:          "test-key",
		OpenAIModel:           "gemini-3.7-flash",
		OpenAIBaseURL:         llmServer.URL,
		TownhallMaxParagraphs: 2,
		DMMaxParagraphs:       10,
		MsgRingBufferSize:     10,
	}

	llmClient := llm.NewClient(cfg, llmServer.Client())
	gw := NewGateway(cfg, llmClient)
	gw.httpClient = besedkaServer.Client()

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	err := gw.DialWebSocket(ctx)
	require.NoError(t, err)

	_, err = gw.FetchBotUser(ctx)
	require.NoError(t, err)

	incomingMsg := models.Message{
		Seq:       1,
		ChatID:    "townhall",
		UserID:    "user-1",
		Content:   fmt.Sprintf("@bot fetch %s for me", targetServer.URL),
		Timestamp: time.Now().Unix(),
	}

	err = gw.ProcessMessage(ctx, incomingMsg)
	require.NoError(t, err)

	select {
	case reply := <-sentMessages:
		assert.Equal(t, models.ClientMessageTypeSend, reply.Type)
		assert.Equal(t, "townhall", reply.ChatID)
		assert.Equal(t, "According to the release notes, Go 1.26 includes major compiler improvements.", reply.Content)
	case <-time.After(1 * time.Second):
		t.Fatal("timed out waiting for bot response")
	}

	gw.Stop()
}

func TestProcessMessage_ImageAttachment(t *testing.T) {
	fakeImageBytes := []byte("fake-png-binary-data")
	fakeImageB64 := "ZmFrZS1wbmctYmluYXJ5LWRhdGE="

	var receivedMultiContent []openai.ChatMessagePart
	var receivedRole string

	llmServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req openai.ChatCompletionRequest
		_ = json.NewDecoder(r.Body).Decode(&req)

		// Check the user message
		for _, msg := range req.Messages {
			if msg.Role == openai.ChatMessageRoleUser && len(msg.MultiContent) > 0 {
				receivedRole = msg.Role
				receivedMultiContent = msg.MultiContent
			}
		}

		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{
					Message: openai.ChatCompletionMessage{
						Role:    openai.ChatMessageRoleAssistant,
						Content: "I analyzed the image.",
					},
				},
			},
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer llmServer.Close()

	sentMessages := make(chan models.ClientMessage, 10)
	besedkaServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/api/images/img-123") {
			assert.Equal(t, "1", r.URL.Query().Get("thumb"))
			w.Header().Set("Content-Type", "image/png")
			_, _ = w.Write(fakeImageBytes)
			return
		}
		if r.URL.Path == "/api/me" {
			_ = json.NewEncoder(w).Encode(models.User{ID: "bot-1", DisplayName: "Bob", UserName: "bot"})
			return
		}
		if r.URL.Path == "/api/users" {
			_ = json.NewEncoder(w).Encode([]models.User{{ID: "u1", DisplayName: "Alice"}})
			return
		}
		if r.URL.Path == "/api/chats" {
			_ = json.NewEncoder(w).Encode([]models.Chat{{ID: "townhall", LastSeq: 0}})
			return
		}
		if r.URL.Path == "/api/chat" {
			upgrader := websocket.Upgrader{}
			c, err := upgrader.Upgrade(w, r, nil)
			if err != nil {
				return
			}
			defer func() { _ = c.Close() }()
			for {
				var cm models.ClientMessage
				if err := c.ReadJSON(&cm); err != nil {
					return
				}
				sentMessages <- cm
			}
		}
	}))
	defer besedkaServer.Close()

	cfg := &config.Config{
		BesedkaURL:            besedkaServer.URL,
		BesedkaAPIKey:         "test-key",
		OpenAIAPIKey:          "test-key",
		OpenAIBaseURL:         llmServer.URL,
		OpenAIModel:           "test-model",
		BotHandle:             "@bot",
		TownhallMaxParagraphs: 2,
		DMMaxParagraphs:       10,
		MsgRingBufferSize:     10,
	}

	llmClient := llm.NewClient(cfg, llmServer.Client())
	gw := NewGateway(cfg, llmClient)
	gw.httpClient = besedkaServer.Client()

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	require.NoError(t, gw.DialWebSocket(ctx))
	_, err := gw.FetchBotUser(ctx)
	require.NoError(t, err)

	incomingMsg := models.Message{
		Seq:       1,
		ChatID:    "townhall",
		UserID:    "u1",
		Content:   "@bot what is in this picture?",
		Timestamp: time.Now().Unix(),
		Attachments: []models.Attachment{
			{
				Type:     models.AttachmentTypeImage,
				Name:     "photo.png",
				MimeType: "image/png",
				FileID:   "img-123",
			},
		},
	}

	err = gw.ProcessMessage(ctx, incomingMsg)
	require.NoError(t, err)

	select {
	case reply := <-sentMessages:
		assert.Equal(t, "townhall", reply.ChatID)
		assert.Equal(t, "I analyzed the image.", reply.Content)
	case <-time.After(1 * time.Second):
		t.Fatal("timed out waiting for bot reply")
	}

	// Verify LLM received multimodal content
	assert.Equal(t, openai.ChatMessageRoleUser, receivedRole)
	require.Len(t, receivedMultiContent, 2)
	assert.Equal(t, openai.ChatMessagePartTypeText, receivedMultiContent[0].Type)
	assert.Equal(t, "Alice: @bot what is in this picture?", receivedMultiContent[0].Text)
	assert.Equal(t, openai.ChatMessagePartTypeImageURL, receivedMultiContent[1].Type)
	assert.Equal(t, "data:image/png;base64,"+fakeImageB64, receivedMultiContent[1].ImageURL.URL)

	// Verify ring buffer entry
	entries := gw.contextManager.GetOrCreate("townhall").Entries()
	require.Len(t, entries, 2) // user + assistant
	assert.Len(t, entries[0].Images, 1)
	assert.Equal(t, "data:image/png;base64,"+fakeImageB64, entries[0].Images[0].URL)

	gw.Stop()
}

func TestProcessMessage_TextAndBinaryAttachments(t *testing.T) {
	sentMessages := make(chan models.ClientMessage, 10)
	besedkaServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/files/f-text" {
			w.Header().Set("Content-Type", "text/plain")
			_, _ = w.Write([]byte("config_key=val123"))
			return
		}
		if r.URL.Path == "/api/files/f-error" {
			http.Error(w, "server error", http.StatusInternalServerError)
			return
		}
		if r.URL.Path == "/api/me" {
			_ = json.NewEncoder(w).Encode(models.User{ID: "bot-1", DisplayName: "Bob", UserName: "bot"})
			return
		}
		if r.URL.Path == "/api/users" {
			_ = json.NewEncoder(w).Encode([]models.User{{ID: "u1", DisplayName: "Alice"}})
			return
		}
		if r.URL.Path == "/api/chat" {
			upgrader := websocket.Upgrader{}
			c, err := upgrader.Upgrade(w, r, nil)
			if err != nil {
				return
			}
			defer func() { _ = c.Close() }()
			for {
				var cm models.ClientMessage
				if err := c.ReadJSON(&cm); err != nil {
					return
				}
				sentMessages <- cm
			}
		}
	}))
	defer besedkaServer.Close()

	llmServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req openai.ChatCompletionRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{
					Message: openai.ChatCompletionMessage{
						Role:    openai.ChatMessageRoleAssistant,
						Content: "Config parsed successfully.",
					},
				},
			},
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer llmServer.Close()

	cfg := &config.Config{
		BesedkaURL:            besedkaServer.URL,
		BesedkaAPIKey:         "test-key",
		OpenAIAPIKey:          "test-key",
		OpenAIBaseURL:         llmServer.URL,
		OpenAIModel:           "test-model",
		BotHandle:             "@bot",
		TownhallMaxParagraphs: 2,
		DMMaxParagraphs:       10,
		MsgRingBufferSize:     10,
	}

	llmClient := llm.NewClient(cfg, llmServer.Client())
	gw := NewGateway(cfg, llmClient)
	gw.httpClient = besedkaServer.Client()

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	require.NoError(t, gw.DialWebSocket(ctx))
	_, err := gw.FetchBotUser(ctx)
	require.NoError(t, err)

	incomingMsg := models.Message{
		Seq:       1,
		ChatID:    "dm_user1",
		UserID:    "u1",
		Content:   "Please check these files",
		Timestamp: time.Now().Unix(),
		Attachments: []models.Attachment{
			{
				Type:     models.AttachmentTypeFile,
				Name:     "settings.conf",
				MimeType: "text/plain",
				FileID:   "f-text",
			},
			{
				Type:     models.AttachmentTypeFile,
				Name:     "data.zip",
				MimeType: "application/zip",
				FileID:   "f-bin",
			},
			{
				Type:     models.AttachmentTypeFile,
				Name:     "missing.txt",
				MimeType: "text/plain",
				FileID:   "f-error",
			},
		},
	}

	err = gw.ProcessMessage(ctx, incomingMsg)
	require.NoError(t, err)

	select {
	case reply := <-sentMessages:
		assert.Equal(t, "dm_user1", reply.ChatID)
		assert.Equal(t, "Config parsed successfully.", reply.Content)
	case <-time.After(1 * time.Second):
		t.Fatal("timed out waiting for bot reply")
	}

	entries := gw.contextManager.GetOrCreate("dm_user1").Entries()
	require.Len(t, entries, 2)
	assert.Contains(t, entries[0].Content, "[Attachment settings.conf]:\n```\nconfig_key=val123\n```")
	assert.Contains(t, entries[0].Content, "[Attachment: data.zip (application/zip, not displayed)]")
	assert.Contains(t, entries[0].Content, "[Attachment: missing.txt (failed to download)]")

	gw.Stop()
}

func TestWarmupChat_WithAttachments(t *testing.T) {
	fakeImageBytes := []byte("thumb-bytes")
	besedkaServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/api/images/img-warmup") {
			w.Header().Set("Content-Type", "image/jpeg")
			_, _ = w.Write(fakeImageBytes)
			return
		}
		if r.URL.Path == "/api/files/f-warmup" {
			w.Header().Set("Content-Type", "text/plain")
			_, _ = w.Write([]byte("historical log line"))
			return
		}
		if r.URL.Path == "/api/chats/townhall/messages" {
			_ = json.NewEncoder(w).Encode([]models.Message{
				{
					Seq:       1,
					ChatID:    "townhall",
					UserID:    "u1",
					Content:   "First with image",
					Timestamp: 100,
					Attachments: []models.Attachment{
						{Type: models.AttachmentTypeImage, Name: "shot.jpg", MimeType: "image/jpeg", FileID: "img-warmup"},
					},
				},
				{
					Seq:       2,
					ChatID:    "townhall",
					UserID:    "u2",
					Content:   "Second with log",
					Timestamp: 101,
					Attachments: []models.Attachment{
						{Type: models.AttachmentTypeFile, Name: "app.log", MimeType: "text/plain", FileID: "f-warmup"},
					},
				},
			})
			return
		}
	}))
	defer besedkaServer.Close()

	cfg := &config.Config{
		BesedkaURL:        besedkaServer.URL,
		MsgRingBufferSize: 10,
	}
	gw := NewGateway(cfg, nil)
	gw.httpClient = besedkaServer.Client()

	gw.WarmupChat(context.Background(), "townhall", 2)

	entries := gw.contextManager.GetOrCreate("townhall").Entries()
	require.Len(t, entries, 2)

	assert.Contains(t, entries[0].Content, "First with image")
	require.Len(t, entries[0].Images, 1)
	assert.Equal(t, "data:image/jpeg;base64,dGh1bWItYnl0ZXM=", entries[0].Images[0].URL)

	assert.Contains(t, entries[1].Content, "Second with log")
	assert.Contains(t, entries[1].Content, "[Attachment app.log]:\n```\nhistorical log line\n```")
}

func TestGateway_EvictionToMemoryIndexing(t *testing.T) {
	tempDir := t.TempDir()

	cfg := &config.Config{
		DataDir:           tempDir,
		MsgRingBufferSize: 3,
	}

	gw := NewGateway(cfg, nil)
	defer gw.Stop()

	// Push 4 messages to ring buffer (capacity 3 -> triggers eviction of msg 1)
	ctx := context.Background()
	now := time.Now().Unix()

	gw.contextManager.Push("townhall", chatcontext.Entry{
		Seq:        10,
		Role:       "user",
		SenderName: "Alice",
		Content:    "Super secret project alpha details in townhall.",
		Timestamp:  now,
	})
	gw.contextManager.Push("townhall", chatcontext.Entry{
		Seq:        11,
		Role:       "assistant",
		SenderName: "Bob",
		Content:    "Acknowledged alpha.",
		Timestamp:  now + 1,
	})
	gw.contextManager.Push("townhall", chatcontext.Entry{
		Seq:        12,
		Role:       "user",
		SenderName: "Charlie",
		Content:    "Alpha is progressing well.",
		Timestamp:  now + 2,
	})

	// 4th push triggers eviction of 1 entry (Msg 10)
	gw.contextManager.Push("townhall", chatcontext.Entry{
		Seq:        13,
		Role:       "assistant",
		SenderName: "Bob",
		Content:    "Alpha update 4.",
		Timestamp:  now + 3,
	})

	// Allow async goroutine to complete
	time.Sleep(100 * time.Millisecond)

	// Verify watermark was updated
	wm, err := gw.MemoryManager().GetWatermark(ctx, "townhall", false)
	require.NoError(t, err)
	assert.Equal(t, int64(10), wm)

	// Verify message is searchable in memory store
	hits, err := gw.MemoryManager().Search(ctx, "secret", "townhall", false, 5)
	require.NoError(t, err)
	require.NotEmpty(t, hits)
	assert.Contains(t, hits[0].Content, "Super secret project alpha")
}

func TestGateway_StartupSequenceCatchup(t *testing.T) {
	tempDir := t.TempDir()

	besedkaServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/me" {
			_ = json.NewEncoder(w).Encode(models.User{ID: "bot_1", Name: "BobBot", DisplayName: "Bob"})
			return
		}
		if r.URL.Path == "/api/users" {
			_ = json.NewEncoder(w).Encode([]models.User{
				{ID: "u1", Name: "Alice", DisplayName: "Alice User"},
			})
			return
		}
		if r.URL.Path == "/api/chats" {
			_ = json.NewEncoder(w).Encode([]models.Chat{
				{ID: "townhall", LastSeq: 3},
				{ID: "dm_user1", LastSeq: 2, IsDM: true},
			})
			return
		}
		if r.URL.Path == "/api/chats/townhall/messages" {
			_ = json.NewEncoder(w).Encode([]models.Message{
				{Seq: 1, ChatID: "townhall", UserID: "u1", Content: "Catchup message 1", Timestamp: 100},
				{Seq: 2, ChatID: "townhall", UserID: "bot_1", Content: "Catchup message 2", Timestamp: 101},
				{Seq: 3, ChatID: "townhall", UserID: "u1", Content: "Catchup message 3", Timestamp: 102},
			})
			return
		}
		if r.URL.Path == "/api/chats/dm_user1/messages" {
			_ = json.NewEncoder(w).Encode([]models.Message{
				{Seq: 1, ChatID: "dm_user1", UserID: "u1", Content: "Private DM catchup 1", Timestamp: 200},
				{Seq: 2, ChatID: "dm_user1", UserID: "bot_1", Content: "Private DM catchup 2", Timestamp: 201},
			})
			return
		}
	}))
	defer besedkaServer.Close()

	cfg := &config.Config{
		BesedkaURL:        besedkaServer.URL,
		DataDir:           tempDir,
		MsgRingBufferSize: 10,
	}

	gw := NewGateway(cfg, nil)
	gw.httpClient = besedkaServer.Client()
	defer gw.Stop()

	ctx := context.Background()

	// Perform context warmup (includes catch-up)
	err := gw.WarmupContext(ctx)
	require.NoError(t, err)

	// Verify townhall watermark is 3
	thWM, err := gw.MemoryManager().GetWatermark(ctx, "townhall", false)
	require.NoError(t, err)
	assert.Equal(t, int64(3), thWM)

	// Verify DM watermark is 2
	dmWM, err := gw.MemoryManager().GetWatermark(ctx, "dm_user1", true)
	require.NoError(t, err)
	assert.Equal(t, int64(2), dmWM)

	// Verify townhall search finds historical messages
	thHits, err := gw.MemoryManager().Search(ctx, "Catchup", "townhall", false, 5)
	require.NoError(t, err)
	assert.GreaterOrEqual(t, len(thHits), 1)

	// Verify DM search finds DM historical messages
	dmHits, err := gw.MemoryManager().Search(ctx, "Private", "dm_user1", true, 5)
	require.NoError(t, err)
	assert.GreaterOrEqual(t, len(dmHits), 1)
	assert.Equal(t, "[Direct Message]", dmHits[0].Source)
}

func TestProcessMessageWithRecallMemoryTool(t *testing.T) {
	tempDir := t.TempDir()

	upgrader := websocket.Upgrader{}
	sentMessages := make(chan models.ClientMessage, 10)

	besedkaServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/me" {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(models.User{ID: "bot-id", UserName: "bot", DisplayName: "BotAssistant"})
			return
		}
		if r.URL.Path == "/api/users" {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode([]models.User{
				{ID: "user-1", UserName: "alice", DisplayName: "Alice"},
			})
			return
		}
		if r.URL.Path == "/api/chats" {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode([]models.Chat{})
			return
		}
		if r.URL.Path == "/api/chat" {
			conn, err := upgrader.Upgrade(w, r, nil)
			require.NoError(t, err)
			defer func() { _ = conn.Close() }()

			for {
				var msg models.ClientMessage
				if err := conn.ReadJSON(&msg); err != nil {
					break
				}
				sentMessages <- msg
			}
		}
	}))
	defer besedkaServer.Close()

	var llmCallCount int32
	llmServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		count := atomic.AddInt32(&llmCallCount, 1)
		w.Header().Set("Content-Type", "application/json")

		if count == 1 {
			var req openai.ChatCompletionRequest
			_ = json.NewDecoder(r.Body).Decode(&req)
			require.NotEmpty(t, req.Tools)

			// Verify recall_memory is registered in tool list
			hasRecallTool := false
			for _, tool := range req.Tools {
				if tool.Function.Name == "recall_memory" {
					hasRecallTool = true
					break
				}
			}
			assert.True(t, hasRecallTool)

			// LLM decides to call recall_memory
			resp := openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role: openai.ChatMessageRoleAssistant,
							ToolCalls: []openai.ToolCall{
								{
									ID:   "call_mem_123",
									Type: openai.ToolTypeFunction,
									Function: openai.FunctionCall{
										Name:      "recall_memory",
										Arguments: `{"query": "database Postgres"}`,
									},
								},
							},
						},
						FinishReason: openai.FinishReasonToolCalls,
					},
				},
			}
			_ = json.NewEncoder(w).Encode(resp)
			return
		}

		// Second call with tool output
		var req openai.ChatCompletionRequest
		_ = json.NewDecoder(r.Body).Decode(&req)

		// Assert tool message is passed back
		hasToolMessage := false
		for _, m := range req.Messages {
			if m.Role == openai.ChatMessageRoleTool && m.ToolCallID == "call_mem_123" {
				hasToolMessage = true
				assert.Contains(t, m.Content, "Postgres 16")
				break
			}
		}
		assert.True(t, hasToolMessage)

		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{
					Message: openai.ChatCompletionMessage{
						Role:    openai.ChatMessageRoleAssistant,
						Content: "According to our memory records, we decided to use Postgres 16 for all production clusters.",
					},
					FinishReason: openai.FinishReasonStop,
				},
			},
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer llmServer.Close()

	cfg := &config.Config{
		BotHandle:         "@bot",
		BesedkaURL:        besedkaServer.URL,
		OpenAIBaseURL:     llmServer.URL,
		OpenAIAPIKey:      "test-key",
		OpenAIModel:       "test-model",
		DataDir:           tempDir,
		MsgRingBufferSize: 10,
	}

	llmClient := llm.NewClient(cfg, llmServer.Client())
	gw := NewGateway(cfg, llmClient)
	gw.httpClient = besedkaServer.Client()
	defer gw.Stop()

	// Pre-index a message into memory
	ctx := context.Background()
	err := gw.MemoryManager().IndexMessages(ctx, "townhall", false, []memory.MessageToStore{
		{
			Seq:        1,
			Timestamp:  time.Now().Unix(),
			ChatID:     "townhall",
			UserID:     "user-1",
			SenderName: "Alice",
			Role:       "user",
			Content:    "We agreed to use Postgres 16 for all production clusters.",
		},
	})
	require.NoError(t, err)

	// Connect WebSocket
	err = gw.DialWebSocket(ctx)
	require.NoError(t, err)

	// Pre-fetch bot user metadata
	_, err = gw.FetchBotUser(ctx)
	require.NoError(t, err)

	incomingMsg := models.Message{
		Seq:       2,
		ChatID:    "townhall",
		UserID:    "user-1",
		Content:   "@bot what database did we decide to use?",
		Timestamp: time.Now().Unix(),
	}

	err = gw.ProcessMessage(ctx, incomingMsg)
	require.NoError(t, err)

	select {
	case reply := <-sentMessages:
		assert.Equal(t, models.ClientMessageTypeSend, reply.Type)
		assert.Equal(t, "townhall", reply.ChatID)
		assert.Contains(t, reply.Content, "Postgres 16")
	case <-time.After(1 * time.Second):
		t.Fatal("timed out waiting for bot reply")
	}

	assert.Equal(t, int32(2), atomic.LoadInt32(&llmCallCount))
}





