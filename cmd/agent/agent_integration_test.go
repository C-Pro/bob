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
	"bob/internal/geoip"
	"bob/internal/llm"
	"bob/internal/models"

	"github.com/fasthttp/websocket"
	openai "github.com/sashabaranov/go-openai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestAgentIntegrationLoop(t *testing.T) {
	// Mock OpenAI completion server
	llmServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.Contains(r.URL.Path, "embeddings") {
			resp := openai.EmbeddingResponse{
				Data: []openai.Embedding{
					{Index: 0, Embedding: make([]float32, 128)},
				},
			}
			_ = json.NewEncoder(w).Encode(resp)
			return
		}

		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{
					Index: 0,
					Message: openai.ChatCompletionMessage{
						Role:    openai.ChatMessageRoleAssistant,
						Content: "I am ready to help!",
					},
				},
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
		OpenAIAPIKey:          "test-openai-key",
		OpenAIModel:           "gemini-3.7-flash",
		OpenAIBaseURL:         llmServer.URL,
		TownhallMaxParagraphs: 2,
		DMMaxParagraphs:       10,
		DataDir:               t.TempDir(),
		EmbeddingModel:        "mock-model",
	}

	llmClient := llm.NewClient(cfg, llmServer.Client())
	gw := gateway.NewGateway(cfg, llmClient)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	go func() {
		_ = gw.Start(ctx)
	}()

	select {
	case reply := <-botReplies:
		assert.Equal(t, models.ClientMessageTypeSend, reply.Type)
		assert.Equal(t, "townhall", reply.ChatID)
		assert.Equal(t, "I am ready to help!", reply.Content)
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for agent response in integration loop")
	}
}

func TestAgentIntegration_MultiTurnContextFlow(t *testing.T) {
	var capturedRequests []openai.ChatCompletionRequest
	llmServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.Contains(r.URL.Path, "embeddings") {
			resp := openai.EmbeddingResponse{
				Data: []openai.Embedding{
					{Index: 0, Embedding: make([]float32, 128)},
				},
			}
			_ = json.NewEncoder(w).Encode(resp)
			return
		}

		var req openai.ChatCompletionRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		capturedRequests = append(capturedRequests, req)

		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{
					Index: 0,
					Message: openai.ChatCompletionMessage{
						Role:    openai.ChatMessageRoleAssistant,
						Content: "Multi-turn reply",
					},
				},
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
		OpenAIBaseURL:         llmServer.URL,
		TownhallMaxParagraphs: 2,
		DMMaxParagraphs:       10,
		MsgRingBufferSize:     100,
		DataDir:               t.TempDir(),
		EmbeddingModel:        "mock-model",
	}

	llmClient := llm.NewClient(cfg, llmServer.Client())
	gw := gateway.NewGateway(cfg, llmClient)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
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
	assert.Equal(t, openai.ChatMessageRoleSystem, messages[0].Role)
	assert.Contains(t, messages[0].Content, "AI Bot")
	assert.Equal(t, "Alice: Initial historical message", messages[1].Content)
	assert.Equal(t, "Bob: Should we deploy to prod today?", messages[2].Content)
	assert.Equal(t, "Alice: @bot what do you think?", messages[3].Content)
}

func TestAgentIntegration_DMContextAndNameResolution(t *testing.T) {
	var capturedRequests []openai.ChatCompletionRequest
	llmServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.Contains(r.URL.Path, "embeddings") {
			resp := openai.EmbeddingResponse{
				Data: []openai.Embedding{
					{Index: 0, Embedding: make([]float32, 128)},
				},
			}
			_ = json.NewEncoder(w).Encode(resp)
			return
		}

		var req openai.ChatCompletionRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		capturedRequests = append(capturedRequests, req)

		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{
					Index: 0,
					Message: openai.ChatCompletionMessage{
						Role:    openai.ChatMessageRoleAssistant,
						Content: "Sure, let's talk about fins.",
					},
				},
			},
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer llmServer.Close()

	upgrader := websocket.Upgrader{}
	besedkaServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/me" {
			_ = json.NewEncoder(w).Encode(models.User{ID: "bot-id", UserName: "bob", DisplayName: "Agent Bob"})
			return
		}
		if r.URL.Path == "/api/users" {
			_ = json.NewEncoder(w).Encode([]models.User{
				{ID: "user-uuid-1234", UserName: "cpro", DisplayName: "C-Pro"},
			})
			return
		}
		if r.URL.Path == "/api/chats" {
			_ = json.NewEncoder(w).Encode([]models.Chat{
				{ID: "townhall", Type: "townhall", LastSeq: 0},
				{ID: "dm_bot_user", Type: "dm", LastSeq: 2},
			})
			return
		}
		if strings.HasPrefix(r.URL.Path, "/api/chats/dm_bot_user/messages") {
			assert.Equal(t, "1", r.URL.Query().Get("fromSeq"))
			assert.Equal(t, "2", r.URL.Query().Get("toSeq"))
			_ = json.NewEncoder(w).Encode([]models.Message{
				{Seq: 1, ChatID: "dm_bot_user", UserID: "user-uuid-1234", Content: "<p>Where to buy freediving fins?</p>", Timestamp: 100},
				{Seq: 2, ChatID: "dm_bot_user", UserID: "bot-id", Content: "<p>Aquamaster in Kuta</p>", Timestamp: 101},
			})
			return
		}
		if strings.HasPrefix(r.URL.Path, "/api/chats/") && strings.HasSuffix(r.URL.Path, "/messages") {
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
		}
	}))
	defer besedkaServer.Close()

	cfg := &config.Config{
		BotHandle:         "@bob",
		BesedkaURL:        besedkaServer.URL,
		OpenAIBaseURL:     llmServer.URL,
		DMMaxParagraphs:   10,
		MsgRingBufferSize: 100,
		DataDir:           t.TempDir(),
		EmbeddingModel:    "mock-model",
	}

	llmClient := llm.NewClient(cfg, llmServer.Client())
	gw := gateway.NewGateway(cfg, llmClient)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	err := gw.DialWebSocket(ctx)
	require.NoError(t, err)

	// Warmup should populate DM history with correct fromSeq/toSeq!
	err = gw.WarmupContext(ctx)
	require.NoError(t, err)

	// Send next turn in DM
	err = gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_bot_user",
		UserID:    "user-uuid-1234",
		Content:   "What did we discuss earlier?",
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)
	require.Len(t, capturedRequests, 1)

	messages := capturedRequests[0].Messages
	require.Len(t, messages, 4)

	// System prompt must have C-Pro (display name), NOT UUID user-uuid-1234
	assert.Equal(t, openai.ChatMessageRoleSystem, messages[0].Role)
	assert.Contains(t, messages[0].Content, "C-Pro")
	assert.NotContains(t, messages[0].Content, "user-uuid-1234")

	// Historical turn 1 from user
	assert.Equal(t, openai.ChatMessageRoleUser, messages[1].Role)
	assert.Equal(t, "C-Pro: Where to buy freediving fins?", messages[1].Content)

	// Historical turn 2 from assistant
	assert.Equal(t, openai.ChatMessageRoleAssistant, messages[2].Role)
	assert.Equal(t, "Aquamaster in Kuta", messages[2].Content)

	// Turn 3
	assert.Equal(t, openai.ChatMessageRoleUser, messages[3].Role)
	assert.Equal(t, "C-Pro: What did we discuss earlier?", messages[3].Content)
}

func TestAgentIntegration_StartupLocationReporting(t *testing.T) {
	// 1. Mock GEOIP Server
	geoServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"status": "success",
			"lat":    -8.4095,
			"lon":    115.1889,
		})
	}))
	defer geoServer.Close()

	// 2. Mock Besedka WebSocket Server
	upgrader := websocket.Upgrader{}
	receivedFrames := make(chan models.ClientMessage, 10)

	besedkaServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/me" {
			_ = json.NewEncoder(w).Encode(models.User{ID: "bot-id", UserName: "bob", DisplayName: "Agent Bob"})
			return
		}
		if r.URL.Path == "/api/users" {
			_ = json.NewEncoder(w).Encode([]models.User{})
			return
		}
		if r.URL.Path == "/api/chats" {
			_ = json.NewEncoder(w).Encode([]models.Chat{})
			return
		}
		if strings.HasPrefix(r.URL.Path, "/api/chats/") {
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

	// 3. Initialize GEOIP and Gateway
	geoClient := geoip.NewClient(
		geoServer.Client(),
		geoip.WithProviders([]geoip.Provider{
			{Name: "mock-ip-api", URL: geoServer.URL, Parser: geoip.ParseIPAPI},
		}),
	)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	loc, err := geoClient.FetchLocation(ctx)
	require.NoError(t, err)
	require.NotNil(t, loc)
	assert.Equal(t, -8.4095, loc.Lat)
	assert.Equal(t, 115.1889, loc.Lng)

	cfg := &config.Config{
		BotHandle:      "@bob",
		BesedkaURL:     besedkaServer.URL,
		DataDir:        t.TempDir(),
		EmbeddingModel: "mock-model",
	}

	gw := gateway.NewGateway(cfg, nil)
	gw.SetLocation(loc)
	gw.SetInitialLocationDelay(20 * time.Millisecond)
	gw.SetLocationInterval(50 * time.Millisecond)

	go func() {
		_ = gw.Start(ctx)
	}()

	// 4. Verify initial location WS frame arrives soon after initial connection
	select {
	case frame := <-receivedFrames:
		assert.Equal(t, models.ClientMessageTypeLocation, frame.Type)
		require.NotNil(t, frame.Location)
		assert.Equal(t, -8.4095, frame.Location.Lat)
		assert.Equal(t, 115.1889, frame.Location.Lng)
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for initial location frame in integration test")
	}

	gw.Stop()
}
