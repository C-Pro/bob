package main

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"bob/internal/bootstrap"
	"bob/internal/config"
	"bob/internal/gateway"
	"bob/internal/geoip"
	"bob/internal/llm"
	"bob/internal/models"
	"bob/internal/objectstore"
	"bob/internal/trajectory"

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
	gw.SetInitialLocationDelay(2 * time.Millisecond)
	gw.SetLocationInterval(10 * time.Millisecond)

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

func TestAgentIntegration_TrajectoryLogger_ToolTurnWithFeedbackAndExport(t *testing.T) {
	// 1. Mock S3 Object Store Server
	var uploadedKey string
	var uploadedBody []byte
	var uploadMu sync.Mutex

	s3Server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPut {
			body, err := io.ReadAll(r.Body)
			if err != nil {
				w.WriteHeader(http.StatusInternalServerError)
				return
			}
			uploadMu.Lock()
			uploadedKey = r.URL.Path
			uploadedBody = body
			uploadMu.Unlock()
			w.WriteHeader(http.StatusOK)
			return
		}
		w.WriteHeader(http.StatusNotFound)
	}))
	defer s3Server.Close()

	// 2. Mock LLM Completion & Embeddings Server
	var llmCallCount int
	var llmMu sync.Mutex

	llmServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.Contains(r.URL.Path, "embeddings") {
			resp := openai.EmbeddingResponse{
				Data: []openai.Embedding{
					{Index: 0, Embedding: make([]float32, 128)},
				},
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(resp)
			return
		}

		llmMu.Lock()
		llmCallCount++
		currentCall := llmCallCount
		llmMu.Unlock()

		w.Header().Set("Content-Type", "application/json")
		if currentCall == 1 {
			// First call triggers tool call 'recall_memory'
			resp := openai.ChatCompletionResponse{
				ID: "call-1",
				Choices: []openai.ChatCompletionChoice{
					{
						Index: 0,
						Message: openai.ChatCompletionMessage{
							Role: openai.ChatMessageRoleAssistant,
							ToolCalls: []openai.ToolCall{
								{
									ID:   "call_recall_1",
									Type: openai.ToolTypeFunction,
									Function: openai.FunctionCall{
										Name:      "recall_memory",
										Arguments: `{"query":"test-memories"}`,
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

		// Second call returns final conversational response
		resp := openai.ChatCompletionResponse{
			ID: "call-2",
			Choices: []openai.ChatCompletionChoice{
				{
					Index: 0,
					Message: openai.ChatCompletionMessage{
						Role:    openai.ChatMessageRoleAssistant,
						Content: "Found test memories successfully.",
					},
				},
			},
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer llmServer.Close()

	// 3. Mock Besedka WebSocket & REST Server
	upgrader := websocket.Upgrader{CheckOrigin: func(r *http.Request) bool { return true }}
	botReplies := make(chan models.ClientMessage, 10)

	besedkaServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch {
		case r.URL.Path == "/api/me":
			_ = json.NewEncoder(w).Encode(models.User{
				ID:          "bot-user-id",
				UserName:    "bot",
				DisplayName: "Bot",
			})
		case r.URL.Path == "/api/users":
			_ = json.NewEncoder(w).Encode([]models.User{
				{ID: "bot-user-id", UserName: "bot", DisplayName: "Bot"},
				{ID: "human-user", UserName: "alice", DisplayName: "Alice"},
			})
		case r.URL.Path == "/api/chats":
			_ = json.NewEncoder(w).Encode([]models.Chat{
				{ID: "townhall", Type: "townhall"},
			})
		case strings.HasPrefix(r.URL.Path, "/api/chats/") && strings.HasSuffix(r.URL.Path, "/messages") && r.Method == http.MethodGet:
			_ = json.NewEncoder(w).Encode([]models.Message{})
		case strings.HasPrefix(r.URL.Path, "/api/chats/") && strings.HasSuffix(r.URL.Path, "/messages") && r.Method == http.MethodPost:
			var req struct {
				Content string `json:"content"`
			}
			_ = json.NewDecoder(r.Body).Decode(&req)
			_ = json.NewEncoder(w).Encode(map[string]interface{}{
				"seq":       10,
				"timestamp": time.Now().Unix(),
			})
		case strings.HasPrefix(r.URL.Path, "/api/chats/") && !strings.Contains(r.URL.Path, "/messages"):
			_ = json.NewEncoder(w).Encode(models.Chat{ID: "townhall", Type: "townhall"})
		case r.URL.Path == "/api/chat":
			conn, err := upgrader.Upgrade(w, r, nil)
			if err != nil {
				return
			}
			defer func() { _ = conn.Close() }()
			for {
				var cm models.ClientMessage
				if err := conn.ReadJSON(&cm); err != nil {
					return
				}
				botReplies <- cm
			}
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer besedkaServer.Close()

	// 4. Initialize Config & Runtime with Trajectory Logger enabled
	tempDir := t.TempDir()
	cfg := &config.Config{
		BotHandle:             "@bot",
		BesedkaURL:            besedkaServer.URL,
		BesedkaAPIKey:         "test-key",
		OpenAIAPIKey:          "test-openai-key",
		OpenAIModel:           "gemini-3.7-flash",
		OpenAIBaseURL:         llmServer.URL,
		TownhallMaxParagraphs: 2,
		DMMaxParagraphs:       10,
		MsgRingBufferSize:     100,
		DataDir:               tempDir,
		EmbeddingModel:        "mock-model",
		TrajectoryEnabled:     true,
		TrajectoryS3Bucket:    "test-traj-bucket",
		TrajectoryS3Prefix:    "trajectories/",
		TrajectoryFlushDelay:  15 * time.Minute,
		S3Endpoint:            s3Server.URL,
		S3Region:              "us-east-1",
		S3AccessKey:           "TEST_KEY",
		S3SecretKey:           "TEST_SECRET",
		S3PathStyle:           true,
	}
	require.NoError(t, cfg.Validate(true))

	llmClient := llm.NewClient(cfg, llmServer.Client())
	rt, err := bootstrap.New(cfg, bootstrap.Options{
		LLMClient: llmClient,
	})
	require.NoError(t, err)
	defer func() { _ = rt.Close() }()

	gw := gateway.NewGatewayWithRuntime(cfg, rt)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	err = gw.DialWebSocket(ctx)
	require.NoError(t, err)

	// 5. Execute Turn 1: User mentions @bot requesting memories, requiring tool invocation
	err = gw.ProcessMessage(ctx, models.Message{
		Seq:       1,
		ChatID:    "townhall",
		UserID:    "human-user",
		Content:   "@bot find memories",
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)

	// 6. Execute Turn 2: User provides positive feedback via /good command
	err = gw.ProcessMessage(ctx, models.Message{
		Seq:       2,
		ChatID:    "townhall",
		UserID:    "human-user",
		Content:   "/good Great answer!",
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)

	// 7. Initialize Trajectory Exporter and trigger FlushOnce
	s3Client, err := objectstore.New(objectstore.Config{
		Endpoint:  s3Server.URL,
		Region:    cfg.S3Region,
		Bucket:    cfg.TrajectoryBucket(),
		AccessKey: cfg.S3AccessKey,
		SecretKey: cfg.S3SecretKey,
		PathStyle: cfg.S3PathStyle,
	})
	require.NoError(t, err)
	require.NotNil(t, s3Client)

	exporterCfg := trajectory.ExporterConfig{
		Enabled:      true,
		Prefix:       cfg.TrajectoryS3Prefix,
		AllowPrivate: cfg.TrajectoryAllowPrivate,
		BatchLimit:   50,
	}
	trajExporter := trajectory.NewExporter(exporterCfg, rt.StoreProvider, s3Client)
	// Override exporter time into the future to satisfy feedback quiescence delay
	trajExporter.SetNow(func() time.Time {
		return time.Now().UTC().Add(5 * time.Minute)
	})

	err = trajExporter.FlushOnce(ctx)
	require.NoError(t, err)

	// 8. Verify S3 Object Store Upload
	uploadMu.Lock()
	key := uploadedKey
	body := uploadedBody
	uploadMu.Unlock()

	require.NotEmpty(t, key, "expected trajectory upload to mock S3 object store")
	assert.True(t, strings.HasPrefix(key, "/test-traj-bucket/trajectories/"), "expected key path prefix: %s", key)
	assert.True(t, strings.HasSuffix(key, "/r2.json"), "expected uploaded revision 2 key: %s", key)

	// Assert typed unmarshaling
	var uploadedTraj trajectory.Trajectory
	err = json.Unmarshal(body, &uploadedTraj)
	require.NoError(t, err)

	assert.Contains(t, uploadedTraj.UserInput, "find memories")
	assert.NotEmpty(t, uploadedTraj.TaskState, "phi_task should be present")
	assert.Equal(t, "Found test memories successfully.", uploadedTraj.FinalOutput)
	require.Len(t, uploadedTraj.Steps, 1)
	assert.Equal(t, "recall_memory", uploadedTraj.Steps[0].Action)
	require.NotNil(t, uploadedTraj.Steps[0].ExitCode)
	assert.Equal(t, 0, *uploadedTraj.Steps[0].ExitCode)
	assert.True(t, uploadedTraj.Breakdown.HasUserFeedback)
	assert.Equal(t, 1.0, uploadedTraj.Breakdown.UserFeedback)
	assert.Equal(t, "Great answer!", uploadedTraj.Breakdown.FeedbackNote)
	assert.Greater(t, uploadedTraj.Reward, 0.8, "reward should exceed 0.8 on positive feedback and tool success")
	assert.Equal(t, 2, uploadedTraj.Revision)

	// Assert canonical raw JSON schema fields (x, phi_task, steps, y, r_eval)
	var raw map[string]interface{}
	err = json.Unmarshal(body, &raw)
	require.NoError(t, err)

	assert.Contains(t, raw, "x")
	assert.Contains(t, raw, "phi_task")
	assert.Contains(t, raw, "steps")
	assert.Contains(t, raw, "y")
	assert.Contains(t, raw, "r_eval")
	assert.Equal(t, float64(2), raw["revision"])

	gw.Stop()
}
