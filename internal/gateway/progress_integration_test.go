package gateway

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"bob/internal/config"
	"bob/internal/fsm"
	"bob/internal/llm"
	"bob/internal/models"

	"github.com/fasthttp/websocket"
	openai "github.com/sashabaranov/go-openai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestGatewayFSMProgressCardIntegration(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "bob-progress-fsm-test-*")
	require.NoError(t, err)
	defer func() { _ = os.RemoveAll(tempDir) }()

	var llmCallCount int32
	llmServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		count := atomic.AddInt32(&llmCallCount, 1)
		w.Header().Set("Content-Type", "application/json")

		if count == 1 {
			// First call returns a tool call
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
										Arguments: `{"query":"project details"}`,
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

		// Second call returns final text response
		resp := openai.ChatCompletionResponse{
			ID: "call-2",
			Choices: []openai.ChatCompletionChoice{
				{
					Index: 0,
					Message: openai.ChatCompletionMessage{
						Role:    openai.ChatMessageRoleAssistant,
						Content: "Found the project details successfully.",
					},
				},
			},
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer llmServer.Close()

	type postedMsg struct {
		ChatID  string
		Content string
		Type    models.MessageType
		Prog    *models.ProgressData
	}

	var mu sync.Mutex
	var postedMessages []postedMsg
	var seqGen int64 = 100

	upgrader := websocket.Upgrader{CheckOrigin: func(r *http.Request) bool { return true }}
	besedkaServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.URL.Path == "/api/me":
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(models.User{
				ID:       "bot_user",
				UserName: "bot",
			})
		case r.URL.Path == "/api/users":
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode([]models.User{
				{ID: "bot_user", UserName: "bot"},
				{ID: "u1", UserName: "alice", DisplayName: "Alice"},
			})
		case strings.HasPrefix(r.URL.Path, "/api/chats/") && strings.HasSuffix(r.URL.Path, "/messages") && r.Method == http.MethodGet:
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode([]models.Message{})
		case strings.HasPrefix(r.URL.Path, "/api/chats/") && strings.HasSuffix(r.URL.Path, "/messages") && r.Method == http.MethodPost:
			var req struct {
				Content     string               `json:"content"`
				Type        models.MessageType   `json:"type"`
				Progress    *models.ProgressData `json:"progress"`
				Attachments []models.Attachment  `json:"attachments"`
			}
			_ = json.NewDecoder(r.Body).Decode(&req)
			parts := strings.Split(r.URL.Path, "/")
			chatID := parts[3]

			mu.Lock()
			seqGen++
			assignedSeq := seqGen
			postedMessages = append(postedMessages, postedMsg{
				ChatID:  chatID,
				Content: req.Content,
				Type:    req.Type,
				Prog:    req.Progress,
			})
			mu.Unlock()

			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]interface{}{
				"seq":       assignedSeq,
				"timestamp": time.Now().Unix(),
			})
		case r.URL.Path == "/api/chat":
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
			}
		}
	}))
	defer besedkaServer.Close()

	cfg := &config.Config{
		BesedkaURL:                besedkaServer.URL,
		BesedkaAPIKey:             "test-key",
		OpenAIAPIKey:              "test-key",
		OpenAIBaseURL:             llmServer.URL,
		OpenAIModel:               "test-model",
		BotHandle:                 "@bot",
		DataDir:                   tempDir,
		TownhallToolMaxIterations: 10,
		DMToolMaxIterations:       20,
		TownhallMaxParagraphs:     5,
		DMMaxParagraphs:           10,
		MsgRingBufferSize:         10,
	}

	llmClient := llm.NewClient(cfg, llmServer.Client())
	gw := NewGateway(cfg, llmClient)
	defer gw.Stop()
	gw.httpClient = besedkaServer.Client()

	storeProv := NewMemoryStoreProvider(gw.MemoryManager(), filepath.Join(cfg.DataDir, "fsm"))
	fsmEngine := fsm.NewEngine(storeProv, llmClient, gw.ToolsRegistry(), fsm.WithDefaultModel(cfg.OpenAIModel))
	gw.SetFSMEngine(fsmEngine)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	require.NoError(t, gw.DialWebSocket(ctx))
	_, err = gw.FetchBotUser(ctx)
	require.NoError(t, err)

	// Process message in a DM chat
	dmChatID := "dm_u1_bot_user"
	err = gw.ProcessMessage(ctx, models.Message{
		ChatID:    dmChatID,
		UserID:    "u1",
		Content:   "@bot find my project details",
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)

	// Verify all posted progress messages
	mu.Lock()
	defer mu.Unlock()

	var progressCards []postedMsg
	for _, m := range postedMessages {
		if m.Type == models.MessageTypeProgress {
			progressCards = append(progressCards, m)
		}
	}

	// Must have:
	// 1. Root card creation (CardStatus = running, Title = "Working on your request...")
	// 2. Child step running (ParentSeq = 101, Status = running, Title = "Recall Memory")
	// 3. Child step completed (ParentSeq = 101, Status = completed)
	// 4. Terminal card status (ParentSeq = 101, CardStatus = completed)
	require.GreaterOrEqual(t, len(progressCards), 4)

	// 1. Root card
	assert.Equal(t, dmChatID, progressCards[0].ChatID)
	assert.Equal(t, models.ProgressStatusRunning, progressCards[0].Prog.CardStatus)
	assert.Equal(t, int64(0), progressCards[0].Prog.ParentSeq)
	assert.Equal(t, "Working on your request...", progressCards[0].Prog.Title)

	// 2. Step running
	assert.Equal(t, int64(101), progressCards[1].Prog.ParentSeq)
	require.NotNil(t, progressCards[1].Prog.Step)
	assert.Equal(t, models.ProgressStatusRunning, progressCards[1].Prog.Step.Status)
	assert.Equal(t, "Recall Memory", progressCards[1].Prog.Step.Title)

	// 3. Step completed
	assert.Equal(t, int64(101), progressCards[2].Prog.ParentSeq)
	require.NotNil(t, progressCards[2].Prog.Step)
	assert.Equal(t, models.ProgressStatusCompleted, progressCards[2].Prog.Step.Status)

	// 4. Root card completed
	lastCard := progressCards[len(progressCards)-1]
	assert.Equal(t, int64(101), lastCard.Prog.ParentSeq)
	assert.Equal(t, models.ProgressStatusCompleted, lastCard.Prog.CardStatus)
	assert.Nil(t, lastCard.Prog.Step)
}

func TestGatewayFSMProgressCard_TownhallWriteDeniedGraceful(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "bob-progress-th-denied-*")
	require.NoError(t, err)
	defer func() { _ = os.RemoveAll(tempDir) }()

	var llmCallCount int32
	llmServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		count := atomic.AddInt32(&llmCallCount, 1)
		w.Header().Set("Content-Type", "application/json")

		if count == 1 {
			resp := openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role: openai.ChatMessageRoleAssistant,
							ToolCalls: []openai.ToolCall{
								{
									ID:   "call_recall_th",
									Type: openai.ToolTypeFunction,
									Function: openai.FunctionCall{
										Name:      "recall_memory",
										Arguments: `{"query":"townhall topic"}`,
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

		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{
					Message: openai.ChatCompletionMessage{
						Role:    openai.ChatMessageRoleAssistant,
						Content: "Here is the townhall answer despite progress failure.",
					},
				},
			},
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer llmServer.Close()

	upgrader := websocket.Upgrader{CheckOrigin: func(r *http.Request) bool { return true }}
	besedkaServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.URL.Path == "/api/me":
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(models.User{ID: "bot_user", UserName: "bot"})
		case r.URL.Path == "/api/users":
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode([]models.User{{ID: "bot_user", UserName: "bot"}, {ID: "u1", UserName: "alice"}})
		case strings.HasPrefix(r.URL.Path, "/api/chats/") && strings.HasSuffix(r.URL.Path, "/messages") && r.Method == http.MethodGet:
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode([]models.Message{})
		case strings.HasPrefix(r.URL.Path, "/api/chats/") && strings.HasSuffix(r.URL.Path, "/messages") && r.Method == http.MethodPost:
			var req struct {
				Type models.MessageType `json:"type"`
			}
			_ = json.NewDecoder(r.Body).Decode(&req)
			if req.Type == models.MessageTypeProgress {
				// Simulate 403 Forbidden write permission error in townhall
				w.WriteHeader(http.StatusForbidden)
				_ = json.NewEncoder(w).Encode(map[string]string{"error": "write permission denied in townhall"})
				return
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]interface{}{
				"seq":       int64(201),
				"timestamp": time.Now().Unix(),
			})
		case r.URL.Path == "/api/chat":
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
			}
		}
	}))
	defer besedkaServer.Close()

	cfg := &config.Config{
		BesedkaURL:                besedkaServer.URL,
		BesedkaAPIKey:             "test-key",
		OpenAIAPIKey:              "test-key",
		OpenAIBaseURL:             llmServer.URL,
		OpenAIModel:               "test-model",
		BotHandle:                 "@bot",
		DataDir:                   tempDir,
		TownhallToolMaxIterations: 10,
		DMToolMaxIterations:       20,
		TownhallMaxParagraphs:     5,
		DMMaxParagraphs:           10,
		MsgRingBufferSize:         10,
	}

	llmClient := llm.NewClient(cfg, llmServer.Client())
	gw := NewGateway(cfg, llmClient)
	defer gw.Stop()
	gw.httpClient = besedkaServer.Client()

	storeProv := NewMemoryStoreProvider(gw.MemoryManager(), filepath.Join(cfg.DataDir, "fsm"))
	fsmEngine := fsm.NewEngine(storeProv, llmClient, gw.ToolsRegistry(), fsm.WithDefaultModel(cfg.OpenAIModel))
	gw.SetFSMEngine(fsmEngine)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	require.NoError(t, gw.DialWebSocket(ctx))
	_, err = gw.FetchBotUser(ctx)
	require.NoError(t, err)

	// In Townhall, progress card fails with 403, but user task must succeed without error
	err = gw.ProcessMessage(ctx, models.Message{
		ChatID:    "townhall",
		UserID:    "u1",
		Content:   "@bot recall townhall topic",
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)
}

func TestGatewayFSMProgressCard_NoToolsEmitsNoProgress(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "bob-no-progress-*")
	require.NoError(t, err)
	defer func() { _ = os.RemoveAll(tempDir) }()

	llmServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{
					Message: openai.ChatCompletionMessage{
						Role:    openai.ChatMessageRoleAssistant,
						Content: "Direct text answer without tools.",
					},
				},
			},
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer llmServer.Close()

	var mu sync.Mutex
	var postedProgressCount int

	upgrader := websocket.Upgrader{CheckOrigin: func(r *http.Request) bool { return true }}
	besedkaServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.URL.Path == "/api/me":
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(models.User{ID: "bot_user", UserName: "bot"})
		case r.URL.Path == "/api/users":
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode([]models.User{{ID: "bot_user", UserName: "bot"}, {ID: "u1", UserName: "alice"}})
		case strings.HasPrefix(r.URL.Path, "/api/chats/") && strings.HasSuffix(r.URL.Path, "/messages") && r.Method == http.MethodGet:
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode([]models.Message{})
		case strings.HasPrefix(r.URL.Path, "/api/chats/") && strings.HasSuffix(r.URL.Path, "/messages") && r.Method == http.MethodPost:
			var req struct {
				Type models.MessageType `json:"type"`
			}
			_ = json.NewDecoder(r.Body).Decode(&req)
			if req.Type == models.MessageTypeProgress {
				mu.Lock()
				postedProgressCount++
				mu.Unlock()
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]interface{}{
				"seq":       int64(301),
				"timestamp": time.Now().Unix(),
			})
		case r.URL.Path == "/api/chat":
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
			}
		}
	}))
	defer besedkaServer.Close()

	cfg := &config.Config{
		BesedkaURL:                besedkaServer.URL,
		BesedkaAPIKey:             "test-key",
		OpenAIAPIKey:              "test-key",
		OpenAIBaseURL:             llmServer.URL,
		OpenAIModel:               "test-model",
		BotHandle:                 "@bot",
		DataDir:                   tempDir,
		TownhallToolMaxIterations: 10,
		DMToolMaxIterations:       20,
		TownhallMaxParagraphs:     5,
		DMMaxParagraphs:           10,
		MsgRingBufferSize:         10,
	}

	llmClient := llm.NewClient(cfg, llmServer.Client())
	gw := NewGateway(cfg, llmClient)
	defer gw.Stop()
	gw.httpClient = besedkaServer.Client()

	storeProv := NewMemoryStoreProvider(gw.MemoryManager(), filepath.Join(cfg.DataDir, "fsm"))
	fsmEngine := fsm.NewEngine(storeProv, llmClient, gw.ToolsRegistry(), fsm.WithDefaultModel(cfg.OpenAIModel))
	gw.SetFSMEngine(fsmEngine)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	require.NoError(t, gw.DialWebSocket(ctx))
	_, err = gw.FetchBotUser(ctx)
	require.NoError(t, err)

	err = gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_u1_bot",
		UserID:    "u1",
		Content:   "@bot what is your name?",
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)

	mu.Lock()
	defer mu.Unlock()
	assert.Equal(t, 0, postedProgressCount, "no progress messages should be posted when 0 tools are called")
}

