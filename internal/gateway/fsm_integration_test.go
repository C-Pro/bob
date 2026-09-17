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

	"bob/internal/config"
	"bob/internal/fsm"
	"bob/internal/llm"
	"bob/internal/models"

	"github.com/fasthttp/websocket"
	openai "github.com/sashabaranov/go-openai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	_ "modernc.org/sqlite"
)

func TestGateway_FSMToolLoop_TownhallIterationCap(t *testing.T) {
	tempDir := t.TempDir()
	var llmCallCount int32

	llmServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		count := atomic.AddInt32(&llmCallCount, 1)

		var req openai.ChatCompletionRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		// Check if last message is synthesis prompt
		if len(req.Messages) > 0 {
			lastMsg := req.Messages[len(req.Messages)-1]
			if strings.Contains(lastMsg.Content, "tool execution limit") {
				resp := openai.ChatCompletionResponse{
					Choices: []openai.ChatCompletionChoice{
						{
							Message: openai.ChatCompletionMessage{
								Role:    openai.ChatMessageRoleAssistant,
								Content: fmt.Sprintf("Townhall synthesis reached after %d calls", count),
							},
						},
					},
				}
				_ = json.NewEncoder(w).Encode(resp)
				return
			}
		}

		// Otherwise return a mock search tool call
		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{
					Message: openai.ChatCompletionMessage{
						Role: openai.ChatMessageRoleAssistant,
						ToolCalls: []openai.ToolCall{
							{
								ID:   fmt.Sprintf("call_%d", count),
								Type: openai.ToolTypeFunction,
								Function: openai.FunctionCall{
									Name:      "web_search",
									Arguments: fmt.Sprintf(`{"query": "iteration %d"}`, count),
								},
							},
						},
					},
				},
			},
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer llmServer.Close()

	sentMsgs := make(chan models.ClientMessage, 10)
	besedkaServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/me":
			_ = json.NewEncoder(w).Encode(models.User{ID: "bot_1", DisplayName: "Bob", UserName: "bot"})
		case "/api/users":
			_ = json.NewEncoder(w).Encode([]models.User{{ID: "u1", DisplayName: "Alice"}})
		case "/api/chats":
			_ = json.NewEncoder(w).Encode([]models.Chat{{ID: "townhall"}})
		case "/api/chat":
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
				sentMsgs <- cm
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

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	require.NoError(t, gw.DialWebSocket(ctx))
	_, err := gw.FetchBotUser(ctx)
	require.NoError(t, err)

	err = gw.ProcessMessage(ctx, models.Message{
		ChatID:    "townhall",
		UserID:    "u1",
		Content:   "@bot run search loop",
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)

	// Townhall iteration cap is 10: 10 iterations + 1 synthesis call = 11 LLM calls
	assert.Equal(t, int32(11), atomic.LoadInt32(&llmCallCount))

	// Verify reply was sent over websocket
	select {
	case reply := <-sentMsgs:
		assert.Contains(t, reply.Content, "Townhall synthesis reached after 11 calls")
	case <-time.After(3 * time.Second):
		t.Fatal("timed out waiting for bot response")
	}

	// Verify run was recorded in SQLite townhall.db
	var runID string
	var status, currentState string
	var iteration, maxIterations int
	activeDBs := gw.MemoryManager().ActiveDBs()
	require.Contains(t, activeDBs, "townhall.db")
	err = activeDBs["townhall.db"].QueryRowContext(ctx, "SELECT id, status, current_state, iteration, max_iterations FROM fsm_runs").Scan(&runID, &status, &currentState, &iteration, &maxIterations)
	require.NoError(t, err)
	assert.Equal(t, 10, maxIterations)
	assert.Equal(t, 10, iteration)
	assert.Equal(t, "COMPLETED", status)
	assert.Equal(t, "COMPLETED", currentState)
}

func TestGateway_FSMToolLoop_DMIterationCap(t *testing.T) {
	tempDir := t.TempDir()
	var llmCallCount int32

	llmServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		count := atomic.AddInt32(&llmCallCount, 1)

		var req openai.ChatCompletionRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		if len(req.Messages) > 0 {
			lastMsg := req.Messages[len(req.Messages)-1]
			if strings.Contains(lastMsg.Content, "tool execution limit") {
				resp := openai.ChatCompletionResponse{
					Choices: []openai.ChatCompletionChoice{
						{
							Message: openai.ChatCompletionMessage{
								Role:    openai.ChatMessageRoleAssistant,
								Content: fmt.Sprintf("DM synthesis reached after %d calls", count),
							},
						},
					},
				}
				_ = json.NewEncoder(w).Encode(resp)
				return
			}
		}

		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{
					Message: openai.ChatCompletionMessage{
						Role: openai.ChatMessageRoleAssistant,
						ToolCalls: []openai.ToolCall{
							{
								ID:   fmt.Sprintf("call_%d", count),
								Type: openai.ToolTypeFunction,
								Function: openai.FunctionCall{
									Name:      "web_search",
									Arguments: fmt.Sprintf(`{"query": "dm iteration %d"}`, count),
								},
							},
						},
					},
				},
			},
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer llmServer.Close()

	sentMsgs := make(chan models.ClientMessage, 10)
	besedkaServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/me":
			_ = json.NewEncoder(w).Encode(models.User{ID: "bot_1", DisplayName: "Bob", UserName: "bot"})
		case "/api/users":
			_ = json.NewEncoder(w).Encode([]models.User{{ID: "u1", DisplayName: "Alice"}})
		case "/api/chats":
			_ = json.NewEncoder(w).Encode([]models.Chat{{ID: "dm_user1"}})
		case "/api/chat":
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
				sentMsgs <- cm
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

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	require.NoError(t, gw.DialWebSocket(ctx))
	_, err := gw.FetchBotUser(ctx)
	require.NoError(t, err)

	err = gw.ProcessMessage(ctx, models.Message{
		ChatID:    "user1",
		UserID:    "u1",
		Content:   "run dm search loop",
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)

	// DM iteration cap is 20: 20 iterations + 1 synthesis call = 21 LLM calls
	assert.Equal(t, int32(21), atomic.LoadInt32(&llmCallCount))

	select {
	case reply := <-sentMsgs:
		assert.Contains(t, reply.Content, "DM synthesis reached after 21 calls")
	case <-time.After(3 * time.Second):
		t.Fatal("timed out waiting for bot response")
	}

	// Verify run recorded in dm_user1.db
	var runID string
	var status, currentState string
	var iteration, maxIterations int
	activeDBs := gw.MemoryManager().ActiveDBs()
	require.Contains(t, activeDBs, "dm_user1.db")
	err = activeDBs["dm_user1.db"].QueryRowContext(ctx, "SELECT id, status, current_state, iteration, max_iterations FROM fsm_runs").Scan(&runID, &status, &currentState, &iteration, &maxIterations)
	require.NoError(t, err)
	assert.Equal(t, 20, maxIterations)
	assert.Equal(t, 20, iteration)
	assert.Equal(t, "COMPLETED", status)
	assert.Equal(t, "COMPLETED", currentState)
}

func TestGateway_RunMaintenance(t *testing.T) {
	tempDir := t.TempDir()
	cfg := &config.Config{
		DataDir:          tempDir,
		FSMRetentionDays: 7,
	}

	gw := NewGateway(cfg, nil)
	defer gw.Stop()

	ctx := context.Background()
	provider := NewMemoryStoreProvider(gw.MemoryManager(), tempDir)

	thStore, err := provider.GetStore(ctx, "townhall", false)
	require.NoError(t, err)
	dmStore, err := provider.GetStore(ctx, "alice", true)
	require.NoError(t, err)

	now := time.Now().Unix()
	tenDaysAgo := now - (10 * 86400)
	oneDayAgo := now - (1 * 86400)

	// In townhall.db: 1 expired completed run, 1 recent completed run, 1 active run
	expiredRun := &fsm.FSMRun{
		ID:            "run_expired_th",
		ChatID:        "townhall",
		UserID:        "user1",
		FSMType:       fsm.FSMTypeToolLoop,
		Status:        fsm.RunStatusCompleted,
		CurrentState:  fsm.StateCompleted,
		Iteration:     2,
		MaxIterations: 10,
		ContextJSON:   "[]",
		CreatedAt:     tenDaysAgo,
		UpdatedAt:     tenDaysAgo,
	}
	require.NoError(t, thStore.CreateRun(ctx, expiredRun))
	require.NoError(t, thStore.CreateSteps(ctx, []fsm.FSMStep{
		{
			ID:            "step_exp_1",
			RunID:         expiredRun.ID,
			Iteration:     1,
			StepIndex:     0,
			ToolName:      "web_search",
			ToolCallID:    "call_1",
			ArgsJSON:      "{}",
			ExecutionMode: fsm.ExecutionModeParallel,
			Status:        fsm.StepStatusCompleted,
		},
	}))

	recentRun := &fsm.FSMRun{
		ID:            "run_recent_th",
		ChatID:        "townhall",
		UserID:        "user1",
		FSMType:       fsm.FSMTypeToolLoop,
		Status:        fsm.RunStatusCompleted,
		CurrentState:  fsm.StateCompleted,
		Iteration:     1,
		MaxIterations: 10,
		ContextJSON:   "[]",
		CreatedAt:     oneDayAgo,
		UpdatedAt:     oneDayAgo,
	}
	require.NoError(t, thStore.CreateRun(ctx, recentRun))

	activeRun := &fsm.FSMRun{
		ID:            "run_active_th",
		ChatID:        "townhall",
		UserID:        "user1",
		FSMType:       fsm.FSMTypeToolLoop,
		Status:        fsm.RunStatusRunning,
		CurrentState:  fsm.StateExecuteSteps,
		Iteration:     1,
		MaxIterations: 10,
		ContextJSON:   "[]",
		CreatedAt:     tenDaysAgo,
		UpdatedAt:     tenDaysAgo,
	}
	require.NoError(t, thStore.CreateRun(ctx, activeRun))

	// In dm_alice.db: 1 expired failed run, 1 waiting run
	expiredFailedRun := &fsm.FSMRun{
		ID:            "run_expired_dm",
		ChatID:        "dm_alice",
		UserID:        "alice",
		FSMType:       fsm.FSMTypeToolLoop,
		Status:        fsm.RunStatusFailed,
		CurrentState:  fsm.StateFailed,
		Iteration:     3,
		MaxIterations: 20,
		ContextJSON:   "[]",
		CreatedAt:     tenDaysAgo,
		UpdatedAt:     tenDaysAgo,
	}
	require.NoError(t, dmStore.CreateRun(ctx, expiredFailedRun))

	waitingRun := &fsm.FSMRun{
		ID:            "run_waiting_dm",
		ChatID:        "dm_alice",
		UserID:        "alice",
		FSMType:       fsm.FSMTypeToolLoop,
		Status:        fsm.RunStatusWaiting,
		CurrentState:  fsm.StateWaiting,
		Iteration:     2,
		MaxIterations: 20,
		ContextJSON:   "[]",
		CreatedAt:     tenDaysAgo,
		UpdatedAt:     tenDaysAgo,
	}
	require.NoError(t, dmStore.CreateRun(ctx, waitingRun))

	// Run maintenance
	gw.RunMaintenance(ctx)

	// In townhall.db: expiredRun deleted, steps deleted, recentRun & activeRun preserved
	_, err = thStore.GetRun(ctx, expiredRun.ID)
	assert.ErrorIs(t, err, fsm.ErrRunNotFound)

	steps, err := thStore.GetStepsForIteration(ctx, expiredRun.ID, 1)
	require.NoError(t, err)
	assert.Empty(t, steps)

	gotRecent, err := thStore.GetRun(ctx, recentRun.ID)
	require.NoError(t, err)
	assert.NotNil(t, gotRecent)

	gotActive, err := thStore.GetRun(ctx, activeRun.ID)
	require.NoError(t, err)
	assert.NotNil(t, gotActive)

	// In dm_alice.db: expiredFailedRun deleted, waitingRun preserved
	_, err = dmStore.GetRun(ctx, expiredFailedRun.ID)
	assert.ErrorIs(t, err, fsm.ErrRunNotFound)

	gotWaiting, err := dmStore.GetRun(ctx, waitingRun.ID)
	require.NoError(t, err)
	assert.NotNil(t, gotWaiting)
}

type failingStoreProvider struct{}

func (f *failingStoreProvider) GetStore(ctx context.Context, chatID string, isDM bool) (*fsm.Store, error) {
	return nil, fmt.Errorf("simulated sqlite database failure")
}

func (f *failingStoreProvider) ActiveStores(ctx context.Context) ([]*fsm.Store, error) {
	return nil, nil
}

func TestGateway_FSMToolLoop_FallbackToVolatileOnFSMFailure(t *testing.T) {
	tempDir := t.TempDir()
	var llmCallCount int32

	llmServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		count := atomic.AddInt32(&llmCallCount, 1)

		var req openai.ChatCompletionRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		if count == 1 {
			// First call from volatile loop: request a tool call
			resp := openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role: openai.ChatMessageRoleAssistant,
							ToolCalls: []openai.ToolCall{
								{
									ID:   "call_fallback",
									Type: openai.ToolTypeFunction,
									Function: openai.FunctionCall{
										Name:      "web_search",
										Arguments: `{"query":"test"}`,
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

		// Second call from volatile loop: return final answer
		resp := openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{
					Message: openai.ChatCompletionMessage{
						Role:    openai.ChatMessageRoleAssistant,
						Content: "Volatile fallback successful reply.",
					},
				},
			},
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer llmServer.Close()

	sentMsgs := make(chan models.ClientMessage, 10)
	besedkaServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/me":
			_ = json.NewEncoder(w).Encode(models.User{ID: "bot_1", DisplayName: "Bob", UserName: "bot"})
		case "/api/users":
			_ = json.NewEncoder(w).Encode([]models.User{{ID: "u1", DisplayName: "Alice"}})
		case "/api/chats":
			_ = json.NewEncoder(w).Encode([]models.Chat{{ID: "townhall"}})
		case "/api/chat":
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
				sentMsgs <- cm
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

	// Inject FSM Engine with failing store provider to trigger infrastructure error
	gw.mu.Lock()
	gw.fsmEngine = fsm.NewEngine(&failingStoreProvider{}, llmClient, gw.toolsRegistry)
	gw.mu.Unlock()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	require.NoError(t, gw.DialWebSocket(ctx))
	_, err := gw.FetchBotUser(ctx)
	require.NoError(t, err)

	err = gw.ProcessMessage(ctx, models.Message{
		ChatID:    "townhall",
		UserID:    "u1",
		Content:   "@bot search test",
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)

	select {
	case reply := <-sentMsgs:
		assert.Contains(t, reply.Content, "Volatile fallback successful reply.")
	case <-time.After(3 * time.Second):
		t.Fatal("timed out waiting for fallback bot response")
	}
}

