package gateway

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"bob/internal/agentapi"
	"bob/internal/cli"
	"bob/internal/config"
	"bob/internal/knowledge"
	"bob/internal/llm"
	"bob/internal/models"
	"bob/internal/sandbox"
	"bob/internal/scheduler"
	"bob/internal/tools"

	"github.com/fasthttp/websocket"
	openai "github.com/sashabaranov/go-openai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type uiE2EHarness struct {
	gw            *Gateway
	sbxMgr        *sandbox.Manager
	wsCh          chan models.ClientMessage
	progressPosts int32
	progressSeq   int64
	msgGetPosts   int32
	mu            sync.Mutex
	llmReqs       []openai.ChatCompletionRequest
}

func (h *uiE2EHarness) nextSend(t *testing.T) models.ClientMessage {
	t.Helper()
	deadline := time.After(4 * time.Second)
	for {
		select {
		case cm := <-h.wsCh:
			if cm.Type != models.ClientMessageTypeSend {
				continue
			}
			return cm
		case <-deadline:
			t.Fatal("timed out waiting for outgoing besedka WS send frame")
			return models.ClientMessage{}
		}
	}
}

func (h *uiE2EHarness) llmCallCount() int {
	h.mu.Lock()
	defer h.mu.Unlock()
	return len(h.llmReqs)
}

func (h *uiE2EHarness) lastLLMRequest() openai.ChatCompletionRequest {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.llmReqs[len(h.llmReqs)-1]
}

func setupUIE2EHarness(
	t *testing.T,
	llmHandler func(callIndex int, req openai.ChatCompletionRequest) openai.ChatCompletionResponse,
	townhallHistory ...models.Message,
) *uiE2EHarness {
	t.Helper()

	h := &uiE2EHarness{
		wsCh:        make(chan models.ClientMessage, 64),
		progressSeq: 1000,
	}

	llmSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req openai.ChatCompletionRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		h.mu.Lock()
		h.llmReqs = append(h.llmReqs, req)
		callIdx := len(h.llmReqs)
		h.mu.Unlock()

		resp := llmHandler(callIdx, req)
		if resp.Model == "" {
			resp.Model = "test-model"
		}
		if resp.Object == "" {
			resp.Object = "chat.completion"
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	t.Cleanup(llmSrv.Close)

	besedkaSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.URL.Path == "/api/me":
			_ = json.NewEncoder(w).Encode(models.User{ID: "bot-1", UserName: "bob", DisplayName: "Bob"})
		case r.URL.Path == "/api/users":
			_ = json.NewEncoder(w).Encode([]models.User{
				{ID: "alice", UserName: "alice", DisplayName: "Alice"},
				{ID: "mallory", UserName: "mallory", DisplayName: "Mallory"},
				{ID: "bot-1", UserName: "bob", DisplayName: "Bob"},
			})
		case r.URL.Path == "/api/chats":
			_ = json.NewEncoder(w).Encode([]models.Chat{
				{ID: "townhall", Name: "Townhall", Type: "townhall"},
				{ID: "dm_alice_bot-1", IsDM: true, Type: "dm", UserIDs: []string{"alice", "bot-1"}, TargetUserID: "alice"},
				{ID: "dm_mallory_bot-1", IsDM: true, Type: "dm", UserIDs: []string{"mallory", "bot-1"}, TargetUserID: "mallory"},
			})
		case r.Method == http.MethodGet && strings.HasPrefix(r.URL.Path, "/api/chats/") && strings.HasSuffix(r.URL.Path, "/messages"):
			atomic.AddInt32(&h.msgGetPosts, 1)
			if strings.Contains(r.URL.Path, "/townhall/") {
				fromSeq, _ := strconv.ParseInt(r.URL.Query().Get("fromSeq"), 10, 64)
				toSeq, _ := strconv.ParseInt(r.URL.Query().Get("toSeq"), 10, 64)
				var filtered []models.Message
				for _, m := range townhallHistory {
					if fromSeq > 0 && m.Seq < fromSeq {
						continue
					}
					if toSeq > 0 && m.Seq > toSeq {
						continue
					}
					filtered = append(filtered, m)
				}
				_ = json.NewEncoder(w).Encode(filtered)
				return
			}
			_ = json.NewEncoder(w).Encode([]models.Message{})
		case r.Method == http.MethodPost && strings.HasPrefix(r.URL.Path, "/api/chats/") && strings.HasSuffix(r.URL.Path, "/messages"):
			atomic.AddInt32(&h.progressPosts, 1)
			seq := atomic.AddInt64(&h.progressSeq, 1)
			_ = json.NewEncoder(w).Encode(map[string]int64{"seq": seq})
		case r.URL.Path == "/api/chat":
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
				h.wsCh <- cm
			}
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(besedkaSrv.Close)

	cfg := &config.Config{
		DataDir:                   t.TempDir(),
		BesedkaURL:                besedkaSrv.URL,
		OpenAIAPIKey:              "test-key",
		OpenAIBaseURL:             llmSrv.URL + "/v1",
		OpenAIModel:               "test-model",
		BotHandle:                 "@bob",
		MsgRingBufferSize:         50,
		TownhallMaxParagraphs:     5,
		DMMaxParagraphs:           10,
		TownhallToolMaxIterations: 10,
		DMToolMaxIterations:       20,
		MaxAttachmentSizeBytes:    25 * 1024 * 1024,
		SandboxEnabled:            true,
		SandboxDrivers:            []string{"bwrap"},
		SandboxMaxLifetime:        30 * time.Minute,
		SandboxDefaultExecTimeout: 1 * time.Minute,
		SandboxMaxExecTimeout:     10 * time.Minute,
	}

	llmClient := llm.NewClient(cfg, llmSrv.Client())
	gw := NewGateway(cfg, llmClient)
	t.Cleanup(gw.Stop)
	gw.httpClient = besedkaSrv.Client()

	sbxMgr := sandbox.NewManager(cfg.SandboxConfig(), []sandbox.Driver{&mockGatewaySandboxDriver{}})
	t.Cleanup(func() { _ = sbxMgr.Close() })
	gw.SetSandboxManager(sbxMgr)

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	t.Cleanup(cancel)

	require.NoError(t, gw.DialWebSocket(ctx))
	_, err := gw.FetchBotUser(ctx)
	require.NoError(t, err)
	_, err = gw.FetchUsers(ctx)
	require.NoError(t, err)
	chats, err := gw.FetchChats(ctx)
	require.NoError(t, err)
	gw.chatCache.SetAll(chats)

	if len(townhallHistory) > 0 {
		var maxSeq int64
		for _, m := range townhallHistory {
			if m.Seq > maxSeq {
				maxSeq = m.Seq
			}
		}
		gw.WarmupChat(ctx, "townhall", maxSeq)
	}

	h.gw = gw
	h.sbxMgr = sbxMgr
	return h
}

func TestUIEndToEnd_SandboxApprovalKeepDestroyDenyAndStaleCard(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	h := setupUIE2EHarness(t, func(call int, _ openai.ChatCompletionRequest) openai.ChatCompletionResponse {
		switch call {
		case 1:
			return openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{{
					Message: openai.ChatCompletionMessage{
						Role: openai.ChatMessageRoleAssistant,
						ToolCalls: []openai.ToolCall{{
							ID:   "call_sbx_a",
							Type: openai.ToolTypeFunction,
							Function: openai.FunctionCall{
								Name:      "sandbox_request",
								Arguments: `{"driver":"bwrap","network":"none","reason":"Run check A"}`,
							},
						}},
					},
				}},
			}
		case 2:
			return openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{{
					Message: openai.ChatCompletionMessage{
						Role:    openai.ChatMessageRoleAssistant,
						Content: "Requesting sandbox A now.",
					},
				}},
			}
		case 3:
			return openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{{
					Message: openai.ChatCompletionMessage{
						Role: openai.ChatMessageRoleAssistant,
						ToolCalls: []openai.ToolCall{{
							ID:   "call_sbx_b",
							Type: openai.ToolTypeFunction,
							Function: openai.FunctionCall{
								Name:      "sandbox_request",
								Arguments: `{"driver":"bwrap","network":"none","reason":"Run check B"}`,
							},
						}},
					},
				}},
			}
		case 4:
			return openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{{
					Message: openai.ChatCompletionMessage{
						Role:    openai.ChatMessageRoleAssistant,
						Content: "Requesting sandbox B now.",
					},
				}},
			}
		default:
			return openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{{
					Message: openai.ChatCompletionMessage{
						Role:    openai.ChatMessageRoleAssistant,
						Content: "Sandbox B diagnostics passed.",
					},
				}},
			}
		}
	})

	// 1. DM request triggers sandbox_request A approval card
	require.NoError(t, h.gw.ProcessMessage(ctx, models.Message{
		Seq: 1, ChatID: "dm_alice_bot-1", UserID: "alice", Type: models.MessageTypeText, Content: "Run script A",
	}))
	cardA := h.nextSend(t)
	assert.Contains(t, cardA.Content, "Sandbox Approval Requested")
	require.NotNil(t, cardA.UI)
	require.Len(t, cardA.UI.Elements, 2)
	approveAID := cardA.UI.Elements[0].ID
	denyAID := cardA.UI.Elements[1].ID
	require.True(t, strings.HasPrefix(approveAID, "bob:sandbox:approve:"))
	require.True(t, strings.HasPrefix(denyAID, "bob:sandbox:deny:"))

	// 2. Deny A via button click -> 0 extra LLM calls
	callsAfterA := h.llmCallCount()
	require.NoError(t, h.gw.ProcessMessage(ctx, models.Message{
		Seq: 2, ChatID: "dm_alice_bot-1", UserID: "alice", Type: models.MessageTypeUIResponse,
		UIResponse: &models.UIResponseData{ParentSeq: 1, SelectedIDs: []string{denyAID}},
	}))
	denyAck := h.nextSend(t)
	assert.Contains(t, denyAck.Content, "Sandbox request denied")
	assert.Equal(t, callsAfterA, h.llmCallCount())

	// 3. New DM request creates replacement sandbox B
	require.NoError(t, h.gw.ProcessMessage(ctx, models.Message{
		Seq: 3, ChatID: "dm_alice_bot-1", UserID: "alice", Type: models.MessageTypeText, Content: "Run script B",
	}))
	cardB := h.nextSend(t)
	require.NotNil(t, cardB.UI)
	approveBID := cardB.UI.Elements[0].ID
	require.NotEqual(t, approveAID, approveBID)

	// 4. Clicking stale A Approve button must NOT approve B
	callsBeforeStale := h.llmCallCount()
	require.NoError(t, h.gw.ProcessMessage(ctx, models.Message{
		Seq: 4, ChatID: "dm_alice_bot-1", UserID: "alice", Type: models.MessageTypeUIResponse,
		UIResponse: &models.UIResponseData{ParentSeq: 1, SelectedIDs: []string{approveAID}},
	}))
	staleWarn := h.nextSend(t)
	assert.Contains(t, staleWarn.Content, "outdated")
	assert.Equal(t, callsBeforeStale, h.llmCallCount())
	sbxB, exists := h.sbxMgr.GetStatus("alice")
	require.True(t, exists)
	assert.Equal(t, sandbox.StatusPendingApproval, sbxB.Status)

	// 5. Clicking B's Approve triggers continuation turn + attaches bound Destroy/Keep UI
	require.NoError(t, h.gw.ProcessMessage(ctx, models.Message{
		Seq: 5, ChatID: "dm_alice_bot-1", UserID: "alice", Type: models.MessageTypeUIResponse,
		UIResponse: &models.UIResponseData{ParentSeq: 4, SelectedIDs: []string{approveBID}},
	}))
	contReply := h.nextSend(t)
	assert.Contains(t, contReply.Content, "Sandbox B diagnostics passed.")
	require.NotNil(t, contReply.UI)
	require.Len(t, contReply.UI.Elements, 2)
	assert.Equal(t, "bob:sandbox:destroy:"+sbxB.ID, contReply.UI.Elements[0].ID)
	assert.Equal(t, "bob:sandbox:keep:"+sbxB.ID, contReply.UI.Elements[1].ID)

	// 6. Keep Running acknowledges with NO LLM call
	callsBeforeKeep := h.llmCallCount()
	require.NoError(t, h.gw.ProcessMessage(ctx, models.Message{
		Seq: 6, ChatID: "dm_alice_bot-1", UserID: "alice", Type: models.MessageTypeUIResponse,
		UIResponse: &models.UIResponseData{ParentSeq: 5, SelectedIDs: []string{"bob:sandbox:keep:" + sbxB.ID}},
	}))
	keepAck := h.nextSend(t)
	assert.Contains(t, keepAck.Content, "Keeping your sandbox running")
	assert.Equal(t, callsBeforeKeep, h.llmCallCount())

	// 7. Destroy terminates sandbox B
	require.NoError(t, h.gw.ProcessMessage(ctx, models.Message{
		Seq: 7, ChatID: "dm_alice_bot-1", UserID: "alice", Type: models.MessageTypeUIResponse,
		UIResponse: &models.UIResponseData{ParentSeq: 5, SelectedIDs: []string{"bob:sandbox:destroy:" + sbxB.ID}},
	}))
	destroyAck := h.nextSend(t)
	assert.Contains(t, destroyAck.Content, "Sandbox terminated")
	_, exists = h.sbxMgr.GetStatus("alice")
	assert.False(t, exists)
}

func TestUIEndToEnd_SingleAndMultiApprovalProposals(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	h := setupUIE2EHarness(t, func(call int, _ openai.ChatCompletionRequest) openai.ChatCompletionResponse {
		switch call {
		case 1:
			return openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{{
					Message: openai.ChatCompletionMessage{
						Role: openai.ChatMessageRoleAssistant,
						ToolCalls: []openai.ToolCall{{
							ID:   "call_mem",
							Type: openai.ToolTypeFunction,
							Function: openai.FunctionCall{
								Name:      "propose_memory",
								Arguments: `{"type":"preference","content":"Use concise bullet summaries","confidence":0.95}`,
							},
						}},
					},
				}},
			}
		case 2:
			return openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{{
					Message: openai.ChatCompletionMessage{
						Role:    openai.ChatMessageRoleAssistant,
						Content: "I proposed saving that memory preference.",
					},
				}},
			}
		case 3:
			return openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{{
					Message: openai.ChatCompletionMessage{
						Role: openai.ChatMessageRoleAssistant,
						ToolCalls: []openai.ToolCall{
							{
								ID:   "call_skill",
								Type: openai.ToolTypeFunction,
								Function: openai.FunctionCall{
									Name:      "propose_skill",
									Arguments: `{"name":"release_checks","description":"Verify release gates","triggers":["release"],"tags":["qa"],"instructions_markdown":"# Release\nRun make check"}`,
								},
							},
							{
								ID:   "call_sched",
								Type: openai.ToolTypeFunction,
								Function: openai.FunctionCall{
									Name:      "schedule_task",
									Arguments: `{"schedule_name":"nightly_qa","type":"interval","schedule_spec":"3600","instruction":"Check QA status","run_timeout_seconds":60,"max_turns":10}`,
								},
							},
						},
					},
				}},
			}
		default:
			return openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{{
					Message: openai.ChatCompletionMessage{
						Role:    openai.ChatMessageRoleAssistant,
						Content: "Both the skill and schedule have been proposed.",
					},
				}},
			}
		}
	})

	// 1. Single proposal attaches approval UI directly to main reply
	require.NoError(t, h.gw.ProcessMessage(ctx, models.Message{
		Seq: 10, ChatID: "dm_alice_bot-1", UserID: "alice", Type: models.MessageTypeText, Content: "Remember I prefer concise bullets",
	}))
	singleReply := h.nextSend(t)
	assert.Contains(t, singleReply.Content, "I proposed saving that memory preference.")
	require.NotNil(t, singleReply.UI)
	require.Len(t, singleReply.UI.Elements, 2)
	memApproveID := singleReply.UI.Elements[0].ID
	require.True(t, strings.HasPrefix(memApproveID, "bob:mem:approve:"))
	memVerID := strings.TrimPrefix(memApproveID, "bob:mem:approve:")

	require.NoError(t, h.gw.ProcessMessage(ctx, models.Message{
		Seq: 11, ChatID: "dm_alice_bot-1", UserID: "alice", Type: models.MessageTypeUIResponse,
		UIResponse: &models.UIResponseData{ParentSeq: 10, SelectedIDs: []string{memApproveID}},
	}))
	assert.Contains(t, h.nextSend(t).Content, "approved and activated")

	kStore, err := h.gw.StoreProvider().GetKnowledgeStore(ctx, "dm_alice_bot-1", true)
	require.NoError(t, err)
	memVer, err := kStore.GetMemoryVersion(ctx, memVerID)
	require.NoError(t, err)
	assert.Equal(t, knowledge.VersionStatusApproved, memVer.Status)

	// 2. Two proposals in one turn send clean main reply + 2 follow-up approval cards
	require.NoError(t, h.gw.ProcessMessage(ctx, models.Message{
		Seq: 12, ChatID: "dm_alice_bot-1", UserID: "alice", Type: models.MessageTypeText, Content: "Create release skill and nightly check",
	}))
	mainMulti := h.nextSend(t)
	assert.Contains(t, mainMulti.Content, "Both the skill and schedule have been proposed.")
	assert.Nil(t, mainMulti.UI, "multi-proposal turn should send approvals as separate follow-up cards")

	card1 := h.nextSend(t)
	card2 := h.nextSend(t)
	require.NotNil(t, card1.UI)
	require.NotNil(t, card2.UI)
	skillApproveID := card1.UI.Elements[0].ID
	schedDenyID := card2.UI.Elements[1].ID
	require.True(t, strings.HasPrefix(skillApproveID, "bob:skill:approve:"))
	require.True(t, strings.HasPrefix(schedDenyID, "bob:sched:deny:"))
	skillVerID := strings.TrimPrefix(skillApproveID, "bob:skill:approve:")
	schedID := strings.TrimPrefix(schedDenyID, "bob:sched:deny:")

	require.NoError(t, h.gw.ProcessMessage(ctx, models.Message{
		Seq: 13, ChatID: "dm_alice_bot-1", UserID: "alice", Type: models.MessageTypeUIResponse,
		UIResponse: &models.UIResponseData{ParentSeq: 12, SelectedIDs: []string{skillApproveID}},
	}))
	assert.Contains(t, h.nextSend(t).Content, "approved and activated")

	require.NoError(t, h.gw.ProcessMessage(ctx, models.Message{
		Seq: 14, ChatID: "dm_alice_bot-1", UserID: "alice", Type: models.MessageTypeUIResponse,
		UIResponse: &models.UIResponseData{ParentSeq: 12, SelectedIDs: []string{schedDenyID}},
	}))
	assert.Contains(t, h.nextSend(t).Content, "Schedule request denied")

	skillVer, err := kStore.GetSkillVersion(ctx, skillVerID)
	require.NoError(t, err)
	assert.Equal(t, knowledge.VersionStatusApproved, skillVer.Status)

	schedStore, err := h.gw.StoreProvider().GetSchedulerStore(ctx, "dm_alice_bot-1", true)
	require.NoError(t, err)
	sched, err := schedStore.GetSchedule(ctx, schedID)
	require.NoError(t, err)
	assert.Equal(t, scheduler.ScheduleStatusDenied, sched.Status)
}

func TestUIEndToEnd_FreeFormAskUserFastFinishAndSelection(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	h := setupUIE2EHarness(t, func(call int, req openai.ChatCompletionRequest) openai.ChatCompletionResponse {
		switch call {
		case 1:
			// Must expose ask_user in interactive Townhall toolset
			hasAskUser := false
			for _, tool := range req.Tools {
				if tool.Function != nil && tool.Function.Name == tools.AskUserToolName {
					hasAskUser = true
					break
				}
			}
			assert.True(t, hasAskUser, "interactive Townhall request must expose ask_user tool")
			return openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{{
					Message: openai.ChatCompletionMessage{
						Role: openai.ChatMessageRoleAssistant,
						ToolCalls: []openai.ToolCall{{
							ID:   "call_ask_deploy",
							Type: openai.ToolTypeFunction,
							Function: openai.FunctionCall{
								Name: tools.AskUserToolName,
								Arguments: `{
									"message": "Choose deployment options:",
									"elements": [
										{"id": "us", "type": "radio", "label": "US", "default": true},
										{"id": "eu", "type": "radio", "label": "EU"},
										{"id": "notify", "type": "checkbox", "label": "Notify ops"}
									]
								}`,
							},
						}},
					},
				}},
			}
		case 2:
			return openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{{
					Message: openai.ChatCompletionMessage{
						Role:    openai.ChatMessageRoleAssistant,
						Content: "Deploying to EU with notifications enabled.",
					},
				}},
			}
		default:
			t.Fatalf("unexpected extra LLM call %d", call)
			return openai.ChatCompletionResponse{}
		}
	})

	// 1. User mentions @bob in Townhall -> 1 LLM call, 0 progress HTTP calls, WS delivers UI prompt
	require.NoError(t, h.gw.ProcessMessage(ctx, models.Message{
		Seq: 39, ChatID: "townhall", UserID: "alice", Type: models.MessageTypeText, Content: "@bob help me deploy",
	}))
	askMsg := h.nextSend(t)
	assert.Equal(t, 1, h.llmCallCount(), "ask_user must complete in a single LLM call via FSM fast-finish")
	assert.Equal(t, int32(0), atomic.LoadInt32(&h.progressPosts), "ask_user-only turn must suppress root progress card")
	assert.Equal(t, "Choose deployment options:", askMsg.Content)
	require.NotNil(t, askMsg.UI)
	require.Len(t, askMsg.UI.Elements, 3)

	// 2. Simulate Besedka WebSocket self-echo assigning Seq=40 to Bob's UI prompt
	require.NoError(t, h.gw.ProcessMessage(ctx, models.Message{
		Seq: 40, ChatID: "townhall", UserID: "bot-1", Type: models.MessageTypeText,
		Content: askMsg.Content, UI: askMsg.UI,
	}))

	// 3. User clicks EU + Notify and submits ui_response referencing ParentSeq=40
	require.NoError(t, h.gw.ProcessMessage(ctx, models.Message{
		Seq: 41, ChatID: "townhall", UserID: "alice", Type: models.MessageTypeUIResponse,
		Content:    "<p>@bob Selected: EU, Notify ops</p>",
		RawContent: "@bob Selected: EU, Notify ops",
		UIResponse: &models.UIResponseData{ParentSeq: 40, SelectedIDs: []string{"eu", "notify"}},
	}))
	finalReply := h.nextSend(t)
	assert.Equal(t, "Deploying to EU with notifications enabled.", finalReply.Content)
	assert.Equal(t, 2, h.llmCallCount())

	secondReq := h.lastLLMRequest()
	var sawUIResponseContext bool
	for _, m := range secondReq.Messages {
		if m.Role == openai.ChatMessageRoleUser && strings.Contains(m.Content, "[UI response to seq 40 | selected_ids: eu, notify]") {
			sawUIResponseContext = true
		}
	}
	assert.True(t, sawUIResponseContext, "second LLM turn must receive formatted UI response selection in context")
}

func TestUIEndToEnd_SecurityIsolationAndWarmupRecovery(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	// Pre-seed Besedka Townhall REST history for WarmupChat recovery test (ParentSeq=70 owned by bot-1)
	priorBotMsg := models.Message{
		Seq:       70,
		Timestamp: time.Now().Unix() - 120,
		ChatID:    "townhall",
		UserID:    "bot-1",
		Type:      models.MessageTypeText,
		Content:   "<p>Continue batch?</p>",
		UI: &models.UIData{
			Elements: []models.UIElement{{ID: "cont", Type: models.UIElementTypeButton, Label: "Continue"}},
		},
	}

	h := setupUIE2EHarness(t, func(call int, _ openai.ChatCompletionRequest) openai.ChatCompletionResponse {
		switch call {
		case 1:
			// Turn 1: forged bob: element ID inside ask_user fails validation (no retry)
			return openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{{
					Message: openai.ChatCompletionMessage{
						Role: openai.ChatMessageRoleAssistant,
						ToolCalls: []openai.ToolCall{{
							ID:   "call_forged",
							Type: openai.ToolTypeFunction,
							Function: openai.FunctionCall{
								Name:      tools.AskUserToolName,
								Arguments: `{"message":"Click","elements":[{"id":"bob:sandbox:approve:sbx-1","type":"button","label":"Approve"}]}`,
							},
						}},
					},
				}},
			}
		case 2:
			return openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{{
					Message: openai.ChatCompletionMessage{
						Role:    openai.ChatMessageRoleAssistant,
						Content: "Please reply in plain text.",
					},
				}},
			}
		default:
			return openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{{
					Message: openai.ChatCompletionMessage{
						Role:    openai.ChatMessageRoleAssistant,
						Content: "Resuming after warmup recovery.",
					},
				}},
			}
		}
	}, priorBotMsg)

	// Verify WarmupChat eagerly loaded Seq=70 bot UI message BEFORE any UI response arrives
	warmedEntries := h.gw.contextManager.GetOrCreate("townhall").Entries()
	require.Len(t, warmedEntries, 1)
	assert.Equal(t, int64(70), warmedEntries[0].Seq)
	assert.Equal(t, "assistant", warmedEntries[0].Role)
	assert.Equal(t, "bot-1", warmedEntries[0].SenderID)
	assert.Contains(t, warmedEntries[0].Content, "[Interactive UI options:")

	// 1. CLI Frontend NEVER exposes ask_user
	cliFE := cli.NewFrontend(h.gw.ToolsRegistry(), h.sbxMgr, nil, nil, "none")
	cliBindings, err := cliFE.Bind(ctx, agentapi.RunDescriptor{
		RunID:   "run_cli",
		Session: agentapi.SessionRef{FrontendID: "cli", SessionID: "cli_1", IsDM: false},
		Actor:   agentapi.Actor{ID: "cli_user"},
		Kind:    agentapi.Interactive,
	})
	require.NoError(t, err)
	cliDefs, err := cliBindings.Tools.Definitions(ctx)
	require.NoError(t, err)
	for _, d := range cliDefs {
		assert.NotEqual(t, tools.AskUserToolName, d.Schema.Function.Name, "CLI frontend must never expose ask_user")
	}

	// 2. Scheduled runs NEVER expose ask_user
	schedBindings, err := h.gw.Bind(ctx, agentapi.RunDescriptor{
		RunID:   "sched_run_1",
		Session: agentapi.SessionRef{FrontendID: "besedka", SessionID: "dm_alice_bot-1", ScopeID: "dm_alice_bot-1", IsDM: true},
		Actor:   agentapi.Actor{ID: "alice"},
		Kind:    agentapi.Scheduled,
	})
	require.NoError(t, err)
	schedDefs, err := schedBindings.Tools.Definitions(ctx)
	require.NoError(t, err)
	for _, d := range schedDefs {
		assert.NotEqual(t, tools.AskUserToolName, d.Schema.Function.Name, "scheduled runs must never expose ask_user")
	}

	// 3. Cross-chat schedule approval & Townhall spoofed sandbox action rejected cleanly
	mallorySchedStore, err := h.gw.StoreProvider().GetSchedulerStore(ctx, "dm_mallory_bot-1", true)
	require.NoError(t, err)
	now := time.Now().Unix()
	require.NoError(t, mallorySchedStore.CreateSchedule(ctx, &scheduler.Schedule{
		ID: "sched_mallory_e2e", Name: "mallory_e2e", ChatID: "dm_mallory_bot-1", UserID: "mallory",
		ScheduleType: scheduler.ScheduleTypeInterval, IntervalSeconds: 600,
		Instruction: "Run", Status: scheduler.ScheduleStatusPendingApproval,
		RunTimeoutSeconds: 60, MaxTurns: 10, NextRunAt: now + 600, CreatedAt: now, UpdatedAt: now,
	}, nil))

	require.NoError(t, h.gw.ProcessMessage(ctx, models.Message{
		Seq: 71, ChatID: "dm_alice_bot-1", UserID: "alice", Type: models.MessageTypeUIResponse,
		UIResponse: &models.UIResponseData{ParentSeq: 70, SelectedIDs: []string{"bob:sched:approve:sched_mallory_e2e"}},
	}))
	assert.Contains(t, h.nextSend(t).Content, "Schedule not found")
	stillPending, err := mallorySchedStore.GetSchedule(ctx, "sched_mallory_e2e")
	require.NoError(t, err)
	assert.Equal(t, scheduler.ScheduleStatusPendingApproval, stillPending.Status)

	require.NoError(t, h.gw.ProcessMessage(ctx, models.Message{
		Seq: 72, ChatID: "townhall", UserID: "alice", Type: models.MessageTypeUIResponse,
		UIResponse: &models.UIResponseData{ParentSeq: 70, SelectedIDs: []string{"bob:sandbox:approve:sbx-1"}},
	}))
	assert.Contains(t, h.nextSend(t).Content, "only available in direct messages")
	assert.Equal(t, 0, h.llmCallCount())

	// 4. Forged bob: ID inside ask_user fails validation (maxAttempts=1) and falls back cleanly on LLM call 2
	require.NoError(t, h.gw.ProcessMessage(ctx, models.Message{
		Seq: 73, ChatID: "townhall", UserID: "alice", Type: models.MessageTypeText, Content: "@bob ask with buttons",
	}))
	fallbackMsg := h.nextSend(t)
	assert.Equal(t, "Please reply in plain text.", fallbackMsg.Content)
	assert.Nil(t, fallbackMsg.UI)
	assert.Equal(t, 2, h.llmCallCount())

	// 5. ui_response in Townhall resolves ownership for Seq=70 from warmed buffer without extra REST lookups.
	getsBeforeResp := atomic.LoadInt32(&h.msgGetPosts)
	require.NoError(t, h.gw.ProcessMessage(ctx, models.Message{
		Seq: 74, ChatID: "townhall", UserID: "alice", Type: models.MessageTypeUIResponse,
		Content:    "<p>@bob Selected: Continue</p>",
		RawContent: "@bob Selected: Continue",
		UIResponse: &models.UIResponseData{ParentSeq: 70, SelectedIDs: []string{"cont"}},
	}))
	warmedReply := h.nextSend(t)
	assert.Equal(t, "Resuming after warmup recovery.", warmedReply.Content)
	assert.Equal(t, 3, h.llmCallCount())
	assert.Equal(t, getsBeforeResp, atomic.LoadInt32(&h.msgGetPosts), "warmed ring buffer should resolve Seq=70 without fallback REST fetch")
}

// Compile-time guard verifying Gateway satisfies agentapi.Frontend.
var _ agentapi.Frontend = (*Gateway)(nil)
