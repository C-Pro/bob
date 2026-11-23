package gateway

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"bob/internal/agentapi"
	"bob/internal/chatcontext"
	"bob/internal/config"
	"bob/internal/knowledge"
	"bob/internal/llm"
	"bob/internal/models"
	"bob/internal/sandbox"
	"bob/internal/scheduler"

	"github.com/fasthttp/websocket"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestToBesedkaUIData(t *testing.T) {
	assert.Nil(t, ToBesedkaUIData(nil))
	assert.Nil(t, ToBesedkaUIData([]agentapi.UIElement{}))

	ui := ToBesedkaUIData([]agentapi.UIElement{
		{
			ID:          "opt_a",
			Type:        agentapi.UIElementRadio,
			Label:       "Option A",
			Description: "First choice",
			Default:     true,
		},
		{
			ID:    "go",
			Type:  agentapi.UIElementButton,
			Label: "Proceed",
			Style: agentapi.UIElementStylePrimary,
		},
	})
	require.NotNil(t, ui)
	require.Len(t, ui.Elements, 2)
	assert.Equal(t, models.UIElementTypeRadio, ui.Elements[0].Type)
	assert.Equal(t, "opt_a", ui.Elements[0].ID)
	assert.True(t, ui.Elements[0].Default)
	assert.Equal(t, models.UIElementTypeButton, ui.Elements[1].Type)
	assert.Equal(t, models.UIElementStylePrimary, ui.Elements[1].Style)
}

func TestGateway_SendMessageWithUI_WebSocket(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	rawFrames := make(chan string, 10)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/chat" {
			http.NotFound(w, r)
			return
		}
		upgrader := websocket.Upgrader{}
		c, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = c.Close() }()
		for {
			_, data, err := c.ReadMessage()
			if err != nil {
				return
			}
			rawFrames <- string(data)
		}
	}))
	defer srv.Close()

	gw := NewGateway(&config.Config{BesedkaURL: srv.URL}, nil)
	defer gw.Stop()
	require.NoError(t, gw.DialWebSocket(ctx))

	// 1. Send valid UI message (pre-filled answered metadata must be stripped without mutating caller input)
	inputUI := &models.UIData{
		Elements: []models.UIElement{
			{ID: " approve ", Type: models.UIElementTypeButton, Label: " Approve ", Style: models.UIElementStylePrimary},
			{ID: " deny ", Type: models.UIElementTypeButton, Label: " Deny ", Style: models.UIElementStyleDanger},
		},
		Answered:    true,
		SelectedIDs: []string{"approve"},
		ResponderID: "alice",
	}
	require.NoError(t, gw.SendMessageWithUI("townhall", "Deploy **v1.2**?", inputUI))

	select {
	case raw := <-rawFrames:
		var msg models.ClientMessage
		require.NoError(t, json.Unmarshal([]byte(raw), &msg))
		assert.Equal(t, models.ClientMessageTypeSend, msg.Type)
		assert.Equal(t, models.MessageTypeText, msg.MessageType)
		assert.Equal(t, "townhall", msg.ChatID)
		assert.Equal(t, "Deploy **v1.2**?", msg.Content)
		require.NotNil(t, msg.UI)
		assert.False(t, msg.UI.Answered)
		assert.Empty(t, msg.UI.SelectedIDs)
		assert.Empty(t, msg.UI.ResponderID)
		require.Len(t, msg.UI.Elements, 2)
		assert.Equal(t, "approve", msg.UI.Elements[0].ID)
		assert.Equal(t, "Approve", msg.UI.Elements[0].Label)
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for UI websocket frame")
	}

	assert.True(t, inputUI.Answered, "caller UIData must not be mutated")

	// 2. Send with attachments and UI
	atts := []models.Attachment{{Type: models.AttachmentTypeFile, Name: "plan.txt", MimeType: "text/plain", FileID: "f_1"}}
	require.NoError(t, gw.SendMessageWithAttachmentsAndUI("dm_1", "Review plan:", atts, &models.UIData{
		Elements: []models.UIElement{{ID: "ok", Type: models.UIElementTypeButton, Label: "Looks good"}},
	}))
	select {
	case raw := <-rawFrames:
		var msg models.ClientMessage
		require.NoError(t, json.Unmarshal([]byte(raw), &msg))
		assert.Equal(t, models.MessageTypeText, msg.MessageType)
		require.Len(t, msg.Attachments, 1)
		assert.Equal(t, "f_1", msg.Attachments[0].FileID)
		require.NotNil(t, msg.UI)
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for attachment+UI websocket frame")
	}

	// 3. Invalid UI payloads must fail before writing to WebSocket
	for _, bad := range []*models.UIData{
		{Elements: []models.UIElement{}},
		{Elements: []models.UIElement{{ID: "x", Type: models.UIElementTypeButton, Label: "X"}, {ID: "x", Type: models.UIElementTypeButton, Label: "Y"}}},
		{Elements: []models.UIElement{{ID: "x", Type: models.UIElementTypeButton, Label: "X", Default: true}}},
	} {
		err := gw.SendMessageWithUI("dm_1", "bad", bad)
		require.Error(t, err)
	}
	select {
	case unexpected := <-rawFrames:
		t.Fatalf("unexpected frame written for invalid UI: %s", unexpected)
	case <-time.After(50 * time.Millisecond):
	}

	// 4. Regression check: SendMessageWithAttachments (ui == nil) omits "ui" and "messageType"
	require.NoError(t, gw.SendMessageWithAttachments("dm_1", "Plain attachment", atts))
	select {
	case raw := <-rawFrames:
		assert.NotContains(t, raw, `"ui"`)
		assert.NotContains(t, raw, `"messageType"`)
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for plain attachment frame")
	}
}

func TestProcessMessage_SelfEchoWithUI_DeduplicatesAndBackfillsSeq(t *testing.T) {
	gw := NewGateway(&config.Config{
		BotHandle:         "@bob",
		MsgRingBufferSize: 20,
	}, nil)
	gw.botUserID = "bot-1"
	gw.botUser = models.User{ID: "bot-1", UserName: "bob", DisplayName: "Bob"}

	ui := &models.UIData{
		Elements: []models.UIElement{
			{ID: "yes", Type: models.UIElementTypeButton, Label: "Yes", Style: models.UIElementStylePrimary},
			{ID: "no", Type: models.UIElementTypeButton, Label: "No", Style: models.UIElementStyleDanger},
		},
	}
	outgoingContent := FormatOutgoingMessageContent("Proceed with deployment?", ui, "")
	gw.contextManager.Push("townhall", chatcontext.Entry{
		Role:       "assistant",
		SenderID:   "bot-1",
		SenderName: "Bob",
		Content:    outgoingContent,
		Timestamp:  time.Now().Unix(),
	})

	echo := models.Message{
		Seq:        42,
		Timestamp:  time.Now().Unix(),
		ChatID:     "townhall",
		UserID:     "bot-1",
		Type:       models.MessageTypeText,
		Content:    "<p>Proceed with deployment?</p>",
		RawContent: "Proceed with deployment?",
		UI:         ui,
	}
	require.NoError(t, gw.ProcessMessage(context.Background(), echo))

	entries := gw.contextManager.GetOrCreate("townhall").Entries()
	require.Len(t, entries, 1, "self-echo must be deduplicated")
	assert.Equal(t, int64(42), entries[0].Seq, "deduplicated echo must backfill Seq onto ring buffer entry")

	// Interleaved user message between outgoing Seq=0 push and incoming echo must still reconcile cleanly,
	// and a second identical prompt with a new Seq must be preserved rather than collapsed.
	gw.contextManager.Push("townhall", chatcontext.Entry{
		Role:       "assistant",
		SenderID:   "bot-1",
		SenderName: "Bob",
		Content:    outgoingContent,
		Timestamp:  time.Now().Unix(),
	})
	gw.contextManager.Push("townhall", chatcontext.Entry{
		Seq:        43,
		Role:       "user",
		SenderID:   "alice",
		SenderName: "Alice",
		Content:    "One sec",
		Timestamp:  time.Now().Unix(),
	})
	secondEcho := echo
	secondEcho.Seq = 44
	require.NoError(t, gw.ProcessMessage(context.Background(), secondEcho))

	entries = gw.contextManager.GetOrCreate("townhall").Entries()
	require.Len(t, entries, 3)
	assert.Equal(t, int64(42), entries[0].Seq)
	assert.Equal(t, int64(44), entries[1].Seq)
	assert.Equal(t, int64(43), entries[2].Seq)
}

func TestProcessMessage_UIResponse_DMAndTownhallRouting(t *testing.T) {
	var mu sync.Mutex
	var recordedPrompts []string
	wsSentCh := make(chan models.ClientMessage, 10)

	llmSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Messages []struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"messages"`
		}
		require.NoError(t, json.NewDecoder(r.Body).Decode(&req))
		mu.Lock()
		for _, m := range req.Messages {
			if m.Role == "user" {
				recordedPrompts = append(recordedPrompts, m.Content)
			}
		}
		mu.Unlock()

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"Acknowledged selection."}}]}`))
	}))
	defer llmSrv.Close()

	besedkaSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/users":
			_ = json.NewEncoder(w).Encode([]models.User{
				{ID: "alice", UserName: "alice", DisplayName: "Alice"},
				{ID: "bot-1", UserName: "bob", DisplayName: "Bob"},
				{ID: "bot-2", UserName: "deploybot", DisplayName: "Deploy Bot"},
			})
		case "/api/chats":
			_ = json.NewEncoder(w).Encode([]models.Chat{
				{ID: "townhall", Name: "Townhall", Type: "townhall"},
				{ID: "dm_alice_bot-1", IsDM: true, Type: "dm", UserIDs: []string{"alice", "bot-1"}, TargetUserID: "alice"},
			})
		case "/api/chats/townhall/messages":
			fromSeq := r.URL.Query().Get("fromSeq")
			toSeq := r.URL.Query().Get("toSeq")
			switch {
			case fromSeq == "2" && toSeq == "2":
				// Evicted parent owned by another bot (bot-2)
				_ = json.NewEncoder(w).Encode([]models.Message{
					{Seq: 2, ChatID: "townhall", UserID: "bot-2", Content: "Other bot prompt"},
				})
			case fromSeq == "3" && toSeq == "3":
				// Evicted parent owned by Bob (bot-1)
				_ = json.NewEncoder(w).Encode([]models.Message{
					{Seq: 3, ChatID: "townhall", UserID: "bot-1", Content: "Bob evicted prompt"},
				})
			case fromSeq == "20" && toSeq == "20":
				// Parent exists on server before WS self-echo landed in ring buffer
				_ = json.NewEncoder(w).Encode([]models.Message{
					{Seq: 20, ChatID: "townhall", UserID: "bot-1", Content: "Proceed?"},
				})
			default:
				_ = json.NewEncoder(w).Encode([]models.Message{})
			}
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
				wsSentCh <- cm
			}
		default:
			http.NotFound(w, r)
		}
	}))
	defer besedkaSrv.Close()

	cfg := &config.Config{
		BesedkaURL:        besedkaSrv.URL,
		OpenAIAPIKey:      "test-key",
		OpenAIBaseURL:     llmSrv.URL + "/v1",
		OpenAIModel:       "test-model",
		BotHandle:         "@custom",
		MsgRingBufferSize: 20,
	}
	llmClient := llm.NewClient(cfg, nil)
	gw := NewGateway(cfg, llmClient)
	gw.botUserID = "bot-1"
	gw.botUser = models.User{ID: "bot-1", UserName: "bob", DisplayName: "Bob"}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	require.NoError(t, gw.DialWebSocket(ctx))
	defer gw.Stop()

	_, err := gw.FetchUsers(ctx)
	require.NoError(t, err)
	chats, err := gw.FetchChats(ctx)
	require.NoError(t, err)
	gw.chatCache.SetAll(chats)

	// 1. DM ui_response with empty Content/RawContent must NOT be dropped and must trigger LLM with enriched metadata
	dmResp := models.Message{
		Seq:       5,
		Timestamp: time.Now().Unix(),
		ChatID:    "dm_alice_bot-1",
		UserID:    "alice",
		Type:      models.MessageTypeUIResponse,
		Content:   "",
		UIResponse: &models.UIResponseData{
			ParentSeq:   4,
			SelectedIDs: []string{"us_east", "notify"},
		},
	}
	require.NoError(t, gw.ProcessMessage(ctx, dmResp))

	mu.Lock()
	require.NotEmpty(t, recordedPrompts)
	assert.Contains(t, recordedPrompts[len(recordedPrompts)-1], "[UI response to seq 4 | selected_ids: us_east, notify]")
	recordedPrompts = nil
	mu.Unlock()

	select {
	case cm := <-wsSentCh:
		assert.Equal(t, "dm_alice_bot-1", cm.ChatID)
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for DM reply over websocket")
	}

	// 2. Townhall ui_response answering Bob's prompt triggers Bob even when BOT_HANDLE (@custom) != botUser.UserName (bob)
	gw.contextManager.Push("townhall", chatcontext.Entry{
		Seq:        10,
		Role:       "assistant",
		SenderID:   "bot-1",
		SenderName: "Bob",
		Content:    "Deploy now?",
		Timestamp:  time.Now().Unix(),
	})
	gw.contextManager.Push("townhall", chatcontext.Entry{
		Seq:        11,
		Role:       "user",
		SenderID:   "bot-2",
		SenderName: "Deploy Bot",
		Content:    "Restart staging?",
		Timestamp:  time.Now().Unix(),
	})

	townhallForBob := models.Message{
		Seq:        12,
		Timestamp:  time.Now().Unix(),
		ChatID:     "townhall",
		UserID:     "alice",
		Type:       models.MessageTypeUIResponse,
		Content:    "<p>@bob Selected: Approve</p>",
		RawContent: "@bob Selected: Approve",
		UIResponse: &models.UIResponseData{
			ParentSeq:   10,
			SelectedIDs: []string{"approve"},
		},
	}
	require.NoError(t, gw.ProcessMessage(ctx, townhallForBob))

	mu.Lock()
	require.NotEmpty(t, recordedPrompts)
	assert.Contains(t, recordedPrompts[len(recordedPrompts)-1], "[UI response to seq 10 | selected_ids: approve]")
	recordedPrompts = nil
	mu.Unlock()

	select {
	case cm := <-wsSentCh:
		assert.Equal(t, "townhall", cm.ChatID)
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for townhall reply over websocket")
	}

	// 3. Townhall ui_response answering another bot's prompt (ParentSeq=11 owned by bot-2) is ignored even if it mentions @bob
	townhallForOtherBot := models.Message{
		Seq:        13,
		Timestamp:  time.Now().Unix(),
		ChatID:     "townhall",
		UserID:     "alice",
		Type:       models.MessageTypeUIResponse,
		Content:    "<p>@bob @deploybot Selected: Restart</p>",
		RawContent: "@bob @deploybot Selected: Restart",
		UIResponse: &models.UIResponseData{
			ParentSeq:   11,
			SelectedIDs: []string{"restart"},
		},
	}
	require.NoError(t, gw.ProcessMessage(ctx, townhallForOtherBot))

	mu.Lock()
	assert.Empty(t, recordedPrompts, "click on another bot's prompt must not invoke Bob")
	mu.Unlock()

	select {
	case unexpected := <-wsSentCh:
		t.Fatalf("unexpected reply sent for another bot's UI response: %+v", unexpected)
	case <-time.After(50 * time.Millisecond):
	}

	thEntries := gw.contextManager.GetOrCreate("townhall").Entries()
	last := thEntries[len(thEntries)-1]
	assert.Equal(t, int64(13), last.Seq)
	assert.Contains(t, last.Content, "[UI response to seq 11 | selected_ids: restart]")

	// 4. Evicted parent owned by another bot (Seq=2 not in ring buffer) must NOT invoke Bob even if content mentions @bob
	evictedOther := models.Message{
		Seq:        14,
		Timestamp:  time.Now().Unix(),
		ChatID:     "townhall",
		UserID:     "alice",
		Type:       models.MessageTypeUIResponse,
		Content:    "<p>@deploybot Selected: Ask @bob</p>",
		RawContent: "@deploybot Selected: Ask @bob",
		UIResponse: &models.UIResponseData{
			ParentSeq:   2,
			SelectedIDs: []string{"ask_bob"},
		},
	}
	require.NoError(t, gw.ProcessMessage(ctx, evictedOther))
	mu.Lock()
	assert.Empty(t, recordedPrompts, "evicted parent owned by another bot must not invoke Bob")
	mu.Unlock()

	// 5. Evicted parent owned by Bob (Seq=3 resolved via REST) invokes Bob
	evictedBob := models.Message{
		Seq:        15,
		Timestamp:  time.Now().Unix(),
		ChatID:     "townhall",
		UserID:     "alice",
		Type:       models.MessageTypeUIResponse,
		Content:    "<p>Selected: Yes</p>",
		RawContent: "Selected: Yes",
		UIResponse: &models.UIResponseData{
			ParentSeq:   3,
			SelectedIDs: []string{"yes"},
		},
	}
	require.NoError(t, gw.ProcessMessage(ctx, evictedBob))
	mu.Lock()
	require.NotEmpty(t, recordedPrompts)
	assert.Contains(t, recordedPrompts[len(recordedPrompts)-1], "[UI response to seq 3 | selected_ids: yes]")
	recordedPrompts = nil
	mu.Unlock()
	select {
	case cm := <-wsSentCh:
		assert.Equal(t, "townhall", cm.ChatID)
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for evicted Bob parent reply")
	}

	// 6. Response processed before parent self-echo arrives in ring buffer (Seq=20 resolved via REST) invokes Bob
	outOfOrderResp := models.Message{
		Seq:        21,
		Timestamp:  time.Now().Unix(),
		ChatID:     "townhall",
		UserID:     "alice",
		Type:       models.MessageTypeUIResponse,
		Content:    "<p>@bob Selected: Proceed</p>",
		RawContent: "@bob Selected: Proceed",
		UIResponse: &models.UIResponseData{
			ParentSeq:   20,
			SelectedIDs: []string{"proceed"},
		},
	}
	require.NoError(t, gw.ProcessMessage(ctx, outOfOrderResp))
	mu.Lock()
	require.NotEmpty(t, recordedPrompts)
	assert.Contains(t, recordedPrompts[len(recordedPrompts)-1], "[UI response to seq 20 | selected_ids: proceed]")
	recordedPrompts = nil
	mu.Unlock()
	select {
	case cm := <-wsSentCh:
		assert.Equal(t, "townhall", cm.ChatID)
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for out-of-order parent reply")
	}

	// 7. Unresolvable parent (Seq=999 missing on server) fails closed even if content mentions @bob
	unknownParent := models.Message{
		Seq:        22,
		Timestamp:  time.Now().Unix(),
		ChatID:     "townhall",
		UserID:     "alice",
		Type:       models.MessageTypeUIResponse,
		Content:    "<p>@bob Selected: Mystery</p>",
		RawContent: "@bob Selected: Mystery",
		UIResponse: &models.UIResponseData{
			ParentSeq:   999,
			SelectedIDs: []string{"mystery"},
		},
	}
	require.NoError(t, gw.ProcessMessage(ctx, unknownParent))
	mu.Lock()
	assert.Empty(t, recordedPrompts, "unresolvable parent must fail closed")
	mu.Unlock()
}

func TestWarmupChat_PreservesUIAndUIResponse(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/chats/townhall/messages":
			_ = json.NewEncoder(w).Encode([]models.Message{
				{
					Seq:        1,
					Timestamp:  1700000001,
					ChatID:     "townhall",
					UserID:     "bot-1",
					Type:       models.MessageTypeText,
					Content:    "<p>Pick region:</p>",
					RawContent: "Pick region:",
					UI: &models.UIData{
						Elements: []models.UIElement{
							{ID: "us", Type: models.UIElementTypeRadio, Label: "US", Default: true},
							{ID: "eu", Type: models.UIElementTypeRadio, Label: "EU"},
						},
						Answered:    true,
						SelectedIDs: []string{"eu"},
						ResponderID: "alice",
					},
				},
				{
					Seq:        2,
					Timestamp:  1700000002,
					ChatID:     "townhall",
					UserID:     "alice",
					Type:       models.MessageTypeUIResponse,
					Content:    "<p>@bob Selected: EU</p>",
					RawContent: "@bob Selected: EU",
					UIResponse: &models.UIResponseData{
						ParentSeq:   1,
						SelectedIDs: []string{"eu"},
					},
				},
			})
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	gw := NewGateway(&config.Config{
		BesedkaURL:        srv.URL,
		MsgRingBufferSize: 10,
	}, nil)
	gw.botUserID = "bot-1"
	gw.botUser = models.User{ID: "bot-1", UserName: "bob", DisplayName: "Bob"}
	gw.userCache.Set(models.User{ID: "alice", UserName: "alice", DisplayName: "Alice"})

	gw.WarmupChat(context.Background(), "townhall", 2)

	entries := gw.contextManager.GetOrCreate("townhall").Entries()
	require.Len(t, entries, 2)

	assert.Equal(t, int64(1), entries[0].Seq)
	assert.Equal(t, "assistant", entries[0].Role)
	assert.Equal(t, "bot-1", entries[0].SenderID)
	assert.True(t, strings.Contains(entries[0].Content, `Pick region:`))
	assert.True(t, strings.Contains(entries[0].Content, `[Interactive UI options: radio "US" (id=us, default); radio "EU" (id=eu)]`))

	assert.Equal(t, int64(2), entries[1].Seq)
	assert.Equal(t, "user", entries[1].Role)
	assert.Contains(t, entries[1].Content, "@bob Selected: EU")
	assert.Contains(t, entries[1].Content, "[UI response to seq 1 | selected_ids: eu]")
}

func TestBuildDeliveryUIPlan_PrecedenceAndDeduplication(t *testing.T) {
	memBytes, err := json.Marshal(agentapi.ApprovalRequestPayload{
		Kind:     agentapi.ApprovalKindMemory,
		TargetID: "mem_1@1",
		Label:    "fact",
		Summary:  "User likes dark mode",
	})
	require.NoError(t, err)

	skillBytes, err := json.Marshal(agentapi.ApprovalRequestPayload{
		Kind:     agentapi.ApprovalKindSkill,
		TargetID: "skill_1@1",
		Label:    "run-tests",
		Summary:  "Runs Go test suite",
	})
	require.NoError(t, err)

	promptBytes, err := json.Marshal(agentapi.InteractivePromptPayload{
		Message: "Which target should I deploy?",
		Elements: []agentapi.UIElement{
			{ID: "staging", Type: agentapi.UIElementButton, Label: "Staging", Style: agentapi.UIElementStylePrimary},
			{ID: "prod", Type: agentapi.UIElementButton, Label: "Prod", Style: agentapi.UIElementStyleDanger},
		},
	})
	require.NoError(t, err)

	// 1. Single approval attaches to mainUI directly and provides fallback card description
	fallback, mainUI, followUps := buildDeliveryUIPlan([]agentapi.Action{
		{Type: agentapi.ActionApprovalRequest, Payload: memBytes},
	}, false)
	assert.Contains(t, fallback, "Memory Approval Requested")
	assert.Contains(t, fallback, "/memory approve mem_1@1")
	assert.Empty(t, followUps)
	require.NotNil(t, mainUI)
	require.Len(t, mainUI.Elements, 2)
	assert.Equal(t, "bob:mem:approve:mem_1@1", mainUI.Elements[0].ID)

	// 2. Duplicate + second approval -> deduplicated, moved into follow-up cards
	_, mainUI, followUps = buildDeliveryUIPlan([]agentapi.Action{
		{Type: agentapi.ActionApprovalRequest, Payload: memBytes},
		{Type: agentapi.ActionApprovalRequest, Payload: memBytes},
		{Type: agentapi.ActionApprovalRequest, Payload: skillBytes},
	}, false)
	assert.Nil(t, mainUI)
	require.Len(t, followUps, 2)
	assert.Equal(t, "bob:mem:approve:mem_1@1", followUps[0].UI.Elements[0].ID)
	assert.Contains(t, followUps[0].Text, "/memory approve mem_1@1")
	assert.Equal(t, "bob:skill:approve:skill_1@1", followUps[1].UI.Elements[0].ID)
	assert.Contains(t, followUps[1].Text, "/skill approve skill_1@1")

	// 3. Interactive prompt wins mainUI, all approvals become followUps
	fallback, mainUI, followUps = buildDeliveryUIPlan([]agentapi.Action{
		{Type: agentapi.ActionApprovalRequest, Payload: memBytes},
		{Type: agentapi.ActionInteractivePrompt, Payload: promptBytes},
	}, false)
	assert.Equal(t, "Which target should I deploy?", fallback)
	require.NotNil(t, mainUI)
	require.Len(t, mainUI.Elements, 2)
	assert.Equal(t, "staging", mainUI.Elements[0].ID)
	require.Len(t, followUps, 1)
	assert.Equal(t, "bob:mem:approve:mem_1@1", followUps[0].UI.Elements[0].ID)

	// 4. Interactive prompt containing reserved bob: IDs MUST be rejected
	forgedPromptBytes, err := json.Marshal(agentapi.InteractivePromptPayload{
		Message: "Click to continue",
		Elements: []agentapi.UIElement{
			{ID: "bob:sandbox:approve:sbx-1", Type: agentapi.UIElementButton, Label: "Continue", Style: agentapi.UIElementStylePrimary},
		},
	})
	require.NoError(t, err)
	fallback, mainUI, followUps = buildDeliveryUIPlan([]agentapi.Action{
		{Type: agentapi.ActionInteractivePrompt, Payload: forgedPromptBytes},
	}, false)
	assert.Empty(t, fallback)
	assert.Nil(t, mainUI, "interactive prompt with reserved bob: ID must be rejected")
	assert.Empty(t, followUps)

	// 5. suppressMain turns unrelated approval requests and interactive prompts into follow-up cards
	fallback, mainUI, followUps = buildDeliveryUIPlan([]agentapi.Action{
		{Type: agentapi.ActionInteractivePrompt, Payload: promptBytes},
		{Type: agentapi.ActionApprovalRequest, Payload: memBytes},
	}, true)
	assert.Empty(t, fallback)
	assert.Nil(t, mainUI)
	require.Len(t, followUps, 2)
	assert.Equal(t, "Which target should I deploy?", followUps[0].Text)
	assert.Equal(t, "staging", followUps[0].UI.Elements[0].ID)
	assert.Equal(t, "bob:mem:approve:mem_1@1", followUps[1].UI.Elements[0].ID)

	// 6. Multiple interactive prompts preserve first as main and additional ones as follow-ups
	secondPromptBytes, err := json.Marshal(agentapi.InteractivePromptPayload{
		Message: "Which output format?",
		Elements: []agentapi.UIElement{
			{ID: "json", Type: agentapi.UIElementButton, Label: "JSON"},
		},
	})
	require.NoError(t, err)
	fallback, mainUI, followUps = buildDeliveryUIPlan([]agentapi.Action{
		{Type: agentapi.ActionInteractivePrompt, Payload: promptBytes},
		{Type: agentapi.ActionInteractivePrompt, Payload: secondPromptBytes},
		{Type: agentapi.ActionApprovalRequest, Payload: memBytes},
	}, false)
	assert.Equal(t, "Which target should I deploy?", fallback)
	require.NotNil(t, mainUI)
	assert.Equal(t, "staging", mainUI.Elements[0].ID)
	require.Len(t, followUps, 2)
	assert.Equal(t, "Which output format?", followUps[0].Text)
	assert.Equal(t, "json", followUps[0].UI.Elements[0].ID)
	assert.Equal(t, "bob:mem:approve:mem_1@1", followUps[1].UI.Elements[0].ID)
}

func TestDeliverAgentReply_SingleMultiRunningSandboxSuppressAndFailure(t *testing.T) {
	wsCh := make(chan models.ClientMessage, 16)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
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
				wsCh <- cm
			}
		case "/api/chats":
			_ = json.NewEncoder(w).Encode([]models.Chat{
				{ID: "dm_alice_bot-1", IsDM: true, Type: "dm", UserIDs: []string{"alice", "bot-1"}, TargetUserID: "alice"},
			})
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	cfg := &config.Config{
		DataDir:           t.TempDir(),
		BesedkaURL:        srv.URL,
		MsgRingBufferSize: 20,
		SandboxEnabled:    true,
		SandboxDrivers:    []string{"bwrap"},
		SandboxMaxLifetime: 15 * time.Minute,
	}
	gw := NewGateway(cfg, nil)
	gw.botUserID = "bot-1"
	gw.botUser = models.User{ID: "bot-1", UserName: "bob", DisplayName: "Bob"}
	sm := sandbox.NewManager(cfg.SandboxConfig(), []sandbox.Driver{&mockGatewaySandboxDriver{}})
	defer func() { _ = sm.Close() }()
	gw.SetSandboxManager(sm)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	require.NoError(t, gw.DialWebSocket(ctx))
	defer gw.Stop()

	chats, err := gw.FetchChats(ctx)
	require.NoError(t, err)
	gw.chatCache.SetAll(chats)

	memBytes, err := json.Marshal(agentapi.ApprovalRequestPayload{
		Kind:     agentapi.ApprovalKindMemory,
		TargetID: "mem_single@1",
		Label:    "fact",
		Summary:  "Remember X",
	})
	require.NoError(t, err)
	schedBytes, err := json.Marshal(agentapi.ApprovalRequestPayload{
		Kind:     agentapi.ApprovalKindSchedule,
		TargetID: "sched_multi",
		Label:    "daily_sync",
		Summary:  "Run sync",
	})
	require.NoError(t, err)

	// 1. SuppressReply alone skips WS send and records short entry;
	// mixed SuppressReply + unrelated ActionApprovalRequest still delivers the unrelated follow-up approval card!
	require.NoError(t, gw.Deliver(ctx, agentapi.Completion{
		Run: agentapi.RunDescriptor{
			Session: agentapi.SessionRef{SessionID: "dm_alice_bot-1", IsDM: true},
			Actor:   agentapi.Actor{ID: "alice"},
		},
		Status: agentapi.RunCompleted,
		Result: agentapi.Result{
			Content: "Should not send",
			Actions: []agentapi.Action{{Type: agentapi.ActionSuppressReply}},
		},
	}))
	select {
	case unexpected := <-wsCh:
		t.Fatalf("unexpected WS message on SuppressReply: %+v", unexpected)
	case <-time.After(30 * time.Millisecond):
	}

	require.NoError(t, gw.Deliver(ctx, agentapi.Completion{
		Run: agentapi.RunDescriptor{
			Session: agentapi.SessionRef{SessionID: "dm_alice_bot-1", IsDM: true},
			Actor:   agentapi.Actor{ID: "alice"},
		},
		Status: agentapi.RunCompleted,
		Result: agentapi.Result{
			Content: "Suppressed main reply",
			Actions: []agentapi.Action{
				{Type: agentapi.ActionApprovalRequest, Payload: memBytes},
				{Type: agentapi.ActionSuppressReply},
			},
		},
	}))
	mixedFollowUp := <-wsCh
	assert.NotContains(t, mixedFollowUp.Content, "Suppressed main reply")
	assert.Contains(t, mixedFollowUp.Content, "Memory Approval Requested")
	require.NotNil(t, mixedFollowUp.UI)
	assert.Equal(t, "bob:mem:approve:mem_single@1", mixedFollowUp.UI.Elements[0].ID)

	// 1a. SuppressReply + ActionInteractivePrompt preserves and delivers the interactive question as a follow-up
	q1Bytes, err := json.Marshal(agentapi.InteractivePromptPayload{
		Message: "Pick region:",
		Elements: []agentapi.UIElement{
			{ID: "us", Type: agentapi.UIElementButton, Label: "US"},
		},
	})
	require.NoError(t, err)
	q2Bytes, err := json.Marshal(agentapi.InteractivePromptPayload{
		Message: "Pick format:",
		Elements: []agentapi.UIElement{
			{ID: "json", Type: agentapi.UIElementButton, Label: "JSON"},
		},
	})
	require.NoError(t, err)
	require.NoError(t, gw.Deliver(ctx, agentapi.Completion{
		Run: agentapi.RunDescriptor{
			Session: agentapi.SessionRef{SessionID: "dm_alice_bot-1", IsDM: true},
			Actor:   agentapi.Actor{ID: "alice"},
		},
		Status: agentapi.RunCompleted,
		Result: agentapi.Result{
			Content: "Suppressed main reply",
			Actions: []agentapi.Action{
				{Type: agentapi.ActionInteractivePrompt, Content: "Pick region:", Payload: q1Bytes},
				{Type: agentapi.ActionSuppressReply},
			},
		},
	}))
	suppressedAskMsg := <-wsCh
	assert.Equal(t, "Pick region:", suppressedAskMsg.Content)
	require.NotNil(t, suppressedAskMsg.UI)
	assert.Equal(t, "us", suppressedAskMsg.UI.Elements[0].ID)

	// 1b. Multiple ask_user prompts deliver both questions separately without overwriting earlier one
	require.NoError(t, gw.Deliver(ctx, agentapi.Completion{
		Run: agentapi.RunDescriptor{
			Session: agentapi.SessionRef{SessionID: "dm_alice_bot-1", IsDM: true},
			Actor:   agentapi.Actor{ID: "alice"},
		},
		Status: agentapi.RunCompleted,
		Result: agentapi.Result{
			Content: "Pick region:",
			Actions: []agentapi.Action{
				{Type: agentapi.ActionInteractivePrompt, Content: "Pick region:", Payload: q1Bytes},
				{Type: agentapi.ActionInteractivePrompt, Content: "Pick format:", Payload: q2Bytes},
			},
		},
	}))
	firstAskMsg := <-wsCh
	secondAskMsg := <-wsCh
	assert.Equal(t, "Pick region:", firstAskMsg.Content)
	require.NotNil(t, firstAskMsg.UI)
	assert.Equal(t, "us", firstAskMsg.UI.Elements[0].ID)
	assert.Equal(t, "Pick format:", secondAskMsg.Content)
	require.NotNil(t, secondAskMsg.UI)
	assert.Equal(t, "json", secondAskMsg.UI.Elements[0].ID)

	// 1c. Single approval with empty model completion uses formatApprovalCardText fallback
	require.NoError(t, gw.Deliver(ctx, agentapi.Completion{
		Run: agentapi.RunDescriptor{
			Session: agentapi.SessionRef{SessionID: "dm_alice_bot-1", IsDM: true},
			Actor:   agentapi.Actor{ID: "alice"},
		},
		Status: agentapi.RunCompleted,
		Result: agentapi.Result{
			Content: "",
			Actions: []agentapi.Action{{Type: agentapi.ActionApprovalRequest, Payload: memBytes}},
		},
	}))
	emptyCompMsg := <-wsCh
	assert.Contains(t, emptyCompMsg.Content, "Memory Approval Requested")
	assert.Contains(t, emptyCompMsg.Content, "Remember X")
	assert.Contains(t, emptyCompMsg.Content, "/memory approve mem_single@1")
	require.NotNil(t, emptyCompMsg.UI)
	assert.Equal(t, "bob:mem:approve:mem_single@1", emptyCompMsg.UI.Elements[0].ID)

	// 2. RunFailed must NOT attach any approval UI or running-sandbox buttons
	sbx, err := sm.RequestSandbox(ctx, "alice", "dm_alice_bot-1", sandbox.RequestParams{
		Driver: sandbox.DriverBwrap, NetworkMode: sandbox.NetworkNone, Reason: "test",
	})
	require.NoError(t, err)
	_, err = sm.ApproveSandbox(ctx, "alice")
	require.NoError(t, err)

	require.NoError(t, gw.Deliver(ctx, agentapi.Completion{
		Run: agentapi.RunDescriptor{
			Session: agentapi.SessionRef{SessionID: "dm_alice_bot-1", IsDM: true},
			Actor:   agentapi.Actor{ID: "alice"},
		},
		Status: agentapi.RunFailed,
		Result: agentapi.Result{
			Actions: []agentapi.Action{{Type: agentapi.ActionApprovalRequest, Payload: memBytes}},
		},
	}))
	failedMsg := <-wsCh
	assert.Contains(t, failedMsg.Content, "encountered an issue")
	assert.Nil(t, failedMsg.UI, "failed runs must not attach interactive UI")

	// 3. Running sandbox + single approval -> approval UI keeps precedence (does NOT clobber mainUI)
	require.NoError(t, gw.Deliver(ctx, agentapi.Completion{
		Run: agentapi.RunDescriptor{
			Session: agentapi.SessionRef{SessionID: "dm_alice_bot-1", IsDM: true},
			Actor:   agentapi.Actor{ID: "alice"},
		},
		Status: agentapi.RunCompleted,
		Result: agentapi.Result{
			Content: "Done analyzing.",
			Actions: []agentapi.Action{{Type: agentapi.ActionApprovalRequest, Payload: memBytes}},
		},
	}))
	singleMsg := <-wsCh
	assert.Contains(t, singleMsg.Content, "/sandbox destroy")
	require.NotNil(t, singleMsg.UI)
	assert.Equal(t, "bob:mem:approve:mem_single@1", singleMsg.UI.Elements[0].ID)

	// 4. Running sandbox + multi-approval -> main gets NewSandboxRunningUI, plus 2 follow-up cards
	require.NoError(t, gw.Deliver(ctx, agentapi.Completion{
		Run: agentapi.RunDescriptor{
			Session: agentapi.SessionRef{SessionID: "dm_alice_bot-1", IsDM: true},
			Actor:   agentapi.Actor{ID: "alice"},
		},
		Status: agentapi.RunCompleted,
		Result: agentapi.Result{
			Content: "Queued two proposals.",
			Actions: []agentapi.Action{
				{Type: agentapi.ActionApprovalRequest, Payload: memBytes},
				{Type: agentapi.ActionApprovalRequest, Payload: schedBytes},
			},
		},
	}))
	mainMsg := <-wsCh
	require.NotNil(t, mainMsg.UI)
	require.Len(t, mainMsg.UI.Elements, 2)
	assert.Equal(t, "bob:sandbox:destroy:"+sbx.ID, mainMsg.UI.Elements[0].ID)
	assert.Equal(t, "bob:sandbox:keep:"+sbx.ID, mainMsg.UI.Elements[1].ID)

	card1 := <-wsCh
	card2 := <-wsCh
	require.NotNil(t, card1.UI)
	require.NotNil(t, card2.UI)
	assert.Equal(t, "bob:mem:approve:mem_single@1", card1.UI.Elements[0].ID)
	assert.Equal(t, "bob:sched:approve:sched_multi", card2.UI.Elements[0].ID)

	// 5. Follow-up card delivery errors are aggregated and returned (never swallowed as nil)
	require.NoError(t, sm.DestroyForID(ctx, "alice", sbx.ID))
	gw.Stop()
	delivErr := gw.Deliver(ctx, agentapi.Completion{
		Run: agentapi.RunDescriptor{
			Session: agentapi.SessionRef{SessionID: "dm_alice_bot-1", IsDM: true},
			Actor:   agentapi.Actor{ID: "alice"},
		},
		Status: agentapi.RunCompleted,
		Result: agentapi.Result{
			Actions: []agentapi.Action{
				{Type: agentapi.ActionApprovalRequest, Payload: memBytes},
				{Type: agentapi.ActionApprovalRequest, Payload: schedBytes},
			},
		},
	})
	require.Error(t, delivErr)
	assert.Contains(t, delivErr.Error(), "follow-up approval card 1")
	assert.Contains(t, delivErr.Error(), "follow-up approval card 2")
}

func TestHandleUIResponseAction_ApprovalsAndSecurity(t *testing.T) {
	var llmMu sync.Mutex
	var llmCalls int

	llmSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		llmMu.Lock()
		llmCalls++
		llmMu.Unlock()
		resp := map[string]interface{}{
			"id":      "chatcmpl-ui-action",
			"object":  "chat.completion",
			"created": time.Now().Unix(),
			"model":   "test-model",
			"choices": []map[string]interface{}{
				{
					"index":         0,
					"finish_reason": "stop",
					"message": map[string]interface{}{
						"role":    "assistant",
						"content": "Sandbox task continued!",
					},
				},
			},
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer llmSrv.Close()

	wsCh := make(chan models.ClientMessage, 32)
	besedkaSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/users":
			_ = json.NewEncoder(w).Encode([]models.User{
				{ID: "alice", UserName: "alice", DisplayName: "Alice"},
				{ID: "mallory", UserName: "mallory", DisplayName: "Mallory"},
				{ID: "bot-1", UserName: "bob", DisplayName: "Bob"},
			})
		case "/api/chats":
			_ = json.NewEncoder(w).Encode([]models.Chat{
				{ID: "townhall", Name: "Townhall", Type: "townhall"},
				{ID: "dm_alice_bot-1", IsDM: true, Type: "dm", UserIDs: []string{"alice", "bot-1"}, TargetUserID: "alice"},
				{ID: "dm_mallory_bot-1", IsDM: true, Type: "dm", UserIDs: []string{"mallory", "bot-1"}, TargetUserID: "mallory"},
			})
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
				wsCh <- cm
			}
		default:
			http.NotFound(w, r)
		}
	}))
	defer besedkaSrv.Close()

	cfg := &config.Config{
		DataDir:            t.TempDir(),
		BesedkaURL:         besedkaSrv.URL,
		OpenAIAPIKey:       "test-key",
		OpenAIBaseURL:      llmSrv.URL + "/v1",
		OpenAIModel:        "test-model",
		BotHandle:          "@bob",
		MsgRingBufferSize:  20,
		SandboxEnabled:     true,
		SandboxDrivers:     []string{"bwrap"},
		SandboxMaxLifetime: 15 * time.Minute,
	}
	llmClient := llm.NewClient(cfg, nil)
	gw := NewGateway(cfg, llmClient)
	gw.botUserID = "bot-1"
	gw.botUser = models.User{ID: "bot-1", UserName: "bob", DisplayName: "Bob"}

	storeProv := NewMemoryStoreProvider(gw.MemoryManager(), cfg.DataDir)
	gw.SetStoreProvider(storeProv)
	sm := sandbox.NewManager(cfg.SandboxConfig(), []sandbox.Driver{&mockGatewaySandboxDriver{}})
	defer func() { _ = sm.Close() }()
	gw.SetSandboxManager(sm)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	require.NoError(t, gw.DialWebSocket(ctx))
	defer gw.Stop()

	_, err := gw.FetchUsers(ctx)
	require.NoError(t, err)
	chats, err := gw.FetchChats(ctx)
	require.NoError(t, err)
	gw.chatCache.SetAll(chats)

	nextWS := func() models.ClientMessage {
		t.Helper()
		select {
		case cm := <-wsCh:
			return cm
		case <-time.After(2 * time.Second):
			t.Fatal("timed out waiting for websocket message")
			return models.ClientMessage{}
		}
	}

	// 1. Townhall spoofed bob:sandbox:approve:<id> is rejected
	gw.contextManager.Push("townhall", chatcontext.Entry{
		Seq: 1, Role: "assistant", SenderID: "bot-1", SenderName: "Bob", Content: "Hi",
	})
	require.NoError(t, gw.ProcessMessage(ctx, models.Message{
		Seq: 2, ChatID: "townhall", UserID: "alice", Type: models.MessageTypeUIResponse,
		UIResponse: &models.UIResponseData{ParentSeq: 1, SelectedIDs: []string{"bob:sandbox:approve:sbx-1"}},
	}))
	thWarn := nextWS()
	assert.Contains(t, thWarn.Content, "only available in direct messages")

	// 2. Non-owner DM click is rejected
	require.NoError(t, gw.ProcessMessage(ctx, models.Message{
		Seq: 3, ChatID: "dm_alice_bot-1", UserID: "mallory", Type: models.MessageTypeUIResponse,
		UIResponse: &models.UIResponseData{ParentSeq: 1, SelectedIDs: []string{"bob:sandbox:approve:sbx-1"}},
	}))
	ownerWarn := nextWS()
	assert.Contains(t, ownerWarn.Content, "unauthorized")

	// 3. Invalid and multiple reserved action IDs warn without invoking LLM
	require.NoError(t, gw.ProcessMessage(ctx, models.Message{
		Seq: 4, ChatID: "dm_alice_bot-1", UserID: "alice", Type: models.MessageTypeUIResponse,
		UIResponse: &models.UIResponseData{ParentSeq: 1, SelectedIDs: []string{"bob:unknown:action"}},
	}))
	assert.Contains(t, nextWS().Content, "Unrecognized or invalid UI action")

	require.NoError(t, gw.ProcessMessage(ctx, models.Message{
		Seq: 5, ChatID: "dm_alice_bot-1", UserID: "alice", Type: models.MessageTypeUIResponse,
		UIResponse: &models.UIResponseData{ParentSeq: 1, SelectedIDs: []string{"bob:sandbox:approve:sbx-1", "bob:sandbox:deny:sbx-1"}},
	}))
	assert.Contains(t, nextWS().Content, "submitted individually")

	llmMu.Lock()
	assert.Equal(t, 0, llmCalls)
	llmMu.Unlock()

	// 4. Stale sandbox card reproduction: restricted sandbox A denied, replacement sandbox B requested;
	// clicking old sandbox A's Approve button must NOT approve sandbox B.
	sbxA, err := sm.RequestSandbox(ctx, "alice", "dm_alice_bot-1", sandbox.RequestParams{
		Driver: sandbox.DriverBwrap, NetworkMode: sandbox.NetworkNone, Reason: "Restricted A",
	})
	require.NoError(t, err)
	require.NoError(t, sm.DenySandboxForID("alice", sbxA.ID))

	sbxB, err := sm.RequestSandbox(ctx, "alice", "dm_alice_bot-1", sandbox.RequestParams{
		Driver: sandbox.DriverBwrap, NetworkMode: sandbox.NetworkNone, Reason: "Replacement B",
	})
	require.NoError(t, err)
	require.NotEqual(t, sbxA.ID, sbxB.ID)

	require.NoError(t, gw.ProcessMessage(ctx, models.Message{
		Seq: 55, ChatID: "dm_alice_bot-1", UserID: "alice", Type: models.MessageTypeUIResponse,
		UIResponse: &models.UIResponseData{ParentSeq: 5, SelectedIDs: []string{"bob:sandbox:approve:" + sbxA.ID}},
	}))
	assert.Contains(t, nextWS().Content, "outdated")
	curB, ok := sm.GetStatus("alice")
	require.True(t, ok)
	assert.Equal(t, sbxB.ID, curB.ID)
	assert.Equal(t, sandbox.StatusPendingApproval, curB.Status)

	// Approve B via its own bound button triggers continuation & attaches Keep/Destroy UI bound to B;
	// Keep does NOT invoke LLM; Destroy terminates B.
	require.NoError(t, gw.ProcessMessage(ctx, models.Message{
		Seq: 6, ChatID: "dm_alice_bot-1", UserID: "alice", Type: models.MessageTypeUIResponse,
		UIResponse: &models.UIResponseData{ParentSeq: 5, SelectedIDs: []string{"bob:sandbox:approve:" + sbxB.ID}},
	}))
	var contReply models.ClientMessage
	for {
		cm := nextWS()
		if cm.Type == models.ClientMessageTypeSend && strings.Contains(cm.Content, "Sandbox task continued!") {
			contReply = cm
			break
		}
	}
	require.NotNil(t, contReply.UI)
	assert.Equal(t, "bob:sandbox:destroy:"+sbxB.ID, contReply.UI.Elements[0].ID)
	assert.Equal(t, "bob:sandbox:keep:"+sbxB.ID, contReply.UI.Elements[1].ID)

	llmMu.Lock()
	callsBeforeKeep := llmCalls
	llmMu.Unlock()
	assert.Equal(t, 1, callsBeforeKeep)

	require.NoError(t, gw.ProcessMessage(ctx, models.Message{
		Seq: 7, ChatID: "dm_alice_bot-1", UserID: "alice", Type: models.MessageTypeUIResponse,
		UIResponse: &models.UIResponseData{ParentSeq: 6, SelectedIDs: []string{"bob:sandbox:keep:" + sbxB.ID}},
	}))
	keepMsg := nextWS()
	assert.Contains(t, keepMsg.Content, "Keeping your sandbox running")
	llmMu.Lock()
	assert.Equal(t, callsBeforeKeep, llmCalls, "sandbox keep must not invoke LLM")
	llmMu.Unlock()

	require.NoError(t, gw.ProcessMessage(ctx, models.Message{
		Seq: 8, ChatID: "dm_alice_bot-1", UserID: "alice", Type: models.MessageTypeUIResponse,
		UIResponse: &models.UIResponseData{ParentSeq: 6, SelectedIDs: []string{"bob:sandbox:destroy:" + sbxB.ID}},
	}))
	destroyMsg := nextWS()
	assert.Contains(t, destroyMsg.Content, "Sandbox terminated")

	// Replacement sandbox C cannot be destroyed or kept via stale sandbox B card IDs
	sbxC, err := sm.RequestSandbox(ctx, "alice", "dm_alice_bot-1", sandbox.RequestParams{
		Driver: sandbox.DriverBwrap, NetworkMode: sandbox.NetworkNone, Reason: "Replacement C",
	})
	require.NoError(t, err)
	_, err = sm.ApproveSandboxForID(ctx, "alice", sbxC.ID)
	require.NoError(t, err)

	require.NoError(t, gw.ProcessMessage(ctx, models.Message{
		Seq: 81, ChatID: "dm_alice_bot-1", UserID: "alice", Type: models.MessageTypeUIResponse,
		UIResponse: &models.UIResponseData{ParentSeq: 6, SelectedIDs: []string{"bob:sandbox:destroy:" + sbxB.ID}},
	}))
	assert.Contains(t, nextWS().Content, "outdated")
	require.NoError(t, gw.ProcessMessage(ctx, models.Message{
		Seq: 82, ChatID: "dm_alice_bot-1", UserID: "alice", Type: models.MessageTypeUIResponse,
		UIResponse: &models.UIResponseData{ParentSeq: 6, SelectedIDs: []string{"bob:sandbox:keep:" + sbxB.ID}},
	}))
	assert.Contains(t, nextWS().Content, "outdated")
	curC, ok := sm.GetStatus("alice")
	require.True(t, ok)
	assert.Equal(t, sbxC.ID, curC.ID)
	assert.Equal(t, sandbox.StatusRunning, curC.Status)
	require.NoError(t, sm.DestroyForID(ctx, "alice", sbxC.ID))

	// 5. Sandbox Deny button click denies pending request without invoking LLM
	sbxDeny, err := sm.RequestSandbox(ctx, "alice", "dm_alice_bot-1", sandbox.RequestParams{
		Driver: sandbox.DriverBwrap, NetworkMode: sandbox.NetworkNone, Reason: "Deny check",
	})
	require.NoError(t, err)
	require.NoError(t, gw.ProcessMessage(ctx, models.Message{
		Seq: 9, ChatID: "dm_alice_bot-1", UserID: "alice", Type: models.MessageTypeUIResponse,
		UIResponse: &models.UIResponseData{ParentSeq: 8, SelectedIDs: []string{"bob:sandbox:deny:" + sbxDeny.ID}},
	}))
	denyMsg := nextWS()
	assert.Contains(t, denyMsg.Content, "Sandbox request denied")

	// 6. Memory and Skill Approve/Reject clicks update knowledge store
	kStore, err := storeProv.GetKnowledgeStore(ctx, "dm_alice_bot-1", true)
	require.NoError(t, err)

	_, memVer1, err := kStore.ProposeMemory(ctx, &knowledge.KnowledgeItem{
		ChatID: "dm_alice_bot-1", UserID: "alice",
	}, &knowledge.MemoryVersion{
		Type: knowledge.MemoryTypeFact, Content: "Likes Go", Confidence: 0.9,
		Provenance: knowledge.Provenance{ChatID: "dm_alice_bot-1", UserID: "alice"},
	})
	require.NoError(t, err)
	_, memVer2, err := kStore.ProposeMemory(ctx, &knowledge.KnowledgeItem{
		ChatID: "dm_alice_bot-1", UserID: "alice",
	}, &knowledge.MemoryVersion{
		Type: knowledge.MemoryTypeFact, Content: "Bad fact", Confidence: 0.3,
		Provenance: knowledge.Provenance{ChatID: "dm_alice_bot-1", UserID: "alice"},
	})
	require.NoError(t, err)
	_, skillVer1, err := kStore.ProposeSkill(ctx, &knowledge.KnowledgeItem{
		ChatID: "dm_alice_bot-1", UserID: "alice",
	}, &knowledge.SkillVersion{
		Name: "lint-code", Description: "Runs linter", InstructionsMarkdown: "# Lint",
		Provenance: knowledge.Provenance{ChatID: "dm_alice_bot-1", UserID: "alice"},
	})
	require.NoError(t, err)

	// Unicode NBSP / vertical-tab smuggled suffix must fail closed and NEVER approve memVer1
	require.NoError(t, gw.ProcessMessage(ctx, models.Message{
		Seq: 95, ChatID: "dm_alice_bot-1", UserID: "alice", Type: models.MessageTypeUIResponse,
		UIResponse: &models.UIResponseData{ParentSeq: 9, SelectedIDs: []string{"bob:mem:approve:" + memVer1.ID + "\u00a0extra"}},
	}))
	assert.Contains(t, nextWS().Content, "Unrecognized or invalid UI action")
	unchangedMem, err := kStore.GetMemoryVersion(ctx, memVer1.ID)
	require.NoError(t, err)
	assert.Equal(t, knowledge.VersionStatusProposed, unchangedMem.Status)

	require.NoError(t, gw.ProcessMessage(ctx, models.Message{
		Seq: 10, ChatID: "dm_alice_bot-1", UserID: "alice", Type: models.MessageTypeUIResponse,
		UIResponse: &models.UIResponseData{ParentSeq: 9, SelectedIDs: []string{"bob:mem:approve:" + memVer1.ID}},
	}))
	assert.Contains(t, nextWS().Content, "approved and activated")

	require.NoError(t, gw.ProcessMessage(ctx, models.Message{
		Seq: 11, ChatID: "dm_alice_bot-1", UserID: "alice", Type: models.MessageTypeUIResponse,
		UIResponse: &models.UIResponseData{ParentSeq: 9, SelectedIDs: []string{"bob:mem:reject:" + memVer2.ID}},
	}))
	assert.Contains(t, nextWS().Content, "has been rejected")

	require.NoError(t, gw.ProcessMessage(ctx, models.Message{
		Seq: 12, ChatID: "dm_alice_bot-1", UserID: "alice", Type: models.MessageTypeUIResponse,
		UIResponse: &models.UIResponseData{ParentSeq: 9, SelectedIDs: []string{"bob:skill:approve:" + skillVer1.ID}},
	}))
	assert.Contains(t, nextWS().Content, "approved and activated")

	gotMem1, err := kStore.GetMemoryVersion(ctx, memVer1.ID)
	require.NoError(t, err)
	assert.Equal(t, knowledge.VersionStatusApproved, gotMem1.Status)
	gotMem2, err := kStore.GetMemoryVersion(ctx, memVer2.ID)
	require.NoError(t, err)
	assert.Equal(t, knowledge.VersionStatusRejected, gotMem2.Status)
	gotSkill1, err := kStore.GetSkillVersion(ctx, skillVer1.ID)
	require.NoError(t, err)
	assert.Equal(t, knowledge.VersionStatusApproved, gotSkill1.Status)

	// 7. Schedule Approve + cross-chat guard
	aliceSchedStore, err := storeProv.GetSchedulerStore(ctx, "dm_alice_bot-1", true)
	require.NoError(t, err)
	now := time.Now().Unix()
	aliceSched := &scheduler.Schedule{
		ID: "sched_alice_1", Name: "alice_job", ChatID: "dm_alice_bot-1", UserID: "alice",
		ScheduleType: scheduler.ScheduleTypeInterval, IntervalSeconds: 600,
		Instruction: "Ping", Status: scheduler.ScheduleStatusPendingApproval,
		RunTimeoutSeconds: 60, MaxTurns: 5, NextRunAt: now + 600, CreatedAt: now, UpdatedAt: now,
	}
	require.NoError(t, aliceSchedStore.CreateSchedule(ctx, aliceSched, nil))

	mallorySchedStore, err := storeProv.GetSchedulerStore(ctx, "dm_mallory_bot-1", true)
	require.NoError(t, err)
	mallorySched := &scheduler.Schedule{
		ID: "sched_mallory_1", Name: "mallory_job", ChatID: "dm_mallory_bot-1", UserID: "mallory",
		ScheduleType: scheduler.ScheduleTypeInterval, IntervalSeconds: 600,
		Instruction: "Evil", Status: scheduler.ScheduleStatusPendingApproval,
		RunTimeoutSeconds: 60, MaxTurns: 5, NextRunAt: now + 600, CreatedAt: now, UpdatedAt: now,
	}
	require.NoError(t, mallorySchedStore.CreateSchedule(ctx, mallorySched, nil))

	// Forged cross-chat schedule approval in alice's chat using mallory's schedule ID fails closed
	require.NoError(t, gw.ProcessMessage(ctx, models.Message{
		Seq: 13, ChatID: "dm_alice_bot-1", UserID: "alice", Type: models.MessageTypeUIResponse,
		UIResponse: &models.UIResponseData{ParentSeq: 12, SelectedIDs: []string{fmt.Sprintf("bob:sched:approve:%s", mallorySched.ID)}},
	}))
	assert.Contains(t, nextWS().Content, "Schedule not found")
	stillPending, err := mallorySchedStore.GetSchedule(ctx, mallorySched.ID)
	require.NoError(t, err)
	assert.Equal(t, scheduler.ScheduleStatusPendingApproval, stillPending.Status)

	// Legitimate schedule approval succeeds
	require.NoError(t, gw.ProcessMessage(ctx, models.Message{
		Seq: 14, ChatID: "dm_alice_bot-1", UserID: "alice", Type: models.MessageTypeUIResponse,
		UIResponse: &models.UIResponseData{ParentSeq: 12, SelectedIDs: []string{fmt.Sprintf("bob:sched:approve:%s", aliceSched.ID)}},
	}))
	assert.Contains(t, nextWS().Content, "Schedule approved and activated")
	approvedSched, err := aliceSchedStore.GetSchedule(ctx, aliceSched.ID)
	require.NoError(t, err)
	assert.Equal(t, scheduler.ScheduleStatusActive, approvedSched.Status)

	// 8. Non-reserved SelectedIDs return handled=false and fall through to normal LLM handling
	handled, err := gw.handleUIResponseAction(ctx, models.Message{
		ChatID: "dm_alice_bot-1", UserID: "alice", Type: models.MessageTypeUIResponse,
		UIResponse: &models.UIResponseData{ParentSeq: 14, SelectedIDs: []string{"opt_a", "opt_b"}},
	}, true, "Alice")
	require.NoError(t, err)
	assert.False(t, handled)

	llmMu.Lock()
	assert.Equal(t, 1, llmCalls, fmt.Sprintf("expected only 1 LLM call (for sandbox continuation), got %d", llmCalls))
	llmMu.Unlock()
}
