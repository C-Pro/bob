package gateway

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"bob/internal/agentapi"
	"bob/internal/chatcontext"
	"bob/internal/config"
	"bob/internal/llm"
	"bob/internal/models"

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
