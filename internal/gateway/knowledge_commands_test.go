package gateway

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"bob/internal/chatcontext"
	"bob/internal/config"
	"bob/internal/knowledge"
	"bob/internal/models"

	"github.com/fasthttp/websocket"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func setupKnowledgeTestGateway(t *testing.T) (*Gateway, *knowledge.Store, chan models.ClientMessage, func()) {
	t.Helper()

	upgrader := websocket.Upgrader{}
	receivedMsgs := make(chan models.ClientMessage, 50)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/me" {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(models.User{ID: "bot-id-qa", UserName: "bot", DisplayName: "Bob Bot"})
			return
		}
		if r.URL.Path == "/api/users" {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode([]models.User{
				{ID: "bot-id-qa", UserName: "bot", DisplayName: "Bob Bot"},
				{ID: "user_alice", UserName: "alice", DisplayName: "Alice"},
			})
			return
		}
		if r.URL.Path == "/api/chats" {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode([]models.Chat{
				{ID: "townhall", Type: "townhall"},
				{ID: "dm_user_alice", Type: "dm", IsDM: true, TargetUserID: "user_alice"},
				{ID: "group_chat_1", Type: "group", IsDM: false},
			})
			return
		}
		if strings.HasPrefix(r.URL.Path, "/api/chats/") && strings.HasSuffix(r.URL.Path, "/messages") {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode([]models.Message{})
			return
		}

		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		go func() {
			defer func() { _ = conn.Close() }()
			for {
				var clientMsg models.ClientMessage
				if err := conn.ReadJSON(&clientMsg); err != nil {
					break
				}
				receivedMsgs <- clientMsg
			}
		}()
	}))

	tempDir := t.TempDir()
	linkTestModels(t, tempDir)

	cfg := &config.Config{
		BotHandle:         "@bot",
		BesedkaURL:        server.URL,
		DataDir:           tempDir,
		MsgRingBufferSize: 50,
	}

	gw := NewGateway(cfg, nil)
	gw.httpClient = server.Client()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	err := gw.DialWebSocket(ctx)
	require.NoError(t, err)

	kStore, err := gw.StoreProvider().GetKnowledgeStore(ctx, "dm_user_alice", true)
	require.NoError(t, err)

	cleanup := func() {
		gw.Stop()
		server.Close()
	}

	return gw, kStore, receivedMsgs, cleanup
}

func recvKnowledgeMsg(t *testing.T, ch chan models.ClientMessage) models.ClientMessage {
	t.Helper()
	select {
	case msg := <-ch:
		return msg
	case <-time.After(3 * time.Second):
		t.Fatal("timed out waiting for client message")
		return models.ClientMessage{}
	}
}

func TestKnowledgeCommands_TownhallRejection(t *testing.T) {
	gw, _, ch, cleanup := setupKnowledgeTestGateway(t)
	defer cleanup()

	ctx := context.Background()

	// 1. /memory in townhall
	err := gw.ProcessMessage(ctx, models.Message{
		ChatID:    "townhall",
		UserID:    "user_alice",
		Content:   "@bot /memory list",
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)

	msg := recvKnowledgeMsg(t, ch)
	assert.Contains(t, msg.Content, "only available in private Direct Messages")

	// 2. /skill in townhall
	err = gw.ProcessMessage(ctx, models.Message{
		ChatID:    "townhall",
		UserID:    "user_alice",
		Content:   "@bot /skill list",
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)

	msg = recvKnowledgeMsg(t, ch)
	assert.Contains(t, msg.Content, "only available in private Direct Messages")
}

func TestKnowledgeCommands_BotSenderIgnored(t *testing.T) {
	gw, _, ch, cleanup := setupKnowledgeTestGateway(t)
	defer cleanup()

	ctx := context.Background()

	// Message from bot itself
	err := gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_user_alice",
		UserID:    "bot-id-qa",
		Content:   "/memory list",
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)

	select {
	case msg := <-ch:
		t.Fatalf("expected no response to bot sender, got: %s", msg.Content)
	case <-time.After(500 * time.Millisecond):
		// Expected timeout, no reply sent
	}
}

func TestMemoryCommands_Lifecycle(t *testing.T) {
	gw, kStore, ch, cleanup := setupKnowledgeTestGateway(t)
	defer cleanup()

	ctx := context.Background()

	// 1. Help message
	err := gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_user_alice",
		UserID:    "user_alice",
		Content:   "/memory help",
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)
	msg := recvKnowledgeMsg(t, ch)
	assert.Contains(t, msg.Content, "Memory Commands:")

	// 2. List empty active
	err = gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_user_alice",
		UserID:    "user_alice",
		Content:   "/memory list",
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)
	msg = recvKnowledgeMsg(t, ch)
	assert.Contains(t, msg.Content, "No memories found with status \"active\"")

	// 3. Propose a memory in the store
	msgSeq := int64(10)
	item, ver, err := kStore.ProposeMemory(ctx, &knowledge.KnowledgeItem{
		ChatID: "dm_user_alice",
		UserID: "user_alice",
	}, &knowledge.MemoryVersion{
		Type:       knowledge.MemoryTypeFact,
		Content:    "Alice prefers dark mode in all editors",
		Confidence: 0.99,
		Provenance: knowledge.Provenance{
			SourceMessageSeq: &msgSeq,
			FSMRunID:         "fsm_run_01",
		},
	})
	require.NoError(t, err)

	// 4. List pending
	err = gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_user_alice",
		UserID:    "user_alice",
		Content:   "/memory list pending",
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)
	msg = recvKnowledgeMsg(t, ch)
	assert.Contains(t, msg.Content, item.ID)
	assert.Contains(t, msg.Content, "Alice prefers dark mode")

	// 5. Show item overview
	err = gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_user_alice",
		UserID:    "user_alice",
		Content:   "/memory show " + item.ID,
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)
	msg = recvKnowledgeMsg(t, ch)
	assert.Contains(t, msg.Content, item.ID)
	assert.Contains(t, msg.Content, "Revisions:")
	assert.Contains(t, msg.Content, "@1")

	// 6. Show specific version
	err = gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_user_alice",
		UserID:    "user_alice",
		Content:   "/memory show " + ver.ID,
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)
	msg = recvKnowledgeMsg(t, ch)
	assert.Contains(t, msg.Content, ver.ID)
	assert.Contains(t, msg.Content, "Alice prefers dark mode in all editors")
	assert.Contains(t, msg.Content, "Source Message Seq")
	assert.Contains(t, msg.Content, "10")
	assert.Contains(t, msg.Content, "fsm_run_01")

	// 7. Approve memory version
	err = gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_user_alice",
		UserID:    "user_alice",
		Content:   "/memory approve " + ver.ID,
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)
	msg = recvKnowledgeMsg(t, ch)
	assert.Contains(t, msg.Content, "approved and activated")

	// 8. List active now returns it
	err = gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_user_alice",
		UserID:    "user_alice",
		Content:   "/memory list active",
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)
	msg = recvKnowledgeMsg(t, ch)
	assert.Contains(t, msg.Content, item.ID)
	assert.Contains(t, msg.Content, "active")

	// 9. Forget memory item: test explicit required notice
	err = gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_user_alice",
		UserID:    "user_alice",
		Content:   "/memory forget " + item.ID,
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)
	msg = recvKnowledgeMsg(t, ch)
	assert.Contains(t, msg.Content, fmt.Sprintf("Memory `%s` has been forgotten and erased from structured memory.", item.ID))
	assert.Contains(t, msg.Content, "*Note: Past conversation transcripts and raw message history remain unchanged.*")

	// 10. Reject flow on a second memory
	_, ver2, err := kStore.ProposeMemory(ctx, &knowledge.KnowledgeItem{
		ChatID: "dm_user_alice",
		UserID: "user_alice",
	}, &knowledge.MemoryVersion{
		Type:    knowledge.MemoryTypeFact,
		Content: "Second fact to be rejected",
	})
	require.NoError(t, err)

	err = gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_user_alice",
		UserID:    "user_alice",
		Content:   "/memory reject " + ver2.ID,
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)
	msg = recvKnowledgeMsg(t, ch)
	assert.Contains(t, msg.Content, "has been rejected")
}

func TestSkillCommands_Lifecycle(t *testing.T) {
	gw, kStore, ch, cleanup := setupKnowledgeTestGateway(t)
	defer cleanup()

	ctx := context.Background()

	// 1. Help message
	err := gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_user_alice",
		UserID:    "user_alice",
		Content:   "/skill help",
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)
	msg := recvKnowledgeMsg(t, ch)
	assert.Contains(t, msg.Content, "Skill Commands:")

	// 2. Propose skill
	item, ver, err := kStore.ProposeSkill(ctx, &knowledge.KnowledgeItem{
		ChatID: "dm_user_alice",
		UserID: "user_alice",
	}, &knowledge.SkillVersion{
		Name:                 "run_integration",
		Description:          "Runs integration tests",
		Triggers:             []string{"test", "integration"},
		Tags:                 []string{"ci", "dev"},
		InstructionsMarkdown: "```bash\ngo test -v -tags integration ./...\n```",
	})
	require.NoError(t, err)

	// 3. List pending
	err = gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_user_alice",
		UserID:    "user_alice",
		Content:   "/skill list pending",
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)
	msg = recvKnowledgeMsg(t, ch)
	assert.Contains(t, msg.Content, item.ID)
	assert.Contains(t, msg.Content, "run_integration")

	// 4. Show skill item
	err = gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_user_alice",
		UserID:    "user_alice",
		Content:   "/skill show " + item.ID,
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)
	msg = recvKnowledgeMsg(t, ch)
	assert.Contains(t, msg.Content, item.ID)
	assert.Contains(t, msg.Content, "run_integration")

	// 5. Show skill version
	err = gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_user_alice",
		UserID:    "user_alice",
		Content:   "/skill show " + ver.ID,
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)
	msg = recvKnowledgeMsg(t, ch)
	assert.Contains(t, msg.Content, ver.ID)
	assert.Contains(t, msg.Content, "Instructions:")
	assert.Contains(t, msg.Content, "go test -v -tags integration")

	// 6. Approve skill
	err = gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_user_alice",
		UserID:    "user_alice",
		Content:   "/skill approve " + ver.ID,
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)
	msg = recvKnowledgeMsg(t, ch)
	assert.Contains(t, msg.Content, "approved and activated")

	// 7. Disable skill
	err = gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_user_alice",
		UserID:    "user_alice",
		Content:   "/skill disable " + item.ID,
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)
	msg = recvKnowledgeMsg(t, ch)
	assert.Contains(t, msg.Content, "has been disabled")

	// 8. Enable skill
	err = gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_user_alice",
		UserID:    "user_alice",
		Content:   "/skill enable " + item.ID,
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)
	msg = recvKnowledgeMsg(t, ch)
	assert.Contains(t, msg.Content, "has been enabled")

	// 9. Archive skill
	err = gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_user_alice",
		UserID:    "user_alice",
		Content:   "/skill archive " + item.ID,
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)
	msg = recvKnowledgeMsg(t, ch)
	assert.Contains(t, msg.Content, "has been archived")

	// 10. Delete skill
	err = gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_user_alice",
		UserID:    "user_alice",
		Content:   "/skill delete " + item.ID,
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)
	msg = recvKnowledgeMsg(t, ch)
	assert.Contains(t, msg.Content, "has been permanently deleted")

	// 11. Reject flow
	_, ver2, err := kStore.ProposeSkill(ctx, &knowledge.KnowledgeItem{
		ChatID: "dm_user_alice",
		UserID: "user_alice",
	}, &knowledge.SkillVersion{
		Name:                 "rejected_skill",
		Description:          "to be rejected",
		InstructionsMarkdown: "do nothing",
	})
	require.NoError(t, err)

	err = gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_user_alice",
		UserID:    "user_alice",
		Content:   "/skill reject " + ver2.ID,
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)
	msg = recvKnowledgeMsg(t, ch)
	assert.Contains(t, msg.Content, "has been rejected")
}

func TestKnowledgeCommands_OwnerAuthorization(t *testing.T) {
	gw, _, ch, cleanup := setupKnowledgeTestGateway(t)
	defer cleanup()

	ctx := context.Background()

	// Eve attempts memory command in Alice's DM
	err := gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_user_alice",
		UserID:    "user_eve",
		Content:   "/memory list",
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)
	msg := recvKnowledgeMsg(t, ch)
	assert.Contains(t, msg.Content, "only the owner of this DM can manage memories and skills")

	// Eve attempts skill command in Alice's DM
	err = gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_user_alice",
		UserID:    "user_eve",
		Content:   "/skill list",
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)
	msg = recvKnowledgeMsg(t, ch)
	assert.Contains(t, msg.Content, "only the owner of this DM can manage memories and skills")
}

func TestKnowledgeCommands_GroupChatRejection(t *testing.T) {
	gw, _, ch, cleanup := setupKnowledgeTestGateway(t)
	defer cleanup()

	ctx := context.Background()

	// Memory in group chat
	err := gw.ProcessMessage(ctx, models.Message{
		ChatID:    "group_chat_1",
		UserID:    "user_alice",
		Content:   "/memory list",
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)
	msg := recvKnowledgeMsg(t, ch)
	assert.Contains(t, msg.Content, "only available in private Direct Messages")

	// Skill in group chat
	err = gw.ProcessMessage(ctx, models.Message{
		ChatID:    "group_chat_1",
		UserID:    "user_alice",
		Content:   "/skill list",
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)
	msg = recvKnowledgeMsg(t, ch)
	assert.Contains(t, msg.Content, "only available in private Direct Messages")
}

func TestKnowledgeCommands_CrossKindRejection(t *testing.T) {
	gw, kStore, ch, cleanup := setupKnowledgeTestGateway(t)
	defer cleanup()

	ctx := context.Background()

	// 1. Propose memory
	memItem, memVer, err := kStore.ProposeMemory(ctx, &knowledge.KnowledgeItem{
		ChatID: "dm_user_alice",
		UserID: "user_alice",
	}, &knowledge.MemoryVersion{
		Type:    knowledge.MemoryTypeFact,
		Content: "Cross kind test fact",
	})
	require.NoError(t, err)

	// Try to approve memory via /skill
	err = gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_user_alice",
		UserID:    "user_alice",
		Content:   "/skill approve " + memVer.ID,
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)
	msg := recvKnowledgeMsg(t, ch)
	assert.Contains(t, msg.Content, "skill IDs must begin with \"skill_\"")

	// Try to show memory via /skill
	err = gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_user_alice",
		UserID:    "user_alice",
		Content:   "/skill show " + memItem.ID,
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)
	msg = recvKnowledgeMsg(t, ch)
	assert.Contains(t, msg.Content, "skill IDs must begin with \"skill_\"")

	// 2. Propose skill
	skillItem, skillVer, err := kStore.ProposeSkill(ctx, &knowledge.KnowledgeItem{
		ChatID: "dm_user_alice",
		UserID: "user_alice",
	}, &knowledge.SkillVersion{
		Name:                 "cross_skill",
		Description:          "testing cross skill",
		InstructionsMarkdown: "steps",
	})
	require.NoError(t, err)

	// Try to approve skill via /memory
	err = gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_user_alice",
		UserID:    "user_alice",
		Content:   "/memory approve " + skillVer.ID,
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)
	msg = recvKnowledgeMsg(t, ch)
	assert.Contains(t, msg.Content, "memory IDs must begin with \"mem_\"")

	// Try to show skill via /memory
	err = gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_user_alice",
		UserID:    "user_alice",
		Content:   "/memory show " + skillItem.ID,
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)
	msg = recvKnowledgeMsg(t, ch)
	assert.Contains(t, msg.Content, "memory IDs must begin with \"mem_\"")
}

func TestKnowledgeCommands_DestructiveVersionReferenceRejection(t *testing.T) {
	gw, _, ch, cleanup := setupKnowledgeTestGateway(t)
	defer cleanup()

	ctx := context.Background()

	// /memory forget with revision reference
	err := gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_user_alice",
		UserID:    "user_alice",
		Content:   "/memory forget mem_123@1",
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)
	msg := recvKnowledgeMsg(t, ch)
	assert.Contains(t, msg.Content, "operates on entire memory items, not specific versions")

	// /skill delete with revision reference
	err = gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_user_alice",
		UserID:    "user_alice",
		Content:   "/skill delete skill_123@1",
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)
	msg = recvKnowledgeMsg(t, ch)
	assert.Contains(t, msg.Content, "operates on entire skill items, not specific versions")

	// /skill disable with revision reference
	err = gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_user_alice",
		UserID:    "user_alice",
		Content:   "/skill disable skill_123@1",
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)
	msg = recvKnowledgeMsg(t, ch)
	assert.Contains(t, msg.Content, "operates on entire skill items, not specific versions")
}

func TestKnowledgeCommands_InvalidStatusFilters(t *testing.T) {
	gw, _, ch, cleanup := setupKnowledgeTestGateway(t)
	defer cleanup()

	ctx := context.Background()

	// /memory list invalid
	err := gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_user_alice",
		UserID:    "user_alice",
		Content:   "/memory list bogus_filter",
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)
	msg := recvKnowledgeMsg(t, ch)
	assert.Contains(t, msg.Content, "Invalid status filter \"bogus_filter\"")

	// /skill list invalid (e.g. expired is valid for memory, but not for skill)
	err = gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_user_alice",
		UserID:    "user_alice",
		Content:   "/skill list expired",
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)
	msg = recvKnowledgeMsg(t, ch)
	assert.Contains(t, msg.Content, "Invalid status filter \"expired\"")
}

func TestKnowledgeCommands_MultiRevisionPendingList(t *testing.T) {
	gw, kStore, ch, cleanup := setupKnowledgeTestGateway(t)
	defer cleanup()

	ctx := context.Background()

	// 1. Propose memory v1 and approve it
	item, ver1, err := kStore.ProposeMemory(ctx, &knowledge.KnowledgeItem{
		ChatID: "dm_user_alice",
		UserID: "user_alice",
	}, &knowledge.MemoryVersion{
		Type:    knowledge.MemoryTypeFact,
		Content: "Initial fact v1",
	})
	require.NoError(t, err)

	err = kStore.ApproveVersion(ctx, ver1.ID, "user_alice")
	require.NoError(t, err)

	// 2. Propose revision v2 (now proposed/pending while item is active with v1)
	_, ver2, err := kStore.ProposeMemory(ctx, &knowledge.KnowledgeItem{
		ID:     item.ID,
		ChatID: "dm_user_alice",
		UserID: "user_alice",
	}, &knowledge.MemoryVersion{
		Type:    knowledge.MemoryTypeFact,
		Content: "Updated fact v2 proposal",
	})
	require.NoError(t, err)
	assert.Equal(t, 2, ver2.Revision)

	// 3. /memory list pending should show @2 and its proposed content, NOT @1!
	err = gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_user_alice",
		UserID:    "user_alice",
		Content:   "/memory list pending",
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)
	msg := recvKnowledgeMsg(t, ch)
	assert.Contains(t, msg.Content, "@2")
	assert.Contains(t, msg.Content, "Updated fact v2 proposal")
	assert.Contains(t, msg.Content, "proposed")
}

func TestKnowledgeCommands_CaseInsensitivityAndBoundaries(t *testing.T) {
	gw, _, ch, cleanup := setupKnowledgeTestGateway(t)
	defer cleanup()

	ctx := context.Background()

	// 1. /Memory list (uppercase M)
	err := gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_user_alice",
		UserID:    "user_alice",
		Content:   "/Memory list",
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)
	msg := recvKnowledgeMsg(t, ch)
	assert.Contains(t, strings.ToLower(msg.Content), "memories")

	// 2. /SKILL list (all uppercase)
	err = gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_user_alice",
		UserID:    "user_alice",
		Content:   "/SKILL list",
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)
	msg = recvKnowledgeMsg(t, ch)
	assert.Contains(t, strings.ToLower(msg.Content), "skills")

	// 3. /memorylane (boundary check: not intercepted as /memory)
	// Because LLM client is nil, an unhandled message produces an error or attempt to call LLM
	_ = gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_user_alice",
		UserID:    "user_alice",
		Content:   "/memorylane is great",
		Timestamp: time.Now().Unix(),
	})
	// Should NOT return a memory command reply
	select {
	case msg := <-ch:
		assert.NotContains(t, msg.Content, "🧠 **Memories**")
		assert.NotContains(t, msg.Content, "Memory Commands:")
	case <-time.After(300 * time.Millisecond):
		// No reply sent (LLM client nil)
	}
}

func TestKnowledgeCommands_TranscriptPreservation(t *testing.T) {
	gw, kStore, ch, cleanup := setupKnowledgeTestGateway(t)
	defer cleanup()

	ctx := context.Background()

	// Push historical conversation entries to context manager
	gw.contextManager.Push("dm_user_alice", chatcontext.Entry{
		Role:    "user",
		Content: "Remember that my secret code is 4242",
		Seq:     1,
	})
	gw.contextManager.Push("dm_user_alice", chatcontext.Entry{
		Role:    "assistant",
		Content: "I have recorded that.",
		Seq:     2,
	})

	// Propose and approve memory
	item, ver, err := kStore.ProposeMemory(ctx, &knowledge.KnowledgeItem{
		ChatID: "dm_user_alice",
		UserID: "user_alice",
	}, &knowledge.MemoryVersion{
		Type:    knowledge.MemoryTypeFact,
		Content: "Secret code is 4242",
	})
	require.NoError(t, err)

	err = kStore.ApproveVersion(ctx, ver.ID, "user_alice")
	require.NoError(t, err)

	// Forget memory
	err = gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_user_alice",
		UserID:    "user_alice",
		Content:   "/memory forget " + item.ID,
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)
	msg := recvKnowledgeMsg(t, ch)
	assert.Contains(t, msg.Content, "Past conversation transcripts and raw message history remain unchanged")

	// Verify transcript entries in ring buffer / context manager are intact!
	entries := gw.contextManager.GetOrCreate("dm_user_alice").Entries()
	require.Len(t, entries, 3)
	assert.Equal(t, "Remember that my secret code is 4242", entries[0].Content)
	assert.Equal(t, "I have recorded that.", entries[1].Content)
	assert.Contains(t, entries[2].Content, "/memory forget")
}
