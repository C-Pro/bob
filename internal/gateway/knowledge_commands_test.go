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
	"bob/internal/tools"

	"github.com/fasthttp/websocket"
	openai "github.com/sashabaranov/go-openai"
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

	_, err = gw.FetchBotUser(ctx)
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

	// Verify discoverable via SearchSkills when active
	_, err = gw.StoreProvider().ReconcileChat(ctx, "dm_user_alice", true, 10)
	require.NoError(t, err)
	searcher, err := gw.StoreProvider().GetKnowledgeSearcher(ctx, "dm_user_alice", true, "user_alice")
	require.NoError(t, err)
	skills, err := searcher.SearchSkills(ctx, "integration", nil, 10)
	require.NoError(t, err)
	assert.Len(t, skills, 1)

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

	// Verify undiscoverable when disabled
	skills, err = searcher.SearchSkills(ctx, "integration", nil, 10)
	require.NoError(t, err)
	assert.Empty(t, skills)

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

	// Reconcile and verify discoverable again
	_, err = gw.StoreProvider().ReconcileChat(ctx, "dm_user_alice", true, 10)
	require.NoError(t, err)
	skills, err = searcher.SearchSkills(ctx, "integration", nil, 10)
	require.NoError(t, err)
	assert.Len(t, skills, 1)

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

	// Verify deletion queue drains and vector records are removed
	_, err = gw.StoreProvider().ReconcileChat(ctx, "dm_user_alice", true, 10)
	require.NoError(t, err)
	skills, err = searcher.SearchSkills(ctx, "integration", nil, 10)
	require.NoError(t, err)
	assert.Empty(t, skills)
	pendingDels, err := kStore.GetPendingIndexDeletions(ctx, 10)
	require.NoError(t, err)
	assert.Empty(t, pendingDels)

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

func TestKnowledgeCommands_ListExpiredMemory_NotFallbackToActive(t *testing.T) {
	gw, kStore, ch, cleanup := setupKnowledgeTestGateway(t)
	defer cleanup()

	ctx := context.Background()

	pastTime := time.Now().Unix() - 3600
	item, ver, err := kStore.ProposeMemory(ctx, &knowledge.KnowledgeItem{
		ChatID: "dm_user_alice",
		UserID: "user_alice",
	}, &knowledge.MemoryVersion{
		Type:      knowledge.MemoryTypeFact,
		Content:   "Ephemeral temporary code 9999",
		ExpiresAt: &pastTime,
	})
	require.NoError(t, err)

	err = kStore.ApproveVersion(ctx, ver.ID, "user_alice")
	require.NoError(t, err)

	// 1. /memory list expired should find the memory and render its status as expired
	err = gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_user_alice",
		UserID:    "user_alice",
		Content:   "/memory list expired",
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)
	msg := recvKnowledgeMsg(t, ch)
	assert.Contains(t, msg.Content, "🧠 **Memories** (expired)")
	assert.Contains(t, msg.Content, item.ID)
	assert.Contains(t, msg.Content, "Ephemeral temporary code 9999")
	assert.Contains(t, msg.Content, "| expired |")

	// 2. /memory list active should NOT find this memory because it is timestamp-expired
	err = gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_user_alice",
		UserID:    "user_alice",
		Content:   "/memory list active",
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)
	msg = recvKnowledgeMsg(t, ch)
	assert.Contains(t, msg.Content, `No memories found with status "active"`)
}

func TestGateway_SkillDisable_Enable_LatePurgeSafe(t *testing.T) {
	gw, kStore, ch, cleanup := setupKnowledgeTestGateway(t)
	defer cleanup()

	ctx := context.Background()

	// Propose and approve skill
	item, ver, err := kStore.ProposeSkill(ctx, &knowledge.KnowledgeItem{
		ChatID: "dm_user_alice",
		UserID: "user_alice",
	}, &knowledge.SkillVersion{
		Name:                 "race_skill",
		Description:          "testing concurrent enable vs purge",
		InstructionsMarkdown: "# instructions",
	})
	require.NoError(t, err)

	err = gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_user_alice",
		UserID:    "user_alice",
		Content:   "/skill approve " + ver.ID,
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)
	_ = recvKnowledgeMsg(t, ch)

	// Now disable skill
	err = gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_user_alice",
		UserID:    "user_alice",
		Content:   "/skill disable " + item.ID,
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)
	msg := recvKnowledgeMsg(t, ch)
	assert.Contains(t, msg.Content, "has been disabled")

	// Immediately re-enable skill
	err = gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_user_alice",
		UserID:    "user_alice",
		Content:   "/skill enable " + item.ID,
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)
	msg = recvKnowledgeMsg(t, ch)
	assert.Contains(t, msg.Content, "has been enabled")

	// Even if delayed purge of disabled skill runs now:
	indexer, err := gw.StoreProvider().GetKnowledgeIndexer(ctx, "dm_user_alice", true)
	require.NoError(t, err)
	purged, err := indexer.PurgePendingDeletion(ctx, ver.ID)
	require.NoError(t, err)
	assert.False(t, purged, "delayed purge must not purge active re-enabled skill")

	// /skill list active must list the skill
	err = gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_user_alice",
		UserID:    "user_alice",
		Content:   "/skill list active",
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)
	msg = recvKnowledgeMsg(t, ch)
	assert.Contains(t, msg.Content, "race_skill")
	assert.Contains(t, msg.Content, "| active |")
}

func TestGateway_VerifyDMOwner_EdgeCases(t *testing.T) {
	gw, _, _, cleanup := setupKnowledgeTestGateway(t)
	defer cleanup()

	ctx := context.Background()

	// 1. Empty chatID or townhall
	assert.ErrorContains(t, gw.VerifyDMOwner(ctx, "", "user_alice"), "only available in private Direct Messages")
	assert.ErrorContains(t, gw.VerifyDMOwner(ctx, "   ", "user_alice"), "only available in private Direct Messages")
	assert.ErrorContains(t, gw.VerifyDMOwner(ctx, "townhall", "user_alice"), "only available in private Direct Messages")

	// 2. Empty userID
	assert.ErrorContains(t, gw.VerifyDMOwner(ctx, "dm_chat", ""), "empty user identity")
	assert.ErrorContains(t, gw.VerifyDMOwner(ctx, "dm_chat", "   "), "empty user identity")

	// 3. Chat not in cache and no http client -> fetch error
	assert.ErrorContains(t, gw.VerifyDMOwner(ctx, "non_existent_chat", "user_alice"), "unable to verify chat metadata")

	// 4. Chat is group chat (not DM)
	gw.chatCache.Set(models.Chat{
		ID:      "group_1",
		Type:    "group",
		IsDM:    false,
		UserIDs: []string{"bot-id-qa", "user_alice"},
	})
	assert.ErrorContains(t, gw.VerifyDMOwner(ctx, "group_1", "user_alice"), "only available in private Direct Messages")

	// 5. Multi-human participants (2 humans in UserIDs)
	gw.chatCache.Set(models.Chat{
		ID:           "multi_dm",
		Type:         "dm",
		IsDM:         true,
		TargetUserID: "user_alice",
		UserIDs:      []string{"bot-id-qa", "user_alice", "user_eve"},
	})
	assert.ErrorContains(t, gw.VerifyDMOwner(ctx, "multi_dm", "user_alice"), "exactly one human participant")

	// 6. Zero human participants in UserIDs
	gw.chatCache.Set(models.Chat{
		ID:           "zero_human_dm",
		Type:         "dm",
		IsDM:         true,
		TargetUserID: "user_alice",
		UserIDs:      []string{"bot-id-qa"},
	})
	assert.ErrorContains(t, gw.VerifyDMOwner(ctx, "zero_human_dm", "user_alice"), "exactly one human participant")

	// 7. TargetUserID mismatch with single human in UserIDs
	gw.chatCache.Set(models.Chat{
		ID:           "mismatch_dm",
		Type:         "dm",
		IsDM:         true,
		TargetUserID: "user_alice",
		UserIDs:      []string{"bot-id-qa", "user_eve"},
	})
	assert.ErrorContains(t, gw.VerifyDMOwner(ctx, "mismatch_dm", "user_alice"), "target user mismatch with chat participant list")

	// 8. No TargetUserID and no UserIDs (unconstrained DM)
	gw.chatCache.Set(models.Chat{
		ID:   "no_owner_dm",
		Type: "dm",
		IsDM: true,
	})
	assert.ErrorContains(t, gw.VerifyDMOwner(ctx, "no_owner_dm", "user_alice"), "unable to verify human owner of this DM")

	// 9. Non-owner caller
	gw.chatCache.Set(models.Chat{
		ID:           "valid_dm",
		Type:         "dm",
		IsDM:         true,
		TargetUserID: "user_alice",
		UserIDs:      []string{"bot-id-qa", "user_alice"},
	})
	assert.ErrorContains(t, gw.VerifyDMOwner(ctx, "valid_dm", "user_eve"), "only the owner of this DM can manage memories and skills")

	// 10. Valid owner caller
	assert.NoError(t, gw.VerifyDMOwner(ctx, "valid_dm", "user_alice"))

	// 11. Valid owner caller where TargetUserID is empty but UserIDs has single human
	gw.chatCache.Set(models.Chat{
		ID:      "inferred_owner_dm",
		Type:    "dm",
		IsDM:    true,
		UserIDs: []string{"bot-id-qa", "user_alice"},
	})
	assert.NoError(t, gw.VerifyDMOwner(ctx, "inferred_owner_dm", "user_alice"))
}

func TestKnowledgeTools_GroupChatRejection_DefinitionsAndExecution(t *testing.T) {
	gw, _, _, cleanup := setupKnowledgeTestGateway(t)
	defer cleanup()

	ctx := context.Background()

	knowledgeToolNames := []string{
		"propose_memory", "discover_memories", "load_memory",
		"propose_skill", "discover_skills", "load_skill",
	}

	hasKnowledgeTool := func(defs []openai.Tool) bool {
		for _, d := range defs {
			if d.Function != nil {
				for _, name := range knowledgeToolNames {
					if d.Function.Name == name {
						return true
					}
				}
			}
		}
		return false
	}

	// 1. Group chat tool definitions must not expose any knowledge tools
	groupDefs := gw.ToolDefinitions(ctx, "group_chat_1", false)
	assert.False(t, hasKnowledgeTool(groupDefs), "group chat must not expose knowledge tools")

	// 2. Townhall tool definitions must not expose any knowledge tools
	townhallDefs := gw.ToolDefinitions(ctx, "townhall", false)
	assert.False(t, hasKnowledgeTool(townhallDefs), "townhall must not expose knowledge tools")

	// 3. Spoofed DM in context for group chat: ToolDefinitions must fail-closed via VerifyDMOwner
	spoofedSession := tools.ChatSessionContext{
		ChatID: "group_chat_1",
		UserID: "user_alice",
		IsDM:   true,
	}
	spoofedCtx := tools.WithChatSession(ctx, spoofedSession)
	spoofedDefs := gw.ToolDefinitions(spoofedCtx, "group_chat_1", true)
	assert.False(t, hasKnowledgeTool(spoofedDefs), "spoofed group chat must have knowledge tools stripped")

	// 4. Direct execution of knowledge tools in non-DM session must fail closed
	nonDMSession := tools.NewChatSessionContext("group_chat_1", "user_alice", false, tools.DefaultKnowledgeBudgetLimits())
	nonDMCtx := tools.WithChatSession(ctx, nonDMSession)

	gw.mu.Lock()
	registry := gw.toolsRegistry
	gw.mu.Unlock()
	require.NotNil(t, registry)

	_, err := registry.Execute(nonDMCtx, "propose_memory", `{"content":"secret preference"}`)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "only available in direct messages")

	_, err = registry.Execute(nonDMCtx, "load_memory", `{"item_id":"mem_123"}`)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "only available in direct messages")

	// 5. Direct execution with spoofed isDM=true in group chat must fail via DMAuthorizer
	spoofedExecSession := tools.NewChatSessionContext("group_chat_1", "user_alice", true, tools.DefaultKnowledgeBudgetLimits())
	spoofedExecCtx := tools.WithChatSession(ctx, spoofedExecSession)

	_, err = registry.Execute(spoofedExecCtx, "propose_memory", `{"content":"secret preference"}`)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "only available in private Direct Messages")

	_, err = registry.Execute(spoofedExecCtx, "load_memory", `{"item_id":"mem_123"}`)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "only available in private Direct Messages")

	// 6. Mismatched session.ChatID between method argument and context must fail closed
	mismatchedSession := tools.ChatSessionContext{
		ChatID: "dm_user_alice",
		UserID: "user_alice",
		IsDM:   true,
	}
	mismatchedCtx := tools.WithChatSession(ctx, mismatchedSession)
	mismatchedDefs := gw.ToolDefinitions(mismatchedCtx, "group_chat_1", false)
	assert.False(t, hasKnowledgeTool(mismatchedDefs), "mismatched chatID must fail closed and strip knowledge tools")
}

func TestProcessMessage_UncachedGroupChat_NoMention(t *testing.T) {
	gw, _, ch, cleanup := setupKnowledgeTestGateway(t)
	defer cleanup()

	ctx := context.Background()

	// Send an unmentioned message in an uncached chat ("new_group_chat")
	err := gw.ProcessMessage(ctx, models.Message{
		ChatID:    "new_group_chat",
		UserID:    "user_alice",
		Content:   "Hello everyone in this new group!",
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)

	// Since it is an uncached non-DM chat without mention, no reply should be sent
	select {
	case msg := <-ch:
		t.Fatalf("unexpected message sent for unmentioned uncached chat: %s", msg.Content)
	case <-time.After(200 * time.Millisecond):
		// Expected: no reply triggered
	}
}

func TestProcessMessage_UnknownDMChat_NoMention(t *testing.T) {
	gw, _, ch, cleanup := setupKnowledgeTestGateway(t)
	defer cleanup()

	ctx := context.Background()

	// An unknown chat whose ID begins with "dm_" ("dm_unknown_channel") not in cache or server
	err := gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_unknown_channel",
		UserID:    "user_alice",
		Content:   "Hello in unknown chat",
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)

	// Since lookup fails, must fail closed to public/group behavior and NOT reply without mention
	select {
	case msg := <-ch:
		t.Fatalf("unexpected message sent for unmentioned unknown dm_* chat: %s", msg.Content)
	case <-time.After(200 * time.Millisecond):
		// Expected: no reply triggered
	}
}

func TestProcessMessage_GroupChatWithDMPrefix_NoMention(t *testing.T) {
	gw, _, ch, cleanup := setupKnowledgeTestGateway(t)
	defer cleanup()

	ctx := context.Background()

	// A chat whose ID begins with "dm_" but authoritative metadata marks it as a group chat
	gw.chatCache.Set(models.Chat{
		ID:      "dm_project_group",
		Type:    "group",
		IsDM:    false,
		UserIDs: []string{"bot-id-qa", "user_alice", "user_bob"},
	})

	err := gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_project_group",
		UserID:    "user_alice",
		Content:   "Status update for the project team",
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)

	// Authoritative group chat must not trigger reply without mention despite "dm_" prefix
	select {
	case msg := <-ch:
		t.Fatalf("unexpected message sent for group chat with dm_ prefix: %s", msg.Content)
	case <-time.After(200 * time.Millisecond):
		// Expected: no reply triggered
	}
}
