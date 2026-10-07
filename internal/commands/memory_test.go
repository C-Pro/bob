package commands_test

import (
	"context"
	"database/sql"
	"errors"
	"path/filepath"
	"testing"

	"bob/internal/agentapi"
	"bob/internal/commands"
	"bob/internal/knowledge"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	_ "modernc.org/sqlite"
)

func setupTestKnowledgeStore(t *testing.T) (*knowledge.Store, *sql.DB) {
	t.Helper()
	dbPath := filepath.Join(t.TempDir(), "knowledge_test.db")
	db, err := sql.Open("sqlite", "file:"+dbPath+"?_auto_vacuum=INCREMENTAL&_pragma=journal_mode(WAL)&_pragma=foreign_keys(ON)&_pragma=busy_timeout(5000)&_pragma=synchronous(NORMAL)")
	require.NoError(t, err)

	err = knowledge.EnsureKnowledgeSchema(context.Background(), db)
	require.NoError(t, err)

	t.Cleanup(func() { _ = db.Close() })
	return knowledge.NewStore(db), db
}

func TestMemoryHandler_Authorization(t *testing.T) {
	ctx := context.Background()
	store, _ := setupTestKnowledgeStore(t)

	session := agentapi.SessionRef{
		FrontendID: "besedka",
		SessionID:  "chat_dm_1",
		ScopeID:    "chat_dm_1",
	}
	actor := agentapi.Actor{ID: "user_alice"}

	// 1. Nil authorizer fails closed
	hNil := commands.NewMemoryHandler(store, nil, nil, "bot_1")
	res, err := hNil.Handle(ctx, commands.Request{
		Session:    session,
		Actor:      actor,
		Command:    "memory",
		Subcommand: "list",
	})
	require.NoError(t, err)
	assert.Contains(t, res.Reply, "Authorization failed: authorizer not configured")

	// 2. Authorizer rejects
	hDenied := commands.NewMemoryHandler(store, nil, commands.AuthorizerFunc(func(_ context.Context, _ agentapi.SessionRef, _ agentapi.Actor, _, _ string) error {
		return errors.New("only DM owner allowed")
	}), "bot_1")
	res, err = hDenied.Handle(ctx, commands.Request{
		Session:    session,
		Actor:      actor,
		Command:    "memory",
		Subcommand: "list",
	})
	require.NoError(t, err)
	assert.Contains(t, res.Reply, "only DM owner allowed")

	// 3. Bot self-invocation is ignored
	allowAuth := commands.AuthorizerFunc(func(_ context.Context, _ agentapi.SessionRef, _ agentapi.Actor, _, _ string) error {
		return nil
	})
	hBot := commands.NewMemoryHandler(store, nil, allowAuth, "bot_1")
	res, err = hBot.Handle(ctx, commands.Request{
		Session:    session,
		Actor:      agentapi.Actor{ID: "bot_1"},
		Command:    "memory",
		Subcommand: "list",
	})
	require.NoError(t, err)
	assert.True(t, res.IsIgnored)

	// 4. Store nil
	hNoStore := commands.NewMemoryHandler(nil, nil, allowAuth, "bot_1")
	res, err = hNoStore.Handle(ctx, commands.Request{
		Session:    session,
		Actor:      actor,
		Command:    "memory",
		Subcommand: "list",
	})
	require.NoError(t, err)
	assert.Contains(t, res.Reply, "Knowledge storage is unavailable")
}

func TestMemoryHandler_ListAndHelp(t *testing.T) {
	ctx := context.Background()
	store, _ := setupTestKnowledgeStore(t)
	allowAuth := commands.AuthorizerFunc(func(_ context.Context, _ agentapi.SessionRef, _ agentapi.Actor, _, _ string) error {
		return nil
	})
	h := commands.NewMemoryHandler(store, nil, allowAuth, "bot_1")

	session := agentapi.SessionRef{
		FrontendID: "besedka",
		SessionID:  "chat_dm_1",
		ScopeID:    "chat_dm_1",
	}
	actor := agentapi.Actor{ID: "user_alice"}

	// Help on empty/unknown subcommand
	res, err := h.Handle(ctx, commands.Request{
		Session:    session,
		Actor:      actor,
		Command:    "memory",
		Subcommand: "",
	})
	require.NoError(t, err)
	assert.Contains(t, res.Reply, "Memory Commands:")

	// List when empty
	res, err = h.Handle(ctx, commands.Request{
		Session:    session,
		Actor:      actor,
		Command:    "memory",
		Subcommand: "list",
	})
	require.NoError(t, err)
	assert.Contains(t, res.Reply, "No memories found")

	// Invalid filter
	res, err = h.Handle(ctx, commands.Request{
		Session:    session,
		Actor:      actor,
		Command:    "memory",
		Subcommand: "list",
		Args:       []string{"bogus"},
	})
	require.NoError(t, err)
	assert.Contains(t, res.Reply, "Invalid status filter")

	// Propose memory and list pending
	item, ver, err := store.ProposeMemory(ctx, &knowledge.KnowledgeItem{
		ChatID: "chat_dm_1",
		UserID: "user_alice",
	}, &knowledge.MemoryVersion{
		Type:       knowledge.MemoryTypeFact,
		Content:    "Alice lives in Zurich",
		Confidence: 1.0,
	})
	require.NoError(t, err)
	assert.NotEmpty(t, item.ID)

	res, err = h.Handle(ctx, commands.Request{
		Session:    session,
		Actor:      actor,
		Command:    "memory",
		Subcommand: "list",
		Args:       []string{"pending"},
	})
	require.NoError(t, err)
	assert.Contains(t, res.Reply, item.ID)
	assert.Contains(t, res.Reply, "Alice lives in Zurich")

	_ = ver
}

func TestMemoryHandler_ShowApproveRejectForget(t *testing.T) {
	ctx := context.Background()
	store, _ := setupTestKnowledgeStore(t)
	allowAuth := commands.AuthorizerFunc(func(_ context.Context, _ agentapi.SessionRef, _ agentapi.Actor, _, _ string) error {
		return nil
	})
	h := commands.NewMemoryHandler(store, nil, allowAuth, "bot_1")

	session := agentapi.SessionRef{
		FrontendID: "besedka",
		SessionID:  "chat_dm_1",
		ScopeID:    "chat_dm_1",
	}
	actor := agentapi.Actor{ID: "user_alice"}

	// 1. Propose memory
	item, ver, err := store.ProposeMemory(ctx, &knowledge.KnowledgeItem{
		ChatID: "chat_dm_1",
		UserID: "user_alice",
	}, &knowledge.MemoryVersion{
		Type:       knowledge.MemoryTypeFact,
		Content:    "Alice prefers Go",
		Confidence: 0.9,
	})
	require.NoError(t, err)

	// 2. Show without args
	res, err := h.Handle(ctx, commands.Request{
		Session:    session,
		Actor:      actor,
		Command:    "memory",
		Subcommand: "show",
	})
	require.NoError(t, err)
	assert.Contains(t, res.Reply, "Please specify a memory ID")

	// 3. Show item
	res, err = h.Handle(ctx, commands.Request{
		Session:    session,
		Actor:      actor,
		Command:    "memory",
		Subcommand: "show",
		Args:       []string{item.ID},
	})
	require.NoError(t, err)
	assert.Contains(t, res.Reply, item.ID)
	assert.Contains(t, res.Reply, "Memory Item:")

	// 4. Show version
	res, err = h.Handle(ctx, commands.Request{
		Session:    session,
		Actor:      actor,
		Command:    "memory",
		Subcommand: "show",
		Args:       []string{ver.ID},
	})
	require.NoError(t, err)
	assert.Contains(t, res.Reply, ver.ID)
	assert.Contains(t, res.Reply, "Memory Version:")
	assert.Contains(t, res.Reply, "Alice prefers Go")

	// 5. Approve
	res, err = h.Handle(ctx, commands.Request{
		Session:    session,
		Actor:      actor,
		Command:    "memory",
		Subcommand: "approve",
		Args:       []string{ver.ID},
	})
	require.NoError(t, err)
	assert.Contains(t, res.Reply, "has been approved and activated")

	// 6. Propose revision 2 and reject
	_, ver2, err := store.ProposeMemory(ctx, &knowledge.KnowledgeItem{
		ID:     item.ID,
		ChatID: "chat_dm_1",
		UserID: "user_alice",
	}, &knowledge.MemoryVersion{
		Type:       knowledge.MemoryTypeFact,
		Content:    "Alice prefers Rust",
		Confidence: 0.5,
	})
	require.NoError(t, err)

	res, err = h.Handle(ctx, commands.Request{
		Session:    session,
		Actor:      actor,
		Command:    "memory",
		Subcommand: "reject",
		Args:       []string{ver2.ID},
	})
	require.NoError(t, err)
	assert.Contains(t, res.Reply, "has been rejected")

	// 7. Forget
	res, err = h.Handle(ctx, commands.Request{
		Session:    session,
		Actor:      actor,
		Command:    "memory",
		Subcommand: "forget",
		Args:       []string{item.ID},
	})
	require.NoError(t, err)
	assert.Contains(t, res.Reply, "has been forgotten and erased")
}
