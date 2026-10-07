package commands_test

import (
	"context"
	"errors"
	"testing"

	"bob/internal/agentapi"
	"bob/internal/commands"
	"bob/internal/knowledge"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSkillHandler_Authorization(t *testing.T) {
	ctx := context.Background()
	store, _ := setupTestKnowledgeStore(t)

	session := agentapi.SessionRef{
		FrontendID: "besedka",
		SessionID:  "chat_dm_1",
		ScopeID:    "chat_dm_1",
	}
	actor := agentapi.Actor{ID: "user_alice"}

	// 1. Nil authorizer fails closed
	hNil := commands.NewSkillHandler(store, nil, nil, "bot_1")
	res, err := hNil.Handle(ctx, commands.Request{
		Session:    session,
		Actor:      actor,
		Command:    "skill",
		Subcommand: "",
		Args:       nil,
	})
	require.NoError(t, err)
	assert.Contains(t, res.Reply, "Authorization failed: authorizer not configured")

	// 2. Authorizer rejects
	hDenied := commands.NewSkillHandler(store, nil, commands.AuthorizerFunc(func(_ context.Context, _ agentapi.SessionRef, _ agentapi.Actor, _, _ string) error {
		return errors.New("only DM owner allowed")
	}), "bot_1")
	res, err = hDenied.Handle(ctx, commands.Request{
		Session:    session,
		Actor:      actor,
		Command:    "skill",
		Subcommand: "list",
	})
	require.NoError(t, err)
	assert.Contains(t, res.Reply, "only DM owner allowed")

	// 3. Bot self-invocation is ignored
	allowAuth := commands.AuthorizerFunc(func(_ context.Context, _ agentapi.SessionRef, _ agentapi.Actor, _, _ string) error {
		return nil
	})
	hBot := commands.NewSkillHandler(store, nil, allowAuth, "bot_1")
	res, err = hBot.Handle(ctx, commands.Request{
		Session:    session,
		Actor:      agentapi.Actor{ID: "bot_1"},
		Command:    "skill",
		Subcommand: "list",
	})
	require.NoError(t, err)
	assert.True(t, res.IsIgnored)

	// 4. Store nil
	hNoStore := commands.NewSkillHandler(nil, nil, allowAuth, "bot_1")
	res, err = hNoStore.Handle(ctx, commands.Request{
		Session:    session,
		Actor:      actor,
		Command:    "skill",
		Subcommand: "list",
	})
	require.NoError(t, err)
	assert.Contains(t, res.Reply, "Knowledge storage is unavailable")
}

func TestSkillHandler_ListAndHelp(t *testing.T) {
	ctx := context.Background()
	store, _ := setupTestKnowledgeStore(t)
	allowAuth := commands.AuthorizerFunc(func(_ context.Context, _ agentapi.SessionRef, _ agentapi.Actor, _, _ string) error {
		return nil
	})
	h := commands.NewSkillHandler(store, nil, allowAuth, "bot_1")

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
		Command:    "skill",
		Subcommand: "",
	})
	require.NoError(t, err)
	assert.Contains(t, res.Reply, "Skill Commands:")

	// List when empty
	res, err = h.Handle(ctx, commands.Request{
		Session:    session,
		Actor:      actor,
		Command:    "skill",
		Subcommand: "list",
	})
	require.NoError(t, err)
	assert.Contains(t, res.Reply, "No skills found")

	// Invalid filter
	res, err = h.Handle(ctx, commands.Request{
		Session:    session,
		Actor:      actor,
		Command:    "skill",
		Subcommand: "list",
		Args:       []string{"invalid"},
	})
	require.NoError(t, err)
	assert.Contains(t, res.Reply, "Invalid status filter")

	// Propose skill
	item, ver, err := store.ProposeSkill(ctx, &knowledge.KnowledgeItem{
		ChatID: "chat_dm_1",
		UserID: "user_alice",
	}, &knowledge.SkillVersion{
		Name:                 "go-test",
		Description:          "Runs tests",
		InstructionsMarkdown: "Use `go test`",
	})
	require.NoError(t, err)
	assert.NotEmpty(t, item.ID)
	_ = ver

	res, err = h.Handle(ctx, commands.Request{
		Session:    session,
		Actor:      actor,
		Command:    "skill",
		Subcommand: "list",
		Args:       []string{"pending"},
	})
	require.NoError(t, err)
	assert.Contains(t, res.Reply, item.ID)
	assert.Contains(t, res.Reply, "go-test")
}

func TestSkillHandler_ShowApproveReject(t *testing.T) {
	ctx := context.Background()
	store, _ := setupTestKnowledgeStore(t)
	allowAuth := commands.AuthorizerFunc(func(_ context.Context, _ agentapi.SessionRef, _ agentapi.Actor, _, _ string) error {
		return nil
	})
	h := commands.NewSkillHandler(store, nil, allowAuth, "bot_1")

	session := agentapi.SessionRef{
		FrontendID: "besedka",
		SessionID:  "chat_dm_1",
		ScopeID:    "chat_dm_1",
	}
	actor := agentapi.Actor{ID: "user_alice"}

	// Propose skill
	item, ver, err := store.ProposeSkill(ctx, &knowledge.KnowledgeItem{
		ChatID: "chat_dm_1",
		UserID: "user_alice",
	}, &knowledge.SkillVersion{
		Name:                 "deploy-app",
		Description:          "Deploys application",
		InstructionsMarkdown: "Run deployment script",
	})
	require.NoError(t, err)

	// Show item
	res, err := h.Handle(ctx, commands.Request{
		Session:    session,
		Actor:      actor,
		Command:    "skill",
		Subcommand: "show",
		Args:       []string{item.ID},
	})
	require.NoError(t, err)
	assert.Contains(t, res.Reply, item.ID)
	assert.Contains(t, res.Reply, "Skill Item:")

	// Show version
	res, err = h.Handle(ctx, commands.Request{
		Session:    session,
		Actor:      actor,
		Command:    "skill",
		Subcommand: "show",
		Args:       []string{ver.ID},
	})
	require.NoError(t, err)
	assert.Contains(t, res.Reply, ver.ID)
	assert.Contains(t, res.Reply, "Skill Version:")

	// Approve
	res, err = h.Handle(ctx, commands.Request{
		Session:    session,
		Actor:      actor,
		Command:    "skill",
		Subcommand: "approve",
		Args:       []string{ver.ID},
	})
	require.NoError(t, err)
	assert.Contains(t, res.Reply, "has been approved and activated")

	// Propose revision 2 and reject
	_, ver2, err := store.ProposeSkill(ctx, &knowledge.KnowledgeItem{
		ID:     item.ID,
		ChatID: "chat_dm_1",
		UserID: "user_alice",
	}, &knowledge.SkillVersion{
		Name:                 "deploy-app-v2",
		Description:          "Bad version",
		InstructionsMarkdown: "Wrong script",
	})
	require.NoError(t, err)

	res, err = h.Handle(ctx, commands.Request{
		Session:    session,
		Actor:      actor,
		Command:    "skill",
		Subcommand: "reject",
		Args:       []string{ver2.ID},
	})
	require.NoError(t, err)
	assert.Contains(t, res.Reply, "has been rejected")
}

func TestSkillHandler_Lifecycle(t *testing.T) {
	ctx := context.Background()
	store, _ := setupTestKnowledgeStore(t)
	allowAuth := commands.AuthorizerFunc(func(_ context.Context, _ agentapi.SessionRef, _ agentapi.Actor, _, _ string) error {
		return nil
	})
	h := commands.NewSkillHandler(store, nil, allowAuth, "bot_1")

	session := agentapi.SessionRef{
		FrontendID: "besedka",
		SessionID:  "chat_dm_1",
		ScopeID:    "chat_dm_1",
	}
	actor := agentapi.Actor{ID: "user_alice"}

	item, ver, err := store.ProposeSkill(ctx, &knowledge.KnowledgeItem{
		ChatID: "chat_dm_1",
		UserID: "user_alice",
	}, &knowledge.SkillVersion{
		Name:                 "build-tool",
		Description:          "Build tool",
		InstructionsMarkdown: "Execute build",
	})
	require.NoError(t, err)

	err = store.ApproveVersion(ctx, ver.ID, "user_alice")
	require.NoError(t, err)

	// Disable
	res, err := h.Handle(ctx, commands.Request{
		Session:    session,
		Actor:      actor,
		Command:    "skill",
		Subcommand: "disable",
		Args:       []string{item.ID},
	})
	require.NoError(t, err)
	assert.Contains(t, res.Reply, "has been disabled")

	// Enable
	res, err = h.Handle(ctx, commands.Request{
		Session:    session,
		Actor:      actor,
		Command:    "skill",
		Subcommand: "enable",
		Args:       []string{item.ID},
	})
	require.NoError(t, err)
	assert.Contains(t, res.Reply, "has been enabled")

	// Archive
	res, err = h.Handle(ctx, commands.Request{
		Session:    session,
		Actor:      actor,
		Command:    "skill",
		Subcommand: "archive",
		Args:       []string{item.ID},
	})
	require.NoError(t, err)
	assert.Contains(t, res.Reply, "has been archived")

	// Delete
	res, err = h.Handle(ctx, commands.Request{
		Session:    session,
		Actor:      actor,
		Command:    "skill",
		Subcommand: "delete",
		Args:       []string{item.ID},
	})
	require.NoError(t, err)
	assert.Contains(t, res.Reply, "has been permanently deleted")
}
