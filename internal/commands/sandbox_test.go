package commands_test

import (
	"context"
	"testing"
	"time"

	"bob/internal/agentapi"
	"bob/internal/commands"
	"bob/internal/models"
	"bob/internal/sandbox"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type testDriver struct {
	driverType sandbox.DriverType
}

func (d *testDriver) Type() sandbox.DriverType {
	return d.driverType
}

func (d *testDriver) Available(_ context.Context) bool {
	return true
}

func (d *testDriver) Create(_ context.Context, _ *sandbox.UserSandbox, _ string) error {
	return nil
}

func (d *testDriver) Exec(_ context.Context, _ *sandbox.UserSandbox, _ []string, _ time.Duration) (*sandbox.ExecResult, error) {
	return &sandbox.ExecResult{ExitCode: 0, Stdout: "ok"}, nil
}

func (d *testDriver) Destroy(_ context.Context, _ *sandbox.UserSandbox) error {
	return nil
}

func setupTestSandboxManager(t *testing.T) *sandbox.Manager {
	t.Helper()
	cfg := sandbox.Config{
		DataDir:            t.TempDir(),
		Enabled:            true,
		Drivers:            []string{"bwrap"},
		MaxLifetime:        30 * time.Minute,
		DefaultExecTimeout: 1 * time.Minute,
		MaxExecTimeout:     10 * time.Minute,
	}
	driver := &testDriver{driverType: sandbox.DriverBwrap}
	mgr := sandbox.NewManager(cfg, []sandbox.Driver{driver})
	t.Cleanup(func() { _ = mgr.Close() })
	return mgr
}

func TestSandboxHandler_NilManagerAndHelp(t *testing.T) {
	ctx := context.Background()
	session := agentapi.SessionRef{
		FrontendID: "besedka",
		SessionID:  "chat_dm_1",
		ScopeID:    "chat_dm_1",
	}
	actor := agentapi.Actor{ID: "user_alice"}

	// 1. Nil manager
	hNil := commands.NewSandboxHandler(nil)
	res, err := hNil.Handle(ctx, commands.Request{
		Session: session,
		Actor:   actor,
	})
	require.NoError(t, err)
	assert.Contains(t, res.Reply, "Sandbox execution is disabled")

	// 2. Help
	mgr := setupTestSandboxManager(t)
	h := commands.NewSandboxHandler(mgr)
	res, err = h.Handle(ctx, commands.Request{
		Session: session,
		Actor:   actor,
	})
	require.NoError(t, err)
	assert.Contains(t, res.Reply, "Sandbox Commands:")
}

func TestSandboxHandler_StatusAndLifecycle(t *testing.T) {
	ctx := context.Background()
	mgr := setupTestSandboxManager(t)
	h := commands.NewSandboxHandler(mgr)

	session := agentapi.SessionRef{
		FrontendID: "besedka",
		SessionID:  "chat_dm_1",
		ScopeID:    "chat_dm_1",
	}
	actor := agentapi.Actor{ID: "user_alice"}

	// 1. Status with no sandbox
	res, err := h.Handle(ctx, commands.Request{
		Session:    session,
		Actor:      actor,
		Subcommand: "status",
	})
	require.NoError(t, err)
	assert.Contains(t, res.Reply, "do not have an active or pending sandbox")

	// 2. Request a sandbox
	_, err = mgr.RequestSandbox(ctx, "user_alice", "chat_dm_1", sandbox.RequestParams{
		Driver:      sandbox.DriverBwrap,
		NetworkMode: sandbox.NetworkNone,
		Reason:      "Run code",
	})
	require.NoError(t, err)

	// Status when pending
	res, err = h.Handle(ctx, commands.Request{
		Session:    session,
		Actor:      actor,
		Subcommand: "status",
	})
	require.NoError(t, err)
	assert.Contains(t, res.Reply, "Awaiting Your Approval")

	// 3. Approve sandbox -> continuation claimed
	res, err = h.Handle(ctx, commands.Request{
		Session:    session,
		Actor:      actor,
		Subcommand: "approve",
	})
	require.NoError(t, err)
	require.NotNil(t, res.Continuation)
	assert.Equal(t, "user_alice", res.Continuation.ActorID)
	assert.Equal(t, "Run code", res.Continuation.Reason)

	// Second approval returns already approved
	res, err = h.Handle(ctx, commands.Request{
		Session:    session,
		Actor:      actor,
		Subcommand: "approve",
	})
	require.NoError(t, err)
	assert.Nil(t, res.Continuation)
	assert.Contains(t, res.Reply, "already approved and running")

	// Status when running
	res, err = h.Handle(ctx, commands.Request{
		Session:    session,
		Actor:      actor,
		Subcommand: "status",
	})
	require.NoError(t, err)
	assert.Contains(t, res.Reply, "Active Sandbox Status")

	// 4. Destroy active sandbox
	res, err = h.Handle(ctx, commands.Request{
		Session:    session,
		Actor:      actor,
		Subcommand: "destroy",
	})
	require.NoError(t, err)
	assert.True(t, res.RecordAssistantEntry)
	assert.Contains(t, res.Reply, "Sandbox terminated")

	// 5. Request new sandbox and deny
	_, err = mgr.RequestSandbox(ctx, "user_alice", "chat_dm_1", sandbox.RequestParams{
		Driver:      sandbox.DriverBwrap,
		NetworkMode: sandbox.NetworkNone,
		Reason:      "Run another command",
	})
	require.NoError(t, err)

	res, err = h.Handle(ctx, commands.Request{
		Session:    session,
		Actor:      actor,
		Subcommand: "deny",
	})
	require.NoError(t, err)
	assert.True(t, res.RecordAssistantEntry)
	assert.Contains(t, res.Reply, "Sandbox request denied")
}

func TestSandboxHandler_StatusAttachesUI(t *testing.T) {
	ctx := context.Background()
	mgr := setupTestSandboxManager(t)
	h := commands.NewSandboxHandler(mgr)

	session := agentapi.SessionRef{FrontendID: "besedka", SessionID: "chat_dm_1", ScopeID: "chat_dm_1"}
	actor := agentapi.Actor{ID: "user_alice"}

	// No sandbox -> UI is nil
	res, err := h.Handle(ctx, commands.Request{Session: session, Actor: actor, Subcommand: "status"})
	require.NoError(t, err)
	assert.Nil(t, res.UI)

	// Pending approval -> Approve/Deny UI bound to sandbox ID
	sbxPending, err := mgr.RequestSandbox(ctx, "user_alice", "chat_dm_1", sandbox.RequestParams{
		Driver:      sandbox.DriverBwrap,
		NetworkMode: sandbox.NetworkNone,
		Reason:      "Pending test",
	})
	require.NoError(t, err)

	res, err = h.Handle(ctx, commands.Request{Session: session, Actor: actor, Subcommand: "status"})
	require.NoError(t, err)
	require.NotNil(t, res.UI)
	require.Len(t, res.UI.Elements, 2)
	assert.Equal(t, "bob:sandbox:approve:"+sbxPending.ID, res.UI.Elements[0].ID)
	assert.Equal(t, "bob:sandbox:deny:"+sbxPending.ID, res.UI.Elements[1].ID)

	// Stale TargetID approve is rejected and leaves pending sandbox untouched
	staleRes, err := h.Handle(ctx, commands.Request{
		Session:    session,
		Actor:      actor,
		Subcommand: "approve",
		TargetID:   "sbx-stale",
	})
	require.NoError(t, err)
	assert.Contains(t, staleRes.Reply, "outdated")
	assert.Nil(t, staleRes.Continuation)

	// Running -> Destroy-only UI bound to sandbox ID
	approveRes, err := h.Handle(ctx, commands.Request{
		Session:    session,
		Actor:      actor,
		Subcommand: "approve",
		TargetID:   sbxPending.ID,
	})
	require.NoError(t, err)
	require.NotNil(t, approveRes.Continuation)

	res, err = h.Handle(ctx, commands.Request{Session: session, Actor: actor, Subcommand: "status"})
	require.NoError(t, err)
	require.NotNil(t, res.UI)
	require.Len(t, res.UI.Elements, 1)
	assert.Equal(t, models.UIElementTypeButton, res.UI.Elements[0].Type)
	assert.Equal(t, "bob:sandbox:destroy:"+sbxPending.ID, res.UI.Elements[0].ID)
}
