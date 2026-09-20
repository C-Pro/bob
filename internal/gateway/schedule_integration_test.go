package gateway

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"bob/internal/config"
	"bob/internal/fsm"
	"bob/internal/models"
	"bob/internal/sandbox"
	"bob/internal/scheduler"
	"bob/internal/tools"

	"github.com/fasthttp/websocket"
	openai "github.com/sashabaranov/go-openai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	_ "modernc.org/sqlite"
)

func TestSchedule_EndToEndLifecycle(t *testing.T) {
	upgrader := websocket.Upgrader{}
	receivedMsgs := make(chan models.ClientMessage, 50)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/me" {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(models.User{ID: "bot-id", UserName: "bot", DisplayName: "Bob"})
			return
		}
		if r.URL.Path == "/api/users" {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode([]models.User{
				{ID: "bot-id", UserName: "bot", DisplayName: "Bob"},
				{ID: "user1", UserName: "user1", DisplayName: "Alice"},
			})
			return
		}
		if r.URL.Path == "/api/chats" {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode([]models.Chat{
				{ID: "dm_user1", Name: "Alice"},
			})
			return
		}
		if strings.HasPrefix(r.URL.Path, "/api/chats/") && strings.HasSuffix(r.URL.Path, "/messages") {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode([]models.Message{})
			return
		}
		if r.URL.Path == "/api/chat" {
			conn, err := upgrader.Upgrade(w, r, nil)
			if err != nil {
				return
			}
			defer func() { _ = conn.Close() }()
			for {
				var msg models.ClientMessage
				if err := conn.ReadJSON(&msg); err != nil {
					return
				}
				receivedMsgs <- msg
			}
		}
		http.NotFound(w, r)
	}))
	defer server.Close()

	dataDir := t.TempDir()
	cfg := &config.Config{
		BesedkaURL:                 server.URL,
		BesedkaAPIKey:              "test-token",
		BotHandle:                  "@bot",
		TownhallMaxParagraphs:      3,
		DMMaxParagraphs:            5,
		MsgRingBufferSize:          20,
		DataDir:                    dataDir,
		SandboxEnabled:             true,
		SandboxDrivers:             []string{"bwrap"},
		SandboxAllowedImages:       []string{"alpine:latest"},
		SandboxAllowedNetworkModes: []string{"none", "restricted"},
		SandboxCPULimit:            1.0,
		SandboxMemoryLimitMB:       256,
		SandboxMaxLifetime:         30 * time.Minute,
		SchedulerMinRunTimeout:     time.Minute,
		SchedulerMaxRunTimeout:     time.Hour,
		SchedulerMinMaxTurns:       10,
		SchedulerMaxMaxTurns:       100,
	}

	mockDriver := &mockGatewaySandboxDriver{}
	sandboxMgr := sandbox.NewManager(cfg.SandboxConfig(), []sandbox.Driver{mockDriver})

	gw := NewGateway(cfg, nil)
	gw.SetSandboxManager(sandboxMgr)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	err := gw.DialWebSocket(ctx)
	require.NoError(t, err)

	recv := func() models.ClientMessage {
		select {
		case msg := <-receivedMsgs:
			return msg
		case <-time.After(2 * time.Second):
			t.Fatal("timed out waiting for websocket message")
			return models.ClientMessage{}
		}
	}

	// 1. Tool execution: schedule_task with elevated sandbox permissions
	storeProv := gw.StoreProvider()
	require.NotNil(t, storeProv)

	rawPermissions := `{"sandbox":{"driver":"bwrap","network":"restricted","domains":["api.backup.org"]}}`
	scheduleArgs := fmt.Sprintf(`{
		"schedule_name": "db_backup",
		"type": "interval",
		"schedule_spec": "600",
		"instruction": "Dump SQLite database and post sha256 checksum",
		"run_timeout_seconds": 300,
		"max_turns": 25,
		"permissions": %s
	}`, rawPermissions)

	sessionCtx := tools.ChatSessionContext{
		ChatID: "dm_user1",
		UserID: "user1",
		IsDM:   true,
	}
	sessionRunCtx := tools.WithChatSession(ctx, sessionCtx)

	toolOutput, err := gw.ToolsRegistry().Execute(sessionRunCtx, "schedule_task", scheduleArgs)
	require.NoError(t, err)
	assert.Contains(t, toolOutput, "awaiting user approval")
	assert.Contains(t, toolOutput, "/schedule approve db_backup")

	// Verify schedule stored as PENDING_APPROVAL
	st, err := storeProv.GetSchedulerStore(ctx, "dm_user1", true)
	require.NoError(t, err)

	sched, err := st.GetScheduleByName(ctx, "dm_user1", "db_backup")
	require.NoError(t, err)
	assert.Equal(t, scheduler.ScheduleStatusPendingApproval, sched.Status)

	grant, err := st.GetGrant(ctx, sched.ID)
	require.NoError(t, err)
	assert.True(t, grant.Verify())

	// 2. User approves via /schedule approve db_backup
	err = gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_user1",
		UserID:    "user1",
		Content:   "/schedule approve db_backup",
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)

	approvalReply := recv()
	assert.Contains(t, strings.ToLower(approvalReply.Content), "approved")
	assert.Contains(t, approvalReply.Content, "db_backup")

	sched, err = st.GetSchedule(ctx, sched.ID)
	require.NoError(t, err)
	assert.Equal(t, scheduler.ScheduleStatusActive, sched.Status)

	// 3. Setup mock FSM to verify execution and tool exposure
	var fsmExecuted atomic.Bool
	var toolsMu sync.Mutex
	var toolsExposed []openai.Tool
	var sandboxRunningDuringFSM atomic.Bool

	mockFSM := FSMRunnerFunc(func(execCtx context.Context, req fsm.ToolLoopRequest) (*fsm.ToolLoopResult, error) {
		fsmExecuted.Store(true)
		toolsMu.Lock()
		toolsExposed = req.Tools
		toolsMu.Unlock()

		if sbx, ok := sandboxMgr.GetStatus("user1"); ok && sbx.Status == sandbox.StatusRunning {
			sandboxRunningDuringFSM.Store(true)
		}

		return &fsm.ToolLoopResult{
			Content: "Database dumped. Checksum: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
		}, nil
	})

	invoker := gw.SchedulerInvoker()
	require.NotNil(t, invoker)
	invoker.SetFSM(mockFSM)

	// Update schedule next_run_at to past so engine claims it immediately
	past := time.Now().Unix() - 10
	_, err = st.DB().ExecContext(ctx, "UPDATE schedules SET next_run_at = ? WHERE id = ?", past, sched.ID)
	require.NoError(t, err)

	// 4. Trigger single poll cycle on scheduler engine
	gw.SchedulerEngine().PollOnce(ctx)

	// Wait briefly for asynchronous execution worker to complete
	require.Eventually(t, func() bool {
		return fsmExecuted.Load()
	}, 3*time.Second, 50*time.Millisecond)

	assert.True(t, sandboxRunningDuringFSM.Load(), "sandbox should be actively running during scheduled execution")

	// Verify result message posted to Besedka chat
	resultMsg := recv()
	assert.Contains(t, resultMsg.Content, "⏱️ **Scheduled Task [db_backup] completed:**")
	assert.Contains(t, resultMsg.Content, "Checksum: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")

	// Verify ephemeral sandbox was destroyed immediately upon completion
	sbx, found := sandboxMgr.GetStatus("user1")
	assert.True(t, !found || sbx.Status != sandbox.StatusRunning, "ephemeral sandbox must be destroyed after scheduled run")

	// Verify tools exposed to scheduled run: sandbox_exec present, sandbox_request hidden
	toolsMu.Lock()
	exposed := append([]openai.Tool(nil), toolsExposed...)
	toolsMu.Unlock()

	hasExec := false
	for _, tool := range exposed {
		if tool.Function.Name == "sandbox_exec" {
			hasExec = true
		}
		assert.NotEqual(t, "sandbox_request", tool.Function.Name)
	}
	assert.True(t, hasExec, "sandbox_exec should be exposed to pre-approved run")

	// Verify schedule run count and next_run_at updated
	updatedSched, err := st.GetSchedule(ctx, sched.ID)
	require.NoError(t, err)
	assert.Equal(t, 1, updatedSched.RunCount)
	assert.Equal(t, "SUCCESS", updatedSched.LastStatus)
	assert.True(t, updatedSched.NextRunAt > time.Now().Unix())

	// 5. User cancels schedule via /schedule cancel db_backup
	err = gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_user1",
		UserID:    "user1",
		Content:   "/schedule cancel db_backup",
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)

	cancelReply := recv()
	assert.Contains(t, strings.ToLower(cancelReply.Content), "cancelled")

	cancelledSched, err := st.GetSchedule(ctx, sched.ID)
	require.NoError(t, err)
	assert.Equal(t, scheduler.ScheduleStatusCancelled, cancelledSched.Status)

	// Verify grant was cleaned up
	_, err = st.GetGrant(ctx, sched.ID)
	assert.ErrorIs(t, err, scheduler.ErrGrantNotFound)
}

func TestSchedule_NoSandbox_RequiresApproval(t *testing.T) {
	upgrader := websocket.Upgrader{}
	receivedMsgs := make(chan models.ClientMessage, 50)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/me" {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(models.User{ID: "bot-id", UserName: "bot", DisplayName: "Bob"})
			return
		}
		if r.URL.Path == "/api/users" {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode([]models.User{
				{ID: "bot-id", UserName: "bot", DisplayName: "Bob"},
				{ID: "user_bob", UserName: "user_bob", DisplayName: "Bob Tester"},
			})
			return
		}
		if r.URL.Path == "/api/chats" {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode([]models.Chat{
				{ID: "dm_user_bob", Name: "Bob Tester"},
			})
			return
		}
		if strings.HasPrefix(r.URL.Path, "/api/chats/") && strings.HasSuffix(r.URL.Path, "/messages") {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode([]models.Message{})
			return
		}
		if r.URL.Path == "/api/chat" {
			conn, err := upgrader.Upgrade(w, r, nil)
			if err != nil {
				return
			}
			defer func() { _ = conn.Close() }()
			for {
				var msg models.ClientMessage
				if err := conn.ReadJSON(&msg); err != nil {
					return
				}
				receivedMsgs <- msg
			}
		}
		http.NotFound(w, r)
	}))
	defer server.Close()

	dataDir := t.TempDir()
	cfg := &config.Config{
		BesedkaURL:             server.URL,
		BesedkaAPIKey:          "test-token",
		BotHandle:              "@bot",
		TownhallMaxParagraphs: 3,
		DMMaxParagraphs:       5,
		MsgRingBufferSize:     20,
		DataDir:               dataDir,
		SchedulerMinRunTimeout: time.Minute,
		SchedulerMaxRunTimeout: time.Hour,
		SchedulerMinMaxTurns:   10,
		SchedulerMaxMaxTurns:   100,
	}

	gw := NewGateway(cfg, nil)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	err := gw.DialWebSocket(ctx)
	require.NoError(t, err)

	recv := func() models.ClientMessage {
		select {
		case msg := <-receivedMsgs:
			return msg
		case <-time.After(2 * time.Second):
			t.Fatal("timed out waiting for websocket message")
			return models.ClientMessage{}
		}
	}

	// 1. Agent schedules task w/o sandbox permissions
	scheduleArgs := `{
		"schedule_name": "daily_brief",
		"type": "interval",
		"schedule_spec": "600",
		"instruction": "Fetch daily tech headlines and summarize",
		"run_timeout_seconds": 180,
		"max_turns": 15
	}`

	sessionCtx := tools.ChatSessionContext{
		ChatID: "dm_user_bob",
		UserID: "user_bob",
		IsDM:   true,
	}
	sessionRunCtx := tools.WithChatSession(ctx, sessionCtx)

	toolOutput, err := gw.ToolsRegistry().Execute(sessionRunCtx, "schedule_task", scheduleArgs)
	require.NoError(t, err)
	assert.Contains(t, toolOutput, "PENDING_APPROVAL")
	assert.Contains(t, toolOutput, "/schedule approve daily_brief")

	// Verify in store as PENDING_APPROVAL and grant is nil
	storeProv := gw.StoreProvider()
	require.NotNil(t, storeProv)
	st, err := storeProv.GetSchedulerStore(ctx, "dm_user_bob", true)
	require.NoError(t, err)

	sched, err := st.GetScheduleByName(ctx, "dm_user_bob", "daily_brief")
	require.NoError(t, err)
	assert.Equal(t, scheduler.ScheduleStatusPendingApproval, sched.Status)

	_, err = st.GetGrant(ctx, sched.ID)
	assert.ErrorIs(t, err, scheduler.ErrGrantNotFound)

	// Mock FSM runner
	var fsmExecuted atomic.Bool
	mockFSM := FSMRunnerFunc(func(execCtx context.Context, req fsm.ToolLoopRequest) (*fsm.ToolLoopResult, error) {
		fsmExecuted.Store(true)
		return &fsm.ToolLoopResult{
			Content: "Daily brief completed without sandbox.",
		}, nil
	})

	invoker := gw.SchedulerInvoker()
	require.NotNil(t, invoker)
	invoker.SetFSM(mockFSM)

	// 2. While PENDING_APPROVAL, engine should NOT execute it even if time is in past
	past := time.Now().Unix() - 10
	_, err = st.DB().ExecContext(ctx, "UPDATE schedules SET next_run_at = ? WHERE id = ?", past, sched.ID)
	require.NoError(t, err)

	gw.SchedulerEngine().PollOnce(ctx)
	time.Sleep(100 * time.Millisecond)
	assert.False(t, fsmExecuted.Load(), "schedule in PENDING_APPROVAL must not be executed")

	// 3. User approves schedule via /schedule approve daily_brief
	err = gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_user_bob",
		UserID:    "user_bob",
		Content:   "/schedule approve daily_brief",
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)

	approvalReply := recv()
	assert.Contains(t, strings.ToLower(approvalReply.Content), "approved")
	assert.Contains(t, approvalReply.Content, "daily_brief")

	sched, err = st.GetSchedule(ctx, sched.ID)
	require.NoError(t, err)
	assert.Equal(t, scheduler.ScheduleStatusActive, sched.Status)

	// 4. Now fast-forward next_run_at and verify execution
	_, err = st.DB().ExecContext(ctx, "UPDATE schedules SET next_run_at = ? WHERE id = ?", past, sched.ID)
	require.NoError(t, err)

	gw.SchedulerEngine().PollOnce(ctx)

	require.Eventually(t, func() bool {
		return fsmExecuted.Load()
	}, 3*time.Second, 50*time.Millisecond)

	resultMsg := recv()
	assert.Contains(t, resultMsg.Content, "⏱️ **Scheduled Task [daily_brief] completed:**")
	assert.Contains(t, resultMsg.Content, "Daily brief completed without sandbox.")

	updatedSched, err := st.GetSchedule(ctx, sched.ID)
	require.NoError(t, err)
	assert.Equal(t, 1, updatedSched.RunCount)
	assert.Equal(t, "SUCCESS", updatedSched.LastStatus)
}
