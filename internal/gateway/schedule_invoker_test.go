package gateway

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"bob/internal/chatcontext"
	"bob/internal/config"
	"bob/internal/fsm"
	"bob/internal/models"
	"bob/internal/sandbox"
	"bob/internal/scheduler"
	"bob/internal/tools"

	openai "github.com/sashabaranov/go-openai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	_ "modernc.org/sqlite"
)

func setupInvokerTestStore(t *testing.T) (*scheduler.Store, func()) {
	t.Helper()
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "invoker_test.db")
	db, err := sql.Open("sqlite", dbPath)
	require.NoError(t, err)

	_, err = db.Exec("PRAGMA foreign_keys = ON")
	require.NoError(t, err)

	err = scheduler.EnsureScheduleSchema(context.Background(), db)
	require.NoError(t, err)

	st := scheduler.NewStore(db)
	return st, func() {
		_ = db.Close()
	}
}

func TestChatLocker_TryAcquire_Success(t *testing.T) {
	locker := NewChatLocker()
	ctx := context.Background()

	release1, err := locker.TryAcquire(ctx, "chat1", 100*time.Millisecond)
	require.NoError(t, err)
	require.NotNil(t, release1)

	// Distinct chat should acquire immediately without blocking
	release2, err := locker.TryAcquire(ctx, "chat2", 100*time.Millisecond)
	require.NoError(t, err)
	require.NotNil(t, release2)

	release2()
	release1()

	// Re-acquire on chat1 should now succeed
	release3, err := locker.TryAcquire(ctx, "chat1", 100*time.Millisecond)
	require.NoError(t, err)
	release3()
}

func TestChatLocker_TryAcquire_Timeout(t *testing.T) {
	locker := NewChatLocker()
	ctx := context.Background()

	release, err := locker.TryAcquire(ctx, "chat1", 100*time.Millisecond)
	require.NoError(t, err)

	// Attempting to acquire on same chat with small timeout should fail with ErrChatBusy
	_, err = locker.TryAcquire(ctx, "chat1", 30*time.Millisecond)
	require.Error(t, err)
	assert.True(t, errors.Is(err, scheduler.ErrChatBusy))

	// Releasing allows subsequent acquire
	release()

	release2, err := locker.TryAcquire(ctx, "chat1", 100*time.Millisecond)
	require.NoError(t, err)
	release2()
}

func TestScheduleInvoker_NilSchedule(t *testing.T) {
	invoker := NewScheduleInvoker(InvokerConfig{})
	err := invoker.ExecuteSchedule(context.Background(), nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "schedule cannot be nil")
}

func TestScheduleInvoker_Execute_NoGrant(t *testing.T) {
	ctx := context.Background()
	ctxMgr := chatcontext.NewManager(10)

	var sentMsg string
	var sentChatID string
	sender := MessageSenderFunc(func(chatID, content string) error {
		sentChatID = chatID
		sentMsg = content
		return nil
	})

	var fsmCalled bool
	var capturedReq fsm.ToolLoopRequest
	mockFSM := FSMRunnerFunc(func(ctx context.Context, req fsm.ToolLoopRequest) (*fsm.ToolLoopResult, error) {
		fsmCalled = true
		capturedReq = req
		return &fsm.ToolLoopResult{
			Content: "Weather report: 22C, sunny.",
		}, nil
	})

	toolsRegistry := tools.NewRegistry(nil, nil, nil)

	invoker := NewScheduleInvoker(InvokerConfig{
		FSM:        mockFSM,
		Tools:      toolsRegistry,
		Sender:     sender,
		ContextMgr: ctxMgr,
		Model:      "test-model",
		BotID:      "bot123",
		BotName:    "BobTheBot",
	})

	sched := &scheduler.Schedule{
		ID:          "sched_1",
		ChatID:      "dm_user1",
		UserID:      "user1",
		Name:        "morning_weather",
		Instruction: "Check current weather and report.",
		MaxTurns:    12,
	}

	err := invoker.ExecuteSchedule(ctx, sched)
	require.NoError(t, err)

	assert.True(t, fsmCalled)
	assert.Equal(t, "dm_user1", capturedReq.ChatID)
	assert.Equal(t, "user1", capturedReq.UserID)
	assert.True(t, capturedReq.IsDM)
	assert.Equal(t, "test-model", capturedReq.Model)
	assert.Equal(t, 12, capturedReq.MaxIterations)
	require.Len(t, capturedReq.Messages, 2)
	assert.Contains(t, capturedReq.Messages[0].Content, "morning_weather")
	assert.Contains(t, capturedReq.Messages[0].Content, "Check current weather and report.")

	// Verify no sandbox tools were exposed since no sandbox grant exists
	for _, tool := range capturedReq.Tools {
		assert.False(t, strings.HasPrefix(tool.Function.Name, "sandbox_"), "tool %s should not be exposed without sandbox grant", tool.Function.Name)
	}

	assert.Equal(t, "dm_user1", sentChatID)
	assert.Contains(t, sentMsg, "⏱️ **Scheduled Task [morning_weather] completed:**")
	assert.Contains(t, sentMsg, "Weather report: 22C, sunny.")

	rb := ctxMgr.GetOrCreate("dm_user1")
	entries := rb.Entries()
	require.Len(t, entries, 1)
	assert.Equal(t, "assistant", entries[0].Role)
	assert.Equal(t, "bot123", entries[0].SenderID)
	assert.Equal(t, "BobTheBot", entries[0].SenderName)
	assert.Equal(t, sentMsg, entries[0].Content)
}

func TestScheduleInvoker_Execute_WithPreApprovedSandbox(t *testing.T) {
	ctx := context.Background()
	st, cleanup := setupInvokerTestStore(t)
	defer cleanup()

	cfg := &config.Config{
		SandboxEnabled:       true,
		SandboxDrivers:       []string{"bwrap"},
		SandboxCPULimit:      1.0,
		SandboxMemoryLimitMB: 256,
		SandboxMaxLifetime:   30 * time.Minute,
		DataDir:              t.TempDir(),
	}
	mockDriver := &mockGatewaySandboxDriver{}
	sandboxMgr := sandbox.NewManager(cfg.SandboxConfig(), []sandbox.Driver{mockDriver})

	rawGrantJSON := `{"sandbox":{"driver":"bwrap","network":"restricted","domains":["api.example.com"],"mounts":[{"path":"data","read_only":true}]}}`
	paramsHash := scheduler.ComputeRawJSONHash(rawGrantJSON)

	now := time.Now().Unix()
	sched := &scheduler.Schedule{
		ID:                "sched_sbx",
		ChatID:            "dm_user1",
		UserID:            "user1",
		Name:              "data_backup",
		Instruction:       "Run backup script in sandbox",
		ScheduleType:      scheduler.ScheduleTypeInterval,
		IntervalSeconds:   600,
		NextRunAt:         now,
		Status:            scheduler.ScheduleStatusActive,
		RunTimeoutSeconds: 60,
		MaxTurns:          20,
		CreatedAt:         now,
		UpdatedAt:         now,
	}

	grant := &scheduler.ScheduleGrant{
		ScheduleID:            sched.ID,
		PermissionRequestJSON: rawGrantJSON,
		ParamsHash:            paramsHash,
		GrantedBy:             "user1",
		GrantedAt:             now,
		ValidUntil:            now + 86400,
	}
	err := st.CreateSchedule(ctx, sched, grant)
	require.NoError(t, err)

	toolsRegistry := tools.NewRegistry(nil, nil, sandboxMgr)

	var fsmExecuted bool
	var activeSandboxDuringFSM bool
	var toolsExposed []openai.Tool
	mockFSM := FSMRunnerFunc(func(ctx context.Context, req fsm.ToolLoopRequest) (*fsm.ToolLoopResult, error) {
		fsmExecuted = true
		toolsExposed = req.Tools
		// Check that sandbox is active during FSM execution
		if sbx, ok := sandboxMgr.GetStatus("user1"); ok && sbx.Status == sandbox.StatusRunning {
			activeSandboxDuringFSM = true
		}
		return &fsm.ToolLoopResult{
			Content: "Backup finished successfully.",
		}, nil
	})

	invoker := NewScheduleInvoker(InvokerConfig{
		FSM:        mockFSM,
		Tools:      toolsRegistry,
		Sandbox:    sandboxMgr,
		Model:      "test-model",
		SchedulerStoreProv: func(ctx context.Context, chatID string, isDM bool) (*scheduler.Store, error) {
			return st, nil
		},
	})

	err = invoker.ExecuteSchedule(ctx, sched)
	require.NoError(t, err)
	assert.True(t, fsmExecuted)
	assert.True(t, activeSandboxDuringFSM, "sandbox should be actively running during FSM execution")

	// Verify sandbox is destroyed after execution
	sbx, found := sandboxMgr.GetStatus("user1")
	assert.True(t, !found || sbx.Status != sandbox.StatusRunning, "sandbox should be destroyed after schedule execution")

	// Verify tool definitions: sandbox_exec is present, sandbox_request and sandbox_destroy are hidden
	hasExec := false
	for _, tool := range toolsExposed {
		name := tool.Function.Name
		if name == "sandbox_exec" {
			hasExec = true
		}
		assert.NotEqual(t, "sandbox_request", name, "sandbox_request should be hidden for scheduled pre-approved runs")
		assert.NotEqual(t, "sandbox_destroy", name, "sandbox_destroy should be hidden to let invoker manage lifecycle")
		assert.NotEqual(t, "schedule_task", name, "schedule_task must be excluded from scheduled runs")
		assert.NotEqual(t, "cancel_schedule", name, "cancel_schedule must be excluded from scheduled runs")
	}
	assert.True(t, hasExec, "sandbox_exec must be exposed to scheduled run")
}

func TestScheduleInvoker_Execute_TamperedGrant(t *testing.T) {
	ctx := context.Background()
	st, cleanup := setupInvokerTestStore(t)
	defer cleanup()

	now := time.Now().Unix()
	sched := &scheduler.Schedule{
		ID:           "sched_tampered",
		ChatID:       "dm_user1",
		UserID:       "user1",
		Name:         "tampered_task",
		Instruction:  "Do something",
		Status:       scheduler.ScheduleStatusActive,
		NextRunAt:    now,
		CreatedAt:    now,
		UpdatedAt:    now,
	}

	// Tampered grant: params_hash does not match JSON
	grant := &scheduler.ScheduleGrant{
		ScheduleID:            sched.ID,
		PermissionRequestJSON: `{"sandbox":{"driver":"docker"}}`,
		ParamsHash:            "bad_hash_1234567890abcdef",
		GrantedBy:             "user1",
		GrantedAt:             now,
		ValidUntil:            now + 3600,
	}
	err := st.CreateSchedule(ctx, sched, grant)
	require.NoError(t, err)

	invoker := NewScheduleInvoker(InvokerConfig{
		SchedulerStoreProv: func(ctx context.Context, chatID string, isDM bool) (*scheduler.Store, error) {
			return st, nil
		},
	})

	err = invoker.ExecuteSchedule(ctx, sched)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "security violation: schedule grant verification failed")
}

func TestScheduleInvoker_Execute_ExpiredGrant(t *testing.T) {
	ctx := context.Background()
	st, cleanup := setupInvokerTestStore(t)
	defer cleanup()

	rawJSON := `{"sandbox":{"driver":"bwrap"}}`
	hash := scheduler.ComputeRawJSONHash(rawJSON)
	past := time.Now().Unix() - 100

	sched := &scheduler.Schedule{
		ID:           "sched_expired",
		ChatID:       "dm_user1",
		UserID:       "user1",
		Name:         "expired_task",
		Instruction:  "Do something",
		Status:       scheduler.ScheduleStatusActive,
		NextRunAt:    past,
		CreatedAt:    past,
		UpdatedAt:    past,
	}

	grant := &scheduler.ScheduleGrant{
		ScheduleID:            sched.ID,
		PermissionRequestJSON: rawJSON,
		ParamsHash:            hash,
		GrantedBy:             "user1",
		GrantedAt:             past,
		ValidUntil:            past, // Expired
	}
	err := st.CreateSchedule(ctx, sched, grant)
	require.NoError(t, err)

	invoker := NewScheduleInvoker(InvokerConfig{
		SchedulerStoreProv: func(ctx context.Context, chatID string, isDM bool) (*scheduler.Store, error) {
			return st, nil
		},
	})

	err = invoker.ExecuteSchedule(ctx, sched)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "schedule grant has expired")
}

func TestScheduleInvoker_Execute_TownhallSandboxForbidden(t *testing.T) {
	ctx := context.Background()
	st, cleanup := setupInvokerTestStore(t)
	defer cleanup()

	cfg := &config.Config{
		SandboxEnabled:       true,
		SandboxDrivers:       []string{"bwrap"},
		SandboxCPULimit:      1.0,
		SandboxMemoryLimitMB: 256,
		DataDir:              t.TempDir(),
	}
	mockDriver := &mockGatewaySandboxDriver{}
	sandboxMgr := sandbox.NewManager(cfg.SandboxConfig(), []sandbox.Driver{mockDriver})

	rawJSON := `{"sandbox":{"driver":"bwrap"}}`
	hash := scheduler.ComputeRawJSONHash(rawJSON)
	now := time.Now().Unix()

	sched := &scheduler.Schedule{
		ID:           "sched_townhall_sbx",
		ChatID:       "townhall",
		UserID:       "user1",
		Name:         "townhall_sandbox",
		Instruction:  "Do something",
		Status:       scheduler.ScheduleStatusActive,
		NextRunAt:    now,
		CreatedAt:    now,
		UpdatedAt:    now,
	}

	grant := &scheduler.ScheduleGrant{
		ScheduleID:            sched.ID,
		PermissionRequestJSON: rawJSON,
		ParamsHash:            hash,
		GrantedBy:             "user1",
		GrantedAt:             now,
		ValidUntil:            now + 3600,
	}
	err := st.CreateSchedule(ctx, sched, grant)
	require.NoError(t, err)

	invoker := NewScheduleInvoker(InvokerConfig{
		Sandbox: sandboxMgr,
		SchedulerStoreProv: func(ctx context.Context, chatID string, isDM bool) (*scheduler.Store, error) {
			return st, nil
		},
	})

	err = invoker.ExecuteSchedule(ctx, sched)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "sandbox execution is not permitted in Townhall")
}

func TestScheduleInvoker_Execute_ChatBusy(t *testing.T) {
	ctx := context.Background()
	locker := NewChatLocker()

	// Acquire lock externally to simulate an active interactive message
	release, err := locker.TryAcquire(ctx, "dm_busy", 100*time.Millisecond)
	require.NoError(t, err)
	defer release()

	invoker := NewScheduleInvoker(InvokerConfig{
		LockTimeout: 30 * time.Millisecond,
	}, locker)

	sched := &scheduler.Schedule{
		ID:          "sched_busy",
		ChatID:      "dm_busy",
		UserID:      "user1",
		Name:        "busy_task",
		Instruction: "Do something",
	}

	err = invoker.ExecuteSchedule(ctx, sched)
	require.Error(t, err)
	assert.True(t, errors.Is(err, scheduler.ErrChatBusy))
}

type mockAttachmentSenderWithProcessor struct {
	sentChatID      string
	sentContent     string
	sentAttachments []models.Attachment
}

func (m *mockAttachmentSenderWithProcessor) SendMessage(chatID, content string) error {
	m.sentChatID = chatID
	m.sentContent = content
	return nil
}

func (m *mockAttachmentSenderWithProcessor) SendMessageWithAttachments(chatID, content string, atts []models.Attachment) error {
	m.sentChatID = chatID
	m.sentContent = content
	m.sentAttachments = atts
	return nil
}

func (m *mockAttachmentSenderWithProcessor) ProcessAttachments(_ context.Context, atts []models.Attachment) (string, []chatcontext.ImageAttachment) {
	var sb strings.Builder
	for _, a := range atts {
		fmt.Fprintf(&sb, "\n\n[Attachment: %s (id: %s, type: %s)]", a.Name, a.FileID, a.MimeType)
	}
	return sb.String(), nil
}

func TestScheduleInvoker_Execute_WithStagedAttachmentsAndContextDedup(t *testing.T) {
	ctx := context.Background()
	st, cleanup := setupInvokerTestStore(t)
	defer cleanup()

	cfg := &config.Config{
		SandboxEnabled:       true,
		SandboxDrivers:       []string{"bwrap"},
		SandboxCPULimit:      1.0,
		SandboxMemoryLimitMB: 256,
		SandboxMaxLifetime:   30 * time.Minute,
		DataDir:              t.TempDir(),
	}
	mockDriver := &mockGatewaySandboxDriver{}
	sandboxMgr := sandbox.NewManager(cfg.SandboxConfig(), []sandbox.Driver{mockDriver})

	rawGrantJSON := `{"sandbox":{"driver":"bwrap","network":"none"}}`
	paramsHash := scheduler.ComputeRawJSONHash(rawGrantJSON)

	now := time.Now().Unix()
	sched := &scheduler.Schedule{
		ID:                "sched_att",
		ChatID:            "dm_user1",
		UserID:            "user1",
		Name:              "report_gen",
		Instruction:       "Generate report in sandbox",
		ScheduleType:      scheduler.ScheduleTypeInterval,
		IntervalSeconds:   600,
		NextRunAt:         now,
		Status:            scheduler.ScheduleStatusActive,
		RunTimeoutSeconds: 60,
		MaxTurns:          20,
		CreatedAt:         now,
		UpdatedAt:         now,
	}

	grant := &scheduler.ScheduleGrant{
		ScheduleID:            sched.ID,
		PermissionRequestJSON: rawGrantJSON,
		ParamsHash:            paramsHash,
		GrantedBy:             "user1",
		GrantedAt:             now,
		ValidUntil:            now + 86400,
	}
	err := st.CreateSchedule(ctx, sched, grant)
	require.NoError(t, err)

	contextMgr := chatcontext.NewManager(10)
	sender := &mockAttachmentSenderWithProcessor{}

	mockFSM := FSMRunnerFunc(func(ctx context.Context, req fsm.ToolLoopRequest) (*fsm.ToolLoopResult, error) {
		session, ok := tools.ChatSessionFromContext(ctx)
		require.True(t, ok)
		err := session.StageAttachment(models.Attachment{
			FileID:   "file_report_99",
			Name:     "report.pdf",
			MimeType: "application/pdf",
			Type:     models.AttachmentTypeFile,
		})
		require.NoError(t, err)
		return &fsm.ToolLoopResult{
			Content:     "Generated report successfully.",
			Attachments: session.GetStagedAttachments(),
		}, nil
	})

	invoker := NewScheduleInvoker(InvokerConfig{
		FSM:                 mockFSM,
		Tools:               tools.NewRegistry(nil, nil, sandboxMgr),
		Sandbox:             sandboxMgr,
		Sender:              sender,
		AttachmentProcessor: sender,
		ContextMgr:          contextMgr,
		BotID:               "bot_1",
		BotName:             "Bob",
		Model:               "test-model",
		SchedulerStoreProv: func(ctx context.Context, chatID string, isDM bool) (*scheduler.Store, error) {
			return st, nil
		},
	})

	err = invoker.ExecuteSchedule(ctx, sched)
	require.NoError(t, err)

	// Verify sender received attachments and proper payload
	require.Len(t, sender.sentAttachments, 1)
	assert.Equal(t, "file_report_99", sender.sentAttachments[0].FileID)
	assert.Equal(t, "report.pdf", sender.sentAttachments[0].Name)
	assert.Contains(t, sender.sentContent, "Generated report successfully.")

	// Verify ContextMgr has enriched attachment descriptor
	entries := contextMgr.GetOrCreate("dm_user1").Entries()
	require.Len(t, entries, 1)
	assert.Equal(t, "assistant", entries[0].Role)
	assert.Equal(t, "bot_1", entries[0].SenderID)
	assert.Contains(t, entries[0].Content, "Generated report successfully.")
	assert.Contains(t, entries[0].Content, "[Attachment: report.pdf (id: file_report_99, type: application/pdf)]")

	// Now simulate Besedka echoing back this message to Gateway.ProcessMessage
	// Verify that the echo matches normalizeForDedup and is discarded as a duplicate
	gw := &Gateway{
		cfg:            cfg,
		botUserID:      "bot_1",
		botUser:        models.User{ID: "bot_1", DisplayName: "Bob", UserName: "bot"},
		contextManager: contextMgr,
	}

	echoMsg := models.Message{
		Seq:       10,
		ChatID:    "dm_user1",
		UserID:    "bot_1",
		Content:   sender.sentContent,
		Timestamp: time.Now().Unix(),
		Attachments: []models.Attachment{
			{
				Type:     models.AttachmentTypeFile,
				Name:     "report.pdf",
				MimeType: "application/pdf",
				FileID:   "file_report_99",
			},
		},
	}

	err = gw.ProcessMessage(ctx, echoMsg)
	require.NoError(t, err)

	// Ring buffer should STILL have exactly 1 entry because deduplication recognized the echo!
	entriesAfterEcho := contextMgr.GetOrCreate("dm_user1").Entries()
	assert.Len(t, entriesAfterEcho, 1, "echoed message with attachments must be deduplicated and not added again")
}
