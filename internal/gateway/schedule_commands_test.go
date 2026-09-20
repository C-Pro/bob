package gateway

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"bob/internal/config"
	"bob/internal/models"
	"bob/internal/scheduler"

	"github.com/fasthttp/websocket"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func setupScheduleTestGateway(t *testing.T) (*Gateway, *scheduler.Store, chan models.ClientMessage, func()) {
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
				{ID: "user_bob_human", UserName: "bob_human", DisplayName: "Bob Human"},
			})
			return
		}
		if r.URL.Path == "/api/chats" {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode([]models.Chat{
				{ID: "townhall", Type: "townhall"},
				{ID: "dm_user_alice", Type: "dm", IsDM: true},
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
		BotHandle:          "@bot",
		BesedkaURL:         server.URL,
		DataDir:            tempDir,
		MsgRingBufferSize:  50,
		SchedulerMinRunTimeout: 1 * time.Minute,
		SchedulerMaxRunTimeout: 1 * time.Hour,
		SchedulerMinMaxTurns:   10,
		SchedulerMaxMaxTurns:   100,
	}

	gw := NewGateway(cfg, nil)
	gw.httpClient = server.Client()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	err := gw.DialWebSocket(ctx)
	require.NoError(t, err)

	schedStore, err := gw.StoreProvider().GetSchedulerStore(ctx, "dm_user_alice", true)
	require.NoError(t, err)

	cleanup := func() {
		gw.Stop()
		server.Close()
	}

	return gw, schedStore, receivedMsgs, cleanup
}

func recvScheduleMsg(t *testing.T, ch chan models.ClientMessage) models.ClientMessage {
	t.Helper()
	select {
	case msg := <-ch:
		return msg
	case <-time.After(3 * time.Second):
		t.Fatal("timed out waiting for client message")
		return models.ClientMessage{}
	}
}

func TestScheduleCommand_HelpAndEmptyList(t *testing.T) {
	gw, _, ch, cleanup := setupScheduleTestGateway(t)
	defer cleanup()

	ctx := context.Background()

	// Help command
	err := gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_user_alice",
		UserID:    "user_alice",
		Content:   "/schedule help",
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)

	msg := recvScheduleMsg(t, ch)
	assert.Contains(t, msg.Content, "Schedule Commands:")
	assert.Contains(t, msg.Content, "/schedule approve")

	// Empty list
	err = gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_user_alice",
		UserID:    "user_alice",
		Content:   "/schedule list",
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)

	msg = recvScheduleMsg(t, ch)
	assert.Contains(t, msg.Content, "No active schedules found for this chat")
}

func TestScheduleCommand_ApproveFlow(t *testing.T) {
	gw, store, ch, cleanup := setupScheduleTestGateway(t)
	defer cleanup()

	ctx := context.Background()
	now := time.Now().Unix()

	rawPerms := `{"sandbox":{"driver":"docker","network":"restricted"}}`
	sched := &scheduler.Schedule{
		ID:                "s_app_1",
		Name:              "daily_scraper",
		ChatID:            "dm_user_alice",
		UserID:            "user_alice",
		ScheduleType:      scheduler.ScheduleTypeInterval,
		IntervalSeconds:   3600,
		Instruction:       "Scrape headlines daily",
		Status:            scheduler.ScheduleStatusPendingApproval,
		NextRunAt:         now + 3600,
		RunTimeoutSeconds: 300,
		MaxTurns:          20,
	}
	grant := &scheduler.ScheduleGrant{
		ID:                    "g_app_1",
		ScheduleID:            "s_app_1",
		PermissionRequestJSON: rawPerms,
		ParamsHash:            scheduler.ComputeRawJSONHash(rawPerms),
		ValidUntil:            now + 86400*30,
	}
	require.NoError(t, store.CreateSchedule(ctx, sched, grant))

	// 1. Bot cannot self-approve
	err := gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_user_alice",
		UserID:    "bot-id-qa",
		Content:   "/schedule approve daily_scraper",
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)
	// Bot self-messages don't execute (ProcessMessage drops bot's own incoming msg),
	// but if passed directly to handleScheduleApprove or if bot handle is called:
	schedCheck, err := store.GetScheduleByName(ctx, "dm_user_alice", "daily_scraper")
	require.NoError(t, err)
	assert.Equal(t, scheduler.ScheduleStatusPendingApproval, schedCheck.Status)

	// 2. Non-owner cannot approve another user's DM schedule
	err = gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_user_alice",
		UserID:    "user_bob_human",
		Content:   "/schedule approve daily_scraper",
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)

	reply := recvScheduleMsg(t, ch)
	assert.Contains(t, reply.Content, "Permission denied")

	// 3. Owner successfully approves
	err = gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_user_alice",
		UserID:    "user_alice",
		Content:   "/schedule approve daily_scraper",
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)

	reply = recvScheduleMsg(t, ch)
	assert.Contains(t, reply.Content, "Schedule approved and activated!")
	assert.Contains(t, reply.Content, "daily_scraper")

	// Verify database status
	schedActive, err := store.GetScheduleByName(ctx, "dm_user_alice", "daily_scraper")
	require.NoError(t, err)
	assert.Equal(t, scheduler.ScheduleStatusActive, schedActive.Status)

	gActive, err := store.GetGrant(ctx, "s_app_1")
	require.NoError(t, err)
	assert.Equal(t, "user_alice", gActive.GrantedBy)
	assert.True(t, gActive.GrantedAt > 0)

	// 4. Repeated approve on active schedule returns info
	err = gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_user_alice",
		UserID:    "user_alice",
		Content:   "/schedule approve daily_scraper",
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)
	reply = recvScheduleMsg(t, ch)
	assert.Contains(t, reply.Content, "not awaiting approval")
}

func TestScheduleCommand_TamperedGrantAndTownhallSandbox(t *testing.T) {
	gw, store, ch, cleanup := setupScheduleTestGateway(t)
	defer cleanup()

	ctx := context.Background()
	now := time.Now().Unix()

	// 1. Tampered grant verification in DM
	schedTampered := &scheduler.Schedule{
		ID:                "s_tamp",
		Name:              "tampered_job",
		ChatID:            "dm_user_alice",
		UserID:            "user_alice",
		ScheduleType:      scheduler.ScheduleTypeInterval,
		IntervalSeconds:   600,
		Instruction:       "Dangerous run",
		Status:            scheduler.ScheduleStatusPendingApproval,
		NextRunAt:         now + 600,
		RunTimeoutSeconds: 120,
		MaxTurns:          10,
	}
	grantTampered := &scheduler.ScheduleGrant{
		ID:                    "g_tamp",
		ScheduleID:            "s_tamp",
		PermissionRequestJSON: `{"sandbox":{"driver":"docker"}}`,
		ParamsHash:            "tampered_hash_value_12345",
		ValidUntil:            now + 86400,
	}
	require.NoError(t, store.CreateSchedule(ctx, schedTampered, grantTampered))

	err := gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_user_alice",
		UserID:    "user_alice",
		Content:   "/schedule approve tampered_job",
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)

	reply := recvScheduleMsg(t, ch)
	assert.Contains(t, reply.Content, "Security alert: Permission request verification failed")

	// 2. Sandbox in Townhall rejection
	thStore, err := gw.StoreProvider().GetSchedulerStore(ctx, "townhall", false)
	require.NoError(t, err)

	rawPerms := `{"sandbox":{"driver":"bwrap"}}`
	schedTH := &scheduler.Schedule{
		ID:                "s_th",
		Name:              "townhall_job",
		ChatID:            "townhall",
		UserID:            "user_alice",
		ScheduleType:      scheduler.ScheduleTypeInterval,
		IntervalSeconds:   600,
		Instruction:       "Townhall sandbox attempt",
		Status:            scheduler.ScheduleStatusPendingApproval,
		NextRunAt:         now + 600,
		RunTimeoutSeconds: 120,
		MaxTurns:          10,
	}
	grantTH := &scheduler.ScheduleGrant{
		ID:                    "g_th",
		ScheduleID:            "s_th",
		PermissionRequestJSON: rawPerms,
		ParamsHash:            scheduler.ComputeRawJSONHash(rawPerms),
		ValidUntil:            now + 86400,
	}
	require.NoError(t, thStore.CreateSchedule(ctx, schedTH, grantTH))

	err = gw.ProcessMessage(ctx, models.Message{
		ChatID:    "townhall",
		UserID:    "user_alice",
		Content:   "@bot /schedule approve townhall_job",
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)

	reply = recvScheduleMsg(t, ch)
	assert.Contains(t, reply.Content, "Elevated sandbox tasks cannot be approved in Townhall")
}

func TestScheduleCommand_DenyAndCancelFlow(t *testing.T) {
	gw, store, ch, cleanup := setupScheduleTestGateway(t)
	defer cleanup()

	ctx := context.Background()
	now := time.Now().Unix()

	sched := &scheduler.Schedule{
		ID:                "s_deny_1",
		Name:              "suspicious_task",
		ChatID:            "dm_user_alice",
		UserID:            "user_alice",
		ScheduleType:      scheduler.ScheduleTypeInterval,
		IntervalSeconds:   300,
		Instruction:       "Run probe",
		Status:            scheduler.ScheduleStatusPendingApproval,
		NextRunAt:         now + 300,
		RunTimeoutSeconds: 60,
		MaxTurns:          10,
	}
	grant := &scheduler.ScheduleGrant{
		ID:                    "g_deny_1",
		ScheduleID:            "s_deny_1",
		PermissionRequestJSON: `{"sandbox":{"driver":"docker"}}`,
		ParamsHash:            scheduler.ComputeRawJSONHash(`{"sandbox":{"driver":"docker"}}`),
		ValidUntil:            now + 86400,
	}
	require.NoError(t, store.CreateSchedule(ctx, sched, grant))

	// Deny
	err := gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_user_alice",
		UserID:    "user_alice",
		Content:   "/schedule deny suspicious_task",
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)

	reply := recvScheduleMsg(t, ch)
	assert.Contains(t, reply.Content, "Schedule request denied")

	sAfterDeny, err := store.GetScheduleByName(ctx, "dm_user_alice", "suspicious_task")
	require.NoError(t, err)
	assert.Equal(t, scheduler.ScheduleStatusDenied, sAfterDeny.Status)

	_, err = store.GetGrant(ctx, "s_deny_1")
	assert.ErrorIs(t, err, scheduler.ErrGrantNotFound)

	// Cancel active schedule
	schedActive := &scheduler.Schedule{
		ID:                "s_cancel_1",
		Name:              "active_task",
		ChatID:            "dm_user_alice",
		UserID:            "user_alice",
		ScheduleType:      scheduler.ScheduleTypeCron,
		CronExpr:          "@daily",
		Instruction:       "Run daily sync",
		Status:            scheduler.ScheduleStatusActive,
		NextRunAt:         now + 3600,
		RunTimeoutSeconds: 120,
		MaxTurns:          10,
	}
	require.NoError(t, store.CreateSchedule(ctx, schedActive, nil))

	err = gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_user_alice",
		UserID:    "user_alice",
		Content:   "/schedule cancel active_task",
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)

	reply = recvScheduleMsg(t, ch)
	assert.Contains(t, reply.Content, "Schedule cancelled")

	sAfterCancel, err := store.GetScheduleByName(ctx, "dm_user_alice", "active_task")
	require.NoError(t, err)
	assert.Equal(t, scheduler.ScheduleStatusCancelled, sAfterCancel.Status)
}

func TestScheduleCommand_PauseResumeListAndInfo(t *testing.T) {
	gw, store, ch, cleanup := setupScheduleTestGateway(t)
	defer cleanup()

	ctx := context.Background()
	now := time.Now().Unix()

	sched := &scheduler.Schedule{
		ID:                "s_pr_1",
		Name:              "metric_collector",
		ChatID:            "dm_user_alice",
		UserID:            "user_alice",
		ScheduleType:      scheduler.ScheduleTypeInterval,
		IntervalSeconds:   600,
		Instruction:       "Collect CPU and memory stats",
		Status:            scheduler.ScheduleStatusActive,
		NextRunAt:         now + 600,
		RunTimeoutSeconds: 120,
		MaxTurns:          15,
	}
	grant := &scheduler.ScheduleGrant{
		ID:                    "g_pr_1",
		ScheduleID:            "s_pr_1",
		PermissionRequestJSON: `{"sandbox":{"driver":"docker","network":"restricted"}}`,
		ParamsHash:            scheduler.ComputeRawJSONHash(`{"sandbox":{"driver":"docker","network":"restricted"}}`),
		GrantedBy:             "user_alice",
		GrantedAt:             now,
		ValidUntil:            now + 86400*7,
	}
	require.NoError(t, store.CreateSchedule(ctx, sched, grant))

	// 1. Pause
	err := gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_user_alice",
		UserID:    "user_alice",
		Content:   "/schedule pause metric_collector",
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)
	reply := recvScheduleMsg(t, ch)
	assert.Contains(t, reply.Content, "Schedule paused")

	// 2. Resume
	err = gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_user_alice",
		UserID:    "user_alice",
		Content:   "/schedule resume metric_collector",
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)
	reply = recvScheduleMsg(t, ch)
	assert.Contains(t, reply.Content, "Schedule resumed")

	// 3. List
	err = gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_user_alice",
		UserID:    "user_alice",
		Content:   "/schedule list",
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)
	reply = recvScheduleMsg(t, ch)
	assert.Contains(t, reply.Content, "| Name | Type | Next Run | Status | Permissions |")
	assert.Contains(t, reply.Content, "metric_collector")
	assert.Contains(t, reply.Content, "Sandbox (docker)")

	// 4. Info
	err = gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_user_alice",
		UserID:    "user_alice",
		Content:   "/schedule info metric_collector",
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)
	reply = recvScheduleMsg(t, ch)
	assert.Contains(t, reply.Content, "Schedule: `metric_collector`")
	assert.Contains(t, reply.Content, "Collect CPU and memory stats")
	assert.Contains(t, reply.Content, "Sandbox Driver:** docker")
	assert.Contains(t, reply.Content, "Network Mode:** restricted")
	assert.Contains(t, reply.Content, "Execution Limits:** Timeout 120s, Max Turns 15")
}
