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

	"bob/internal/config"
	"bob/internal/models"
	"bob/internal/sandbox"

	"github.com/fasthttp/websocket"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// setupGatewayQATestEnvironment initializes a mock Besedka server and Gateway for sandbox command verification.
func setupGatewayQATestEnvironment(t *testing.T) (*Gateway, *sandbox.Manager, chan models.ClientMessage, func()) {
	t.Helper()

	upgrader := websocket.Upgrader{}
	receivedMsgs := make(chan models.ClientMessage, 50)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/me" {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(models.User{ID: "bot-id-qa", UserName: "bot", DisplayName: "Bob QA Bot"})
			return
		}
		if r.URL.Path == "/api/users" {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode([]models.User{
				{ID: "bot-id-qa", UserName: "bot", DisplayName: "Bob QA Bot"},
				{ID: "user1", UserName: "alice", DisplayName: "Alice QA"},
			})
			return
		}
		if r.URL.Path == "/api/chats" {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode([]models.Chat{
				{ID: "townhall", Type: "townhall"},
				{ID: "dm_user1", Type: "dm", IsDM: true},
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

	cfg := &config.Config{
		BotHandle:                  "@bot",
		BesedkaURL:                 server.URL,
		DataDir:                    tempDir,
		EmbeddingModel:             "none",
		SandboxEnabled:             true,
		SandboxDrivers:             []string{"bwrap"},
		SandboxAllowedNetworkModes: []string{"none", "restricted"},
		SandboxMaxLifetime:         30 * time.Minute,
		SandboxDefaultExecTimeout:  1 * time.Minute,
		SandboxMaxExecTimeout:      10 * time.Minute,
	}

	gw := NewGateway(cfg, nil)
	gw.httpClient = server.Client()

	mockDriver := &mockGatewaySandboxDriver{}
	sm := sandbox.NewManager(cfg.SandboxConfig(), []sandbox.Driver{mockDriver})
	gw.SetSandboxManager(sm)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	err := gw.DialWebSocket(ctx)
	require.NoError(t, err)

	cleanup := func() {
		gw.Stop()
		server.Close()
		_ = sm.Close()
	}

	return gw, sm, receivedMsgs, cleanup
}

// TestGateway_SandboxDenyCommand verifies command interception and state handling for /sandbox deny.
func TestGateway_SandboxDenyCommand(t *testing.T) {
	gw, sm, receivedMsgs, cleanup := setupGatewayQATestEnvironment(t)
	defer cleanup()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	recv := func() models.ClientMessage {
		select {
		case msg := <-receivedMsgs:
			return msg
		case <-time.After(1 * time.Second):
			t.Fatal("timed out waiting for message")
			return models.ClientMessage{}
		}
	}

	t.Run("deny with pending sandbox request successfully cancels and pushes context", func(t *testing.T) {
		// 1. Request sandbox
		_, err := sm.RequestSandbox(ctx, "user1", "dm_user1", sandbox.RequestParams{
			Driver:      sandbox.DriverBwrap,
			NetworkMode: sandbox.NetworkNone,
			Reason:      "Compile Rust project",
		})
		require.NoError(t, err)

		// Verify pending
		sbx, exists := sm.GetStatus("user1")
		require.True(t, exists)
		assert.Equal(t, sandbox.StatusPendingApproval, sbx.Status)

		// 2. User sends /sandbox deny in DM
		err = gw.ProcessMessage(ctx, models.Message{
			ChatID:    "dm_user1",
			UserID:    "user1",
			Content:   "/sandbox deny",
			Timestamp: time.Now().Unix(),
		})
		require.NoError(t, err)

		// Verify bot reply
		reply := recv()
		assert.Equal(t, "dm_user1", reply.ChatID)
		assert.Contains(t, reply.Content, "❌ **Sandbox request denied.**")

		// Verify sandbox is purged
		_, existsAfter := sm.GetStatus("user1")
		assert.False(t, existsAfter)

		// Verify assistant context push
		entries := gw.contextManager.GetOrCreate("dm_user1").Entries()
		require.NotEmpty(t, entries)
		lastEntry := entries[len(entries)-1]
		assert.Equal(t, "assistant", lastEntry.Role)
		assert.Contains(t, lastEntry.Content, "Sandbox request denied")

		// 3. Subsequent approve fails
		err = gw.ProcessMessage(ctx, models.Message{
			ChatID:    "dm_user1",
			UserID:    "user1",
			Content:   "/sandbox approve",
			Timestamp: time.Now().Unix(),
		})
		require.NoError(t, err)
		approveReply := recv()
		assert.Contains(t, approveReply.Content, "⚠️ Failed to approve sandbox: no pending sandbox request found to approve")
	})

	t.Run("deny without pending sandbox returns failure notification", func(t *testing.T) {
		err := gw.ProcessMessage(ctx, models.Message{
			ChatID:    "dm_user1",
			UserID:    "user1",
			Content:   "/sandbox deny",
			Timestamp: time.Now().Unix(),
		})
		require.NoError(t, err)

		reply := recv()
		assert.Contains(t, reply.Content, "⚠️ Failed to deny sandbox: no pending sandbox request found to deny")
	})

	t.Run("deny formatting variations", func(t *testing.T) {
		variations := []string{
			"<p>/sandbox deny</p>\n",
			"`/sandbox deny`",
			"   /sandbox   deny   ",
			"/sandbox DENY",
			"/sandbox Deny",
		}

		for _, v := range variations {
			// Create a pending request
			_, err := sm.RequestSandbox(ctx, "user1", "dm_user1", sandbox.RequestParams{
				Driver:      sandbox.DriverBwrap,
				NetworkMode: sandbox.NetworkNone,
				Reason:      "Run benchmark",
			})
			require.NoError(t, err)

			err = gw.ProcessMessage(ctx, models.Message{
				ChatID:    "dm_user1",
				UserID:    "user1",
				Content:   v,
				Timestamp: time.Now().Unix(),
			})
			require.NoError(t, err)

			reply := recv()
			assert.Contains(t, reply.Content, "❌ **Sandbox request denied.**")

			_, exists := sm.GetStatus("user1")
			assert.False(t, exists)
		}
	})
}

// TestGateway_SandboxDestroyCommand verifies command interception and resource release for /sandbox destroy.
func TestGateway_SandboxDestroyCommand(t *testing.T) {
	gw, sm, receivedMsgs, cleanup := setupGatewayQATestEnvironment(t)
	defer cleanup()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	recv := func() models.ClientMessage {
		select {
		case msg := <-receivedMsgs:
			return msg
		case <-time.After(1 * time.Second):
			t.Fatal("timed out waiting for message")
			return models.ClientMessage{}
		}
	}

	t.Run("destroy active running sandbox terminates resources", func(t *testing.T) {
		// Request and approve sandbox
		_, err := sm.RequestSandbox(ctx, "user1", "dm_user1", sandbox.RequestParams{
			Driver:      sandbox.DriverBwrap,
			NetworkMode: sandbox.NetworkNone,
			Reason:      "Execute test suite",
		})
		require.NoError(t, err)

		approved, err := sm.ApproveSandbox(ctx, "user1")
		require.NoError(t, err)
		assert.Equal(t, sandbox.StatusRunning, approved.Status)

		// Send /sandbox destroy
		err = gw.ProcessMessage(ctx, models.Message{
			ChatID:    "dm_user1",
			UserID:    "user1",
			Content:   "/sandbox destroy",
			Timestamp: time.Now().Unix(),
		})
		require.NoError(t, err)

		reply := recv()
		assert.Contains(t, reply.Content, "🧹 **Sandbox terminated and resources released.**")

		// Confirm destroyed in manager
		_, exists := sm.GetStatus("user1")
		assert.False(t, exists)

		// Confirm context entry pushed
		entries := gw.contextManager.GetOrCreate("dm_user1").Entries()
		require.NotEmpty(t, entries)
		assert.Equal(t, "assistant", entries[len(entries)-1].Role)
		assert.Contains(t, entries[len(entries)-1].Content, "Sandbox terminated")
	})

	t.Run("destroy when no sandbox exists is idempotent", func(t *testing.T) {
		err := gw.ProcessMessage(ctx, models.Message{
			ChatID:    "dm_user1",
			UserID:    "user1",
			Content:   "/sandbox destroy",
			Timestamp: time.Now().Unix(),
		})
		require.NoError(t, err)

		reply := recv()
		assert.Contains(t, reply.Content, "🧹 **Sandbox terminated and resources released.**")
	})

	t.Run("destroy formatting variations", func(t *testing.T) {
		variations := []string{
			"<p>/sandbox destroy</p>",
			"`/sandbox destroy`",
			"/sandbox DESTROY",
			"/sandbox Destroy",
			"  /sandbox   destroy  ",
		}

		for _, v := range variations {
			err := gw.ProcessMessage(ctx, models.Message{
				ChatID:    "dm_user1",
				UserID:    "user1",
				Content:   v,
				Timestamp: time.Now().Unix(),
			})
			require.NoError(t, err)

			reply := recv()
			assert.Contains(t, reply.Content, "🧹 **Sandbox terminated and resources released.**")
		}
	})
}

// TestGateway_SandboxStatusCommand verifies /sandbox status reporting across all lifecycle states.
func TestGateway_SandboxStatusCommand(t *testing.T) {
	gw, sm, receivedMsgs, cleanup := setupGatewayQATestEnvironment(t)
	defer cleanup()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	recv := func() models.ClientMessage {
		select {
		case msg := <-receivedMsgs:
			return msg
		case <-time.After(1 * time.Second):
			t.Fatal("timed out waiting for message")
			return models.ClientMessage{}
		}
	}

	t.Run("status with no sandbox", func(t *testing.T) {
		err := gw.ProcessMessage(ctx, models.Message{
			ChatID:    "dm_user1",
			UserID:    "user1",
			Content:   "/sandbox status",
			Timestamp: time.Now().Unix(),
		})
		require.NoError(t, err)

		reply := recv()
		assert.Contains(t, reply.Content, "ℹ️ You do not have an active or pending sandbox.")
	})

	t.Run("status with pending sandbox contains approval commands", func(t *testing.T) {
		_, err := sm.RequestSandbox(ctx, "user1", "dm_user1", sandbox.RequestParams{
			Driver:      sandbox.DriverBwrap,
			NetworkMode: sandbox.NetworkNone,
			Reason:      "Analyze log files",
		})
		require.NoError(t, err)

		err = gw.ProcessMessage(ctx, models.Message{
			ChatID:    "dm_user1",
			UserID:    "user1",
			Content:   "/sandbox status",
			Timestamp: time.Now().Unix(),
		})
		require.NoError(t, err)

		reply := recv()
		assert.Contains(t, reply.Content, "⏳ **Sandbox Request Awaiting Your Approval**")
		assert.Contains(t, reply.Content, "bwrap")
		assert.Contains(t, reply.Content, "none")
		assert.Contains(t, reply.Content, "/sandbox approve")
		assert.Contains(t, reply.Content, "/sandbox deny")
	})

	t.Run("status with active sandbox contains driver and network details", func(t *testing.T) {
		_, err := sm.ApproveSandbox(ctx, "user1")
		require.NoError(t, err)

		err = gw.ProcessMessage(ctx, models.Message{
			ChatID:    "dm_user1",
			UserID:    "user1",
			Content:   "/sandbox status",
			Timestamp: time.Now().Unix(),
		})
		require.NoError(t, err)

		reply := recv()
		assert.Contains(t, reply.Content, "🟢 **Active Sandbox Status**")
		assert.Contains(t, reply.Content, "bwrap")
		assert.Contains(t, reply.Content, "none")
		assert.Contains(t, reply.Content, "Time Remaining")
	})

	t.Run("status with expired sandbox", func(t *testing.T) {
		shortCfg := *gw.cfg
		shortCfg.SandboxMaxLifetime = 20 * time.Millisecond
		shortSM := sandbox.NewManager(shortCfg.SandboxConfig(), []sandbox.Driver{&mockGatewaySandboxDriver{}})
		defer func() { _ = shortSM.Close() }()

		gw.SetSandboxManager(shortSM)
		defer gw.SetSandboxManager(sm)

		_, err := shortSM.RequestSandbox(ctx, "user_expired", "dm_user1", sandbox.RequestParams{
			Driver:      sandbox.DriverBwrap,
			NetworkMode: sandbox.NetworkNone,
		})
		require.NoError(t, err)

		_, err = shortSM.ApproveSandbox(ctx, "user_expired")
		require.NoError(t, err)

		// Await expiration
		time.Sleep(30 * time.Millisecond)

		err = gw.ProcessMessage(ctx, models.Message{
			ChatID:    "dm_user1",
			UserID:    "user_expired",
			Content:   "/sandbox status",
			Timestamp: time.Now().Unix(),
		})
		require.NoError(t, err)

		reply := recv()
		assert.Contains(t, reply.Content, "⌛ **Your previous sandbox has expired.**")
	})

	t.Run("status when sandbox manager is nil", func(t *testing.T) {
		gw.mu.Lock()
		gw.sandboxManager = nil
		gw.mu.Unlock()

		err := gw.ProcessMessage(ctx, models.Message{
			ChatID:    "dm_user1",
			UserID:    "user1",
			Content:   "/sandbox status",
			Timestamp: time.Now().Unix(),
		})
		require.NoError(t, err)

		reply := recv()
		assert.Contains(t, reply.Content, "Sandbox execution is disabled on this server.")

		// Restore for other tests
		gw.mu.Lock()
		gw.sandboxManager = sm
		gw.mu.Unlock()
	})
}

// TestGateway_NonDMSandboxCommandRejection verifies that sandbox slash commands sent in
// townhall or group chats are strictly rejected from command interception.
func TestGateway_NonDMSandboxCommandRejection(t *testing.T) {
	gw, sm, receivedMsgs, cleanup := setupGatewayQATestEnvironment(t)
	defer cleanup()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// Setup a pending sandbox for user1
	_, err := sm.RequestSandbox(ctx, "user1", "dm_user1", sandbox.RequestParams{
		Driver:      sandbox.DriverBwrap,
		NetworkMode: sandbox.NetworkNone,
		Reason:      "Compile kernel module",
	})
	require.NoError(t, err)

	t.Run("townhall slash command without mention is completely ignored", func(t *testing.T) {
		err := gw.ProcessMessage(ctx, models.Message{
			ChatID:    "townhall",
			UserID:    "user1",
			Content:   "/sandbox approve",
			Timestamp: time.Now().Unix(),
		})
		require.NoError(t, err)

		// Expect no message sent
		select {
		case msg := <-receivedMsgs:
			t.Fatalf("unexpected message sent to townhall: %s", msg.Content)
		default:
			// Success: no command was intercepted or processed
		}

		// State remains pending
		sbx, exists := sm.GetStatus("user1")
		require.True(t, exists)
		assert.Equal(t, sandbox.StatusPendingApproval, sbx.Status)
	})

	t.Run("townhall slash command with mention does not trigger sandbox execution", func(t *testing.T) {
		// Mentioning bot in townhall: "@bot /sandbox approve"
		err := gw.ProcessMessage(ctx, models.Message{
			ChatID:    "townhall",
			UserID:    "user1",
			Content:   "@bot /sandbox approve",
			Timestamp: time.Now().Unix(),
		})
		require.NoError(t, err)

		// Sandbox must NOT be approved via townhall
		sbx, exists := sm.GetStatus("user1")
		require.True(t, exists)
		assert.Equal(t, sandbox.StatusPendingApproval, sbx.Status)
	})

	t.Run("townhall /sandbox destroy with mention does not destroy DM sandbox", func(t *testing.T) {
		// Approve via DM first
		_, err := sm.ApproveSandbox(ctx, "user1")
		require.NoError(t, err)

		// Send @bot /sandbox destroy in townhall
		err = gw.ProcessMessage(ctx, models.Message{
			ChatID:    "townhall",
			UserID:    "user1",
			Content:   "@bot /sandbox destroy",
			Timestamp: time.Now().Unix(),
		})
		require.NoError(t, err)

		// Sandbox must still be running
		sbx, exists := sm.GetStatus("user1")
		require.True(t, exists)
		assert.Equal(t, sandbox.StatusRunning, sbx.Status)
	})

	t.Run("bot cannot approve or manipulate sandboxes", func(t *testing.T) {
		err := gw.ProcessMessage(ctx, models.Message{
			ChatID:    "dm_user1",
			UserID:    "bot-id-qa", // Message sent with bot's own ID
			Content:   "/sandbox destroy",
			Timestamp: time.Now().Unix(),
		})
		require.NoError(t, err)

		// No message sent
		select {
		case msg := <-receivedMsgs:
			t.Fatalf("unexpected bot self-message: %s", msg.Content)
		default:
			// Success: rejected
		}

		// Sandbox still running
		sbx, exists := sm.GetStatus("user1")
		require.True(t, exists)
		assert.Equal(t, sandbox.StatusRunning, sbx.Status)
	})
}

// TestGateway_RepeatedApprove_AlreadyRunning verifies that sending /sandbox approve
// when a sandbox is already running acknowledges that the sandbox is approved and running
// without re-triggering continuation turns.
func TestGateway_RepeatedApprove_AlreadyRunning(t *testing.T) {
	gw, sm, receivedMsgs, cleanup := setupGatewayQATestEnvironment(t)
	defer cleanup()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	recv := func() models.ClientMessage {
		select {
		case msg := <-receivedMsgs:
			return msg
		case <-time.After(1 * time.Second):
			t.Fatal("timed out waiting for message")
			return models.ClientMessage{}
		}
	}

	_, err := sm.RequestSandbox(ctx, "user1", "dm_user1", sandbox.RequestParams{
		Driver:      sandbox.DriverBwrap,
		NetworkMode: sandbox.NetworkNone,
		Reason:      "Run tests",
	})
	require.NoError(t, err)

	approved, err := sm.ApproveSandbox(ctx, "user1")
	require.NoError(t, err)
	assert.Equal(t, sandbox.StatusRunning, approved.Status)
	assert.True(t, approved.ClaimContinuation(), "First claim succeeds")

	// Send repeated /sandbox approve
	err = gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_user1",
		UserID:    "user1",
		Content:   "/sandbox approve",
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)

	reply := recv()
	assert.Contains(t, reply.Content, "Sandbox is already approved and running.")
}

type blockingGatewaySandboxDriver struct {
	createdCount int
	onCreate     func()
	mu           sync.Mutex
}

func (m *blockingGatewaySandboxDriver) Type() sandbox.DriverType {
	return sandbox.DriverBwrap
}

func (m *blockingGatewaySandboxDriver) Available(ctx context.Context) bool {
	return true
}

func (m *blockingGatewaySandboxDriver) Create(ctx context.Context, sbx *sandbox.UserSandbox, workspace string) error {
	m.mu.Lock()
	m.createdCount++
	m.mu.Unlock()
	if m.onCreate != nil {
		m.onCreate()
	}
	return nil
}

func (m *blockingGatewaySandboxDriver) Exec(ctx context.Context, sbx *sandbox.UserSandbox, cmd []string, timeout time.Duration) (*sandbox.ExecResult, error) {
	return &sandbox.ExecResult{ExitCode: 0}, nil
}

func (m *blockingGatewaySandboxDriver) Destroy(ctx context.Context, sbx *sandbox.UserSandbox) error {
	return nil
}

// TestGateway_ConcurrentApprove_CreationInFlight tests that if a user sends a second /sandbox approve
// while driver creation is still in-flight (e.g. image downloading), both messages succeed, exactly
// one conversation continuation is triggered, and the second message receives an acknowledgment.
func TestGateway_ConcurrentApprove_CreationInFlight(t *testing.T) {
	gw, _, receivedMsgs, cleanup := setupGatewayQATestEnvironment(t)
	defer cleanup()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	recv := func() models.ClientMessage {
		select {
		case msg := <-receivedMsgs:
			return msg
		case <-time.After(3 * time.Second):
			t.Fatal("timed out waiting for message")
			return models.ClientMessage{}
		}
	}

	createStarted := make(chan struct{})
	createBlock := make(chan struct{})

	blockingDriver := &blockingGatewaySandboxDriver{
		onCreate: func() {
			close(createStarted)
			<-createBlock
		},
	}
	sm := sandbox.NewManager(gw.cfg.SandboxConfig(), []sandbox.Driver{blockingDriver})
	defer func() { _ = sm.Close() }()
	gw.SetSandboxManager(sm)

	_, err := sm.RequestSandbox(ctx, "user1", "dm_user1", sandbox.RequestParams{
		Driver:      sandbox.DriverBwrap,
		NetworkMode: sandbox.NetworkNone,
		Reason:      "Inspect airbag repo",
	})
	require.NoError(t, err)

	// User sends first /sandbox approve
	var wg sync.WaitGroup
	var err1, err2 error
	wg.Add(1)
	go func() {
		defer wg.Done()
		err1 = gw.ProcessMessage(ctx, models.Message{
			ChatID:    "dm_user1",
			UserID:    "user1",
			Content:   "/sandbox approve",
			Timestamp: time.Now().Unix(),
		})
	}()

	// Wait until driver.Create is executing
	<-createStarted

	// User sends second /sandbox approve concurrently while status is StatusCreating
	wg.Add(1)
	go func() {
		defer wg.Done()
		err2 = gw.ProcessMessage(ctx, models.Message{
			ChatID:    "dm_user1",
			UserID:    "user1",
			Content:   "/sandbox approve",
			Timestamp: time.Now().Unix(),
		})
	}()

	// Allow goroutine 2 to enter ProcessMessage and block in ApproveSandbox
	time.Sleep(50 * time.Millisecond)

	// Unblock driver.Create
	close(createBlock)
	wg.Wait()

	require.NoError(t, err1)
	require.NoError(t, err2)

	// Collect messages sent by gateway
	var replies []string
	for i := 0; i < 2; i++ {
		msg := recv()
		replies = append(replies, msg.Content)
	}

	hasProceeding := false
	hasAlreadyApproved := false
	for _, r := range replies {
		if strings.Contains(r, "Sandbox created successfully, proceeding with Inspect airbag repo...") {
			hasProceeding = true
		}
		if strings.Contains(r, "Sandbox is already approved and running.") {
			hasAlreadyApproved = true
		}
	}

	assert.True(t, hasProceeding, "Expected one proceeding message")
	assert.True(t, hasAlreadyApproved, "Expected one already approved message")

	// Ensure driver.Create was only called once
	blockingDriver.mu.Lock()
	assert.Equal(t, 1, blockingDriver.createdCount)
	blockingDriver.mu.Unlock()
}

// TestGateway_SandboxApprove_EmitsProgressCardNotTextMessage verifies that when the server
// supports progress cards, /sandbox approve emits a progress card containing the sandbox
// creation confirmation step rather than sending a redundant plain text chat message.
func TestGateway_SandboxApprove_EmitsProgressCardNotTextMessage(t *testing.T) {
	upgrader := websocket.Upgrader{}
	receivedMsgs := make(chan models.ClientMessage, 50)

	var mu sync.Mutex
	var progressRequests []sendProgressRequest
	seqCounter := int64(100)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/me" {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(models.User{ID: "bot-id-qa", UserName: "bot", DisplayName: "Bob QA Bot"})
			return
		}
		if r.URL.Path == "/api/users" {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode([]models.User{
				{ID: "bot-id-qa", UserName: "bot", DisplayName: "Bob QA Bot"},
				{ID: "user1", UserName: "alice", DisplayName: "Alice QA"},
			})
			return
		}
		if r.URL.Path == "/api/chats" {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode([]models.Chat{
				{ID: "townhall", Type: "townhall"},
				{ID: "dm_user1", Type: "dm", IsDM: true},
			})
			return
		}
		if strings.HasPrefix(r.URL.Path, "/api/chats/") && strings.HasSuffix(r.URL.Path, "/messages") {
			if r.Method == http.MethodPost {
				var req sendProgressRequest
				if err := json.NewDecoder(r.Body).Decode(&req); err == nil {
					mu.Lock()
					seqCounter++
					assignedSeq := seqCounter
					progressRequests = append(progressRequests, req)
					mu.Unlock()
					w.Header().Set("Content-Type", "application/json")
					_ = json.NewEncoder(w).Encode(sendProgressResponse{Seq: assignedSeq, Timestamp: time.Now().Unix()})
					return
				}
			}
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
	defer server.Close()

	cfg := &config.Config{
		BotHandle:                  "@bot",
		BesedkaURL:                 server.URL,
		DataDir:                    t.TempDir(),
		EmbeddingModel:             "none",
		SandboxEnabled:             true,
		SandboxDrivers:             []string{"bwrap"},
		SandboxAllowedNetworkModes: []string{"none", "restricted"},
		SandboxMaxLifetime:         30 * time.Minute,
		SandboxDefaultExecTimeout:  1 * time.Minute,
		SandboxMaxExecTimeout:      10 * time.Minute,
	}

	gw := NewGateway(cfg, nil)
	gw.httpClient = server.Client()

	mockDriver := &mockGatewaySandboxDriver{}
	sm := sandbox.NewManager(cfg.SandboxConfig(), []sandbox.Driver{mockDriver})
	defer func() { _ = sm.Close() }()
	gw.SetSandboxManager(sm)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	require.NoError(t, gw.DialWebSocket(ctx))
	defer gw.Stop()

	// 1. Request sandbox
	_, err := sm.RequestSandbox(ctx, "user1", "dm_user1", sandbox.RequestParams{
		Driver:      sandbox.DriverBwrap,
		NetworkMode: sandbox.NetworkNone,
		Reason:      "Inspect airbag repo",
	})
	require.NoError(t, err)

	// 2. User approves sandbox
	err = gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_user1",
		UserID:    "user1",
		Content:   "/sandbox approve",
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)

	// 3. Verify progress card was emitted
	mu.Lock()
	defer mu.Unlock()
	require.NotEmpty(t, progressRequests, "Must have emitted progress message")

	// First progress request is the root card
	rootReq := progressRequests[0]
	assert.Equal(t, models.MessageTypeProgress, rootReq.Type)
	require.NotNil(t, rootReq.Progress)
	assert.Equal(t, int64(0), rootReq.Progress.ParentSeq)
	assert.Equal(t, "Working on your request...", rootReq.Progress.Title)
	assert.Equal(t, models.ProgressStatusRunning, rootReq.Progress.CardStatus)
	require.Len(t, rootReq.Progress.Steps, 1)

	step := rootReq.Progress.Steps[0]
	assert.Equal(t, "sandbox_create", step.ID)
	assert.Equal(t, "Sandbox created", step.Title)
	assert.Equal(t, models.ProgressStatusCompleted, step.Status)
	assert.Equal(t, "Proceeding with: Inspect airbag repo", step.Description)

	// Second progress request completes the card (since llmClient is nil)
	require.Len(t, progressRequests, 2)
	finalReq := progressRequests[1]
	assert.Equal(t, int64(101), finalReq.Progress.ParentSeq)
	assert.Equal(t, models.ProgressStatusCompleted, finalReq.Progress.CardStatus)

	// 4. Verify that NO plain text message was sent over WebSocket
	select {
	case msg := <-receivedMsgs:
		t.Fatalf("unexpected chat message received via websocket: %+v", msg)
	case <-time.After(100 * time.Millisecond):
		// Expected: no redundant text message sent
	}

	// 5. Verify chat context does not contain redundant confirmation text
	entries := gw.contextManager.GetOrCreate("dm_user1").Entries()
	for _, entry := range entries {
		assert.NotContains(t, entry.Content, "Sandbox created successfully")
	}
}

// TestGateway_SandboxApprove_FallbackToTextMessageWhenProgressFails verifies that
// if emitting the progress card fails (e.g. older server version), Bob falls back to
// sending the plain text confirmation message.
func TestGateway_SandboxApprove_FallbackToTextMessageWhenProgressFails(t *testing.T) {
	gw, sm, receivedMsgs, cleanup := setupGatewayQATestEnvironment(t)
	defer cleanup()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	recv := func() models.ClientMessage {
		select {
		case msg := <-receivedMsgs:
			return msg
		case <-time.After(1 * time.Second):
			t.Fatal("timed out waiting for message")
			return models.ClientMessage{}
		}
	}

	_, err := sm.RequestSandbox(ctx, "user1", "dm_user1", sandbox.RequestParams{
		Driver:      sandbox.DriverBwrap,
		NetworkMode: sandbox.NetworkNone,
		Reason:      "Inspect airbag repo",
	})
	require.NoError(t, err)

	// User approves sandbox in an environment where POST /messages returns empty array (fail seq)
	err = gw.ProcessMessage(ctx, models.Message{
		ChatID:    "dm_user1",
		UserID:    "user1",
		Content:   "/sandbox approve",
		Timestamp: time.Now().Unix(),
	})
	require.NoError(t, err)

	reply := recv()
	assert.Contains(t, reply.Content, "Sandbox created successfully, proceeding with Inspect airbag repo...")
}
