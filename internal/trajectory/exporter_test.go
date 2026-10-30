package trajectory

import (
	"bytes"
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"bob/internal/objectstore"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type mockUploader struct {
	mu      sync.Mutex
	uploads map[string][]byte
	putHook func(ctx context.Context, key string, r io.Reader, size int64) error
}

func newMockUploader() *mockUploader {
	return &mockUploader{
		uploads: make(map[string][]byte),
	}
}

func (m *mockUploader) Put(ctx context.Context, key string, r io.Reader, size int64) error {
	m.mu.Lock()
	hook := m.putHook
	m.mu.Unlock()

	if hook != nil {
		if err := hook(ctx, key, r, size); err != nil {
			return err
		}
	}

	data, err := io.ReadAll(r)
	if err != nil {
		return err
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	m.uploads[key] = data
	return nil
}

func (m *mockUploader) getUpload(key string) ([]byte, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	data, ok := m.uploads[key]
	return data, ok
}

func setupTestStore(t *testing.T) (*SQLiteStore, *sql.DB) {
	t.Helper()
	db, err := sql.Open("sqlite", ":memory:")
	require.NoError(t, err)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	err = EnsureSchema(ctx, db)
	require.NoError(t, err)

	store, err := NewStore(db)
	require.NoError(t, err)

	return store, db
}

func TestExporter_Disabled(t *testing.T) {
	store, db := setupTestStore(t)
	defer func() { _ = db.Close() }()

	uploader := newMockUploader()
	cfg := ExporterConfig{
		Enabled: false,
	}
	exporter := NewExporter(cfg, SingleStoreProvider{Store: store}, uploader)

	ctx := context.Background()
	err := exporter.FlushOnce(ctx)
	assert.NoError(t, err)
	assert.Empty(t, uploader.uploads)
}

func TestExporter_MissingUploaderWhenEnabled(t *testing.T) {
	store, db := setupTestStore(t)
	defer func() { _ = db.Close() }()

	cfg := ExporterConfig{
		Enabled: true,
	}
	exporter := NewExporter(cfg, SingleStoreProvider{Store: store}, nil)

	ctx := context.Background()
	err := exporter.FlushOnce(ctx)
	assert.ErrorIs(t, err, ErrUploaderRequired)
}

func TestBuildObjectKey(t *testing.T) {
	fixedTime := time.Date(2026, 10, 29, 14, 30, 0, 0, time.UTC)

	t.Run("valid key with standard prefix", func(t *testing.T) {
		traj := &Trajectory{
			ID:        "traj-1234-uuid",
			Revision:  1,
			CreatedAt: fixedTime,
		}
		key, err := BuildObjectKey("trajectories/", traj)
		require.NoError(t, err)
		assert.Equal(t, "trajectories/2026/10/29/traj-1234-uuid/r1.json", key)
	})

	t.Run("prefix normalization", func(t *testing.T) {
		traj := &Trajectory{
			ID:        "traj-1234-uuid",
			Revision:  2,
			CreatedAt: fixedTime,
		}
		// Leading slash stripped, trailing slash added
		key, err := BuildObjectKey("/custom/prefix", traj)
		require.NoError(t, err)
		assert.Equal(t, "custom/prefix/2026/10/29/traj-1234-uuid/r2.json", key)
	})

	t.Run("empty prefix", func(t *testing.T) {
		traj := &Trajectory{
			ID:        "traj-1234-uuid",
			Revision:  3,
			CreatedAt: fixedTime,
		}
		key, err := BuildObjectKey("", traj)
		require.NoError(t, err)
		assert.Equal(t, "2026/10/29/traj-1234-uuid/r3.json", key)
	})

	t.Run("reject unsafe trajectory ID", func(t *testing.T) {
		unsafeIDs := []string{
			"../malicious",
			"path/traversal",
			"space in id",
			" leading_space",
			"trailing_space ",
			"id\x00null",
			"",
		}
		for _, id := range unsafeIDs {
			traj := &Trajectory{
				ID:        id,
				Revision:  1,
				CreatedAt: fixedTime,
			}
			_, err := BuildObjectKey("trajectories/", traj)
			assert.Error(t, err, "expected error for unsafe ID: %q", id)
		}
	})

	t.Run("reject invalid revision", func(t *testing.T) {
		traj := &Trajectory{
			ID:        "traj-1",
			Revision:  0,
			CreatedAt: fixedTime,
		}
		_, err := BuildObjectKey("trajectories/", traj)
		assert.ErrorIs(t, err, ErrInvalidRevision)
	})

	t.Run("reject zero created_at", func(t *testing.T) {
		traj := &Trajectory{
			ID:       "traj-1",
			Revision: 1,
		}
		_, err := BuildObjectKey("trajectories/", traj)
		assert.ErrorIs(t, err, ErrMissingCreatedAt)
	})
}

func TestIsEligibleForExport(t *testing.T) {
	tests := []struct {
		name         string
		approved     bool
		privacy      string
		allowPrivate bool
		expected     bool
	}{
		{"unapproved public", false, "public", false, false},
		{"approved public", true, "public", false, true},
		{"approved public with allowPrivate", true, "public", true, true},
		{"approved private disallowed", true, "private", false, false},
		{"approved private allowed", true, "private", true, true},
		{"approved internal allowed private", true, "internal", true, false},
		{"approved unknown classification", true, "confidential", true, false},
		{"nil trajectory", false, "", false, false},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var traj *Trajectory
			if tc.name != "nil trajectory" {
				traj = &Trajectory{
					ExportApproved: tc.approved,
					PrivacyLevel:   tc.privacy,
				}
			}
			assert.Equal(t, tc.expected, IsEligibleForExport(traj, tc.allowPrivate))
		})
	}
}

func TestExporter_FlushOnce_Success(t *testing.T) {
	store, db := setupTestStore(t)
	defer func() { _ = db.Close() }()

	ctx := context.Background()
	now := time.Now().UTC()

	// Stage a mature public trajectory
	traj := &Trajectory{
		ID:             "traj-e2e-success",
		RunID:          "run-1",
		FrontendID:     "besedka",
		SessionID:      "townhall",
		ScopeID:        "townhall",
		ActorID:        "user-1",
		PrivacyLevel:   "public",
		ExportApproved: true,
		CreatedAt:      now.Add(-20 * time.Minute),
		CompletedAt:    now.Add(-19 * time.Minute),
		UserInput:      "Hello world",
		FinalOutput:    "Hi there!",
		Reward:         0.40,
	}
	err := store.InsertStaged(ctx, traj, 0)
	require.NoError(t, err)

	uploader := newMockUploader()
	cfg := ExporterConfig{
		Enabled:      true,
		Prefix:       "trajectories/",
		FlushDelay:   15 * time.Minute,
		AllowPrivate: false,
	}
	exporter := NewExporter(cfg, SingleStoreProvider{Store: store}, uploader)

	// Flush
	err = exporter.FlushOnce(ctx)
	require.NoError(t, err)

	// Verify uploaded object key and content
	expectedKey, err := BuildObjectKey(cfg.Prefix, traj)
	require.NoError(t, err)

	uploadedBytes, ok := uploader.getUpload(expectedKey)
	require.True(t, ok, "expected object key to be uploaded: %s", expectedKey)

	var uploadedTraj Trajectory
	err = json.Unmarshal(uploadedBytes, &uploadedTraj)
	require.NoError(t, err)
	assert.Equal(t, traj.ID, uploadedTraj.ID)
	assert.Equal(t, "Hello world", uploadedTraj.UserInput)
	assert.Equal(t, "Hi there!", uploadedTraj.FinalOutput)

	// Verify status in DB is now 'exported'
	persisted, err := store.Get(ctx, traj.ID)
	require.NoError(t, err)
	assert.Equal(t, 1, persisted.Revision)

	// Query DB status directly
	var status string
	var exportedRev int
	err = db.QueryRow("SELECT status, exported_revision FROM agent_trajectories WHERE id = ?", traj.ID).Scan(&status, &exportedRev)
	require.NoError(t, err)
	assert.Equal(t, "exported", status)
	assert.Equal(t, 1, exportedRev)

	// Second flush does not re-upload already exported revision
	delete(uploader.uploads, expectedKey)
	err = exporter.FlushOnce(ctx)
	require.NoError(t, err)
	assert.Empty(t, uploader.uploads)
}

func TestExporter_FlushOnce_RevisionRace(t *testing.T) {
	store, db := setupTestStore(t)
	defer func() { _ = db.Close() }()

	ctx := context.Background()
	now := time.Now().UTC()

	traj := &Trajectory{
		ID:             "traj-race",
		RunID:          "run-race",
		FrontendID:     "besedka",
		SessionID:      "townhall",
		ScopeID:        "townhall",
		ActorID:        "user-race",
		PrivacyLevel:   "public",
		ExportApproved: true,
		CreatedAt:      now.Add(-20 * time.Minute),
		CompletedAt:    now.Add(-19 * time.Minute),
		UserInput:      "Task input",
		FinalOutput:    "Initial answer",
		Reward:         0.40,
	}
	err := store.InsertStaged(ctx, traj, 0)
	require.NoError(t, err)

	uploader := newMockUploader()
	feedbackApplied := make(chan struct{})

	// Hook into Put to simulate user feedback arriving during upload
	uploader.putHook = func(putCtx context.Context, key string, r io.Reader, size int64) error {
		// Apply feedback during upload
		_, fbErr := store.UpdateFeedback(putCtx, FeedbackRequest{
			MessageID:  "msg-fb-race",
			FrontendID: "besedka",
			SessionID:  "townhall",
			ScopeID:    "townhall",
			ActorID:    "user-race",
			Rating:     1.0,
			Note:       "Great answer!",
		})
		require.NoError(t, fbErr)
		close(feedbackApplied)
		return nil
	}

	cfg := ExporterConfig{
		Enabled:      true,
		Prefix:       "trajectories/",
		FlushDelay:   10 * time.Minute,
		AllowPrivate: false,
	}
	exporter := NewExporter(cfg, SingleStoreProvider{Store: store}, uploader)

	// Run first flush: uploads revision 1, but feedback bumped SQLite revision to 2
	err = exporter.FlushOnce(ctx)
	require.NoError(t, err)

	// Verify r1 was uploaded
	keyR1, err := BuildObjectKey(cfg.Prefix, traj)
	require.NoError(t, err)
	_, ok := uploader.getUpload(keyR1)
	require.True(t, ok)

	// In SQLite: exported_revision should be 1, but status must remain 'staged' because revision is 2!
	var status string
	var currentRev, exportedRev int
	err = db.QueryRow("SELECT status, revision, exported_revision FROM agent_trajectories WHERE id = ?", traj.ID).Scan(&status, &currentRev, &exportedRev)
	require.NoError(t, err)
	assert.Equal(t, "staged", status)
	assert.Equal(t, 2, currentRev)
	assert.Equal(t, 1, exportedRev)

	// Now clear the put hook so next upload succeeds normally
	uploader.mu.Lock()
	uploader.putHook = nil
	uploader.mu.Unlock()

	// Advance time past the feedback quiescence window (1 minute)
	exporter.now = func() time.Time {
		return now.Add(5 * time.Minute)
	}

	// Run second flush: uploads revision 2
	err = exporter.FlushOnce(ctx)
	require.NoError(t, err)

	// Verify r2 was uploaded
	traj.Revision = 2
	keyR2, err := BuildObjectKey(cfg.Prefix, traj)
	require.NoError(t, err)
	dataR2, ok := uploader.getUpload(keyR2)
	require.True(t, ok)

	var trajR2 Trajectory
	err = json.Unmarshal(dataR2, &trajR2)
	require.NoError(t, err)
	assert.Equal(t, 2, trajR2.Revision)
	assert.Equal(t, 1.0, trajR2.Breakdown.UserFeedback)

	// Now SQLite status is finally 'exported' with exported_revision = 2
	err = db.QueryRow("SELECT status, revision, exported_revision FROM agent_trajectories WHERE id = ?", traj.ID).Scan(&status, &currentRev, &exportedRev)
	require.NoError(t, err)
	assert.Equal(t, "exported", status)
	assert.Equal(t, 2, currentRev)
	assert.Equal(t, 2, exportedRev)
}

func TestExporter_FailureIsolation_And_StarvationPrevention(t *testing.T) {
	store, db := setupTestStore(t)
	defer func() { _ = db.Close() }()

	ctx := context.Background()
	now := time.Now().UTC()

	// Trajectory 1: older, will fail upload
	trajFail := &Trajectory{
		ID:             "traj-fail",
		RunID:          "run-fail",
		FrontendID:     "besedka",
		SessionID:      "townhall",
		ScopeID:        "townhall",
		ActorID:        "user-1",
		PrivacyLevel:   "public",
		ExportApproved: true,
		CreatedAt:      now.Add(-30 * time.Minute),
		CompletedAt:    now.Add(-29 * time.Minute),
		UserInput:      "Failing task",
	}
	err := store.InsertStaged(ctx, trajFail, 0)
	require.NoError(t, err)

	// Trajectory 2: newer, will succeed
	trajSucceed := &Trajectory{
		ID:             "traj-succeed",
		RunID:          "run-succeed",
		FrontendID:     "besedka",
		SessionID:      "townhall",
		ScopeID:        "townhall",
		ActorID:        "user-2",
		PrivacyLevel:   "public",
		ExportApproved: true,
		CreatedAt:      now.Add(-20 * time.Minute),
		CompletedAt:    now.Add(-19 * time.Minute),
		UserInput:      "Succeeding task",
	}
	err = store.InsertStaged(ctx, trajSucceed, 0)
	require.NoError(t, err)

	uploader := newMockUploader()
	uploader.putHook = func(ctx context.Context, key string, r io.Reader, size int64) error {
		if bytes.Contains([]byte(key), []byte("traj-fail")) {
			return errors.New("simulated S3 transient error")
		}
		return nil
	}

	cfg := ExporterConfig{
		Enabled:    true,
		Prefix:     "trajectories/",
		MaxRetries: 2,
		BatchLimit: 1, // Crucial: limit=1 tests that failing row does not starve newer rows
	}
	exporter := NewExporter(cfg, SingleStoreProvider{Store: store}, uploader)

	// FlushOnce pass 1: with limit=1, trajFail is queried first and fails upload
	err = exporter.FlushOnce(ctx)
	assert.Error(t, err) // reports accumulated error for trajFail
	assert.Contains(t, err.Error(), "traj-fail")

	// TrajSucceed has not been uploaded yet because limit was 1
	keySucceed, err := BuildObjectKey(cfg.Prefix, trajSucceed)
	require.NoError(t, err)
	_, ok := uploader.getUpload(keySucceed)
	assert.False(t, ok, "trajSucceed should not have been uploaded yet in batch of size 1")

	// FlushOnce pass 2: trajFail is excluded via SQL excludeIDs, so trajSucceed is queried and uploaded!
	assert.True(t, exporter.IsBackingOff("traj-fail", now))
	err = exporter.FlushOnce(ctx)
	assert.NoError(t, err, "backoff should exclude failing row, allowing trajSucceed to be exported")

	_, ok = uploader.getUpload(keySucceed)
	assert.True(t, ok, "trajSucceed must be uploaded despite batch limit of 1")
}

func TestExporter_SerializedFlushes(t *testing.T) {
	store, db := setupTestStore(t)
	defer func() { _ = db.Close() }()

	ctx := context.Background()
	now := time.Now().UTC()

	traj := &Trajectory{
		ID:             "traj-serialized",
		RunID:          "run-s",
		FrontendID:     "besedka",
		SessionID:      "townhall",
		ScopeID:        "townhall",
		ActorID:        "user-s",
		PrivacyLevel:   "public",
		ExportApproved: true,
		CreatedAt:      now.Add(-20 * time.Minute),
		CompletedAt:    now.Add(-19 * time.Minute),
		UserInput:      "Serialized test",
	}
	err := store.InsertStaged(ctx, traj, 0)
	require.NoError(t, err)

	uploader := newMockUploader()
	var inFlight int
	var maxInFlight int
	var lock sync.Mutex

	uploader.putHook = func(ctx context.Context, key string, r io.Reader, size int64) error {
		lock.Lock()
		inFlight++
		if inFlight > maxInFlight {
			maxInFlight = inFlight
		}
		lock.Unlock()

		time.Sleep(20 * time.Millisecond)

		lock.Lock()
		inFlight--
		lock.Unlock()
		return nil
	}

	cfg := ExporterConfig{
		Enabled: true,
		Prefix:  "trajectories/",
	}
	exporter := NewExporter(cfg, SingleStoreProvider{Store: store}, uploader)

	// Launch multiple concurrent FlushOnce calls
	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_ = exporter.FlushOnce(ctx)
		}()
	}
	wg.Wait()

	// Ensure maxInFlight was at most 1
	lock.Lock()
	defer lock.Unlock()
	assert.LessOrEqual(t, maxInFlight, 1, "flushes must be strictly serialized")
}

func TestExporter_Stop_BoundedShutdownFlush(t *testing.T) {
	store, db := setupTestStore(t)
	defer func() { _ = db.Close() }()

	ctx := context.Background()
	now := time.Now().UTC()

	traj := &Trajectory{
		ID:             "traj-shutdown",
		RunID:          "run-sd",
		FrontendID:     "besedka",
		SessionID:      "townhall",
		ScopeID:        "townhall",
		ActorID:        "user-sd",
		PrivacyLevel:   "public",
		ExportApproved: true,
		CreatedAt:      now.Add(-20 * time.Minute),
		CompletedAt:    now.Add(-19 * time.Minute),
		UserInput:      "Shutdown test",
	}
	err := store.InsertStaged(ctx, traj, 0)
	require.NoError(t, err)

	uploader := newMockUploader()
	cfg := ExporterConfig{
		Enabled:      true,
		Prefix:       "trajectories/",
		PollInterval: 1 * time.Hour, // long interval so ticker won't fire during test
	}
	exporter := NewExporter(cfg, SingleStoreProvider{Store: store}, uploader)

	// Start exporter in background
	exporter.Start(ctx)
	assert.True(t, exporter.Running())

	// Stop with shutdown context
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	err = exporter.Stop(shutdownCtx)
	require.NoError(t, err)
	assert.False(t, exporter.Running())

	// Trajectory must have been flushed during shutdown
	key, err := BuildObjectKey(cfg.Prefix, traj)
	require.NoError(t, err)
	_, ok := uploader.getUpload(key)
	assert.True(t, ok, "staged trajectory should be exported during bounded shutdown flush")
}

func TestExporter_ContextCancellation(t *testing.T) {
	store, db := setupTestStore(t)
	defer func() { _ = db.Close() }()

	ctx, cancel := context.WithCancel(context.Background())
	now := time.Now().UTC()

	traj := &Trajectory{
		ID:             "traj-cancel",
		RunID:          "run-c",
		FrontendID:     "besedka",
		SessionID:      "townhall",
		ScopeID:        "townhall",
		ActorID:        "user-c",
		PrivacyLevel:   "public",
		ExportApproved: true,
		CreatedAt:      now.Add(-20 * time.Minute),
		CompletedAt:    now.Add(-19 * time.Minute),
		UserInput:      "Cancel test",
	}
	err := store.InsertStaged(ctx, traj, 0)
	require.NoError(t, err)

	uploader := newMockUploader()
	uploader.putHook = func(ctx context.Context, key string, r io.Reader, size int64) error {
		// Cancel context during upload
		cancel()
		<-ctx.Done()
		return ctx.Err()
	}

	cfg := ExporterConfig{
		Enabled:    true,
		Prefix:     "trajectories/",
		MaxRetries: 3,
	}
	exporter := NewExporter(cfg, SingleStoreProvider{Store: store}, uploader)

	err = exporter.FlushOnce(ctx)
	assert.ErrorIs(t, err, context.Canceled)
}

func TestExporter_WithRealObjectStoreClient(t *testing.T) {
	store, db := setupTestStore(t)
	defer func() { _ = db.Close() }()

	ctx := context.Background()
	now := time.Now().UTC()

	traj := &Trajectory{
		ID:             "traj-real-s3",
		RunID:          "run-real-s3",
		FrontendID:     "besedka",
		SessionID:      "townhall",
		ScopeID:        "townhall",
		ActorID:        "user-real",
		PrivacyLevel:   "public",
		ExportApproved: true,
		CreatedAt:      now.Add(-20 * time.Minute),
		CompletedAt:    now.Add(-19 * time.Minute),
		UserInput:      "Real S3 test",
		FinalOutput:    "Success payload",
		Reward:         0.40,
	}
	err := store.InsertStaged(ctx, traj, 0)
	require.NoError(t, err)

	expectedKey, err := BuildObjectKey("trajectories/", traj)
	require.NoError(t, err)

	var receivedPath string
	var receivedBody []byte
	var receivedSha256 string
	var authHeader string

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		authHeader = r.Header.Get("Authorization")
		receivedPath = r.URL.Path
		receivedSha256 = r.Header.Get("X-Amz-Content-Sha256")

		body, readErr := io.ReadAll(r.Body)
		if readErr != nil {
			http.Error(w, readErr.Error(), http.StatusInternalServerError)
			return
		}
		receivedBody = body
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	// Configure real objectstore client with trajectory bucket override
	bucketOverride := "traj-custom-bucket"
	s3Client, err := objectstore.New(objectstore.Config{
		Endpoint:  srv.URL,
		Region:    "us-east-1",
		Bucket:    bucketOverride,
		AccessKey: "TRAJ_ACCESS",
		SecretKey: "TRAJ_SECRET",
		PathStyle: true,
	})
	require.NoError(t, err)
	require.NotNil(t, s3Client)

	cfg := ExporterConfig{
		Enabled:      true,
		Bucket:       bucketOverride,
		Prefix:       "trajectories/",
		FlushDelay:   15 * time.Minute,
		AllowPrivate: false,
	}
	exporter := NewExporter(cfg, SingleStoreProvider{Store: store}, s3Client)

	err = exporter.FlushOnce(ctx)
	require.NoError(t, err)

	// Verify SigV4 Authorization and headers
	assert.True(t, strings.HasPrefix(authHeader, "AWS4-HMAC-SHA256 "), "must have SigV4 Authorization header")
	assert.Contains(t, authHeader, "Credential=TRAJ_ACCESS/")

	// Verify path matches bucket + prefix + YYYY/MM/DD/id/r1.json
	expectedPath := "/" + bucketOverride + "/" + expectedKey
	assert.Equal(t, expectedPath, receivedPath)

	// Verify Sha256 hash matches received body
	expectedSum := sha256.Sum256(receivedBody)
	assert.Equal(t, hex.EncodeToString(expectedSum[:]), receivedSha256)

	// Verify payload contains trajectory fields
	var uploadedTraj Trajectory
	err = json.Unmarshal(receivedBody, &uploadedTraj)
	require.NoError(t, err)
	assert.Equal(t, "traj-real-s3", uploadedTraj.ID)
	assert.Equal(t, "Real S3 test", uploadedTraj.UserInput)
	assert.Equal(t, "Success payload", uploadedTraj.FinalOutput)

	// Verify marked exported in DB
	persisted, err := store.Get(ctx, traj.ID)
	require.NoError(t, err)
	assert.Equal(t, 1, persisted.Revision)
}

func TestExporter_StartStop_Idempotency(t *testing.T) {
	store, db := setupTestStore(t)
	defer func() { _ = db.Close() }()

	uploader := newMockUploader()
	cfg := ExporterConfig{
		Enabled:      true,
		Prefix:       "trajectories/",
		PollInterval: 1 * time.Hour,
	}
	exporter := NewExporter(cfg, SingleStoreProvider{Store: store}, uploader)

	// Stop when stopped is a clean no-op
	assert.NoError(t, exporter.Stop(context.Background()))

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Start transitions to running
	exporter.Start(ctx)
	assert.True(t, exporter.Running())

	// Calling Start again while running is an idempotent no-op
	exporter.Start(ctx)
	assert.True(t, exporter.Running())

	// Stop halts worker
	assert.NoError(t, exporter.Stop(context.Background()))
	assert.False(t, exporter.Running())

	// Restarting after Stop works cleanly
	exporter.Start(ctx)
	assert.True(t, exporter.Running())
	assert.NoError(t, exporter.Stop(context.Background()))
	assert.False(t, exporter.Running())
}

func TestExporter_Stop_DeadlineEnforcedWhileWorkerRunning(t *testing.T) {
	store, db := setupTestStore(t)
	defer func() { _ = db.Close() }()

	ctx := context.Background()

	traj := &Trajectory{
		ID:             "traj-slow-stop",
		RunID:          "run-ss",
		FrontendID:     "besedka",
		SessionID:      "townhall",
		ScopeID:        "townhall",
		ActorID:        "user-ss",
		PrivacyLevel:   "public",
		ExportApproved: true,
		CreatedAt:      time.Now().UTC().Add(-20 * time.Minute),
		CompletedAt:    time.Now().UTC().Add(-19 * time.Minute),
		UserInput:      "Slow stop test",
	}
	err := store.InsertStaged(ctx, traj, 0)
	require.NoError(t, err)

	uploader := newMockUploader()
	blockUpload := make(chan struct{})
	uploader.putHook = func(ctx context.Context, key string, r io.Reader, size int64) error {
		select {
		case <-blockUpload:
			return nil
		case <-time.After(150 * time.Millisecond): // Deliberately slow upload to test deadline enforcement
			return nil
		}
	}

	cfg := ExporterConfig{
		Enabled:       true,
		Prefix:        "trajectories/",
		PollInterval:  1 * time.Hour,
		UploadTimeout: 10 * time.Second,
	}
	exporter := NewExporter(cfg, SingleStoreProvider{Store: store}, uploader)

	// Start worker in background (will pick up traj and block in upload)
	workerCtx, workerCancel := context.WithCancel(context.Background())
	defer workerCancel()
	exporter.Start(workerCtx)

	// Give worker time to enter upload
	time.Sleep(20 * time.Millisecond)

	// Attempt Stop with a tight deadline that expires before upload finishes
	stopCtx, stopCancel := context.WithTimeout(context.Background(), 25*time.Millisecond)
	defer stopCancel()

	err = exporter.Stop(stopCtx)
	// Must not hang indefinitely; should return deadline error
	assert.ErrorIs(t, err, context.DeadlineExceeded)

	// Unblock to allow clean exit
	close(blockUpload)
	workerCancel()
}

func TestExporter_FlushOnce_ContextCanceledLock(t *testing.T) {
	store, db := setupTestStore(t)
	defer func() { _ = db.Close() }()

	uploader := newMockUploader()
	cfg := ExporterConfig{
		Enabled: true,
		Prefix:  "trajectories/",
	}
	exporter := NewExporter(cfg, SingleStoreProvider{Store: store}, uploader)

	// Simulate an active flush holding the semaphore
	exporter.flushSem <- struct{}{}

	// FlushOnce with canceled context returns immediately
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	err := exporter.FlushOnce(ctx)
	assert.ErrorIs(t, err, context.Canceled)

	// Release semaphore
	<-exporter.flushSem
}

func TestExporter_Stop_ConcurrentCallersShareFinalFlush(t *testing.T) {
	store, db := setupTestStore(t)
	defer func() { _ = db.Close() }()

	ctx := context.Background()
	now := time.Now().UTC()

	traj := &Trajectory{
		ID:             "traj-concurrent-stop",
		RunID:          "run-cs",
		FrontendID:     "besedka",
		SessionID:      "townhall",
		ScopeID:        "townhall",
		ActorID:        "user-cs",
		PrivacyLevel:   "public",
		ExportApproved: true,
		CreatedAt:      now.Add(-20 * time.Minute),
		CompletedAt:    now.Add(-19 * time.Minute),
		UserInput:      "Concurrent stop test",
		FinalOutput:    "Success payload",
	}
	require.NoError(t, store.InsertStaged(ctx, traj, 0))

	uploader := newMockUploader()
	flushBlocker := make(chan struct{})
	flushStarted := make(chan struct{})

	// Hook into Put to keep the final flush active until released
	uploader.putHook = func(putCtx context.Context, key string, r io.Reader, size int64) error {
		select {
		case <-flushStarted:
		default:
			close(flushStarted)
		}
		<-flushBlocker
		return nil
	}

	cfg := ExporterConfig{
		Enabled:      true,
		Prefix:       "trajectories/",
		PollInterval: 1 * time.Hour,
	}
	exporter := NewExporter(cfg, SingleStoreProvider{Store: store}, uploader)

	workerCtx, workerCancel := context.WithCancel(context.Background())
	defer workerCancel()
	exporter.Start(workerCtx)

	var wg sync.WaitGroup
	var err1, err2 error

	wg.Add(2)
	// Caller 1 initiates Stop with final flush
	go func() {
		defer wg.Done()
		err1 = exporter.Stop(context.Background())
	}()

	// Wait until upload is in flight
	select {
	case <-flushStarted:
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for flush to start")
	}

	// Caller 2 initiates concurrent Stop
	go func() {
		defer wg.Done()
		err2 = exporter.Stop(context.Background())
	}()

	// Release upload hook so final flush completes
	close(flushBlocker)
	wg.Wait()

	assert.NoError(t, err1)
	assert.NoError(t, err2)
	assert.False(t, exporter.Running())

	// Verify trajectory was uploaded
	key, err := BuildObjectKey("trajectories/", traj)
	require.NoError(t, err)
	_, uploaded := uploader.getUpload(key)
	assert.True(t, uploaded)
}

func TestExporter_Stop_ConcurrentCallersShareWorkerTimeout(t *testing.T) {
	store, db := setupTestStore(t)
	defer func() { _ = db.Close() }()

	ctx := context.Background()
	now := time.Now().UTC()

	traj := &Trajectory{
		ID:             "traj-worker-timeout",
		RunID:          "run-wt",
		FrontendID:     "besedka",
		SessionID:      "townhall",
		ScopeID:        "townhall",
		ActorID:        "user-wt",
		PrivacyLevel:   "public",
		ExportApproved: true,
		CreatedAt:      now.Add(-20 * time.Minute),
		CompletedAt:    now.Add(-19 * time.Minute),
		UserInput:      "Worker timeout test",
	}
	require.NoError(t, store.InsertStaged(ctx, traj, 0))

	uploader := newMockUploader()
	uploadStarted := make(chan struct{})
	blockUpload := make(chan struct{})

	uploader.putHook = func(ctx context.Context, key string, r io.Reader, size int64) error {
		select {
		case <-uploadStarted:
		default:
			close(uploadStarted)
		}
		select {
		case <-blockUpload:
			return nil
		case <-time.After(200 * time.Millisecond):
			return nil
		}
	}

	cfg := ExporterConfig{
		Enabled:      true,
		Prefix:       "trajectories/",
		PollInterval: 1 * time.Hour,
	}
	exporter := NewExporter(cfg, SingleStoreProvider{Store: store}, uploader)

	workerCtx, workerCancel := context.WithCancel(context.Background())
	defer workerCancel()
	exporter.Start(workerCtx)

	// Wait for worker startup pass to enter upload
	select {
	case <-uploadStarted:
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for worker upload to start")
	}

	var wg sync.WaitGroup
	var err1, err2 error

	// Stop caller 1 with tight 20ms deadline (worker is stuck in 200ms upload)
	flushCtx1, cancel1 := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel1()
	// Stop caller 2 with longer deadline to wait on caller 1's completion
	flushCtx2, cancel2 := context.WithTimeout(context.Background(), 500*time.Millisecond)
	defer cancel2()

	wg.Add(2)
	go func() {
		defer wg.Done()
		err1 = exporter.Stop(flushCtx1)
	}()

	time.Sleep(5 * time.Millisecond)

	go func() {
		defer wg.Done()
		err2 = exporter.Stop(flushCtx2)
	}()

	wg.Wait()
	close(blockUpload)

	// Both callers should report context.DeadlineExceeded
	assert.ErrorIs(t, err1, context.DeadlineExceeded)
	assert.ErrorIs(t, err2, context.DeadlineExceeded)
}

