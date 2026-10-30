package gateway

import (
	"context"
	"testing"
	"time"

	"bob/internal/config"
	"bob/internal/memory"
	"bob/internal/models"
	"bob/internal/trajectory"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestGateway_HandleFeedbackCommand(t *testing.T) {
	tempDir := t.TempDir()
	cfg := config.Defaults()
	cfg.DataDir = tempDir

	memMgr := memory.NewManager(cfg, nil)
	defer func() { _ = memMgr.Close() }()

	storeProv := NewMemoryStoreProvider(memMgr, tempDir)

	ctx := context.Background()
	tStore, err := storeProv.GetTrajectoryStore(ctx, "townhall", false)
	require.NoError(t, err)

	now := time.Now().UTC()
	traj := &trajectory.Trajectory{
		ID:             "traj-gw-fb",
		RunID:          "run-gw",
		FrontendID:     "besedka",
		SessionID:      "townhall",
		ScopeID:        "townhall",
		ActorID:        "user-gw-1",
		ExecutionKind:  "interactive",
		PrivacyLevel:   "public",
		ExportApproved: true,
		CreatedAt:      now.Add(-10 * time.Minute),
		CompletedAt:    now.Add(-9 * time.Minute),
		UserInput:      "What is the weather?",
		FinalOutput:    "It is sunny.",
	}
	err = tStore.InsertStaged(ctx, traj, 15*time.Minute)
	require.NoError(t, err)

	gw := &Gateway{
		cfg:           cfg,
		storeProvider: storeProv,
		memoryManager: memMgr,
	}

	msg := models.Message{
		Seq:       101,
		Timestamp: now.Unix(),
		ChatID:    "townhall",
		UserID:    "user-gw-1",
		Content:   "/good Great answer!",
	}

	_ = gw.handleFeedbackCommand(ctx, msg, "/good Great answer!", "User GW", false)

	// Verify feedback was persisted in SQLite
	persisted, err := tStore.Get(ctx, traj.ID)
	require.NoError(t, err)
	assert.True(t, persisted.Breakdown.HasUserFeedback)
	assert.Equal(t, 1.0, persisted.Breakdown.UserFeedback)
	assert.Equal(t, "Great answer!", persisted.Breakdown.FeedbackNote)
	assert.Equal(t, 2, persisted.Revision)
}
