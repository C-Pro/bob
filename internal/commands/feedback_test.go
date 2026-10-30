package commands

import (
	"context"
	"database/sql"
	"testing"
	"time"

	"bob/internal/agentapi"
	"bob/internal/trajectory"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type mockTrajectoryResolver struct {
	store *trajectory.SQLiteStore
}

func (m *mockTrajectoryResolver) GetTrajectoryStore(ctx context.Context, chatID string, isDM bool) (*trajectory.SQLiteStore, error) {
	return m.store, nil
}

func setupTestFeedbackStore(t *testing.T) (*trajectory.SQLiteStore, *sql.DB) {
	t.Helper()
	db, err := sql.Open("sqlite", ":memory:")
	require.NoError(t, err)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	err = trajectory.EnsureSchema(ctx, db)
	require.NoError(t, err)

	store, err := trajectory.NewStore(db)
	require.NoError(t, err)

	return store, db
}

func TestFeedbackHandler_NilStores(t *testing.T) {
	h := NewFeedbackHandler(nil)
	ctx := context.Background()
	req := Request{
		Command: "good",
	}
	res, err := h.Handle(ctx, req)
	require.NoError(t, err)
	assert.Contains(t, res.Reply, "not configured")
}

func TestFeedbackHandler_GoodAndBad_LatestTarget(t *testing.T) {
	store, db := setupTestFeedbackStore(t)
	defer func() { _ = db.Close() }()

	ctx := context.Background()
	now := time.Now().UTC()

	// Stage an interactive trajectory
	traj := &trajectory.Trajectory{
		ID:             "traj-fb-1",
		RunID:          "run-1",
		FrontendID:     "besedka",
		SessionID:      "chat-123",
		ScopeID:        "chat-123",
		ActorID:        "user-abc",
		ExecutionKind:  "interactive",
		PrivacyLevel:   "public",
		ExportApproved: true,
		CreatedAt:      now.Add(-10 * time.Minute),
		CompletedAt:    now.Add(-9 * time.Minute),
		UserInput:      "Hello bot",
		FinalOutput:    "Hello human!",
	}
	err := store.InsertStaged(ctx, traj, 15*time.Minute)
	require.NoError(t, err)

	h := NewFeedbackHandler(&mockTrajectoryResolver{store: store})

	req := Request{
		MessageID: "msg-good-1",
		Session: agentapi.SessionRef{
			FrontendID: "besedka",
			SessionID:  "chat-123",
			ScopeID:    "chat-123",
		},
		Actor: agentapi.Actor{
			ID: "user-abc",
		},
		Command: "good",
		Args:    []string{"Great", "response!"},
	}

	res, err := h.Handle(ctx, req)
	require.NoError(t, err)
	assert.Contains(t, res.Reply, "👍 Feedback recorded")
	assert.Contains(t, res.Reply, "Great response!")

	// Verify rating was recorded
	updated, err := store.Get(ctx, traj.ID)
	require.NoError(t, err)
	assert.True(t, updated.Breakdown.HasUserFeedback)
	assert.Equal(t, 1.0, updated.Breakdown.UserFeedback)
	assert.Equal(t, "Great response!", updated.Breakdown.FeedbackNote)

	// Idempotent retry with same MessageID
	res2, err := h.Handle(ctx, req)
	require.NoError(t, err)
	assert.Contains(t, res2.Reply, "👍 Feedback recorded")

	// Submit /bad superseding rating with new message ID
	reqBad := Request{
		MessageID: "msg-bad-2",
		Session: agentapi.SessionRef{
			FrontendID: "besedka",
			SessionID:  "chat-123",
			ScopeID:    "chat-123",
		},
		Actor: agentapi.Actor{
			ID: "user-abc",
		},
		Command: "bad",
		Args:    []string{"Actually", "wrong"},
	}
	resBad, err := h.Handle(ctx, reqBad)
	require.NoError(t, err)
	assert.Contains(t, resBad.Reply, "👎 Feedback recorded")

	updatedBad, err := store.Get(ctx, traj.ID)
	require.NoError(t, err)
	assert.Equal(t, -1.0, updatedBad.Breakdown.UserFeedback)
	assert.Equal(t, "Actually wrong", updatedBad.Breakdown.FeedbackNote)
}

func TestFeedbackHandler_RateCommand(t *testing.T) {
	store, db := setupTestFeedbackStore(t)
	defer func() { _ = db.Close() }()

	ctx := context.Background()
	now := time.Now().UTC()

	traj := &trajectory.Trajectory{
		ID:             "traj-rate-explicit",
		RunID:          "run-2",
		FrontendID:     "besedka",
		SessionID:      "chat-456",
		ScopeID:        "chat-456",
		ActorID:        "user-xyz",
		ExecutionKind:  "interactive",
		PrivacyLevel:   "public",
		ExportApproved: true,
		CreatedAt:      now.Add(-10 * time.Minute),
		CompletedAt:    now.Add(-9 * time.Minute),
	}
	err := store.InsertStaged(ctx, traj, 15*time.Minute)
	require.NoError(t, err)

	h := NewFeedbackHandler(&mockTrajectoryResolver{store: store})

	t.Run("invalid usage", func(t *testing.T) {
		req := Request{
			Command: "rate",
			Args:    nil,
		}
		res, err := h.Handle(ctx, req)
		require.NoError(t, err)
		assert.Contains(t, res.Reply, "Usage:")
	})

	t.Run("invalid rating string", func(t *testing.T) {
		req := Request{
			Command: "rate",
			Args:    []string{"not-a-number"},
		}
		res, err := h.Handle(ctx, req)
		require.NoError(t, err)
		assert.Contains(t, res.Reply, "Invalid rating value")
	})

	t.Run("explicit target and neutral score", func(t *testing.T) {
		req := Request{
			MessageID: "msg-rate-0",
			Session: agentapi.SessionRef{
				FrontendID: "besedka",
				SessionID:  "chat-456",
				ScopeID:    "chat-456",
			},
			Actor: agentapi.Actor{
				ID: "user-xyz",
			},
			Command: "rate",
			Args:    []string{"0", "traj-rate-explicit", "neutral", "note"},
		}
		res, err := h.Handle(ctx, req)
		require.NoError(t, err)
		assert.Contains(t, res.Reply, "👌 Neutral feedback recorded")

		updated, err := store.Get(ctx, traj.ID)
		require.NoError(t, err)
		assert.True(t, updated.Breakdown.HasUserFeedback)
		assert.Equal(t, 0.0, updated.Breakdown.UserFeedback)
		assert.Equal(t, "neutral note", updated.Breakdown.FeedbackNote)
	})
}

func TestFeedbackHandler_NotFoundAndExpired(t *testing.T) {
	store, db := setupTestFeedbackStore(t)
	defer func() { _ = db.Close() }()

	ctx := context.Background()
	h := NewFeedbackHandler(&mockTrajectoryResolver{store: store})

	// No trajectories in store
	req := Request{
		Session: agentapi.SessionRef{
			FrontendID: "besedka",
			SessionID:  "empty-chat",
			ScopeID:    "empty-chat",
		},
		Actor: agentapi.Actor{
			ID: "user-none",
		},
		Command: "good",
	}
	res, err := h.Handle(ctx, req)
	require.NoError(t, err)
	assert.Contains(t, res.Reply, "No recent assistant turn found")

	// Target expired / pruned trajectory
	traj := &trajectory.Trajectory{
		ID:             "traj-pruned-test",
		RunID:          "run-p",
		FrontendID:     "besedka",
		SessionID:      "prune-chat",
		ScopeID:        "prune-chat",
		ActorID:        "user-p",
		ExecutionKind:  "interactive",
		PrivacyLevel:   "public",
		ExportApproved: true,
		CreatedAt:      time.Now().UTC().Add(-48 * time.Hour),
		CompletedAt:    time.Now().UTC().Add(-47 * time.Hour),
	}
	err = store.InsertStaged(ctx, traj, 0)
	require.NoError(t, err)

	err = store.MarkExported(ctx, traj.ID, 1, time.Now().UTC().Add(-2*time.Hour))
	require.NoError(t, err)

	prunedCount, err := store.PruneExported(ctx, 1*time.Hour)
	require.NoError(t, err)
	assert.Equal(t, int64(1), prunedCount)

	reqPruned := Request{
		Session: agentapi.SessionRef{
			FrontendID: "besedka",
			SessionID:  "prune-chat",
			ScopeID:    "prune-chat",
		},
		Actor: agentapi.Actor{
			ID: "user-p",
		},
		Command: "good",
		Args:    []string{"traj-pruned-test"},
	}
	resPruned, err := h.Handle(ctx, reqPruned)
	require.NoError(t, err)
	assert.Contains(t, resPruned.Reply, "expired from local history")
}
