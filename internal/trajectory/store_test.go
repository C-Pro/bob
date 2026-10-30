package trajectory

import (
	"context"
	"database/sql"
	"encoding/json"
	"path/filepath"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	_ "modernc.org/sqlite"
)

func setupTestDB(t *testing.T) (*sql.DB, *SQLiteStore) {
	t.Helper()
	dbPath := filepath.Join(t.TempDir(), "test_trajectory.db")
	db, err := sql.Open("sqlite", "file:"+dbPath+"?_pragma=journal_mode(WAL)&_pragma=foreign_keys(ON)&_pragma=busy_timeout(5000)")
	require.NoError(t, err)

	store, err := NewSQLiteStore(db, true)
	require.NoError(t, err)

	t.Cleanup(func() {
		_ = store.Close()
	})
	return db, store
}

func TestStore_RecordEventTx_AtomicAndConflict(t *testing.T) {
	db, store := setupTestDB(t)
	ctx := context.Background()

	trajID := uuid.NewString()
	runID := "run-1"

	// 1. Rollback test
	tx, err := db.BeginTx(ctx, nil)
	require.NoError(t, err)

	ev := &Event{
		TrajectoryID: trajID,
		RunID:        runID,
		Generation:   1,
		Iteration:    1,
		Phase:        "loop",
		StepID:       "step-1",
		Attempt:      1,
		EventType:    "tool_attempt",
		Payload:      json.RawMessage(`{"action":"web_search"}`),
	}
	err = store.RecordEventTx(ctx, tx, ev)
	require.NoError(t, err)
	require.NoError(t, tx.Rollback())

	var count int
	err = db.QueryRow("SELECT count(*) FROM agent_trajectory_events WHERE trajectory_id = ?", trajID).Scan(&count)
	require.NoError(t, err)
	assert.Equal(t, 0, count)

	// 2. Commit test
	tx2, err := db.BeginTx(ctx, nil)
	require.NoError(t, err)
	err = store.RecordEventTx(ctx, tx2, ev)
	require.NoError(t, err)
	require.NoError(t, tx2.Commit())

	err = db.QueryRow("SELECT count(*) FROM agent_trajectory_events WHERE trajectory_id = ?", trajID).Scan(&count)
	require.NoError(t, err)
	assert.Equal(t, 1, count)

	// 3. Replay with identical payload: should be idempotent no-op
	tx3, err := db.BeginTx(ctx, nil)
	require.NoError(t, err)
	err = store.RecordEventTx(ctx, tx3, ev)
	require.NoError(t, err)
	require.NoError(t, tx3.Commit())

	// 4. Replay with conflicting payload: should return error
	conflictingEv := *ev
	conflictingEv.Payload = json.RawMessage(`{"action":"web_search_modified"}`)
	tx4, err := db.BeginTx(ctx, nil)
	require.NoError(t, err)
	err = store.RecordEventTx(ctx, tx4, &conflictingEv)
	assert.ErrorContains(t, err, "conflict")
	_ = tx4.Rollback()
}

func TestStore_InsertStaged_InitialAndStaleFinalizer(t *testing.T) {
	_, store := setupTestDB(t)
	ctx := context.Background()

	trajID := uuid.NewString()
	now := time.Now().UTC()

	traj := &Trajectory{
		ID:             trajID,
		RunID:          "run-100",
		FrontendID:     "besedka",
		SessionID:      "chat-1",
		ScopeID:        "townhall",
		ActorID:        "user-1",
		Model:          "gemini-2.5-flash",
		ExecutionKind:  "interactive",
		PrivacyLevel:   "public",
		ExportApproved: true,
		Generation:     2,
		CreatedAt:      now,
		CompletedAt:    now,
		UserInput:      "What is the weather?",
		FinalOutput:    "It is sunny.",
		Breakdown: EvalBreakdown{
			TotalToolCalls:      1,
			SuccessfulToolCalls: 1,
			TerminationReason:   "completed",
		},
	}

	// 1. Initial insert
	err := store.InsertStaged(ctx, traj, 15*time.Minute)
	require.NoError(t, err)

	stored, err := store.Get(ctx, trajID)
	require.NoError(t, err)
	assert.Equal(t, trajID, stored.ID)
	assert.Equal(t, 1, stored.Revision)
	assert.Equal(t, 2, stored.Generation)
	assert.InDelta(t, 0.40, stored.Reward, 0.0001)

	// 2. Stale finalizer arrives with Generation 1 (should be rejected as no-op)
	stale := &Trajectory{
		ID:             trajID,
		RunID:          "run-100",
		FrontendID:     "besedka",
		SessionID:      "chat-1",
		ScopeID:        "townhall",
		ActorID:        "user-1",
		Model:          "gemini-2.5-flash",
		ExecutionKind:  "interactive",
		PrivacyLevel:   "public",
		ExportApproved: true,
		Generation:     1, // older!
		CreatedAt:      now,
		CompletedAt:    now,
		UserInput:      "What is the weather?",
		FinalOutput:    "STALE ERROR",
		Breakdown: EvalBreakdown{
			FSMFailed:         true,
			TerminationReason: "fsm_failed",
		},
	}
	err = store.InsertStaged(ctx, stale, 15*time.Minute)
	require.NoError(t, err)

	afterStale, err := store.Get(ctx, trajID)
	require.NoError(t, err)
	assert.Equal(t, "It is sunny.", afterStale.FinalOutput) // unmolested
	assert.Equal(t, 2, afterStale.Generation)
	assert.Equal(t, 1, afterStale.Revision)
}

func TestStore_InsertStaged_ChangedStepsBumpsRevision(t *testing.T) {
	_, store := setupTestDB(t)
	ctx := context.Background()

	trajID := uuid.NewString()
	now := time.Now().UTC()

	exitZero := 0
	exitOne := 1

	traj := &Trajectory{
		ID:             trajID,
		RunID:          "run-1",
		FrontendID:     "besedka",
		SessionID:      "chat-1",
		ScopeID:        "townhall",
		ActorID:        "user-1",
		Model:          "gemini",
		ExecutionKind:  "interactive",
		PrivacyLevel:   "public",
		ExportApproved: true,
		Generation:     1,
		CreatedAt:      now,
		CompletedAt:    now,
		UserInput:      "Hello",
		FinalOutput:    "Hi there",
		Steps: []Step{
			{Index: 0, Action: "ping", ExitCode: &exitZero, Observation: "pong"},
		},
		Breakdown: EvalBreakdown{
			TotalToolCalls:      1,
			SuccessfulToolCalls: 1,
			TerminationReason:   "completed",
		},
	}

	require.NoError(t, store.InsertStaged(ctx, traj, 15*time.Minute))
	firstGet, err := store.Get(ctx, trajID)
	require.NoError(t, err)
	assert.Equal(t, 1, firstGet.Revision)

	// Same number of steps, but step content/exitCode changed -> must bump revision to 2
	changedTraj := *traj
	changedTraj.Steps = []Step{
		{Index: 0, Action: "ping", ExitCode: &exitOne, Observation: "error: timeout"},
	}
	changedTraj.Breakdown.SuccessfulToolCalls = 0
	changedTraj.Breakdown.FailedToolCalls = 1

	require.NoError(t, store.InsertStaged(ctx, &changedTraj, 15*time.Minute))
	secondGet, err := store.Get(ctx, trajID)
	require.NoError(t, err)
	assert.Equal(t, 2, secondGet.Revision)
	assert.Equal(t, "error: timeout", secondGet.Steps[0].Observation)
}

func TestStore_InsertStaged_FallbackAfterExport(t *testing.T) {
	_, store := setupTestDB(t)
	ctx := context.Background()

	trajID := uuid.NewString()
	now := time.Now().UTC()

	// 1. Initial execution outcome (e.g. FSM failed, Generation 1)
	initial := &Trajectory{
		ID:             trajID,
		RunID:          "run-fail",
		FrontendID:     "besedka",
		SessionID:      "chat-1",
		ScopeID:        "townhall",
		ActorID:        "user-1",
		Model:          "gemini",
		ExecutionKind:  "interactive",
		PrivacyLevel:   "public",
		ExportApproved: true,
		Generation:     1,
		CreatedAt:      now,
		CompletedAt:    now,
		UserInput:      "Calculate 2+2",
		FinalOutput:    "Tool execution failed",
		Breakdown: EvalBreakdown{
			TotalToolCalls:      1,
			SuccessfulToolCalls: 0,
			FailedToolCalls:     1,
			FSMFailed:           true,
			TerminationReason:   "fsm_failed",
		},
	}
	require.NoError(t, store.InsertStaged(ctx, initial, 15*time.Minute))

	// 2. User feedback arrives while staged
	feedbackReq := FeedbackRequest{
		MessageID:  "msg-fb-1",
		FrontendID: "besedka",
		SessionID:  "chat-1",
		ScopeID:    "townhall",
		ActorID:    "user-1",
		Rating:     -1.0,
		Note:       "It failed",
	}
	withFb, err := store.UpdateFeedback(ctx, feedbackReq)
	require.NoError(t, err)
	assert.Equal(t, 2, withFb.Revision)
	assert.True(t, withFb.Breakdown.HasUserFeedback)

	// 3. Exporter uploads revision 2
	require.NoError(t, store.MarkExported(ctx, trajID, 2, now))

	// 4. Fallback execution finishes (Generation 2) with successful direct answer
	fallback := &Trajectory{
		ID:             trajID,
		RunID:          "run-fail",
		FrontendID:     "besedka",
		SessionID:      "chat-1",
		ScopeID:        "townhall",
		ActorID:        "user-1",
		Model:          "gemini",
		ExecutionKind:  "interactive",
		PrivacyLevel:   "public",
		ExportApproved: true,
		Generation:     2,
		CreatedAt:      now,
		CompletedAt:    time.Now().UTC(),
		UserInput:      "Calculate 2+2",
		FinalOutput:    "2 + 2 = 4 (fallback answer)",
		Breakdown: EvalBreakdown{
			TotalToolCalls:    0,
			FSMFailed:         false,
			TerminationReason: "completed",
		},
	}
	require.NoError(t, store.InsertStaged(ctx, fallback, 15*time.Minute))

	// 5. Verify feedback survives (-1.0), revision is 3, status is staged
	res, err := store.Get(ctx, trajID)
	require.NoError(t, err)
	assert.Equal(t, 3, res.Revision)
	assert.True(t, res.Breakdown.HasUserFeedback)
	assert.Equal(t, -1.0, res.Breakdown.UserFeedback)
	assert.Equal(t, "2 + 2 = 4 (fallback answer)", res.FinalOutput)

	// Check staged in DB
	ready, err := store.ListReadyForExport(ctx, time.Now().UTC().Add(20*time.Minute), ListExportOptions{Limit: 10})
	require.NoError(t, err)
	require.Len(t, ready, 1)
	assert.Equal(t, 3, ready[0].Revision)
}

func TestStore_UpdateFeedback_ScopeAndActorAuthorization(t *testing.T) {
	_, store := setupTestDB(t)
	ctx := context.Background()

	trajID := uuid.NewString()
	now := time.Now().UTC()

	traj := &Trajectory{
		ID:             trajID,
		RunID:          "run-2",
		FrontendID:     "besedka",
		SessionID:      "chat-1",
		ScopeID:        "townhall",
		ActorID:        "alice",
		Model:          "gemini",
		ExecutionKind:  "interactive",
		PrivacyLevel:   "public",
		ExportApproved: true,
		Generation:     1,
		CreatedAt:      now,
		CompletedAt:    now,
		UserInput:      "Help me",
		FinalOutput:    "Here is help",
		Breakdown: EvalBreakdown{
			TotalToolCalls:    0,
			TerminationReason: "completed",
		},
	}
	require.NoError(t, store.InsertStaged(ctx, traj, 15*time.Minute))

	// 1. Cross-scope attack (different scope): should be rejected
	reqCrossScope := FeedbackRequest{
		MessageID:  "msg-cross-scope",
		FrontendID: "besedka",
		SessionID:  "chat-1",
		ScopeID:    "wrong-channel",
		ActorID:    "alice",
		TrajectoryID: trajID,
		Rating:     1.0,
	}
	_, err := store.UpdateFeedback(ctx, reqCrossScope)
	assert.ErrorIs(t, err, ErrUnauthorized)

	// 2. Cross-actor unauthorized rating: should be rejected
	reqCrossActor := FeedbackRequest{
		MessageID:  "msg-cross-actor",
		FrontendID: "besedka",
		SessionID:  "chat-1",
		ScopeID:    "townhall",
		ActorID:    "mallory",
		TrajectoryID: trajID,
		Rating:     -1.0,
	}
	_, err = store.UpdateFeedback(ctx, reqCrossActor)
	assert.ErrorIs(t, err, ErrUnauthorized)

	// 3. Valid feedback from alice
	reqValid := FeedbackRequest{
		MessageID:  "msg-valid-1",
		FrontendID: "besedka",
		SessionID:  "chat-1",
		ScopeID:    "townhall",
		ActorID:    "alice",
		TrajectoryID: trajID,
		Rating:     1.0,
	}
	res, err := store.UpdateFeedback(ctx, reqValid)
	require.NoError(t, err)
	assert.Equal(t, 2, res.Revision)
}

func TestStore_MarkExported_FeedbackRaceSequence(t *testing.T) {
	_, store := setupTestDB(t)
	ctx := context.Background()

	trajID := uuid.NewString()
	now := time.Now().UTC()

	traj := &Trajectory{
		ID:             trajID,
		RunID:          "run-race",
		FrontendID:     "besedka",
		SessionID:      "chat-1",
		ScopeID:        "townhall",
		ActorID:        "alice",
		Model:          "gemini",
		ExecutionKind:  "interactive",
		PrivacyLevel:   "public",
		ExportApproved: true,
		Generation:     1,
		CreatedAt:      now,
		CompletedAt:    now,
		UserInput:      "Test race",
		FinalOutput:    "Response",
		Breakdown: EvalBreakdown{
			TotalToolCalls:    0,
			TerminationReason: "completed",
		},
	}
	require.NoError(t, store.InsertStaged(ctx, traj, 0))

	// Step 1: Exporter reads ready list, fetches revision 1
	ready, err := store.ListReadyForExport(ctx, now.Add(1*time.Minute), ListExportOptions{Limit: 10})
	require.NoError(t, err)
	require.Len(t, ready, 1)
	assert.Equal(t, 1, ready[0].Revision)

	// Step 2: User submits feedback while exporter is uploading revision 1
	// This bumps revision to 2 in DB
	_, err = store.UpdateFeedback(ctx, FeedbackRequest{
		MessageID:  "msg-race-fb",
		FrontendID: "besedka",
		SessionID:  "chat-1",
		ScopeID:    "townhall",
		ActorID:    "alice",
		TrajectoryID: trajID,
		Rating:     1.0,
	})
	require.NoError(t, err)

	// Step 3: Exporter finishes upload of revision 1 and calls MarkExported(id, 1)
	err = store.MarkExported(ctx, trajID, 1, now)
	require.NoError(t, err)

	// Step 4: Verify status REMAINS 'staged' because revision in DB is 2!
	// And exported_revision became 1.
	stored, err := store.Get(ctx, trajID)
	require.NoError(t, err)
	assert.Equal(t, 2, stored.Revision)

	readyAfter, err := store.ListReadyForExport(ctx, time.Now().UTC().Add(5*time.Minute), ListExportOptions{Limit: 10})
	require.NoError(t, err)
	require.Len(t, readyAfter, 1)
	assert.Equal(t, 2, readyAfter[0].Revision)

	// Step 5: Exporter uploads revision 2 and acknowledges MarkExported(id, 2)
	err = store.MarkExported(ctx, trajID, 2, now)
	require.NoError(t, err)

	// Now it is fully exported
	readyFinal, err := store.ListReadyForExport(ctx, time.Now().UTC().Add(10*time.Minute), ListExportOptions{Limit: 10})
	require.NoError(t, err)
	assert.Empty(t, readyFinal)
}

func TestStore_ReconcileTerminalRuns(t *testing.T) {
	_, store := setupTestDB(t)
	ctx := context.Background()

	trajID := uuid.NewString()
	runID := "run-reconcile"
	now := time.Now().UTC()

	// 1. Record turn_start event
	startPayload, err := json.Marshal(TurnStartPayload{
		RunID:          runID,
		FrontendID:     "besedka",
		SessionID:      "chat-reconcile",
		ScopeID:        "townhall",
		ActorID:        "bob_user",
		Model:          "gemini-2.5-flash",
		ExecutionKind:  "interactive",
		PrivacyLevel:   "public",
		ExportApproved: true,
		Generation:     1,
		Input:          "Recover me",
		TaskState:      "task context",
	})
	require.NoError(t, err)

	require.NoError(t, store.RecordEvent(ctx, &Event{
		TrajectoryID: trajID,
		RunID:        runID,
		Generation:   1,
		Iteration:    0,
		Phase:        "start",
		EventType:    "turn_start",
		Payload:      startPayload,
		CreatedAt:    now,
	}))

	// 2. Running turn without terminal event: should NOT be reconciled yet
	require.NoError(t, store.ReconcileTerminalRuns(ctx))
	_, err = store.Get(ctx, trajID)
	assert.ErrorIs(t, err, ErrTrajectoryNotFound)

	// 3. Record tool attempt
	stepPayload, err := json.Marshal(Step{
		Index:       0,
		Action:      "search",
		Observation: "found 1 item",
	})
	require.NoError(t, err)
	require.NoError(t, store.RecordEvent(ctx, &Event{
		TrajectoryID: trajID,
		RunID:        runID,
		Generation:   1,
		Iteration:    1,
		Phase:        "loop",
		EventType:    "tool_attempt",
		Payload:      stepPayload,
		CreatedAt:    now.Add(1 * time.Second),
	}))

	// 4. Record terminal event
	termPayload, err := json.Marshal(TurnTerminalPayload{
		Generation: 1,
		Reason:     "completed",
		Failed:     false,
		FinalReply: "Here is your recovery",
	})
	require.NoError(t, err)
	require.NoError(t, store.RecordEvent(ctx, &Event{
		TrajectoryID: trajID,
		RunID:        runID,
		Generation:   1,
		Iteration:    1,
		Phase:        "terminal",
		EventType:    "turn_terminal",
		Payload:      termPayload,
		CreatedAt:    now.Add(2 * time.Second),
	}))

	// 5. Reconcile terminal runs: should materialize trajectory
	require.NoError(t, store.ReconcileTerminalRuns(ctx))

	reconciled, err := store.Get(ctx, trajID)
	require.NoError(t, err)
	assert.Equal(t, trajID, reconciled.ID)
	assert.Equal(t, "chat-reconcile", reconciled.SessionID)
	assert.Equal(t, "Recover me", reconciled.UserInput)
	assert.Equal(t, "Here is your recovery", reconciled.FinalOutput)
	assert.Len(t, reconciled.Steps, 1)
	assert.Equal(t, 1, reconciled.Revision)
	assert.Equal(t, 1, reconciled.Generation)
}

func TestStore_PruneExported_PreventsResurrection(t *testing.T) {
	_, store := setupTestDB(t)
	ctx := context.Background()

	trajID := uuid.NewString()
	now := time.Now().UTC().Add(-2 * time.Hour)

	traj := &Trajectory{
		ID:             trajID,
		RunID:          "run-prune",
		FrontendID:     "besedka",
		SessionID:      "chat-1",
		ScopeID:        "townhall",
		ActorID:        "alice",
		Model:          "gemini",
		ExecutionKind:  "interactive",
		PrivacyLevel:   "public",
		ExportApproved: true,
		Generation:     1,
		CreatedAt:      now,
		CompletedAt:    now,
		UserInput:      "Prune me",
		FinalOutput:    "Pruned response",
		Breakdown: EvalBreakdown{
			TotalToolCalls:    0,
			TerminationReason: "completed",
		},
	}
	require.NoError(t, store.InsertStaged(ctx, traj, 10*time.Minute))

	// Mark exported 1 hour ago
	require.NoError(t, store.MarkExported(ctx, trajID, 1, now.Add(30*time.Minute)))

	// Prune trajectories older than 30 minutes
	prunedCount, err := store.PruneExported(ctx, 30*time.Minute)
	require.NoError(t, err)
	assert.Equal(t, int64(1), prunedCount)

	// 1. Get returns ErrTrajectoryExpired
	_, err = store.Get(ctx, trajID)
	assert.ErrorIs(t, err, ErrTrajectoryExpired)

	// 2. Feedback returns ErrTrajectoryExpired
	_, err = store.UpdateFeedback(ctx, FeedbackRequest{
		MessageID:    "msg-late",
		FrontendID:   "besedka",
		SessionID:    "chat-1",
		ScopeID:      "townhall",
		ActorID:      "alice",
		TrajectoryID: trajID,
		Rating:       1.0,
	})
	assert.ErrorIs(t, err, ErrTrajectoryExpired)

	// 3. Late InsertStaged does not resurrect
	require.NoError(t, store.InsertStaged(ctx, traj, 10*time.Minute))
	_, err = store.Get(ctx, trajID)
	assert.ErrorIs(t, err, ErrTrajectoryExpired)

	// 4. ReconcileTerminalRuns does not resurrect
	require.NoError(t, store.ReconcileTerminalRuns(ctx))
	_, err = store.Get(ctx, trajID)
	assert.ErrorIs(t, err, ErrTrajectoryExpired)
}

func TestStore_ListReadyForExport_Privacy(t *testing.T) {
	_, store := setupTestDB(t)
	ctx := context.Background()

	now := time.Now().UTC()

	// 1. Public approved
	t1 := &Trajectory{
		ID:             "traj-pub",
		RunID:          "run-pub",
		FrontendID:     "besedka",
		SessionID:      "chat-1",
		ScopeID:        "townhall",
		ActorID:        "user-1",
		PrivacyLevel:   "public",
		ExportApproved: true,
		CreatedAt:      now,
		CompletedAt:    now,
		UserInput:      "Public prompt",
		FinalOutput:    "Public reply",
	}
	require.NoError(t, store.InsertStaged(ctx, t1, 0))

	// 2. Private approved
	t2 := &Trajectory{
		ID:             "traj-priv",
		RunID:          "run-priv",
		FrontendID:     "besedka",
		SessionID:      "chat-2",
		ScopeID:        "dm",
		ActorID:        "user-2",
		PrivacyLevel:   "private",
		ExportApproved: true,
		CreatedAt:      now,
		CompletedAt:    now,
		UserInput:      "Private prompt",
		FinalOutput:    "Private reply",
	}
	require.NoError(t, store.InsertStaged(ctx, t2, 0))

	// Query with AllowPrivate = false -> only public
	listPub, err := store.ListReadyForExport(ctx, now.Add(1*time.Minute), ListExportOptions{AllowPrivate: false})
	require.NoError(t, err)
	require.Len(t, listPub, 1)
	assert.Equal(t, "traj-pub", listPub[0].ID)

	// Query with AllowPrivate = true -> both public and private
	listAll, err := store.ListReadyForExport(ctx, now.Add(1*time.Minute), ListExportOptions{AllowPrivate: true})
	require.NoError(t, err)
	require.Len(t, listAll, 2)
}

func TestStore_ReconcileTerminalRuns_PreservesFeedbackAndIncrementsRevision(t *testing.T) {
	db, store := setupTestDB(t)
	ctx := context.Background()

	trajID := uuid.NewString()
	runID := "run-feedback-reconcile"
	now := time.Now().UTC()

	// 1. Record turn_start event
	startPayload, err := json.Marshal(TurnStartPayload{
		RunID:          runID,
		FrontendID:     "besedka",
		SessionID:      "chat-fb-rec",
		ScopeID:        "townhall",
		ActorID:        "user-fb",
		Model:          "gemini-2.5-flash",
		ExecutionKind:  "interactive",
		PrivacyLevel:   "public",
		ExportApproved: true,
		Generation:     1,
		Input:          "Initial question",
		TaskState:      "task context",
	})
	require.NoError(t, err)

	require.NoError(t, store.RecordEvent(ctx, &Event{
		TrajectoryID: trajID,
		RunID:        runID,
		Generation:   1,
		Iteration:    0,
		Phase:        "start",
		EventType:    "turn_start",
		Payload:      startPayload,
		CreatedAt:    now,
	}))

	// 2. Record initial terminal event
	termPayload, err := json.Marshal(TurnTerminalPayload{
		Generation: 1,
		Reason:     "completed",
		Failed:     false,
		FinalReply: "Initial answer",
	})
	require.NoError(t, err)
	require.NoError(t, store.RecordEvent(ctx, &Event{
		TrajectoryID: trajID,
		RunID:        runID,
		Generation:   1,
		Iteration:    1,
		Phase:        "terminal",
		EventType:    "turn_terminal",
		Payload:      termPayload,
		CreatedAt:    now.Add(1 * time.Second),
	}))

	// 3. Reconcile: creates revision 1
	require.NoError(t, store.ReconcileTerminalRuns(ctx))
	t1, err := store.Get(ctx, trajID)
	require.NoError(t, err)
	assert.Equal(t, 1, t1.Revision)
	assert.False(t, t1.Breakdown.HasUserFeedback)

	// 4. User provides positive feedback via /good -> revision 2
	t2, err := store.UpdateFeedback(ctx, FeedbackRequest{
		MessageID:  "msg-fb-1",
		FrontendID: "besedka",
		SessionID:  "chat-fb-rec",
		ScopeID:    "townhall",
		ActorID:    "user-fb",
		Rating:     1.0,
		Note:       "Great response!",
	})
	require.NoError(t, err)
	assert.Equal(t, 2, t2.Revision)
	assert.True(t, t2.Breakdown.HasUserFeedback)
	assert.Equal(t, 1.0, t2.Breakdown.UserFeedback)

	// Verify SQL revision is 2
	var sqlRev int
	err = db.QueryRowContext(ctx, "SELECT revision FROM agent_trajectories WHERE id = ?", trajID).Scan(&sqlRev)
	require.NoError(t, err)
	assert.Equal(t, 2, sqlRev)

	// 5. Subsequent delivery terminal event arrives after completed_at
	termPayload2, err := json.Marshal(TurnTerminalPayload{
		Generation: 1,
		Reason:     "delivery_failed",
		Failed:     true,
		FinalReply: "Initial answer",
	})
	require.NoError(t, err)
	require.NoError(t, store.RecordEvent(ctx, &Event{
		TrajectoryID: trajID,
		RunID:        runID,
		Generation:   1,
		Iteration:    2,
		Phase:        "delivery",
		EventType:    "turn_terminal",
		Payload:      termPayload2,
		CreatedAt:    now.Add(10 * time.Second),
	}))

	// 6. Reconcile again: must merge feedback and advance to revision 3
	require.NoError(t, store.ReconcileTerminalRuns(ctx))
	t3, err := store.Get(ctx, trajID)
	require.NoError(t, err)

	// Assert feedback was NOT erased
	assert.True(t, t3.Breakdown.HasUserFeedback)
	assert.Equal(t, 1.0, t3.Breakdown.UserFeedback)
	assert.Equal(t, "Great response!", t3.Breakdown.FeedbackNote)
	assert.Equal(t, "delivery_failed", t3.Breakdown.TerminationReason)
	assert.True(t, t3.Breakdown.FSMFailed)

	// Assert SQL revision and JSON payload revision equality
	assert.Equal(t, 3, t3.Revision)
	err = db.QueryRowContext(ctx, "SELECT revision FROM agent_trajectories WHERE id = ?", trajID).Scan(&sqlRev)
	require.NoError(t, err)
	assert.Equal(t, 3, sqlRev)

	// 7. Repeated reconciliation without new events: must be an unchanged no-op
	require.NoError(t, store.ReconcileTerminalRuns(ctx))
	t4, err := store.Get(ctx, trajID)
	require.NoError(t, err)
	assert.Equal(t, 3, t4.Revision)
	err = db.QueryRowContext(ctx, "SELECT revision FROM agent_trajectories WHERE id = ?", trajID).Scan(&sqlRev)
	require.NoError(t, err)
	assert.Equal(t, 3, sqlRev)
}

func TestStore_ConfiguredStagingDelay(t *testing.T) {
	db, store := setupTestDB(t)
	ctx := context.Background()

	// Configure custom staging delay
	customDelay := 45 * time.Minute
	store.SetStagingDelay(customDelay)

	trajID := uuid.NewString()
	runID := "run-delay-test"
	now := time.Now().UTC()

	startPayload, err := json.Marshal(TurnStartPayload{
		RunID:          runID,
		FrontendID:     "besedka",
		SessionID:      "chat-delay",
		ScopeID:        "townhall",
		ActorID:        "user-delay",
		Model:          "gemini-2.5-flash",
		ExecutionKind:  "interactive",
		PrivacyLevel:   "public",
		ExportApproved: true,
		Generation:     1,
		Input:          "Delay test",
	})
	require.NoError(t, err)

	require.NoError(t, store.RecordEvent(ctx, &Event{
		TrajectoryID: trajID,
		RunID:        runID,
		Generation:   1,
		Phase:        "start",
		EventType:    "turn_start",
		Payload:      startPayload,
		CreatedAt:    now,
	}))

	termPayload, err := json.Marshal(TurnTerminalPayload{
		Generation: 1,
		Reason:     "completed",
		Failed:     false,
		FinalReply: "Done",
	})
	require.NoError(t, err)

	require.NoError(t, store.RecordEvent(ctx, &Event{
		TrajectoryID: trajID,
		RunID:        runID,
		Generation:   1,
		Phase:        "terminal",
		EventType:    "turn_terminal",
		Payload:      termPayload,
		CreatedAt:    now.Add(1 * time.Second),
	}))

	require.NoError(t, store.ReconcileTerminalRuns(ctx))

	var eligibleAtStr string
	err = db.QueryRowContext(ctx, "SELECT eligible_at FROM agent_trajectories WHERE id = ?", trajID).Scan(&eligibleAtStr)
	require.NoError(t, err)

	eligibleAt, err := time.Parse(timeLayout, eligibleAtStr)
	require.NoError(t, err)

	// Eligible at should be at least now + 40m (honoring the 45m delay, not hardcoded 15m)
	assert.True(t, eligibleAt.After(now.Add(40*time.Minute)), "expected eligible_at to honor 45m delay, got: %v", eligibleAt)
}

func TestStore_InsertStaged_PreservesFeedbackAndNoopOnIdenticalReplay(t *testing.T) {
	db, store := setupTestDB(t)
	ctx := context.Background()

	trajID := uuid.NewString()
	now := time.Now().UTC()

	// 1. Initial InsertStaged -> Revision 1
	traj := &Trajectory{
		ID:             trajID,
		RunID:          "run-replay-fb",
		FrontendID:     "besedka",
		SessionID:      "chat-replay",
		ScopeID:        "townhall",
		ActorID:        "user-r",
		Model:          "gemini-2.5-flash",
		ExecutionKind:  "interactive",
		PrivacyLevel:   "public",
		ExportApproved: true,
		Generation:     1,
		CreatedAt:      now.Add(-10 * time.Minute),
		CompletedAt:    now.Add(-9 * time.Minute),
		UserInput:      "Replay test",
		FinalOutput:    "Answer 1",
		Steps: []Step{
			{Index: 0, Action: "tool_a", Observation: "obs_a"},
		},
		Breakdown: EvalBreakdown{
			TotalToolCalls:      1,
			SuccessfulToolCalls: 1,
			ToolSuccessRate:     1.0,
		},
	}
	require.NoError(t, store.InsertStaged(ctx, traj, 15*time.Minute))

	// 2. User feedback arrives -> Revision 2
	t2, err := store.UpdateFeedback(ctx, FeedbackRequest{
		MessageID:  "msg-replay-1",
		FrontendID: "besedka",
		SessionID:  "chat-replay",
		ScopeID:    "townhall",
		ActorID:    "user-r",
		Rating:     1.0,
		Note:       "Great replay test!",
	})
	require.NoError(t, err)
	assert.Equal(t, 2, t2.Revision)
	assert.True(t, t2.Breakdown.HasUserFeedback)

	var feedbackEligibleAt string
	err = db.QueryRowContext(ctx, "SELECT eligible_at FROM agent_trajectories WHERE id = ?", trajID).Scan(&feedbackEligibleAt)
	require.NoError(t, err)

	// 3. Stale/identical execution replay (e.g. finalizer re-executes with original traj without feedback)
	trajReplay := &Trajectory{
		ID:             trajID,
		RunID:          "run-replay-fb",
		FrontendID:     "besedka",
		SessionID:      "chat-replay",
		ScopeID:        "townhall",
		ActorID:        "user-r",
		Model:          "gemini-2.5-flash",
		ExecutionKind:  "interactive",
		PrivacyLevel:   "public",
		ExportApproved: true,
		Generation:     1,
		CreatedAt:      now.Add(-10 * time.Minute),
		CompletedAt:    now.Add(-9 * time.Minute),
		UserInput:      "Replay test",
		FinalOutput:    "Answer 1",
		Steps: []Step{
			{Index: 0, Action: "tool_a", Observation: "obs_a"},
		},
		Breakdown: EvalBreakdown{
			TotalToolCalls:      1,
			SuccessfulToolCalls: 1,
			ToolSuccessRate:     1.0,
		},
	}
	require.NoError(t, store.InsertStaged(ctx, trajReplay, 15*time.Minute))

	// Assert unchanged revision (still 2, not incremented to 3)
	tAfter, err := store.Get(ctx, trajID)
	require.NoError(t, err)
	assert.Equal(t, 2, tAfter.Revision)
	assert.True(t, tAfter.Breakdown.HasUserFeedback)
	assert.Equal(t, 1.0, tAfter.Breakdown.UserFeedback)
	assert.Equal(t, "Great replay test!", tAfter.Breakdown.FeedbackNote)

	// Assert unchanged eligibility (not reset to now + 15m)
	var finalEligibleAt string
	err = db.QueryRowContext(ctx, "SELECT eligible_at FROM agent_trajectories WHERE id = ?", trajID).Scan(&finalEligibleAt)
	require.NoError(t, err)
	assert.Equal(t, feedbackEligibleAt, finalEligibleAt, "eligible_at should remain unchanged after identical execution replay")
}
