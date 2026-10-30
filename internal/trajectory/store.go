package trajectory

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
)

var (
	// ErrTrajectoryNotFound indicates the requested trajectory ID does not exist.
	ErrTrajectoryNotFound = errors.New("trajectory not found")
	// ErrTrajectoryExpired indicates the trajectory was already pruned from local storage.
	ErrTrajectoryExpired = errors.New("trajectory expired or pruned")
	// ErrUnauthorized indicates the feedback caller does not match session/actor permissions.
	ErrUnauthorized = errors.New("unauthorized feedback actor")
)

const (
	// Fixed-width microsecond timestamp ensuring lexicographical string sorting strictly matches chronological time.
	timeLayout = "2006-01-02T15:04:05.000000Z"
)

func formatTime(t time.Time) string {
	return t.UTC().Format(timeLayout)
}

// parseTime parses timestamps from SQLite or payload strings, handling RFC3339Nano,
// fixed-width microsecond layouts, and driver-stripped trailing zero formats.
func parseTime(s string) (time.Time, error) {
	if t, err := time.Parse(time.RFC3339Nano, s); err == nil {
		return t.UTC(), nil
	}
	if t, err := time.Parse(timeLayout, s); err == nil {
		return t.UTC(), nil
	}
	if t, err := time.Parse(time.RFC3339, s); err == nil {
		return t.UTC(), nil
	}
	if t, err := time.Parse("2006-01-02 15:04:05.999999999", s); err == nil {
		return t.UTC(), nil
	}
	t, err := time.Parse("2006-01-02 15:04:05", s)
	if err != nil {
		return time.Time{}, err
	}
	return t.UTC(), nil
}

// ListExportOptions configures trajectory export queries.
type ListExportOptions struct {
	AllowPrivate bool
	Limit        int
	ExcludeIDs   []string
}

// Store defines operations for storing, mutating, and retrieving trajectories.
type Store interface {
	RecordEvent(ctx context.Context, event *Event) error
	RecordEventTx(ctx context.Context, tx *sql.Tx, event *Event) error
	InsertStaged(ctx context.Context, traj *Trajectory, delay time.Duration) error
	UpdateFeedback(ctx context.Context, req FeedbackRequest) (*Trajectory, error)
	ListReadyForExport(ctx context.Context, now time.Time, opts ListExportOptions) ([]*Trajectory, error)
	MarkExported(ctx context.Context, id string, revision int, exportedAt time.Time) error
	PruneExported(ctx context.Context, olderThan time.Duration) (int64, error)
	Get(ctx context.Context, id string) (*Trajectory, error)
	ReconcileTerminalRuns(ctx context.Context) error
	Close() error
}

// SQLiteStore is a SQLite-backed implementation of Store.
type SQLiteStore struct {
	db           *sql.DB
	ownsDB       bool
	delayMu      sync.RWMutex
	stagingDelay time.Duration
}

// SetStagingDelay sets the default eligibility delay applied to staged trajectories.
func (s *SQLiteStore) SetStagingDelay(d time.Duration) {
	if s == nil {
		return
	}
	s.delayMu.Lock()
	defer s.delayMu.Unlock()
	s.stagingDelay = d
}

func (s *SQLiteStore) getStagingDelay() time.Duration {
	if s == nil {
		return 15 * time.Minute
	}
	s.delayMu.RLock()
	defer s.delayMu.RUnlock()
	if s.stagingDelay <= 0 {
		return 15 * time.Minute
	}
	return s.stagingDelay
}

// NewSQLiteStore initializes and ensures the schema for a SQLiteStore.
func NewSQLiteStore(db *sql.DB, ownsDB bool) (*SQLiteStore, error) {
	if db == nil {
		return nil, errors.New("database connection is nil")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err := EnsureSchema(ctx, db); err != nil {
		return nil, fmt.Errorf("failed to ensure trajectory schema: %w", err)
	}

	return &SQLiteStore{
		db:     db,
		ownsDB: ownsDB,
	}, nil
}

// NewStore initializes a new SQLiteStore wrapping db without closing it on Close.
func NewStore(db *sql.DB) (*SQLiteStore, error) {
	return NewSQLiteStore(db, false)
}

// Close closes the underlying database if owned by this store.
func (s *SQLiteStore) Close() error {
	if s.ownsDB && s.db != nil {
		return s.db.Close()
	}
	return nil
}

// RecordEvent writes an execution lifecycle event directly to a database handle.
func RecordEvent(ctx context.Context, db *sql.DB, event *Event) error {
	if db == nil {
		return errors.New("database is nil")
	}
	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to begin tx: %w", err)
	}
	defer tx.Rollback() //nolint:errcheck

	if err := RecordEventTx(ctx, tx, event); err != nil {
		return err
	}
	return tx.Commit()
}

// RecordEventTx appends an execution lifecycle event to agent_trajectory_events within an external transaction.
func RecordEventTx(ctx context.Context, tx *sql.Tx, event *Event) error {
	if tx == nil {
		return errors.New("transaction is nil")
	}
	if event == nil {
		return errors.New("event is nil")
	}
	if event.ID == "" {
		event.ID = uuid.NewString()
	}
	if event.CreatedAt.IsZero() {
		event.CreatedAt = time.Now().UTC()
	}
	if event.Generation <= 0 {
		event.Generation = 1
	}
	if event.Phase == "" {
		event.Phase = "loop"
	}
	if event.Attempt <= 0 {
		event.Attempt = 1
	}

	payload := event.Payload
	if len(payload) == 0 {
		payload = json.RawMessage("{}")
	}

	// Immutable check: if event with this identity exists, verify identical payload
	var existingPayload []byte
	checkQuery := `
		SELECT payload FROM agent_trajectory_events 
		WHERE run_id = ? AND generation = ? AND iteration = ? AND phase = ? AND step_id = ? AND attempt = ? AND event_type = ?
	`
	err := tx.QueryRowContext(ctx, checkQuery,
		event.RunID, event.Generation, event.Iteration, event.Phase, event.StepID, event.Attempt, event.EventType,
	).Scan(&existingPayload)

	if err == nil {
		// Event already recorded: must have identical payload (replay idempotency)
		if bytes.Equal(existingPayload, payload) {
			return nil
		}
		return fmt.Errorf("conflict: event (%s, gen=%d, iter=%d, phase=%s, step=%s, attempt=%d, type=%s) already exists with different payload",
			event.RunID, event.Generation, event.Iteration, event.Phase, event.StepID, event.Attempt, event.EventType)
	} else if !errors.Is(err, sql.ErrNoRows) {
		return fmt.Errorf("failed to check existing event: %w", err)
	}

	insertQuery := `
		INSERT INTO agent_trajectory_events (
			id, trajectory_id, run_id, generation, iteration, phase, step_id, attempt, event_type, payload, created_at
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
	`
	_, err = tx.ExecContext(ctx, insertQuery,
		event.ID,
		event.TrajectoryID,
		event.RunID,
		event.Generation,
		event.Iteration,
		event.Phase,
		event.StepID,
		event.Attempt,
		event.EventType,
		[]byte(payload),
		formatTime(event.CreatedAt),
	)
	if err != nil {
		return fmt.Errorf("failed to record trajectory event: %w", err)
	}
	return nil
}

// RecordEventTx appends an execution lifecycle event to agent_trajectory_events within an external transaction.
func (s *SQLiteStore) RecordEventTx(ctx context.Context, tx *sql.Tx, event *Event) error {
	return RecordEventTx(ctx, tx, event)
}

// RecordEvent appends an execution lifecycle event in its own transaction.
func (s *SQLiteStore) RecordEvent(ctx context.Context, event *Event) error {
	return s.withRetryTx(ctx, func(tx *sql.Tx) error {
		return RecordEventTx(ctx, tx, event)
	})
}

// InsertStaged stages a trajectory rollout in agent_trajectories.
func (s *SQLiteStore) InsertStaged(ctx context.Context, traj *Trajectory, delay time.Duration) error {
	if traj == nil {
		return errors.New("trajectory is nil")
	}
	if traj.ID == "" {
		return errors.New("trajectory ID cannot be empty")
	}

	return s.withRetryTx(ctx, func(tx *sql.Tx) error {
		return s.insertStagedTx(ctx, tx, traj, delay)
	})
}

func (s *SQLiteStore) insertStagedTx(ctx context.Context, tx *sql.Tx, traj *Trajectory, delay time.Duration) error {
	// 1. Check if trajectory was pruned (tombstone check)
		var prunedCount int
		err := tx.QueryRowContext(ctx, "SELECT count(*) FROM agent_trajectory_pruned WHERE trajectory_id = ?", traj.ID).Scan(&prunedCount)
		if err != nil {
			return fmt.Errorf("failed to check prune tombstone: %w", err)
		}
		if prunedCount > 0 {
			// Already pruned; do not resurrect
			return nil
		}

		now := time.Now().UTC()
		if traj.CreatedAt.IsZero() {
			traj.CreatedAt = now
		}
		if traj.CompletedAt.IsZero() {
			traj.CompletedAt = now
		}
		if traj.Generation <= 0 {
			traj.Generation = 1
		}
		if traj.Revision <= 0 {
			traj.Revision = 1
		}
		if traj.PrivacyLevel == "" {
			traj.PrivacyLevel = "private"
		}
		if traj.ExecutionKind == "" {
			traj.ExecutionKind = "interactive"
		}
		traj.Breakdown.RewardVersion = RewardVersionV1

		// 2. Query existing record
		var existingPayload []byte
		var existingRevision int
		var existingExportedRev int
		var existingGen int
		var existingStatus string
		var existingCreatedAt string

		err = tx.QueryRowContext(ctx, `
			SELECT payload, revision, exported_revision, generation, status, created_at 
			FROM agent_trajectories 
			WHERE id = ?
		`, traj.ID).Scan(&existingPayload, &existingRevision, &existingExportedRev, &existingGen, &existingStatus, &existingCreatedAt)

		if errors.Is(err, sql.ErrNoRows) {
			// Initial insertion
			traj.Reward = CalculateReward(traj.Breakdown)
			traj.Revision = 1

			payloadBytes, err := json.Marshal(traj)
			if err != nil {
				return fmt.Errorf("failed to marshal trajectory payload: %w", err)
			}

			eligibleAt := now.Add(delay).UTC()

			_, err = tx.ExecContext(ctx, `
				INSERT INTO agent_trajectories (
					id, run_id, frontend_id, session_id, scope_id, actor_id, execution_kind,
					privacy_level, export_approved, generation, payload, r_eval,
					status, revision, exported_revision, created_at, completed_at, eligible_at
				) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'staged', ?, ?, ?, ?, ?)
			`,
				traj.ID,
				traj.RunID,
				traj.FrontendID,
				traj.SessionID,
				traj.ScopeID,
				traj.ActorID,
				traj.ExecutionKind,
				traj.PrivacyLevel,
				boolToInt(traj.ExportApproved),
				traj.Generation,
				payloadBytes,
				traj.Reward,
				traj.Revision,
				0,
				formatTime(traj.CreatedAt),
				formatTime(traj.CompletedAt),
				formatTime(eligibleAt),
			)
			if err != nil {
				return fmt.Errorf("failed to insert staged trajectory: %w", err)
			}
			return nil
		} else if err != nil {
			return fmt.Errorf("failed to query existing trajectory: %w", err)
		}

		// Row exists. Check durable execution ordering.
		if traj.Generation < existingGen {
			// Stale finalizer arrives after a newer generation: reject as no-op.
			return nil
		}

		// Merge with any persisted user feedback before checking equality
		var feedbackRating float64
		var feedbackNote string

		row := tx.QueryRowContext(ctx, `
			SELECT rating, note FROM agent_trajectory_feedback 
			WHERE trajectory_id = ? 
			ORDER BY created_at DESC LIMIT 1
		`, traj.ID)
		if err := row.Scan(&feedbackRating, &feedbackNote); err == nil {
			traj.Breakdown.HasUserFeedback = true
			traj.Breakdown.UserFeedback = feedbackRating
			traj.Breakdown.FeedbackNote = feedbackNote
		} else if !errors.Is(err, sql.ErrNoRows) {
			return fmt.Errorf("failed to query trajectory feedback: %w", err)
		}

		// Recompute reward with merged feedback
		traj.Reward = CalculateReward(traj.Breakdown)

		// If same generation, check if substantive fields are unchanged
		if traj.Generation == existingGen && len(existingPayload) > 0 {
			var existingTraj Trajectory
			if err := json.Unmarshal(existingPayload, &existingTraj); err == nil {
				if trajectoriesSubstantivelyEqual(&existingTraj, traj) {
					// No substantive changes: no-op
					return nil
				}
			}
		}

		// Increment revision for changed payload
		traj.Revision = existingRevision + 1

		payloadBytes, err := json.Marshal(traj)
		if err != nil {
			return fmt.Errorf("failed to marshal updated trajectory: %w", err)
		}

		// Re-staging delay begins from re-staging time
		eligibleAt := now.Add(delay).UTC()

		_, err = tx.ExecContext(ctx, `
			UPDATE agent_trajectories
			SET payload = ?,
			    r_eval = ?,
			    status = 'staged',
			    revision = ?,
			    generation = ?,
			    completed_at = ?,
			    eligible_at = ?,
			    privacy_level = ?,
			    export_approved = ?
			WHERE id = ?
		`,
			payloadBytes,
			traj.Reward,
			traj.Revision,
			traj.Generation,
			formatTime(traj.CompletedAt),
			formatTime(eligibleAt),
			traj.PrivacyLevel,
			boolToInt(traj.ExportApproved),
			traj.ID,
		)
		if err != nil {
			return fmt.Errorf("failed to update staged trajectory: %w", err)
		}

		return nil
}

// UpdateFeedback records user feedback and re-stages the affected trajectory.
func (s *SQLiteStore) UpdateFeedback(ctx context.Context, req FeedbackRequest) (*Trajectory, error) {
	if strings.TrimSpace(req.MessageID) == "" {
		return nil, errors.New("message ID cannot be empty")
	}
	if strings.TrimSpace(req.FrontendID) == "" {
		return nil, errors.New("frontend ID cannot be empty")
	}
	if strings.TrimSpace(req.SessionID) == "" {
		return nil, errors.New("session ID cannot be empty")
	}
	if strings.TrimSpace(req.ScopeID) == "" {
		return nil, errors.New("scope ID cannot be empty")
	}
	if strings.TrimSpace(req.ActorID) == "" {
		return nil, errors.New("actor ID cannot be empty")
	}
	if req.Rating < -1.0 || req.Rating > 1.0 {
		return nil, fmt.Errorf("invalid feedback rating: %f; must be within [-1.0, 1.0]", req.Rating)
	}

	var updated *Trajectory

	err := s.withRetryTx(ctx, func(tx *sql.Tx) error {
		// 1. Check for replay idempotency via message_id
		var storedTrajID string
		var storedFeID, storedSessID, storedScopeID, storedActorID string

		err := tx.QueryRowContext(ctx, `
			SELECT trajectory_id, frontend_id, session_id, scope_id, actor_id 
			FROM agent_trajectory_feedback 
			WHERE message_id = ?
		`, req.MessageID).Scan(&storedTrajID, &storedFeID, &storedSessID, &storedScopeID, &storedActorID)

		if err == nil {
			// Message already processed: verify identity including scope
			if storedFeID != req.FrontendID || storedSessID != req.SessionID || storedScopeID != req.ScopeID || storedActorID != req.ActorID {
				return ErrUnauthorized
			}
			// Idempotent return of existing trajectory
			t, err := s.getTx(ctx, tx, storedTrajID)
			if err != nil {
				return err
			}
			updated = t
			return nil
		} else if !errors.Is(err, sql.ErrNoRows) {
			return fmt.Errorf("failed to check feedback replay: %w", err)
		}

		// 2. Resolve target trajectory
		targetID := req.TrajectoryID
		if targetID == "" {
			// Find latest interactive turn by actor in session and scope
			err := tx.QueryRowContext(ctx, `
				SELECT id FROM agent_trajectories 
				WHERE frontend_id = ? AND session_id = ? AND scope_id = ? AND actor_id = ? AND execution_kind = 'interactive'
				ORDER BY completed_at DESC LIMIT 1
			`, req.FrontendID, req.SessionID, req.ScopeID, req.ActorID).Scan(&targetID)
			if errors.Is(err, sql.ErrNoRows) {
				return ErrTrajectoryNotFound
			} else if err != nil {
				return fmt.Errorf("failed to resolve latest trajectory for actor: %w", err)
			}
		}

		// Check if target is pruned
		var prunedCount int
		err = tx.QueryRowContext(ctx, "SELECT count(*) FROM agent_trajectory_pruned WHERE trajectory_id = ?", targetID).Scan(&prunedCount)
		if err != nil {
			return fmt.Errorf("failed to check target prune status: %w", err)
		}
		if prunedCount > 0 {
			return ErrTrajectoryExpired
		}

		// Retrieve target trajectory
		var payloadBytes []byte
		var ownerFe, ownerSess, ownerScope, ownerActor string

		err = tx.QueryRowContext(ctx, `
			SELECT payload, frontend_id, session_id, scope_id, actor_id 
			FROM agent_trajectories 
			WHERE id = ?
		`, targetID).Scan(&payloadBytes, &ownerFe, &ownerSess, &ownerScope, &ownerActor)
		if errors.Is(err, sql.ErrNoRows) {
			return ErrTrajectoryNotFound
		} else if err != nil {
			return fmt.Errorf("failed to fetch target trajectory: %w", err)
		}

		// Authorize: frontend, session, scope must match, and actor must match or be admin/system
		if ownerFe != req.FrontendID || ownerSess != req.SessionID || ownerScope != req.ScopeID {
			return ErrUnauthorized
		}
		if ownerActor != req.ActorID && req.ActorID != "admin" && req.ActorID != "system" {
			return ErrUnauthorized
		}

		// Unmarshal trajectory
		var traj Trajectory
		if err := json.Unmarshal(payloadBytes, &traj); err != nil {
			return fmt.Errorf("failed to unmarshal trajectory payload: %w", err)
		}

		// 3. Record feedback audit entry
		feedbackID := uuid.NewString()
		now := time.Now().UTC()

		_, err = tx.ExecContext(ctx, `
			INSERT INTO agent_trajectory_feedback (
				id, message_id, trajectory_id, frontend_id, session_id, scope_id, actor_id, rating, note, created_at
			) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
		`,
			feedbackID,
			req.MessageID,
			targetID,
			req.FrontendID,
			req.SessionID,
			req.ScopeID,
			req.ActorID,
			req.Rating,
			req.Note,
			formatTime(now),
		)
		if err != nil {
			return fmt.Errorf("failed to insert trajectory feedback: %w", err)
		}

		// 4. Update breakdown and recompute reward
		traj.Breakdown.HasUserFeedback = true
		traj.Breakdown.UserFeedback = req.Rating
		traj.Breakdown.FeedbackNote = req.Note
		traj.Reward = CalculateReward(traj.Breakdown)

		// Increment revision and reset status to 'staged'
		traj.Revision++

		newPayload, err := json.Marshal(&traj)
		if err != nil {
			return fmt.Errorf("failed to marshal updated trajectory: %w", err)
		}

		// Accelerate eligibility: eligible after 1 minute of quiescence from now
		eligibleAt := now.Add(1 * time.Minute)

		_, err = tx.ExecContext(ctx, `
			UPDATE agent_trajectories
			SET payload = ?,
			    r_eval = ?,
			    status = 'staged',
			    revision = ?,
			    eligible_at = ?
			WHERE id = ?
		`,
			newPayload,
			traj.Reward,
			traj.Revision,
			formatTime(eligibleAt),
			targetID,
		)
		if err != nil {
			return fmt.Errorf("failed to update trajectory with feedback: %w", err)
		}

		updated = &traj
		return nil
	})

	if err != nil {
		return nil, err
	}
	return updated, nil
}

// ListReadyForExport returns trajectories that are mature, approved, and match privacy constraints.
func (s *SQLiteStore) ListReadyForExport(ctx context.Context, now time.Time, opts ListExportOptions) ([]*Trajectory, error) {
	limit := opts.Limit
	if limit <= 0 {
		limit = 50
	}

	var queryBuilder strings.Builder
	queryBuilder.WriteString(`
		SELECT payload FROM agent_trajectories
		WHERE status = 'staged'
		  AND eligible_at <= ?
		  AND export_approved = 1
		  AND (privacy_level = 'public' OR (? = 1 AND privacy_level = 'private'))
	`)

	args := []any{
		formatTime(now),
		boolToInt(opts.AllowPrivate),
	}

	if len(opts.ExcludeIDs) > 0 {
		placeholders := make([]string, len(opts.ExcludeIDs))
		for i, id := range opts.ExcludeIDs {
			placeholders[i] = "?"
			args = append(args, id)
		}
		fmt.Fprintf(&queryBuilder, " AND id NOT IN (%s)", strings.Join(placeholders, ", "))
	}

	queryBuilder.WriteString(`
		ORDER BY eligible_at ASC
		LIMIT ?
	`)
	args = append(args, limit)

	rows, err := s.db.QueryContext(ctx, queryBuilder.String(), args...)
	if err != nil {
		return nil, fmt.Errorf("failed to list trajectories for export: %w", err)
	}
	defer func() { _ = rows.Close() }()

	var results []*Trajectory
	for rows.Next() {
		var payloadBytes []byte
		if err := rows.Scan(&payloadBytes); err != nil {
			return nil, fmt.Errorf("failed to scan trajectory payload: %w", err)
		}
		var traj Trajectory
		if err := json.Unmarshal(payloadBytes, &traj); err != nil {
			return nil, fmt.Errorf("failed to unmarshal trajectory: %w", err)
		}
		results = append(results, &traj)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return results, nil
}

// MarkExported marks a specific revision of a trajectory as exported.
func (s *SQLiteStore) MarkExported(ctx context.Context, id string, revision int, exportedAt time.Time) error {
	if id == "" {
		return errors.New("trajectory ID cannot be empty")
	}

	return s.withRetryTx(ctx, func(tx *sql.Tx) error {
		query := `
			UPDATE agent_trajectories
			SET status = CASE WHEN revision = ? THEN 'exported' ELSE 'staged' END,
			    exported_revision = ?,
			    exported_at = ?
			WHERE id = ? AND exported_revision < ? AND revision >= ?
		`
		res, err := tx.ExecContext(ctx, query,
			revision,
			revision,
			formatTime(exportedAt),
			id,
			revision,
			revision,
		)
		if err != nil {
			return fmt.Errorf("failed to mark trajectory exported: %w", err)
		}
		rowsAffected, err := res.RowsAffected()
		if err != nil {
			return err
		}
		if rowsAffected == 0 {
			// Verify if it exists
			var currentRev, currentExpRev int
			err := tx.QueryRowContext(ctx, "SELECT revision, exported_revision FROM agent_trajectories WHERE id = ?", id).Scan(&currentRev, &currentExpRev)
			if errors.Is(err, sql.ErrNoRows) {
				return ErrTrajectoryNotFound
			}
		}
		return nil
	})
}

// PruneExported deletes fully exported trajectories older than olderThan and records tombstones.
func (s *SQLiteStore) PruneExported(ctx context.Context, olderThan time.Duration) (int64, error) {
	cutoff := time.Now().UTC().Add(-olderThan)
	var prunedCount int64

	err := s.withRetryTx(ctx, func(tx *sql.Tx) error {
		rows, err := tx.QueryContext(ctx, `
			SELECT id FROM agent_trajectories 
			WHERE status = 'exported' 
			  AND exported_revision = revision 
			  AND exported_at <= ?
		`, formatTime(cutoff))
		if err != nil {
			return fmt.Errorf("failed to query prunable trajectories: %w", err)
		}
		defer func() { _ = rows.Close() }()

		var ids []string
		for rows.Next() {
			var id string
			if err := rows.Scan(&id); err != nil {
				return err
			}
			ids = append(ids, id)
		}
		if err := rows.Err(); err != nil {
			return err
		}

		nowStr := formatTime(time.Now().UTC())
		for _, id := range ids {
			// Record tombstone
			_, err = tx.ExecContext(ctx, `
				INSERT INTO agent_trajectory_pruned (trajectory_id, pruned_at) VALUES (?, ?)
				ON CONFLICT(trajectory_id) DO UPDATE SET pruned_at = excluded.pruned_at
			`, id, nowStr)
			if err != nil {
				return fmt.Errorf("failed to record tombstone for %s: %w", id, err)
			}

			// Explicitly delete feedback (in case foreign keys are not active on connection)
			_, err = tx.ExecContext(ctx, "DELETE FROM agent_trajectory_feedback WHERE trajectory_id = ?", id)
			if err != nil {
				return fmt.Errorf("failed to delete feedback for %s: %w", id, err)
			}

			// Delete events
			_, err = tx.ExecContext(ctx, "DELETE FROM agent_trajectory_events WHERE trajectory_id = ?", id)
			if err != nil {
				return fmt.Errorf("failed to delete events for %s: %w", id, err)
			}

			// Delete trajectory
			_, err = tx.ExecContext(ctx, "DELETE FROM agent_trajectories WHERE id = ?", id)
			if err != nil {
				return fmt.Errorf("failed to delete trajectory %s: %w", id, err)
			}
		}

		prunedCount = int64(len(ids))
		return nil
	})

	if err != nil {
		return 0, err
	}
	return prunedCount, nil
}

// Get returns the canonical Trajectory for a given ID.
func (s *SQLiteStore) Get(ctx context.Context, id string) (*Trajectory, error) {
	return s.getTx(ctx, nil, id)
}

func (s *SQLiteStore) getTx(ctx context.Context, tx *sql.Tx, id string) (*Trajectory, error) {
	query := "SELECT payload FROM agent_trajectories WHERE id = ?"
	var row *sql.Row
	if tx != nil {
		row = tx.QueryRowContext(ctx, query, id)
	} else {
		row = s.db.QueryRowContext(ctx, query, id)
	}

	var payloadBytes []byte
	err := row.Scan(&payloadBytes)
	if errors.Is(err, sql.ErrNoRows) {
		// Check tombstone
		var prunedCount int
		var qErr error
		if tx != nil {
			qErr = tx.QueryRowContext(ctx, "SELECT count(*) FROM agent_trajectory_pruned WHERE trajectory_id = ?", id).Scan(&prunedCount)
		} else {
			qErr = s.db.QueryRowContext(ctx, "SELECT count(*) FROM agent_trajectory_pruned WHERE trajectory_id = ?", id).Scan(&prunedCount)
		}
		if qErr == nil && prunedCount > 0 {
			return nil, ErrTrajectoryExpired
		}
		return nil, ErrTrajectoryNotFound
	} else if err != nil {
		return nil, err
	}

	var traj Trajectory
	if err := json.Unmarshal(payloadBytes, &traj); err != nil {
		return nil, fmt.Errorf("failed to unmarshal trajectory: %w", err)
	}
	return &traj, nil
}

// ReconcileTerminalRuns materializes missing trajectories from events for runs that reached a terminal state.
func (s *SQLiteStore) ReconcileTerminalRuns(ctx context.Context) error {
	return s.withRetryTx(ctx, func(tx *sql.Tx) error {
		// Find distinct trajectory_ids that have a 'turn_terminal' event, but no agent_trajectories row and no tombstone
		rows, err := tx.QueryContext(ctx, `
			SELECT DISTINCT e.trajectory_id 
			FROM agent_trajectory_events e
			LEFT JOIN agent_trajectories t ON e.trajectory_id = t.id
			LEFT JOIN agent_trajectory_pruned p ON e.trajectory_id = p.trajectory_id
			WHERE e.event_type = 'turn_terminal' AND p.trajectory_id IS NULL AND (t.id IS NULL OR e.generation > t.generation OR (e.generation = t.generation AND e.created_at > t.completed_at))
		`)
		if err != nil {
			return fmt.Errorf("failed to query terminal orphaned events: %w", err)
		}
		defer func() { _ = rows.Close() }()

		var trajIDs []string
		for rows.Next() {
			var id string
			if err := rows.Scan(&id); err != nil {
				return err
			}
			trajIDs = append(trajIDs, id)
		}
		if err := rows.Err(); err != nil {
			return err
		}

		for _, trajID := range trajIDs {
			// Load all events for this trajectory
			evRows, err := tx.QueryContext(ctx, `
				SELECT run_id, generation, iteration, phase, event_type, payload, created_at 
				FROM agent_trajectory_events 
				WHERE trajectory_id = ? 
				ORDER BY generation ASC, iteration ASC, attempt ASC, created_at ASC
			`, trajID)
			if err != nil {
				return fmt.Errorf("failed to load events for trajectory %s: %w", trajID, err)
			}

			var startPayload *TurnStartPayload
			var terminalPayload *TurnTerminalPayload
			var startTime time.Time
			var terminalTime time.Time
			var stepsByGen = make(map[int][]Step)
			var latestOutput string

			for evRows.Next() {
				var runID string
				var gen, iter int
				var phase, evType string
				var payloadBytes []byte
				var createdAtStr string
				if err := evRows.Scan(&runID, &gen, &iter, &phase, &evType, &payloadBytes, &createdAtStr); err != nil {
					_ = evRows.Close()
					return err
				}
				evTime, _ := parseTime(createdAtStr)
				if startTime.IsZero() {
					startTime = evTime
				}

				switch evType {
				case "turn_start":
					var sp TurnStartPayload
					if err := json.Unmarshal(payloadBytes, &sp); err == nil {
						startPayload = &sp
					}
				case "tool_attempt":
					var step Step
					if err := json.Unmarshal(payloadBytes, &step); err == nil {
						stepsByGen[gen] = append(stepsByGen[gen], step)
					}
				case "llm_response":
					var respPayload struct {
						Content string `json:"content"`
					}
					if err := json.Unmarshal(payloadBytes, &respPayload); err == nil {
						latestOutput = respPayload.Content
					}
				case "turn_terminal":
					var tp TurnTerminalPayload
					if err := json.Unmarshal(payloadBytes, &tp); err == nil {
						terminalPayload = &tp
						terminalTime = evTime
						if tp.FinalReply != "" {
							latestOutput = tp.FinalReply
						}
					}
				}
			}
			_ = evRows.Close()

			// Must have a turn_start and turn_terminal to reconstruct durable session context
			if startPayload == nil || terminalPayload == nil {
				continue
			}

			if terminalTime.IsZero() {
				terminalTime = time.Now().UTC()
			}

			// Use steps from terminal generation (or highest generation)
			terminalGen := terminalPayload.Generation
			if terminalGen <= 0 {
				terminalGen = startPayload.Generation
			}
			steps := stepsByGen[terminalGen]

			var succCount, failCount int
			for _, step := range steps {
				if step.ExitCode == nil || *step.ExitCode == 0 {
					succCount++
				} else {
					failCount++
				}
			}

			breakdown := EvalBreakdown{
				TotalToolCalls:      len(steps),
				SuccessfulToolCalls: succCount,
				FailedToolCalls:     failCount,
				FSMFailed:           terminalPayload.Failed,
				TerminationReason:   terminalPayload.Reason,
				RewardVersion:       RewardVersionV1,
			}
			if len(steps) > 0 {
				breakdown.ToolSuccessRate = float64(succCount) / float64(len(steps))
			}

			reconciled := &Trajectory{
				ID:             trajID,
				RunID:          startPayload.RunID,
				FrontendID:     startPayload.FrontendID,
				SessionID:      startPayload.SessionID,
				ScopeID:        startPayload.ScopeID,
				ActorID:        startPayload.ActorID,
				Model:          startPayload.Model,
				ExecutionKind:  startPayload.ExecutionKind,
				PrivacyLevel:   startPayload.PrivacyLevel,
				ExportApproved: startPayload.ExportApproved,
				Generation:     terminalGen,
				CreatedAt:      startTime,
				CompletedAt:    terminalTime,
				UserInput:      startPayload.Input,
				TaskState:      startPayload.TaskState,
				ActiveSkills:   startPayload.ActiveSkills,
				Steps:          steps,
				FinalOutput:    latestOutput,
				Reward:         CalculateReward(breakdown),
				Breakdown:      breakdown,
				Revision:       1,
			}

			if err := s.insertStagedTx(ctx, tx, reconciled, s.getStagingDelay()); err != nil {
				return fmt.Errorf("failed to reconcile staged trajectory %s: %w", trajID, err)
			}
		}

		return nil
	})
}

// withRetryTx executes fn within an explicit SQLite transaction, retrying on busy/locked errors.
func (s *SQLiteStore) withRetryTx(ctx context.Context, fn func(tx *sql.Tx) error) error {
	var err error
	for attempt := 0; attempt < 3; attempt++ {
		if attempt > 0 {
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(time.Duration(10*(attempt*attempt)) * time.Millisecond):
			}
		}

		tx, txErr := s.db.BeginTx(ctx, nil)
		if txErr != nil {
			err = txErr
			if isBusy(err) {
				continue
			}
			return err
		}

		err = fn(tx)
		if err != nil {
			_ = tx.Rollback()
			if isBusy(err) {
				continue
			}
			return err
		}

		if err = tx.Commit(); err != nil {
			if isBusy(err) {
				continue
			}
			return err
		}
		return nil
	}
	return err
}

func isBusy(err error) bool {
	if err == nil {
		return false
	}
	msg := err.Error()
	return strings.Contains(msg, "busy") || strings.Contains(msg, "locked")
}

func boolToInt(b bool) int {
	if b {
		return 1
	}
	return 0
}

func trajectoriesSubstantivelyEqual(a, b *Trajectory) bool {
	if a == nil || b == nil {
		return a == b
	}
	if a.UserInput != b.UserInput || a.FinalOutput != b.FinalOutput || a.TaskState != b.TaskState {
		return false
	}
	if a.Model != b.Model || a.PrivacyLevel != b.PrivacyLevel || a.ExportApproved != b.ExportApproved {
		return false
	}
	if a.Generation != b.Generation {
		return false
	}
	if len(a.ActiveSkills) != len(b.ActiveSkills) {
		return false
	}
	for i := range a.ActiveSkills {
		if a.ActiveSkills[i] != b.ActiveSkills[i] {
			return false
		}
	}
	if len(a.Steps) != len(b.Steps) {
		return false
	}
	for i := range a.Steps {
		sA, sB := a.Steps[i], b.Steps[i]
		if sA.Index != sB.Index || sA.Action != sB.Action || sA.Observation != sB.Observation || sA.Thought != sB.Thought {
			return false
		}
		if (sA.ExitCode == nil) != (sB.ExitCode == nil) {
			return false
		}
		if sA.ExitCode != nil && sB.ExitCode != nil && *sA.ExitCode != *sB.ExitCode {
			return false
		}
		if !actionArgsEqual(sA.ActionArgs, sB.ActionArgs) {
			return false
		}
	}
	if a.Breakdown.TerminationReason != b.Breakdown.TerminationReason ||
		a.Breakdown.BudgetExhausted != b.Breakdown.BudgetExhausted ||
		a.Breakdown.FSMFailed != b.Breakdown.FSMFailed ||
		a.Breakdown.TotalToolCalls != b.Breakdown.TotalToolCalls ||
		a.Breakdown.SuccessfulToolCalls != b.Breakdown.SuccessfulToolCalls ||
		a.Breakdown.FailedToolCalls != b.Breakdown.FailedToolCalls ||
		a.Breakdown.HasUserFeedback != b.Breakdown.HasUserFeedback ||
		a.Breakdown.UserFeedback != b.Breakdown.UserFeedback ||
		a.Breakdown.FeedbackNote != b.Breakdown.FeedbackNote {
		return false
	}
	return true
}

func actionArgsEqual(a, b json.RawMessage) bool {
	normA := len(a) == 0 || bytes.Equal(a, []byte("null"))
	normB := len(b) == 0 || bytes.Equal(b, []byte("null"))
	if normA && normB {
		return true
	}
	return bytes.Equal(a, b)
}
