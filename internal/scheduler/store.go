package scheduler

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"strings"
	"time"
)

var (
	// ErrScheduleNotFound indicates no schedule matched the criteria.
	ErrScheduleNotFound = errors.New("schedule not found")
	// ErrGrantNotFound indicates no permission grant exists for the schedule.
	ErrGrantNotFound = errors.New("schedule grant not found")
	// ErrScheduleNotPending indicates schedule is not awaiting approval.
	ErrScheduleNotPending = errors.New("schedule is not in pending approval state")
)

// Store provides isolated SQLite database operations for schedules and grants.
type Store struct {
	db *sql.DB
}

// NewStore initializes a new Store wrapping the given SQLite database.
func NewStore(db *sql.DB) *Store {
	return &Store{db: db}
}

// DB returns the underlying database connection.
func (s *Store) DB() *sql.DB {
	return s.db
}

// CreateSchedule persists a new schedule and its optional permission grant atomically.
func (s *Store) CreateSchedule(ctx context.Context, sched *Schedule, grant *ScheduleGrant) error {
	if sched == nil {
		return errors.New("schedule cannot be nil")
	}
	sched.Name = strings.TrimSpace(sched.Name)
	if err := ValidateScheduleName(sched.Name); err != nil {
		return err
	}

	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
	}
	defer func() { _ = tx.Rollback() }()

	now := time.Now().Unix()
	if sched.CreatedAt == 0 {
		sched.CreatedAt = now
	}
	sched.UpdatedAt = now

	insertSchedSQL := `
		INSERT INTO schedules (
			id, name, chat_id, user_id, schedule_type, cron_expr, interval_seconds,
			instruction, status, next_run_at, last_run_at, expires_at, max_runs,
			run_count, missed_count, run_timeout_seconds, max_turns, last_status,
			created_at, updated_at
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
	`
	_, err = tx.ExecContext(ctx, insertSchedSQL,
		sched.ID, sched.Name, sched.ChatID, sched.UserID, string(sched.ScheduleType),
		nullString(sched.CronExpr), nullInt(sched.IntervalSeconds),
		sched.Instruction, string(sched.Status), sched.NextRunAt,
		nullInt64(sched.LastRunAt), nullInt64(sched.ExpiresAt), sched.MaxRuns,
		sched.RunCount, sched.MissedCount, sched.RunTimeoutSeconds, sched.MaxTurns,
		nullString(sched.LastStatus), sched.CreatedAt, sched.UpdatedAt,
	)
	if err != nil {
		return fmt.Errorf("failed to insert schedule %s (%s): %w", sched.Name, sched.ID, err)
	}

	if grant != nil {
		insertGrantSQL := `
			INSERT INTO schedule_grants (
				id, schedule_id, permission_request_json, params_hash,
				granted_by, granted_at, valid_until
			) VALUES (?, ?, ?, ?, ?, ?, ?)
		`
		_, err = tx.ExecContext(ctx, insertGrantSQL,
			grant.ID, sched.ID, grant.PermissionRequestJSON, grant.ParamsHash,
			grant.GrantedBy, grant.GrantedAt, grant.ValidUntil,
		)
		if err != nil {
			return fmt.Errorf("failed to insert schedule grant for %s: %w", sched.ID, err)
		}
	}

	return tx.Commit()
}

// GetSchedule retrieves a schedule by its unique ID.
func (s *Store) GetSchedule(ctx context.Context, id string) (*Schedule, error) {
	query := `
		SELECT id, name, chat_id, user_id, schedule_type, cron_expr, interval_seconds,
		       instruction, status, next_run_at, last_run_at, expires_at, max_runs,
		       run_count, missed_count, run_timeout_seconds, max_turns, last_status,
		       created_at, updated_at
		FROM schedules
		WHERE id = ?
	`
	row := s.db.QueryRowContext(ctx, query, id)
	return scanSchedule(row)
}

type querier interface {
	QueryRowContext(ctx context.Context, query string, args ...any) *sql.Row
}

func getScheduleByNameQ(ctx context.Context, q querier, chatID, name string) (*Schedule, error) {
	name = strings.TrimSpace(name)
	query := `
		SELECT id, name, chat_id, user_id, schedule_type, cron_expr, interval_seconds,
		       instruction, status, next_run_at, last_run_at, expires_at, max_runs,
		       run_count, missed_count, run_timeout_seconds, max_turns, last_status,
		       created_at, updated_at
		FROM schedules
		WHERE chat_id = ? AND name = ?
	`
	row := q.QueryRowContext(ctx, query, chatID, name)
	return scanSchedule(row)
}

func getGrantQ(ctx context.Context, q querier, scheduleID string) (*ScheduleGrant, error) {
	query := `
		SELECT id, schedule_id, permission_request_json, params_hash, granted_by, granted_at, valid_until
		FROM schedule_grants
		WHERE schedule_id = ?
	`
	var g ScheduleGrant
	err := q.QueryRowContext(ctx, query, scheduleID).Scan(
		&g.ID, &g.ScheduleID, &g.PermissionRequestJSON, &g.ParamsHash,
		&g.GrantedBy, &g.GrantedAt, &g.ValidUntil,
	)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return nil, ErrGrantNotFound
		}
		return nil, fmt.Errorf("failed to query schedule grant for %s: %w", scheduleID, err)
	}
	return &g, nil
}

// GetScheduleByName retrieves a schedule by chat ID and human-readable name.
func (s *Store) GetScheduleByName(ctx context.Context, chatID, name string) (*Schedule, error) {
	return getScheduleByNameQ(ctx, s.db, chatID, name)
}

// GetGrant retrieves the permission grant associated with a schedule.
func (s *Store) GetGrant(ctx context.Context, scheduleID string) (*ScheduleGrant, error) {
	return getGrantQ(ctx, s.db, scheduleID)
}

// ApproveSchedule activates a pending schedule and confirms the permission grant.
func (s *Store) ApproveSchedule(ctx context.Context, chatID, name, userID string, validUntil int64) (*Schedule, *ScheduleGrant, error) {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to begin transaction: %w", err)
	}
	defer func() { _ = tx.Rollback() }()

	name = strings.TrimSpace(name)
	sched, err := getScheduleByNameQ(ctx, tx, chatID, name)
	if err != nil {
		return nil, nil, err
	}
	if sched.Status != ScheduleStatusPendingApproval {
		return nil, nil, fmt.Errorf("%w: current status is %s", ErrScheduleNotPending, sched.Status)
	}

	grant, err := getGrantQ(ctx, tx, sched.ID)
	if err != nil && !errors.Is(err, ErrGrantNotFound) {
		return nil, nil, err
	}

	now := time.Now().Unix()
	if grant != nil {
		if !grant.Verify() {
			return nil, nil, errors.New("grant params_hash does not match permission_request_json")
		}
		if validUntil > 0 && validUntil <= now {
			return nil, nil, errors.New("valid_until must be in the future")
		}
		grant.GrantedBy = userID
		grant.GrantedAt = now
		if validUntil > 0 {
			grant.ValidUntil = validUntil
		}
		updateGrantSQL := `
			UPDATE schedule_grants
			SET granted_by = ?, granted_at = ?, valid_until = ?
			WHERE schedule_id = ?
		`
		_, err = tx.ExecContext(ctx, updateGrantSQL, grant.GrantedBy, grant.GrantedAt, grant.ValidUntil, sched.ID)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to update schedule grant: %w", err)
		}
	}

	sched.Status = ScheduleStatusActive
	sched.UpdatedAt = now
	updateSchedSQL := `
		UPDATE schedules
		SET status = ?, updated_at = ?
		WHERE id = ? AND status = 'PENDING_APPROVAL'
	`
	res, err := tx.ExecContext(ctx, updateSchedSQL, string(sched.Status), sched.UpdatedAt, sched.ID)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to activate schedule: %w", err)
	}
	n, err := res.RowsAffected()
	if err != nil {
		return nil, nil, fmt.Errorf("failed to check rows affected: %w", err)
	}
	if n == 0 {
		return nil, nil, fmt.Errorf("%w: schedule is no longer pending approval", ErrScheduleNotPending)
	}

	if err := tx.Commit(); err != nil {
		return nil, nil, fmt.Errorf("failed to commit approval: %w", err)
	}
	return sched, grant, nil
}

// DenySchedule rejects a pending schedule and removes any associated grant.
func (s *Store) DenySchedule(ctx context.Context, chatID, name string) (*Schedule, error) {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to begin transaction: %w", err)
	}
	defer func() { _ = tx.Rollback() }()

	name = strings.TrimSpace(name)
	sched, err := getScheduleByNameQ(ctx, tx, chatID, name)
	if err != nil {
		return nil, err
	}
	if sched.Status != ScheduleStatusPendingApproval {
		return nil, fmt.Errorf("%w: current status is %s", ErrScheduleNotPending, sched.Status)
	}

	now := time.Now().Unix()
	sched.Status = ScheduleStatusDenied
	sched.UpdatedAt = now

	res, err := tx.ExecContext(ctx, "UPDATE schedules SET status = ?, updated_at = ? WHERE id = ? AND status = 'PENDING_APPROVAL'",
		string(sched.Status), sched.UpdatedAt, sched.ID)
	if err != nil {
		return nil, fmt.Errorf("failed to update denied schedule status: %w", err)
	}
	n, err := res.RowsAffected()
	if err != nil {
		return nil, fmt.Errorf("failed to check rows affected: %w", err)
	}
	if n == 0 {
		return nil, fmt.Errorf("%w: schedule is no longer pending approval", ErrScheduleNotPending)
	}

	_, err = tx.ExecContext(ctx, "DELETE FROM schedule_grants WHERE schedule_id = ?", sched.ID)
	if err != nil {
		return nil, fmt.Errorf("failed to remove grant for denied schedule: %w", err)
	}

	if err := tx.Commit(); err != nil {
		return nil, fmt.Errorf("failed to commit denial: %w", err)
	}
	return sched, nil
}

// CancelSchedule cancels an active or pending schedule and deletes any associated grant.
func (s *Store) CancelSchedule(ctx context.Context, chatID, name string) (*Schedule, error) {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to begin transaction: %w", err)
	}
	defer func() { _ = tx.Rollback() }()

	name = strings.TrimSpace(name)
	sched, err := getScheduleByNameQ(ctx, tx, chatID, name)
	if err != nil {
		return nil, err
	}
	if sched.Status.IsTerminal() {
		return nil, fmt.Errorf("cannot cancel schedule with terminal status %s", sched.Status)
	}

	now := time.Now().Unix()
	sched.Status = ScheduleStatusCancelled
	sched.UpdatedAt = now

	res, err := tx.ExecContext(ctx, "UPDATE schedules SET status = ?, updated_at = ? WHERE id = ? AND status NOT IN ('CANCELLED', 'COMPLETED', 'DENIED', 'EXPIRED')",
		string(sched.Status), sched.UpdatedAt, sched.ID)
	if err != nil {
		return nil, fmt.Errorf("failed to update cancelled schedule status: %w", err)
	}
	n, err := res.RowsAffected()
	if err != nil {
		return nil, fmt.Errorf("failed to check rows affected: %w", err)
	}
	if n == 0 {
		return nil, fmt.Errorf("schedule %s is already in terminal status", sched.ID)
	}

	_, err = tx.ExecContext(ctx, "DELETE FROM schedule_grants WHERE schedule_id = ?", sched.ID)
	if err != nil {
		return nil, fmt.Errorf("failed to delete grant on cancellation: %w", err)
	}

	if err := tx.Commit(); err != nil {
		return nil, fmt.Errorf("failed to commit cancellation: %w", err)
	}
	return sched, nil
}

// PauseSchedule temporarily pauses an active schedule.
func (s *Store) PauseSchedule(ctx context.Context, chatID, name string) (*Schedule, error) {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to begin transaction: %w", err)
	}
	defer func() { _ = tx.Rollback() }()

	name = strings.TrimSpace(name)
	sched, err := getScheduleByNameQ(ctx, tx, chatID, name)
	if err != nil {
		return nil, err
	}
	if sched.Status != ScheduleStatusActive {
		return nil, fmt.Errorf("cannot pause schedule with status %s", sched.Status)
	}

	now := time.Now().Unix()
	sched.Status = ScheduleStatusPaused
	sched.UpdatedAt = now

	res, err := tx.ExecContext(ctx, "UPDATE schedules SET status = ?, updated_at = ? WHERE id = ? AND status = 'ACTIVE'",
		string(sched.Status), sched.UpdatedAt, sched.ID)
	if err != nil {
		return nil, fmt.Errorf("failed to pause schedule: %w", err)
	}
	n, err := res.RowsAffected()
	if err != nil {
		return nil, fmt.Errorf("failed to check rows affected: %w", err)
	}
	if n == 0 {
		return nil, fmt.Errorf("cannot pause schedule: status is no longer ACTIVE")
	}

	if err := tx.Commit(); err != nil {
		return nil, fmt.Errorf("failed to commit pause: %w", err)
	}
	return sched, nil
}

// ResumeSchedule resumes a paused schedule.
func (s *Store) ResumeSchedule(ctx context.Context, chatID, name string) (*Schedule, error) {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to begin transaction: %w", err)
	}
	defer func() { _ = tx.Rollback() }()

	name = strings.TrimSpace(name)
	sched, err := getScheduleByNameQ(ctx, tx, chatID, name)
	if err != nil {
		return nil, err
	}
	if sched.Status != ScheduleStatusPaused {
		return nil, fmt.Errorf("cannot resume schedule with status %s", sched.Status)
	}

	now := time.Now().Unix()
	sched.Status = ScheduleStatusActive
	sched.UpdatedAt = now

	res, err := tx.ExecContext(ctx, "UPDATE schedules SET status = ?, updated_at = ? WHERE id = ? AND status = 'PAUSED'",
		string(sched.Status), sched.UpdatedAt, sched.ID)
	if err != nil {
		return nil, fmt.Errorf("failed to resume schedule: %w", err)
	}
	n, err := res.RowsAffected()
	if err != nil {
		return nil, fmt.Errorf("failed to check rows affected: %w", err)
	}
	if n == 0 {
		return nil, fmt.Errorf("cannot resume schedule: status is no longer PAUSED")
	}

	if err := tx.Commit(); err != nil {
		return nil, fmt.Errorf("failed to commit resume: %w", err)
	}
	return sched, nil
}

// ListSchedules lists schedules for the chat context.
func (s *Store) ListSchedules(ctx context.Context, chatID string, activeOnly bool) ([]*Schedule, error) {
	var query string
	var rows *sql.Rows
	var err error

	if activeOnly {
		query = `
			SELECT id, name, chat_id, user_id, schedule_type, cron_expr, interval_seconds,
			       instruction, status, next_run_at, last_run_at, expires_at, max_runs,
			       run_count, missed_count, run_timeout_seconds, max_turns, last_status,
			       created_at, updated_at
			FROM schedules
			WHERE chat_id = ? AND status IN ('ACTIVE', 'PENDING_APPROVAL', 'PAUSED')
			ORDER BY next_run_at ASC
		`
		rows, err = s.db.QueryContext(ctx, query, chatID)
	} else {
		query = `
			SELECT id, name, chat_id, user_id, schedule_type, cron_expr, interval_seconds,
			       instruction, status, next_run_at, last_run_at, expires_at, max_runs,
			       run_count, missed_count, run_timeout_seconds, max_turns, last_status,
			       created_at, updated_at
			FROM schedules
			WHERE chat_id = ?
			ORDER BY created_at DESC
		`
		rows, err = s.db.QueryContext(ctx, query, chatID)
	}
	if err != nil {
		return nil, fmt.Errorf("failed to query schedules for chat %s: %w", chatID, err)
	}
	defer func() { _ = rows.Close() }()

	var results []*Schedule
	for rows.Next() {
		sched, err := scanScheduleRow(rows)
		if err != nil {
			return nil, err
		}
		results = append(results, sched)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error reading schedules rows: %w", err)
	}
	return results, nil
}

// ClaimDueSchedules finds and atomically claims active schedules that are due for execution.
// It optimistic CAS-locks claimed rows by advancing next_run_at to claimLockUntil (e.g. now + 300s).
func (s *Store) ClaimDueSchedules(ctx context.Context, cutoffTime int64, claimLockUntil int64, limit int) ([]*Schedule, error) {
	if limit <= 0 {
		limit = 10
	}

	query := `
		SELECT id, name, chat_id, user_id, schedule_type, cron_expr, interval_seconds,
		       instruction, status, next_run_at, last_run_at, expires_at, max_runs,
		       run_count, missed_count, run_timeout_seconds, max_turns, last_status,
		       created_at, updated_at
		FROM schedules
		WHERE status = 'ACTIVE' AND next_run_at <= ?
		ORDER BY next_run_at ASC
		LIMIT ?
	`
	rows, err := s.db.QueryContext(ctx, query, cutoffTime, limit)
	if err != nil {
		return nil, fmt.Errorf("failed to query due schedules: %w", err)
	}
	defer func() { _ = rows.Close() }()

	var candidates []*Schedule
	for rows.Next() {
		sched, err := scanScheduleRow(rows)
		if err != nil {
			return nil, err
		}
		candidates = append(candidates, sched)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error reading due schedule candidates: %w", err)
	}

	var claimed []*Schedule
	now := time.Now().Unix()
	casSQL := `
		UPDATE schedules
		SET next_run_at = ?, updated_at = ?
		WHERE id = ? AND next_run_at = ? AND status = 'ACTIVE'
	`
	for _, cand := range candidates {
		res, err := s.db.ExecContext(ctx, casSQL, claimLockUntil, now, cand.ID, cand.NextRunAt)
		if err != nil {
			continue
		}
		if n, _ := res.RowsAffected(); n == 1 {
			claimed = append(claimed, cand)
		}
	}
	return claimed, nil
}

// RecordExecutionResult records execution outcomes and updates recurrence state.
func (s *Store) RecordExecutionResult(ctx context.Context, scheduleID string, lastStatus string, nextRunAt int64, missedIncr int, shouldEnd bool) error {
	now := time.Now().Unix()
	var updateSQL string
	var err error

	var res sql.Result
	// missedIncr serves double duty: when > 0 the row is a "skip" (downtime recovery),
	// so run_count stays unchanged and only missed_count advances. When == 0 the row is
	// a real execution, so run_count increments by 1.
	if shouldEnd {
		updateSQL = `
			UPDATE schedules
			SET run_count = CASE WHEN ? > 0 THEN run_count ELSE run_count + 1 END,
			    missed_count = missed_count + ?,
			    last_run_at = ?,
			    last_status = ?,
			    status = ?,
			    updated_at = ?
			WHERE id = ? AND status = 'ACTIVE'
		`
		res, err = s.db.ExecContext(ctx, updateSQL, missedIncr, missedIncr, now, lastStatus, string(ScheduleStatusCompleted), now, scheduleID)
	} else {
		if nextRunAt <= 0 {
			return fmt.Errorf("nextRunAt must be positive for continuing schedule %s", scheduleID)
		}
		updateSQL = `
			UPDATE schedules
			SET run_count = CASE WHEN ? > 0 THEN run_count ELSE run_count + 1 END,
			    missed_count = missed_count + ?,
			    last_run_at = ?,
			    last_status = ?,
			    next_run_at = ?,
			    updated_at = ?
			WHERE id = ? AND status = 'ACTIVE'
		`
		res, err = s.db.ExecContext(ctx, updateSQL, missedIncr, missedIncr, now, lastStatus, nextRunAt, now, scheduleID)
	}
	if err != nil {
		return fmt.Errorf("failed to record execution result for %s: %w", scheduleID, err)
	}
	n, err := res.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to check rows affected: %w", err)
	}
	if n == 0 {
		return fmt.Errorf("schedule %s is not active", scheduleID)
	}
	return nil
}

// PruneTerminalSchedules removes completed, cancelled, expired, or denied schedules updated before olderThanUnix.
func (s *Store) PruneTerminalSchedules(ctx context.Context, olderThanUnix int64) (int64, error) {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return 0, fmt.Errorf("failed to begin transaction: %w", err)
	}
	defer func() { _ = tx.Rollback() }()

	// Explicitly delete grants first in case foreign keys were not active on this connection
	if _, err := tx.ExecContext(ctx, `
		DELETE FROM schedule_grants
		WHERE schedule_id IN (
			SELECT id FROM schedules
			WHERE status IN ('COMPLETED', 'CANCELLED', 'EXPIRED', 'DENIED')
			  AND updated_at < ?
		)
	`, olderThanUnix); err != nil {
		return 0, fmt.Errorf("failed to prune terminal schedule grants: %w", err)
	}

	res, err := tx.ExecContext(ctx, `
		DELETE FROM schedules
		WHERE status IN ('COMPLETED', 'CANCELLED', 'EXPIRED', 'DENIED')
		  AND updated_at < ?
	`, olderThanUnix)
	if err != nil {
		return 0, fmt.Errorf("failed to prune terminal schedules: %w", err)
	}

	if err := tx.Commit(); err != nil {
		return 0, fmt.Errorf("failed to commit pruning: %w", err)
	}
	return res.RowsAffected()
}

func scanSchedule(row *sql.Row) (*Schedule, error) {
	var s Schedule
	var st string
	var cronExpr sql.NullString
	var intervalSec sql.NullInt64
	var lastRunAt sql.NullInt64
	var expiresAt sql.NullInt64
	var lastStatus sql.NullString

	err := row.Scan(
		&s.ID, &s.Name, &s.ChatID, &s.UserID, &st, &cronExpr, &intervalSec,
		&s.Instruction, &s.Status, &s.NextRunAt, &lastRunAt, &expiresAt, &s.MaxRuns,
		&s.RunCount, &s.MissedCount, &s.RunTimeoutSeconds, &s.MaxTurns, &lastStatus,
		&s.CreatedAt, &s.UpdatedAt,
	)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return nil, ErrScheduleNotFound
		}
		return nil, fmt.Errorf("failed to scan schedule: %w", err)
	}

	s.ScheduleType = ScheduleType(st)
	if cronExpr.Valid {
		s.CronExpr = cronExpr.String
	}
	if intervalSec.Valid {
		s.IntervalSeconds = int(intervalSec.Int64)
	}
	if lastRunAt.Valid {
		v := lastRunAt.Int64
		s.LastRunAt = &v
	}
	if expiresAt.Valid {
		v := expiresAt.Int64
		s.ExpiresAt = &v
	}
	if lastStatus.Valid {
		s.LastStatus = lastStatus.String
	}
	return &s, nil
}

func scanScheduleRow(rows *sql.Rows) (*Schedule, error) {
	var s Schedule
	var st string
	var cronExpr sql.NullString
	var intervalSec sql.NullInt64
	var lastRunAt sql.NullInt64
	var expiresAt sql.NullInt64
	var lastStatus sql.NullString

	err := rows.Scan(
		&s.ID, &s.Name, &s.ChatID, &s.UserID, &st, &cronExpr, &intervalSec,
		&s.Instruction, &s.Status, &s.NextRunAt, &lastRunAt, &expiresAt, &s.MaxRuns,
		&s.RunCount, &s.MissedCount, &s.RunTimeoutSeconds, &s.MaxTurns, &lastStatus,
		&s.CreatedAt, &s.UpdatedAt,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to scan schedule row: %w", err)
	}

	s.ScheduleType = ScheduleType(st)
	if cronExpr.Valid {
		s.CronExpr = cronExpr.String
	}
	if intervalSec.Valid {
		s.IntervalSeconds = int(intervalSec.Int64)
	}
	if lastRunAt.Valid {
		v := lastRunAt.Int64
		s.LastRunAt = &v
	}
	if expiresAt.Valid {
		v := expiresAt.Int64
		s.ExpiresAt = &v
	}
	if lastStatus.Valid {
		s.LastStatus = lastStatus.String
	}
	return &s, nil
}

func nullString(s string) sql.NullString {
	if s == "" {
		return sql.NullString{}
	}
	return sql.NullString{String: s, Valid: true}
}

func nullInt(i int) sql.NullInt64 {
	if i == 0 {
		return sql.NullInt64{}
	}
	return sql.NullInt64{Int64: int64(i), Valid: true}
}

func nullInt64(i *int64) sql.NullInt64 {
	if i == nil {
		return sql.NullInt64{}
	}
	return sql.NullInt64{Int64: *i, Valid: true}
}
