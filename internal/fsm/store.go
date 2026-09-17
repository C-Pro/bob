package fsm

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"time"
)

var (
	// ErrRunNotFound indicates that an FSM run with the given ID does not exist.
	ErrRunNotFound = errors.New("fsm run not found")
	// ErrStepNotFound indicates that an FSM step with the given ID does not exist.
	ErrStepNotFound = errors.New("fsm step not found")
)

// Store provides persistence methods for FSM runs and steps on a specific SQLite database.
type Store struct {
	db *sql.DB
}

// NewStore creates a new Store bound to a SQLite database handle.
func NewStore(db *sql.DB) *Store {
	return &Store{db: db}
}

// DB returns the underlying database handle.
func (s *Store) DB() *sql.DB {
	return s.db
}

// CreateRun inserts a new FSM run into the database.
func (s *Store) CreateRun(ctx context.Context, run *FSMRun) error {
	if run == nil {
		return fmt.Errorf("run cannot be nil")
	}
	now := time.Now().Unix()
	if run.CreatedAt == 0 {
		run.CreatedAt = now
	}
	if run.UpdatedAt == 0 {
		run.UpdatedAt = now
	}
	if run.Status == "" {
		run.Status = RunStatusPending
	}
	if run.CurrentState == "" {
		run.CurrentState = StateInit
	}

	isDMInt := 0
	if run.IsDM {
		isDMInt = 1
	}

	query := `
		INSERT INTO fsm_runs (
			id, chat_id, user_id, is_dm, fsm_type, status, current_state,
			iteration, max_iterations, context_json, result_json,
			error_text, resume_at, created_at, updated_at
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
	`

	var resumeAt sql.NullInt64
	if run.ResumeAt != nil {
		resumeAt = sql.NullInt64{Int64: *run.ResumeAt, Valid: true}
	}

	_, err := s.db.ExecContext(ctx, query,
		run.ID, run.ChatID, run.UserID, isDMInt, string(run.FSMType), string(run.Status), string(run.CurrentState),
		run.Iteration, run.MaxIterations, run.ContextJSON, run.ResultJSON,
		run.ErrorText, resumeAt, run.CreatedAt, run.UpdatedAt,
	)
	if err != nil {
		return fmt.Errorf("failed to insert fsm_run %s: %w", run.ID, err)
	}

	return nil
}

// GetRun retrieves an FSM run by its unique ID.
func (s *Store) GetRun(ctx context.Context, id string) (*FSMRun, error) {
	query := `
		SELECT id, chat_id, user_id, is_dm, fsm_type, status, current_state,
		       iteration, max_iterations, context_json, result_json,
		       error_text, resume_at, created_at, updated_at
		FROM fsm_runs
		WHERE id = ?
	`

	var run FSMRun
	var isDMInt int
	var fsmType, status, currentState string
	var resumeAt sql.NullInt64
	var resultJSON, errorText sql.NullString

	err := s.db.QueryRowContext(ctx, query, id).Scan(
		&run.ID, &run.ChatID, &run.UserID, &isDMInt, &fsmType, &status, &currentState,
		&run.Iteration, &run.MaxIterations, &run.ContextJSON, &resultJSON,
		&errorText, &resumeAt, &run.CreatedAt, &run.UpdatedAt,
	)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return nil, ErrRunNotFound
		}
		return nil, fmt.Errorf("failed to query fsm_run %s: %w", id, err)
	}

	run.IsDM = isDMInt == 1
	run.FSMType = FSMType(fsmType)
	run.Status = RunStatus(status)
	run.CurrentState = RunState(currentState)
	if resultJSON.Valid {
		run.ResultJSON = resultJSON.String
	}
	if errorText.Valid {
		run.ErrorText = errorText.String
	}
	if resumeAt.Valid {
		val := resumeAt.Int64
		run.ResumeAt = &val
	}

	return &run, nil
}

// UpdateRunState updates the dynamic execution state, status, context, and timestamps of a run.
func (s *Store) UpdateRunState(ctx context.Context, run *FSMRun) error {
	if run == nil {
		return fmt.Errorf("run cannot be nil")
	}
	run.UpdatedAt = time.Now().Unix()

	isDMInt := 0
	if run.IsDM {
		isDMInt = 1
	}

	query := `
		UPDATE fsm_runs
		SET status = ?, current_state = ?, iteration = ?, context_json = ?,
		    result_json = ?, error_text = ?, resume_at = ?, is_dm = ?, updated_at = ?
		WHERE id = ?
	`

	var resumeAt sql.NullInt64
	if run.ResumeAt != nil {
		resumeAt = sql.NullInt64{Int64: *run.ResumeAt, Valid: true}
	}

	res, err := s.db.ExecContext(ctx, query,
		string(run.Status), string(run.CurrentState), run.Iteration, run.ContextJSON,
		run.ResultJSON, run.ErrorText, resumeAt, isDMInt, run.UpdatedAt, run.ID,
	)
	if err != nil {
		return fmt.Errorf("failed to update fsm_run %s: %w", run.ID, err)
	}

	affected, err := res.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to check affected rows for fsm_run %s: %w", run.ID, err)
	}
	if affected == 0 {
		return ErrRunNotFound
	}

	return nil
}

// UpdateRun is an alias for UpdateRunState.
func (s *Store) UpdateRun(ctx context.Context, run *FSMRun) error {
	return s.UpdateRunState(ctx, run)
}

// ListActiveRuns returns all runs currently in PENDING, RUNNING, or WAITING status.
func (s *Store) ListActiveRuns(ctx context.Context) ([]FSMRun, error) {
	query := `
		SELECT id, chat_id, user_id, is_dm, fsm_type, status, current_state,
		       iteration, max_iterations, context_json, result_json,
		       error_text, resume_at, created_at, updated_at
		FROM fsm_runs
		WHERE status IN ('PENDING', 'RUNNING', 'WAITING')
		ORDER BY created_at ASC
	`

	rows, err := s.db.QueryContext(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to list active fsm_runs: %w", err)
	}
	defer func() { _ = rows.Close() }()

	var runs []FSMRun
	for rows.Next() {
		var run FSMRun
		var isDMInt int
		var fsmType, status, currentState string
		var resumeAt sql.NullInt64
		var resultJSON, errorText sql.NullString

		if err := rows.Scan(
			&run.ID, &run.ChatID, &run.UserID, &isDMInt, &fsmType, &status, &currentState,
			&run.Iteration, &run.MaxIterations, &run.ContextJSON, &resultJSON,
			&errorText, &resumeAt, &run.CreatedAt, &run.UpdatedAt,
		); err != nil {
			return nil, fmt.Errorf("failed to scan active fsm_run: %w", err)
		}

		run.IsDM = isDMInt == 1
		run.FSMType = FSMType(fsmType)
		run.Status = RunStatus(status)
		run.CurrentState = RunState(currentState)
		if resultJSON.Valid {
			run.ResultJSON = resultJSON.String
		}
		if errorText.Valid {
			run.ErrorText = errorText.String
		}
		if resumeAt.Valid {
			val := resumeAt.Int64
			run.ResumeAt = &val
		}

		runs = append(runs, run)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error during active runs iteration: %w", err)
	}

	return runs, nil
}

// ListDueWaitingRuns returns runs in WAITING status whose resume_at deadline has arrived.
func (s *Store) ListDueWaitingRuns(ctx context.Context, nowUnix int64) ([]FSMRun, error) {
	query := `
		SELECT id, chat_id, user_id, is_dm, fsm_type, status, current_state,
		       iteration, max_iterations, context_json, result_json,
		       error_text, resume_at, created_at, updated_at
		FROM fsm_runs
		WHERE status = 'WAITING' AND resume_at IS NOT NULL AND resume_at <= ?
		ORDER BY resume_at ASC
	`

	rows, err := s.db.QueryContext(ctx, query, nowUnix)
	if err != nil {
		return nil, fmt.Errorf("failed to list due waiting fsm_runs: %w", err)
	}
	defer func() { _ = rows.Close() }()

	var runs []FSMRun
	for rows.Next() {
		var run FSMRun
		var isDMInt int
		var fsmType, status, currentState string
		var resumeAt sql.NullInt64
		var resultJSON, errorText sql.NullString

		if err := rows.Scan(
			&run.ID, &run.ChatID, &run.UserID, &isDMInt, &fsmType, &status, &currentState,
			&run.Iteration, &run.MaxIterations, &run.ContextJSON, &resultJSON,
			&errorText, &resumeAt, &run.CreatedAt, &run.UpdatedAt,
		); err != nil {
			return nil, fmt.Errorf("failed to scan due waiting fsm_run: %w", err)
		}

		run.IsDM = isDMInt == 1
		run.FSMType = FSMType(fsmType)
		run.Status = RunStatus(status)
		run.CurrentState = RunState(currentState)
		if resultJSON.Valid {
			run.ResultJSON = resultJSON.String
		}
		if errorText.Valid {
			run.ErrorText = errorText.String
		}
		if resumeAt.Valid {
			val := resumeAt.Int64
			run.ResumeAt = &val
		}

		runs = append(runs, run)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error during due waiting runs iteration: %w", err)
	}

	return runs, nil
}

// CreateSteps inserts a batch of FSM steps atomically within a single transaction.
func (s *Store) CreateSteps(ctx context.Context, steps []FSMStep) error {
	if len(steps) == 0 {
		return nil
	}

	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to begin transaction for create steps: %w", err)
	}
	defer func() { _ = tx.Rollback() }()

	query := `
		INSERT INTO fsm_steps (
			id, run_id, iteration, step_index, tool_name, tool_call_id,
			args_json, result_json, execution_mode, status, attempt,
			max_attempts, timeout_seconds, started_at, completed_at, error_text
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
	`

	stmt, err := tx.PrepareContext(ctx, query)
	if err != nil {
		return fmt.Errorf("failed to prepare create step stmt: %w", err)
	}
	defer func() { _ = stmt.Close() }()

	for _, step := range steps {
		var startedAt, completedAt sql.NullInt64
		if step.StartedAt != nil {
			startedAt = sql.NullInt64{Int64: *step.StartedAt, Valid: true}
		}
		if step.CompletedAt != nil {
			completedAt = sql.NullInt64{Int64: *step.CompletedAt, Valid: true}
		}

		status := step.Status
		if status == "" {
			status = StepStatusPending
		}
		mode := step.ExecutionMode
		if mode == "" {
			mode = ExecutionModeSequential
		}
		timeout := step.TimeoutSeconds
		if timeout <= 0 {
			timeout = 30
		}
		maxAttempts := step.MaxAttempts
		if maxAttempts <= 0 {
			maxAttempts = 3
		}

		_, err := stmt.ExecContext(ctx,
			step.ID, step.RunID, step.Iteration, step.StepIndex, step.ToolName, step.ToolCallID,
			step.ArgsJSON, step.ResultJSON, string(mode), string(status), step.Attempt,
			maxAttempts, timeout, startedAt, completedAt, step.ErrorText,
		)
		if err != nil {
			return fmt.Errorf("failed to insert fsm_step %s: %w", step.ID, err)
		}
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("failed to commit create steps transaction: %w", err)
	}

	return nil
}

// GetStep retrieves an FSM step by its unique ID.
func (s *Store) GetStep(ctx context.Context, id string) (*FSMStep, error) {
	query := `
		SELECT id, run_id, iteration, step_index, tool_name, tool_call_id,
		       args_json, result_json, execution_mode, status, attempt,
		       max_attempts, timeout_seconds, started_at, completed_at, error_text
		FROM fsm_steps
		WHERE id = ?
	`

	var step FSMStep
	var mode, status string
	var startedAt, completedAt sql.NullInt64
	var resultJSON, errorText sql.NullString

	err := s.db.QueryRowContext(ctx, query, id).Scan(
		&step.ID, &step.RunID, &step.Iteration, &step.StepIndex, &step.ToolName, &step.ToolCallID,
		&step.ArgsJSON, &resultJSON, &mode, &status, &step.Attempt,
		&step.MaxAttempts, &step.TimeoutSeconds, &startedAt, &completedAt, &errorText,
	)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return nil, ErrStepNotFound
		}
		return nil, fmt.Errorf("failed to query fsm_step %s: %w", id, err)
	}

	step.ExecutionMode = ExecutionMode(mode)
	step.Status = StepStatus(status)
	if resultJSON.Valid {
		step.ResultJSON = resultJSON.String
	}
	if errorText.Valid {
		step.ErrorText = errorText.String
	}
	if startedAt.Valid {
		val := startedAt.Int64
		step.StartedAt = &val
	}
	if completedAt.Valid {
		val := completedAt.Int64
		step.CompletedAt = &val
	}

	return &step, nil
}

// GetStepsForIteration returns all steps associated with a run and iteration, ordered by step_index ASC.
func (s *Store) GetStepsForIteration(ctx context.Context, runID string, iteration int) ([]FSMStep, error) {
	query := `
		SELECT id, run_id, iteration, step_index, tool_name, tool_call_id,
		       args_json, result_json, execution_mode, status, attempt,
		       max_attempts, timeout_seconds, started_at, completed_at, error_text
		FROM fsm_steps
		WHERE run_id = ? AND iteration = ?
		ORDER BY step_index ASC
	`

	return s.querySteps(ctx, query, runID, iteration)
}

// ListStepsByIteration is an alias for GetStepsForIteration.
func (s *Store) ListStepsByIteration(ctx context.Context, runID string, iteration int) ([]FSMStep, error) {
	return s.GetStepsForIteration(ctx, runID, iteration)
}

// GetPendingSteps returns all pending steps for a run and iteration, ordered by step_index ASC.
func (s *Store) GetPendingSteps(ctx context.Context, runID string, iteration int) ([]FSMStep, error) {
	query := `
		SELECT id, run_id, iteration, step_index, tool_name, tool_call_id,
		       args_json, result_json, execution_mode, status, attempt,
		       max_attempts, timeout_seconds, started_at, completed_at, error_text
		FROM fsm_steps
		WHERE run_id = ? AND iteration = ? AND status = 'PENDING'
		ORDER BY step_index ASC
	`

	return s.querySteps(ctx, query, runID, iteration)
}

func (s *Store) querySteps(ctx context.Context, query string, args ...any) ([]FSMStep, error) {
	rows, err := s.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("failed to query fsm_steps: %w", err)
	}
	defer func() { _ = rows.Close() }()

	var steps []FSMStep
	for rows.Next() {
		var step FSMStep
		var mode, status string
		var startedAt, completedAt sql.NullInt64
		var resultJSON, errorText sql.NullString

		if err := rows.Scan(
			&step.ID, &step.RunID, &step.Iteration, &step.StepIndex, &step.ToolName, &step.ToolCallID,
			&step.ArgsJSON, &resultJSON, &mode, &status, &step.Attempt,
			&step.MaxAttempts, &step.TimeoutSeconds, &startedAt, &completedAt, &errorText,
		); err != nil {
			return nil, fmt.Errorf("failed to scan fsm_step: %w", err)
		}

		step.ExecutionMode = ExecutionMode(mode)
		step.Status = StepStatus(status)
		if resultJSON.Valid {
			step.ResultJSON = resultJSON.String
		}
		if errorText.Valid {
			step.ErrorText = errorText.String
		}
		if startedAt.Valid {
			val := startedAt.Int64
			step.StartedAt = &val
		}
		if completedAt.Valid {
			val := completedAt.Int64
			step.CompletedAt = &val
		}

		steps = append(steps, step)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error during fsm_steps iteration: %w", err)
	}

	return steps, nil
}

// UpdateStep updates the execution results, attempt counter, status, and timestamps of a step.
func (s *Store) UpdateStep(ctx context.Context, step *FSMStep) error {
	if step == nil {
		return fmt.Errorf("step cannot be nil")
	}

	query := `
		UPDATE fsm_steps
		SET status = ?, attempt = ?, result_json = ?, error_text = ?,
		    started_at = ?, completed_at = ?
		WHERE id = ?
	`

	var startedAt, completedAt sql.NullInt64
	if step.StartedAt != nil {
		startedAt = sql.NullInt64{Int64: *step.StartedAt, Valid: true}
	}
	if step.CompletedAt != nil {
		completedAt = sql.NullInt64{Int64: *step.CompletedAt, Valid: true}
	}

	res, err := s.db.ExecContext(ctx, query,
		string(step.Status), step.Attempt, step.ResultJSON, step.ErrorText,
		startedAt, completedAt, step.ID,
	)
	if err != nil {
		return fmt.Errorf("failed to update fsm_step %s: %w", step.ID, err)
	}

	affected, err := res.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to check affected rows for fsm_step %s: %w", step.ID, err)
	}
	if affected == 0 {
		return ErrStepNotFound
	}

	return nil
}
