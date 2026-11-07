package fsm

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"bob/internal/agentapi"
	"bob/internal/tools"
	"bob/internal/trajectory"
)

var (
	// ErrRunNotFound indicates that an FSM run with the given ID does not exist.
	ErrRunNotFound = errors.New("fsm run not found")
	// ErrStepNotFound indicates that an FSM step with the given ID does not exist.
	ErrStepNotFound = errors.New("fsm step not found")
	// ErrConcurrentUpdate indicates that an optimistic concurrency conflict occurred while updating an FSM run.
	ErrConcurrentUpdate = errors.New("concurrent fsm run update conflict")
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
			id, chat_id, user_id, is_dm, frontend_id, scope_id, model, execution_kind,
			fsm_type, status, current_state,
			iteration, max_iterations, wait_cycles, context_json, result_json,
			error_text, resume_at, version, source_message_seq, created_at, updated_at
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
	`

	var resumeAt sql.NullInt64
	if run.ResumeAt != nil {
		resumeAt = sql.NullInt64{Int64: *run.ResumeAt, Valid: true}
	}
	var sourceMsgSeq sql.NullInt64
	if run.SourceMessageSeq != nil {
		sourceMsgSeq = sql.NullInt64{Int64: *run.SourceMessageSeq, Valid: true}
	}

	_, err := s.db.ExecContext(ctx, query,
		run.ID, run.ChatID, run.UserID, isDMInt, run.FrontendID, run.ScopeID, run.Model, string(run.ExecutionKind),
		string(run.FSMType), string(run.Status), string(run.CurrentState),
		run.Iteration, run.MaxIterations, run.WaitCycles, run.ContextJSON, run.ResultJSON,
		run.ErrorText, resumeAt, run.Version, sourceMsgSeq, run.CreatedAt, run.UpdatedAt,
	)
	if err != nil {
		return fmt.Errorf("failed to insert fsm_run %s: %w", run.ID, err)
	}

	run.MarkContextSynced()
	return nil
}

type rowScanner interface {
	Scan(dest ...any) error
}

const runSelectColumns = `id, chat_id, user_id, is_dm, frontend_id, scope_id, model, execution_kind,
       fsm_type, status, current_state,
       iteration, max_iterations, wait_cycles, context_json, result_json,
       error_text, resume_at, version, source_message_seq, created_at, updated_at`

func scanRun(scanner rowScanner) (*FSMRun, error) {
	var run FSMRun
	var isDMInt int
	var executionKind string
	var fsmType, status, currentState string
	var resumeAt sql.NullInt64
	var sourceMsgSeq sql.NullInt64
	var resultJSON, errorText sql.NullString

	err := scanner.Scan(
		&run.ID, &run.ChatID, &run.UserID, &isDMInt,
		&run.FrontendID, &run.ScopeID, &run.Model, &executionKind,
		&fsmType, &status, &currentState,
		&run.Iteration, &run.MaxIterations, &run.WaitCycles, &run.ContextJSON, &resultJSON,
		&errorText, &resumeAt, &run.Version, &sourceMsgSeq, &run.CreatedAt, &run.UpdatedAt,
	)
	if err != nil {
		return nil, err
	}

	run.IsDM = isDMInt == 1
	run.ExecutionKind = agentapi.ExecutionKind(executionKind)
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
	if sourceMsgSeq.Valid {
		val := sourceMsgSeq.Int64
		run.SourceMessageSeq = &val
	}
	run.MarkContextSynced()

	return &run, nil
}

// GetRun retrieves an FSM run by its unique ID.
func (s *Store) GetRun(ctx context.Context, id string) (*FSMRun, error) {
	query := `
		SELECT ` + runSelectColumns + `
		FROM fsm_runs
		WHERE id = ?
	`

	run, err := scanRun(s.db.QueryRowContext(ctx, query, id))
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return nil, ErrRunNotFound
		}
		return nil, fmt.Errorf("failed to query fsm_run %s: %w", id, err)
	}

	return run, nil
}

// UpdateRunState updates the dynamic execution state, status, context, and timestamps of a run using optimistic concurrency control.
// If ContextJSON has not changed since last persistent sync, context_json is omitted from the SQL UPDATE to avoid write amplification.
func (s *Store) UpdateRunState(ctx context.Context, run *FSMRun) error {
	if run == nil {
		return fmt.Errorf("run cannot be nil")
	}
	oldVersion := run.Version
	newVersion := oldVersion + 1
	now := time.Now().Unix()

	isDMInt := 0
	if run.IsDM {
		isDMInt = 1
	}

	var resumeAt sql.NullInt64
	if run.ResumeAt != nil {
		resumeAt = sql.NullInt64{Int64: *run.ResumeAt, Valid: true}
	}

	isDirty := run.IsContextDirty()
	var res sql.Result
	var err error

	if isDirty {
		query := `
			UPDATE fsm_runs
			SET status = ?, current_state = ?, iteration = ?, wait_cycles = ?, context_json = ?,
			    result_json = ?, error_text = ?, resume_at = ?, is_dm = ?, version = ?, updated_at = ?
			WHERE id = ? AND version = ?
		`
		res, err = s.db.ExecContext(ctx, query,
			string(run.Status), string(run.CurrentState), run.Iteration, run.WaitCycles, run.ContextJSON,
			run.ResultJSON, run.ErrorText, resumeAt, isDMInt, newVersion, now, run.ID, oldVersion,
		)
	} else {
		query := `
			UPDATE fsm_runs
			SET status = ?, current_state = ?, iteration = ?, wait_cycles = ?,
			    result_json = ?, error_text = ?, resume_at = ?, is_dm = ?, version = ?, updated_at = ?
			WHERE id = ? AND version = ?
		`
		res, err = s.db.ExecContext(ctx, query,
			string(run.Status), string(run.CurrentState), run.Iteration, run.WaitCycles,
			run.ResultJSON, run.ErrorText, resumeAt, isDMInt, newVersion, now, run.ID, oldVersion,
		)
	}
	if err != nil {
		return fmt.Errorf("failed to update fsm_run %s: %w", run.ID, err)
	}

	affected, err := res.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to check affected rows for fsm_run %s: %w", run.ID, err)
	}
	if affected == 0 {
		var exists int
		if checkErr := s.db.QueryRowContext(ctx, "SELECT count(*) FROM fsm_runs WHERE id = ?", run.ID).Scan(&exists); checkErr == nil && exists > 0 {
			return ErrConcurrentUpdate
		}
		return ErrRunNotFound
	}

	run.Version = newVersion
	run.UpdatedAt = now
	run.MarkContextSynced()
	return nil
}

// UpdateRun is an alias for UpdateRunState.
func (s *Store) UpdateRun(ctx context.Context, run *FSMRun) error {
	return s.UpdateRunState(ctx, run)
}

// ListActiveRuns returns all runs currently in PENDING, RUNNING, or WAITING status.
func (s *Store) ListActiveRuns(ctx context.Context) ([]FSMRun, error) {
	query := `
		SELECT ` + runSelectColumns + `
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
		run, err := scanRun(rows)
		if err != nil {
			return nil, fmt.Errorf("failed to scan active fsm_run: %w", err)
		}
		runs = append(runs, *run)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error during active runs iteration: %w", err)
	}

	return runs, nil
}

// ListDueWaitingRuns returns runs in WAITING status whose resume_at deadline has arrived.
func (s *Store) ListDueWaitingRuns(ctx context.Context, nowUnix int64) ([]FSMRun, error) {
	query := `
		SELECT ` + runSelectColumns + `
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
		run, err := scanRun(rows)
		if err != nil {
			return nil, fmt.Errorf("failed to scan due waiting fsm_run: %w", err)
		}
		runs = append(runs, *run)
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
		ON CONFLICT(id) DO NOTHING
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

// ListStepsByRun returns all steps associated with a run, ordered by iteration ASC, step_index ASC.
func (s *Store) ListStepsByRun(ctx context.Context, runID string) ([]FSMStep, error) {
	query := `
		SELECT id, run_id, iteration, step_index, tool_name, tool_call_id,
		       args_json, result_json, execution_mode, status, attempt,
		       max_attempts, timeout_seconds, started_at, completed_at, error_text
		FROM fsm_steps
		WHERE run_id = ?
		ORDER BY iteration ASC, step_index ASC
	`

	return s.querySteps(ctx, query, runID)
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

// UpdateStepTx updates a step within an existing database transaction.
func (s *Store) UpdateStepTx(ctx context.Context, tx *sql.Tx, step *FSMStep) error {
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

	res, err := tx.ExecContext(ctx, query,
		string(step.Status), step.Attempt, step.ResultJSON, step.ErrorText,
		startedAt, completedAt, step.ID,
	)
	if err != nil {
		return fmt.Errorf("failed to update fsm_step %s in tx: %w", step.ID, err)
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

// UpdateStepWithEvent updates a step and atomically logs an outbox event in the same transaction.
func (s *Store) UpdateStepWithEvent(ctx context.Context, step *FSMStep, event *trajectory.Event) error {
	if step == nil {
		return fmt.Errorf("step cannot be nil")
	}

	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
	}
	defer tx.Rollback() //nolint:errcheck

	if err := s.UpdateStepTx(ctx, tx, step); err != nil {
		return err
	}

	if event != nil {
		if err := trajectory.RecordEventTx(ctx, tx, event); err != nil {
			return fmt.Errorf("failed to record step event in tx: %w", err)
		}
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("failed to commit step and event tx: %w", err)
	}

	return nil
}

// UpdateStep updates the execution results, attempt counter, status, and timestamps of a step.
func (s *Store) UpdateStep(ctx context.Context, step *FSMStep) error {
	return s.UpdateStepWithEvent(ctx, step, nil)
}

// RecordEvent logs an outbox event to agent_trajectory_events using the store's database.
func (s *Store) RecordEvent(ctx context.Context, event *trajectory.Event) error {
	return trajectory.RecordEvent(ctx, s.db, event)
}

// RecordEventTx logs an outbox event to agent_trajectory_events within an existing transaction.
func (s *Store) RecordEventTx(ctx context.Context, tx *sql.Tx, event *trajectory.Event) error {
	return trajectory.RecordEventTx(ctx, tx, event)
}

// NormalizeCompactionState validates and normalizes a CompactedTaskState prior to persistence or event creation,
// returning its canonical JSON serialization whose byte length unconditionally matches c.OutputBytes.
func NormalizeCompactionState(c *CompactedTaskState) ([]byte, error) {
	if c == nil {
		return nil, fmt.Errorf("compaction state cannot be nil")
	}
	if c.RunID == "" {
		return nil, fmt.Errorf("compaction run_id cannot be empty")
	}
	if c.Version <= 0 {
		return nil, fmt.Errorf("compaction version must be > 0 (got %d)", c.Version)
	}
	if c.ParentVersion < 0 || c.ParentVersion >= c.Version {
		return nil, fmt.Errorf("compaction parent_version must be in [0, %d) (got %d)", c.Version, c.ParentVersion)
	}
	if c.CompactedUpToIter <= 0 {
		return nil, fmt.Errorf("compaction compacted_up_to_iter must be > 0 (got %d)", c.CompactedUpToIter)
	}
	if c.InputBytes < 0 || c.OutputBytes < 0 {
		return nil, fmt.Errorf("compaction byte counts cannot be negative (input=%d, output=%d)", c.InputBytes, c.OutputBytes)
	}
	if c.ID == "" {
		c.ID = fmt.Sprintf("%s_c_v%d", c.RunID, c.Version)
	}
	if c.CreatedAt == 0 {
		c.CreatedAt = time.Now().Unix()
	}
	if c.RubricVerdict == "" {
		c.RubricVerdict = CompactionVerdictProbedCompact
	}
	if c.RubricVerdict != CompactionVerdictProbedCompact && c.RubricVerdict != CompactionVerdictBackstopForced {
		return nil, fmt.Errorf("invalid persisted compaction rubric_verdict %q", c.RubricVerdict)
	}

	stateBytes, err := json.Marshal(c)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal compacted task state: %w", err)
	}
	c.OutputBytes = len(stateBytes)
	return stateBytes, nil
}

// MakeSnapshotCompactionEvent normalizes a CompactedTaskState and builds a matching trajectory compaction event.
// execIteration is the FSM iteration at which compaction ran (must be >= c.CompactedUpToIter when > 0; defaults to c.CompactedUpToIter when <= 0).
func MakeSnapshotCompactionEvent(c *CompactedTaskState, trajID string, gen, execIteration int, reason string) (*trajectory.Event, error) {
	stateBytes, err := NormalizeCompactionState(c)
	if err != nil {
		return nil, err
	}
	if execIteration <= 0 {
		execIteration = c.CompactedUpToIter
	}
	if execIteration < c.CompactedUpToIter {
		return nil, fmt.Errorf("compaction execution iteration %d cannot be less than compacted_up_to_iter %d", execIteration, c.CompactedUpToIter)
	}
	if trajID == "" {
		trajID = c.RunID
	}
	return trajectory.MakeCompactionEvent(trajID, c.RunID, gen, execIteration, "", trajectory.CompactionEventPayload{
		Version:           c.Version,
		ParentVersion:     c.ParentVersion,
		CompactedUpToIter: c.CompactedUpToIter,
		ShouldCompact:     true,
		RubricVerdict:     c.RubricVerdict,
		Reason:            reason,
		Model:             c.Model,
		InputBytes:        c.InputBytes,
		OutputBytes:       c.OutputBytes,
		StateJSON:         json.RawMessage(stateBytes),
	})
}

// SaveCompaction persists a new versioned CompactedTaskState snapshot for a run.
func (s *Store) SaveCompaction(ctx context.Context, c *CompactedTaskState) error {
	return s.SaveCompactionWithEvent(ctx, c, nil)
}

// SaveCompactionWithEvent persists a new versioned CompactedTaskState snapshot and atomically records a trajectory outbox event.
func (s *Store) SaveCompactionWithEvent(ctx context.Context, c *CompactedTaskState, event *trajectory.Event) error {
	stateBytes, err := NormalizeCompactionState(c)
	if err != nil {
		return err
	}

	if event != nil {
		if event.RunID != c.RunID {
			return fmt.Errorf("compaction event run_id %q does not match compaction run_id %q", event.RunID, c.RunID)
		}
		if event.EventType != "compaction" {
			return fmt.Errorf("compaction event must have event_type \"compaction\" (got %q)", event.EventType)
		}
		if event.Iteration < c.CompactedUpToIter {
			return fmt.Errorf("compaction event iteration %d cannot be less than compacted_up_to_iter %d", event.Iteration, c.CompactedUpToIter)
		}
		var evPayload trajectory.CompactionEventPayload
		if err := json.Unmarshal(event.Payload, &evPayload); err != nil {
			return fmt.Errorf("failed to unmarshal compaction event payload: %w", err)
		}
		if !evPayload.ShouldCompact ||
			evPayload.Version != c.Version ||
			evPayload.ParentVersion != c.ParentVersion ||
			evPayload.CompactedUpToIter != c.CompactedUpToIter ||
			evPayload.RubricVerdict != c.RubricVerdict ||
			evPayload.Model != c.Model ||
			evPayload.InputBytes != c.InputBytes ||
			evPayload.OutputBytes != c.OutputBytes ||
			!bytes.Equal(evPayload.StateJSON, stateBytes) {
			return fmt.Errorf("compaction event payload metadata or state_json does not match normalized snapshot")
		}
	}

	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to begin transaction for save compaction: %w", err)
	}
	defer func() { _ = tx.Rollback() }()

	var prevVersion, prevIter int
	err = tx.QueryRowContext(ctx, `
		SELECT version, compacted_up_to_iter
		FROM fsm_compactions
		WHERE run_id = ?
		ORDER BY version DESC
		LIMIT 1
	`, c.RunID).Scan(&prevVersion, &prevIter)
	if err != nil && !errors.Is(err, sql.ErrNoRows) {
		return fmt.Errorf("failed to check existing compaction for run %s: %w", c.RunID, err)
	}
	if errors.Is(err, sql.ErrNoRows) {
		if c.ParentVersion != 0 {
			return fmt.Errorf("first compaction for run %s must have parent_version 0 (got %d)", c.RunID, c.ParentVersion)
		}
	} else {
		if c.Version <= prevVersion {
			return fmt.Errorf("compaction version %d must be > latest version %d", c.Version, prevVersion)
		}
		if c.ParentVersion != prevVersion {
			return fmt.Errorf("compaction parent_version %d must match latest version %d", c.ParentVersion, prevVersion)
		}
		if c.CompactedUpToIter <= prevIter {
			return fmt.Errorf("compaction compacted_up_to_iter %d must advance beyond %d", c.CompactedUpToIter, prevIter)
		}
	}

	query := `
		INSERT INTO fsm_compactions (
			id, run_id, version, parent_version, compacted_up_to_iter,
			rubric_verdict, state_json, model, input_bytes, output_bytes, created_at
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
	`
	_, err = tx.ExecContext(ctx, query,
		c.ID, c.RunID, c.Version, c.ParentVersion, c.CompactedUpToIter,
		c.RubricVerdict, string(stateBytes), c.Model, c.InputBytes, c.OutputBytes, c.CreatedAt,
	)
	if err != nil {
		return fmt.Errorf("failed to insert fsm_compaction %s: %w", c.ID, err)
	}

	if event != nil {
		if err := trajectory.RecordEventTx(ctx, tx, event); err != nil {
			return fmt.Errorf("failed to record compaction event in tx: %w", err)
		}
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("failed to commit save compaction transaction: %w", err)
	}

	return nil
}

func scanCompaction(scanner rowScanner) (*CompactedTaskState, error) {
	var id, runID, rubricVerdict, stateJSON, model string
	var version, parentVersion, compactedUpToIter, inputBytes, outputBytes int
	var createdAt int64

	err := scanner.Scan(
		&id, &runID, &version, &parentVersion, &compactedUpToIter,
		&rubricVerdict, &stateJSON, &model, &inputBytes, &outputBytes, &createdAt,
	)
	if err != nil {
		return nil, err
	}

	var state CompactedTaskState
	if stateJSON != "" {
		if err := json.Unmarshal([]byte(stateJSON), &state); err != nil {
			return nil, fmt.Errorf("failed to unmarshal fsm_compaction %s state_json: %w", id, err)
		}
	}

	state.ID = id
	state.RunID = runID
	state.Version = version
	state.ParentVersion = parentVersion
	state.CompactedUpToIter = compactedUpToIter
	state.RubricVerdict = rubricVerdict
	state.Model = model
	state.InputBytes = inputBytes
	state.OutputBytes = outputBytes
	state.CreatedAt = createdAt

	return &state, nil
}

// GetLatestCompaction returns the highest-version CompactedTaskState for a run, or (nil, nil) if none exists.
func (s *Store) GetLatestCompaction(ctx context.Context, runID string) (*CompactedTaskState, error) {
	query := `
		SELECT id, run_id, version, parent_version, compacted_up_to_iter,
		       rubric_verdict, state_json, model, input_bytes, output_bytes, created_at
		FROM fsm_compactions
		WHERE run_id = ?
		ORDER BY version DESC
		LIMIT 1
	`

	state, err := scanCompaction(s.db.QueryRowContext(ctx, query, runID))
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return nil, nil
		}
		return nil, fmt.Errorf("failed to query latest fsm_compaction for run %s: %w", runID, err)
	}

	return state, nil
}

// ListCompactions returns all compaction snapshots for a run ordered by version ASC.
func (s *Store) ListCompactions(ctx context.Context, runID string) ([]CompactedTaskState, error) {
	query := `
		SELECT id, run_id, version, parent_version, compacted_up_to_iter,
		       rubric_verdict, state_json, model, input_bytes, output_bytes, created_at
		FROM fsm_compactions
		WHERE run_id = ?
		ORDER BY version ASC
	`

	rows, err := s.db.QueryContext(ctx, query, runID)
	if err != nil {
		return nil, fmt.Errorf("failed to list fsm_compactions for run %s: %w", runID, err)
	}
	defer func() { _ = rows.Close() }()

	var result []CompactedTaskState
	for rows.Next() {
		state, err := scanCompaction(rows)
		if err != nil {
			return nil, err
		}
		result = append(result, *state)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error iterating fsm_compactions for run %s: %w", runID, err)
	}

	return result, nil
}

// HasCompactionEventForIteration reports whether a compaction outbox event has already been recorded for the given run and iteration.
func (s *Store) HasCompactionEventForIteration(ctx context.Context, runID string, iteration int) (bool, error) {
	var count int
	err := s.db.QueryRowContext(ctx, `
		SELECT count(*)
		FROM agent_trajectory_events
		WHERE run_id = ? AND iteration = ? AND event_type = 'compaction'
	`, runID, iteration).Scan(&count)
	if err != nil {
		return false, fmt.Errorf("failed to query compaction events for run %s iter %d: %w", runID, iteration, err)
	}
	return count > 0, nil
}

// ListTaskHistorySteps implements tools.TaskHistoryStore by verifying that runID belongs to
// the specified chat/session and returning all durable FSMSteps for that run.
func (s *Store) ListTaskHistorySteps(ctx context.Context, chatID, userID string, isDM bool, runID string) ([]tools.TaskHistoryStep, error) {
	if s == nil || s.db == nil {
		return nil, errors.New("fsm store is not initialized")
	}
	run, err := s.GetRun(ctx, runID)
	if err != nil {
		return nil, fmt.Errorf("failed to resolve run %s: %w", runID, err)
	}
	if run.ChatID != chatID {
		return nil, fmt.Errorf("run %s does not belong to chat %s", runID, chatID)
	}
	if run.IsDM != isDM {
		return nil, fmt.Errorf("run %s channel scope mismatch", runID)
	}
	if run.IsDM && (run.UserID == "" || userID == "") {
		return nil, fmt.Errorf("run %s requires non-empty DM user owner", runID)
	}
	if run.UserID != userID {
		return nil, fmt.Errorf("run %s does not belong to user %s", runID, userID)
	}

	steps, err := s.ListStepsByRun(ctx, runID)
	if err != nil {
		return nil, err
	}

	res := make([]tools.TaskHistoryStep, 0, len(steps))
	for _, st := range steps {
		var content string
		if st.ResultJSON != "" {
			decoded, decErr := DecodeStoredToolResult(st.ResultJSON)
			if decErr == nil {
				content = decoded.Content
			} else {
				content = st.ResultJSON
			}
		}
		res = append(res, tools.TaskHistoryStep{
			ID:          st.ID,
			RunID:       st.RunID,
			Iteration:   st.Iteration,
			StepIndex:   st.StepIndex,
			ToolName:    st.ToolName,
			ToolCallID:  st.ToolCallID,
			ArgsJSON:    st.ArgsJSON,
			Status:      string(st.Status),
			Content:     content,
			ErrorText:   st.ErrorText,
			StartedAt:   st.StartedAt,
			CompletedAt: st.CompletedAt,
		})
	}
	return res, nil
}
