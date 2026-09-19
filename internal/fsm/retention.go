package fsm

import (
	"context"
	"database/sql"
	"fmt"
	"log/slog"
	"time"
)

// RetentionManager manages the lifecycle and cleanup of expired terminal FSM runs.
type RetentionManager struct {
	db            *sql.DB
	retentionDays int
}

// NewRetentionManager creates a new retention manager for a SQLite database.
func NewRetentionManager(db *sql.DB, retentionDays int) *RetentionManager {
	if retentionDays <= 0 {
		retentionDays = 7
	}
	return &RetentionManager{
		db:            db,
		retentionDays: retentionDays,
	}
}

// PruneTerminalRuns deletes completed, failed, and terminated runs older than cutoffTime.
// Active, running, and waiting workflows are strictly immune to pruning.
// Associated fsm_steps rows are automatically cleaned up via ON DELETE CASCADE.
func (r *RetentionManager) PruneTerminalRuns(ctx context.Context, cutoffTime time.Time) (int64, error) {
	if r.db == nil {
		return 0, fmt.Errorf("db cannot be nil")
	}

	var tableExists int
	err := r.db.QueryRowContext(ctx, "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='fsm_runs'").Scan(&tableExists)
	if err != nil {
		return 0, fmt.Errorf("failed to check fsm_runs table: %w", err)
	}
	if tableExists == 0 {
		return 0, nil
	}

	query := `
		DELETE FROM fsm_runs
		WHERE status IN ('COMPLETED', 'FAILED', 'TERMINATED')
		  AND updated_at < ?
	`

	res, err := r.db.ExecContext(ctx, query, cutoffTime.Unix())
	if err != nil {
		return 0, fmt.Errorf("failed to prune terminal fsm_runs: %w", err)
	}

	affected, err := res.RowsAffected()
	if err != nil {
		return 0, fmt.Errorf("failed to get rows affected by fsm prune: %w", err)
	}

	return affected, nil
}

// IncrementalVacuum releases freed database pages back to the operating system.
// If pages <= 0, a default of 500 pages (approx 2MB) is requested.
// If PRAGMA auto_vacuum is 0 (disabled), a warning is logged and vacuuming is skipped.
func (r *RetentionManager) IncrementalVacuum(ctx context.Context, pages int) error {
	if r.db == nil {
		return fmt.Errorf("db cannot be nil")
	}

	var autoVacuum int
	if err := r.db.QueryRowContext(ctx, "PRAGMA auto_vacuum;").Scan(&autoVacuum); err != nil {
		return fmt.Errorf("failed to check auto_vacuum status: %w", err)
	}
	if autoVacuum == 0 {
		slog.Warn("incremental vacuum skipped because database auto_vacuum is disabled (0)")
		return nil
	}

	if pages <= 0 {
		pages = 500
	}

	query := fmt.Sprintf("PRAGMA incremental_vacuum(%d);", pages) // nosemgrep: go.lang.security.audit.database.string-formatted-query.string-formatted-query
	if _, err := r.db.ExecContext(ctx, query); err != nil {
		return fmt.Errorf("failed to execute incremental_vacuum: %w", err)
	}

	return nil
}

// PruneAndCompact executes a pruning cycle based on retentionDays and performs incremental vacuuming if rows were pruned.
func (r *RetentionManager) PruneAndCompact(ctx context.Context) (int64, error) {
	cutoff := time.Now().AddDate(0, 0, -r.retentionDays)
	pruned, err := r.PruneTerminalRuns(ctx, cutoff)
	if err != nil {
		return 0, err
	}

	if pruned > 0 {
		if err := r.IncrementalVacuum(ctx, 500); err != nil {
			return pruned, fmt.Errorf("pruned %d runs but incremental vacuum failed: %w", pruned, err)
		}
	}

	return pruned, nil
}
