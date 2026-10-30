package trajectory

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"regexp"
	"strings"
	"sync"
	"time"
)

var (
	// ErrExporterDisabled indicates trajectory export is disabled.
	ErrExporterDisabled = errors.New("trajectory exporter: disabled")
	// ErrUploaderRequired indicates object storage client is nil while export is enabled.
	ErrUploaderRequired = errors.New("trajectory exporter: uploader is required when enabled")
	// ErrInvalidTrajectoryID indicates an unsafe or malformed trajectory ID.
	ErrInvalidTrajectoryID = errors.New("trajectory exporter: invalid trajectory ID")
	// ErrInvalidRevision indicates trajectory revision must be >= 1.
	ErrInvalidRevision = errors.New("trajectory exporter: invalid revision")
	// ErrMissingCreatedAt indicates trajectory created_at timestamp is zero.
	ErrMissingCreatedAt = errors.New("trajectory exporter: missing created_at timestamp")
)

var safeIDRegex = regexp.MustCompile(`^[a-zA-Z0-9_\-]+$`)

// ObjectUploader defines the minimal S3 Put interface required by Exporter.
// *objectstore.Client directly satisfies this interface.
type ObjectUploader interface {
	Put(ctx context.Context, key string, r io.Reader, size int64) error
}

// StoreProvider discovers or provides active trajectory stores.
type StoreProvider interface {
	ActiveTrajectoryStores(ctx context.Context) ([]Store, error)
}

// SingleStoreProvider adapts a single Store to StoreProvider.
type SingleStoreProvider struct {
	Store Store
}

// ActiveTrajectoryStores returns the single wrapped store.
func (s SingleStoreProvider) ActiveTrajectoryStores(ctx context.Context) ([]Store, error) {
	if s.Store == nil {
		return nil, nil
	}
	return []Store{s.Store}, nil
}

// ExporterConfig defines settings for the trajectory S3 exporter.
type ExporterConfig struct {
	Enabled       bool
	Bucket        string
	Prefix        string
	FlushDelay    time.Duration
	AllowPrivate  bool
	PollInterval  time.Duration
	BatchLimit    int
	PruneAge      time.Duration
	UploadTimeout time.Duration
	MaxRetries    int
}

// BuildObjectKey constructs the stable partitioned S3 object key:
// <prefix>/YYYY/MM/DD/<trajectory_id>/r<revision>.json
// Date components are derived strictly from traj.CreatedAt.UTC() to guarantee stable paths across revisions.
// IDs containing whitespace or path traversal characters are strictly rejected to prevent collisions.
func BuildObjectKey(prefix string, traj *Trajectory) (string, error) {
	if traj == nil {
		return "", errors.New("trajectory is nil")
	}
	id := traj.ID
	if id == "" || !safeIDRegex.MatchString(id) {
		return "", fmt.Errorf("%w: %q", ErrInvalidTrajectoryID, traj.ID)
	}
	if traj.Revision <= 0 {
		return "", fmt.Errorf("%w: %d", ErrInvalidRevision, traj.Revision)
	}
	if traj.CreatedAt.IsZero() {
		return "", ErrMissingCreatedAt
	}

	normPrefix := strings.TrimPrefix(prefix, "/")
	if normPrefix != "" && !strings.HasSuffix(normPrefix, "/") {
		normPrefix += "/"
	}

	datePart := traj.CreatedAt.UTC().Format("2006/01/02")
	return fmt.Sprintf("%s%s/%s/r%d.json", normPrefix, datePart, id, traj.Revision), nil
}

// IsEligibleForExport validates export approval and privacy classification.
// Snapshots are evaluated at selection time based on their materialized metadata.
func IsEligibleForExport(traj *Trajectory, allowPrivate bool) bool {
	if traj == nil || !traj.ExportApproved {
		return false
	}
	switch traj.PrivacyLevel {
	case "public":
		return true
	case "private":
		return allowPrivate
	default:
		// "internal", unknown, or empty are never exported
		return false
	}
}

type exporterRun struct {
	cancel    context.CancelFunc
	done      chan struct{}
	flushDone chan struct{}
	stopErr   error
}

type lifecycleState int

const (
	stateStopped lifecycleState = iota
	stateRunning
	stateStopping
)

type failureRecord struct {
	revision int
	until    time.Time
}

// Exporter flushes mature, export-approved trajectories to S3/MinIO.
type Exporter struct {
	cfg      ExporterConfig
	provider StoreProvider
	uploader ObjectUploader
	now      func() time.Time

	flushSem chan struct{}

	failMu         sync.Mutex
	failedAttempts map[string]failureRecord

	mu     sync.Mutex
	state  lifecycleState
	curRun *exporterRun
}

// NewExporter creates an initialized Exporter.
func NewExporter(cfg ExporterConfig, provider StoreProvider, uploader ObjectUploader) *Exporter {
	if cfg.Prefix == "" {
		cfg.Prefix = "trajectories/"
	}
	if cfg.PollInterval <= 0 {
		cfg.PollInterval = 1 * time.Minute
	}
	if cfg.BatchLimit <= 0 {
		cfg.BatchLimit = 50
	}
	if cfg.FlushDelay <= 0 {
		cfg.FlushDelay = 15 * time.Minute
	}
	if cfg.UploadTimeout <= 0 {
		cfg.UploadTimeout = 30 * time.Second
	}
	if cfg.MaxRetries <= 0 {
		cfg.MaxRetries = 3
	}

	return &Exporter{
		cfg:            cfg,
		provider:       provider,
		uploader:       uploader,
		now:            time.Now,
		flushSem:       make(chan struct{}, 1),
		failedAttempts: make(map[string]failureRecord),
		state:          stateStopped,
	}
}

// SetNow overrides the time provider used by the exporter (primarily for testing).
func (e *Exporter) SetNow(nowFn func() time.Time) {
	if e == nil {
		return
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if nowFn == nil {
		e.now = time.Now
	} else {
		e.now = nowFn
	}
}

func (e *Exporter) getNow() time.Time {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.now == nil {
		return time.Now()
	}
	return e.now()
}

// Running reports whether the background worker is currently active.
func (e *Exporter) Running() bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.state == stateRunning
}

// Start launches the background worker loop.
func (e *Exporter) Start(ctx context.Context) {
	e.mu.Lock()
	if e.state != stateStopped {
		e.mu.Unlock()
		return
	}
	e.state = stateRunning
	workerCtx, cancel := context.WithCancel(ctx)
	run := &exporterRun{
		cancel:    cancel,
		done:      make(chan struct{}),
		flushDone: make(chan struct{}),
	}
	e.curRun = run
	go e.runLoop(workerCtx, run)
	e.mu.Unlock()
}

// Stop gracefully stops the background worker, waits for any in-flight pass to finish,
// and executes a final bounded flush if a valid context is provided.
func (e *Exporter) Stop(flushCtx context.Context) error {
	e.mu.Lock()
	if e.state == stateStopped {
		e.mu.Unlock()
		return nil
	}
	if e.state == stateStopping {
		run := e.curRun
		e.mu.Unlock()
		if run != nil {
			if flushCtx != nil {
				select {
				case <-run.flushDone:
					return run.stopErr
				case <-flushCtx.Done():
					return flushCtx.Err()
				}
			} else {
				<-run.flushDone
				return run.stopErr
			}
		}
		return nil
	}

	e.state = stateStopping
	run := e.curRun
	run.cancel()
	e.mu.Unlock()

	defer func() {
		close(run.flushDone)
	}()

	// Wait for worker with respect to flushCtx deadline if provided
	var waitErr error
	if flushCtx != nil {
		select {
		case <-run.done:
		case <-flushCtx.Done():
			waitErr = flushCtx.Err()
		}
	} else {
		<-run.done
	}

	// Keep stateStopping through the final flush so Start cannot overlap shutdown
	var flushErr error
	if waitErr == nil && flushCtx != nil && flushCtx.Err() == nil {
		flushErr = e.FlushOnce(flushCtx)
	}

	var stopErr error
	if waitErr != nil {
		stopErr = waitErr
	} else if flushErr != nil {
		stopErr = flushErr
	}
	run.stopErr = stopErr

	e.mu.Lock()
	// Only transition to stateStopped if worker has actually exited
	select {
	case <-run.done:
		e.state = stateStopped
		e.curRun = nil
	default:
		// Worker is still in progress (timed out); remain stateStopping until it exits
		go func(r *exporterRun) {
			<-r.done
			e.mu.Lock()
			e.state = stateStopped
			e.curRun = nil
			e.mu.Unlock()
		}(run)
	}
	e.mu.Unlock()

	return stopErr
}

func (e *Exporter) runLoop(ctx context.Context, run *exporterRun) {
	defer func() {
		close(run.done)
	}()

	// Initial immediate pass
	if err := e.FlushOnce(ctx); err != nil && !errors.Is(err, context.Canceled) {
		slog.Warn("trajectory export startup pass encountered errors", "error", err)
	}

	ticker := time.NewTicker(e.cfg.PollInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			if err := e.FlushOnce(ctx); err != nil && !errors.Is(err, context.Canceled) {
				slog.Warn("trajectory export pass encountered errors", "error", err)
			}
		}
	}
}

// FlushOnce scans all active stores, reconciling terminal runs and uploading mature trajectories.
func (e *Exporter) FlushOnce(ctx context.Context) error {
	if !e.cfg.Enabled {
		return nil
	}
	if e.uploader == nil {
		return ErrUploaderRequired
	}
	if e.provider == nil {
		return nil
	}

	// Serialize flushes with context-aware acquisition to prevent hanging on cancellation
	select {
	case e.flushSem <- struct{}{}:
		defer func() { <-e.flushSem }()
	case <-ctx.Done():
		return ctx.Err()
	}

	stores, err := e.provider.ActiveTrajectoryStores(ctx)
	if err != nil {
		return fmt.Errorf("failed to discover active trajectory stores: %w", err)
	}

	var allErrs []error
	selectionNow := e.getNow().UTC()

	for _, store := range stores {
		if err := ctx.Err(); err != nil {
			allErrs = append(allErrs, err)
			break
		}
		if store == nil {
			continue
		}

		if storeErr := e.flushStore(ctx, store, selectionNow); storeErr != nil {
			allErrs = append(allErrs, storeErr)
		}
	}

	return errors.Join(allErrs...)
}

func (e *Exporter) flushStore(ctx context.Context, store Store, selectionNow time.Time) error {
	var storeErrs []error

	// 1. Reconcile any terminal runs before querying.
	// Failure to reconcile reports a warning but does not block exporting already staged rows.
	if err := store.ReconcileTerminalRuns(ctx); err != nil {
		slog.Warn("failed to reconcile terminal runs before trajectory export", "error", err)
		storeErrs = append(storeErrs, fmt.Errorf("reconciliation error: %w", err))
	}

	// 2. Query mature, approved candidates, excluding currently backed-off IDs
	// so repeated failures on the oldest row do not starve later eligible rows.
	excludeIDs := e.getBackedOffIDs(selectionNow)
	opts := ListExportOptions{
		AllowPrivate: e.cfg.AllowPrivate,
		Limit:        e.cfg.BatchLimit,
		ExcludeIDs:   excludeIDs,
	}
	trajs, err := store.ListReadyForExport(ctx, selectionNow, opts)
	if err != nil {
		storeErrs = append(storeErrs, fmt.Errorf("failed to list trajectories for export: %w", err))
		return errors.Join(storeErrs...)
	}

	// 3. Process each candidate
	for _, traj := range trajs {
		if err := ctx.Err(); err != nil {
			storeErrs = append(storeErrs, err)
			break
		}
		if traj == nil {
			continue
		}

		if !IsEligibleForExport(traj, e.cfg.AllowPrivate) {
			continue
		}

		key, err := BuildObjectKey(e.cfg.Prefix, traj)
		if err != nil {
			slog.Error("invalid trajectory metadata for object key", "id", traj.ID, "error", err)
			storeErrs = append(storeErrs, err)
			continue
		}

		// Exact snapshot serialization
		payload, err := json.Marshal(traj)
		if err != nil {
			storeErrs = append(storeErrs, fmt.Errorf("failed to marshal trajectory %s: %w", traj.ID, err))
			continue
		}

		uploadErr := e.uploadWithRetry(ctx, key, payload)
		if uploadErr != nil {
			slog.Warn("failed to upload trajectory to object store", "id", traj.ID, "key", key, "error", uploadErr)
			e.recordFailure(traj.ID, traj.Revision)
			storeErrs = append(storeErrs, fmt.Errorf("upload failed for %s: %w", traj.ID, uploadErr))
			continue
		}

		// Fresh timestamp post-upload for acknowledgment
		ackTime := e.getNow().UTC()
		if markErr := store.MarkExported(ctx, traj.ID, traj.Revision, ackTime); markErr != nil {
			slog.Warn("failed to mark trajectory exported", "id", traj.ID, "revision", traj.Revision, "error", markErr)
			storeErrs = append(storeErrs, fmt.Errorf("failed to mark %s exported: %w", traj.ID, markErr))
			continue
		}

		e.ClearFailure(traj.ID)
		slog.Debug("successfully exported trajectory", "id", traj.ID, "revision", traj.Revision, "key", key)
	}

	// 4. Prune exported trajectories older than PruneAge
	if e.cfg.PruneAge > 0 {
		if _, pruneErr := store.PruneExported(ctx, e.cfg.PruneAge); pruneErr != nil {
			slog.Warn("failed to prune exported trajectories", "error", pruneErr)
			storeErrs = append(storeErrs, fmt.Errorf("prune error: %w", pruneErr))
		}
	}

	return errors.Join(storeErrs...)
}

func (e *Exporter) uploadWithRetry(ctx context.Context, key string, payload []byte) error {
	var lastErr error
	maxRetries := e.cfg.MaxRetries
	if maxRetries < 1 {
		maxRetries = 1
	}

	for attempt := 0; attempt < maxRetries; attempt++ {
		if attempt > 0 {
			backoffDelay := time.Duration(100<<uint(attempt-1)) * time.Millisecond
			if backoffDelay > 2*time.Second {
				backoffDelay = 2 * time.Second
			}
			t := time.NewTimer(backoffDelay)
			select {
			case <-ctx.Done():
				t.Stop()
				return ctx.Err()
			case <-t.C:
			}
		}

		uploadCtx, cancel := context.WithTimeout(ctx, e.cfg.UploadTimeout)
		err := e.uploader.Put(uploadCtx, key, bytes.NewReader(payload), int64(len(payload)))
		cancel()

		if err == nil {
			return nil
		}

		lastErr = err
		if ctx.Err() != nil {
			return ctx.Err()
		}
	}

	return lastErr
}

func (e *Exporter) getBackedOffIDs(now time.Time) []string {
	e.failMu.Lock()
	defer e.failMu.Unlock()
	var ids []string
	for id, rec := range e.failedAttempts {
		if now.Before(rec.until) {
			ids = append(ids, id)
		} else {
			delete(e.failedAttempts, id)
		}
	}
	return ids
}

// IsBackingOff reports whether a trajectory ID is currently backing off due to a recent failure.
func (e *Exporter) IsBackingOff(id string, now time.Time) bool {
	e.failMu.Lock()
	defer e.failMu.Unlock()
	rec, exists := e.failedAttempts[id]
	if !exists {
		return false
	}
	if now.After(rec.until) {
		delete(e.failedAttempts, id)
		return false
	}
	return true
}

func (e *Exporter) recordFailure(id string, revision int) {
	e.failMu.Lock()
	defer e.failMu.Unlock()
	// Backoff 2 minutes from failure time
	now := e.getNow().UTC()
	e.failedAttempts[id] = failureRecord{
		revision: revision,
		until:    now.Add(2 * time.Minute),
	}
}

// ClearFailure removes backoff for a given trajectory ID.
func (e *Exporter) ClearFailure(id string) {
	e.failMu.Lock()
	defer e.failMu.Unlock()
	delete(e.failedAttempts, id)
}
