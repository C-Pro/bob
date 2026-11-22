package memory

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"io"
	"strings"
	"sync"

	"bob/internal/llm"

	"github.com/liliang-cn/cortexdb/v2/pkg/cortexdb"
)

// VectorSpaceIdentity identifies the embedding model space and dimensionality stored in a SQLite DB.
type VectorSpaceIdentity struct {
	EmbedderID string
	Dims       int
}

// VectorSpaceMismatchError indicates that a database's stored vectors were produced by an
// incompatible embedding model or dimensionality than the active query/indexing embedder.
type VectorSpaceMismatchError struct {
	StoredEmbedderID string
	StoredDims       int
	ActiveEmbedderID string
	ActiveDims       int
}

func (e *VectorSpaceMismatchError) Error() string {
	return fmt.Sprintf(
		"vector space mismatch (db=%s:%d, active=%s:%d): run bob --regenerate-vectors to migrate stored embeddings",
		e.StoredEmbedderID, e.StoredDims, e.ActiveEmbedderID, e.ActiveDims,
	)
}

type embedErrorSink struct {
	mu  sync.Mutex
	err error
}

func (s *embedErrorSink) set(err error) {
	if s == nil || err == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.err == nil {
		s.err = err
	}
}

func (s *embedErrorSink) get() error {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.err
}

type embedErrorSinkKey struct{}
type expectedDimsKey struct{}

func withEmbedContext(ctx context.Context, expectedDims int, captureErr bool) (context.Context, *embedErrorSink) {
	if ctx == nil {
		ctx = context.Background()
	}
	if expectedDims > 0 {
		ctx = context.WithValue(ctx, expectedDimsKey{}, expectedDims)
	}
	var sink *embedErrorSink
	if captureErr {
		sink = &embedErrorSink{}
		ctx = context.WithValue(ctx, embedErrorSinkKey{}, sink)
	}
	return ctx, sink
}

func expectedDimsFromContext(ctx context.Context) int {
	if ctx == nil {
		return 0
	}
	d, _ := ctx.Value(expectedDimsKey{}).(int)
	return d
}

func embedErrorSinkFromContext(ctx context.Context) *embedErrorSink {
	if ctx == nil {
		return nil
	}
	sink, _ := ctx.Value(embedErrorSinkKey{}).(*embedErrorSink)
	return sink
}

// strictAwareEmbedder wraps cortexdb.Embedder to capture embedding errors and enforce
// DB dimensionality invariants per-call via context before CortexDB can swallow errors.
type strictAwareEmbedder struct {
	inner cortexdb.Embedder
}

func wrapEmbedder(inner cortexdb.Embedder) cortexdb.Embedder {
	if isNilEmbedder(inner) {
		return nil
	}
	if _, already := inner.(*strictAwareEmbedder); already {
		return inner
	}
	return &strictAwareEmbedder{inner: inner}
}

func unwrapEmbedder(e cortexdb.Embedder) cortexdb.Embedder {
	for {
		w, ok := e.(*strictAwareEmbedder)
		if !ok || w == nil {
			return e
		}
		e = w.inner
	}
}

func (w *strictAwareEmbedder) Dim() int {
	return w.inner.Dim()
}

func (w *strictAwareEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	vec, err := w.inner.Embed(ctx, text)
	sink := embedErrorSinkFromContext(ctx)
	if err != nil {
		sink.set(err)
		return nil, err
	}
	if expected := expectedDimsFromContext(ctx); expected > 0 && len(vec) != expected {
		mismatch := &VectorSpaceMismatchError{
			StoredEmbedderID: DescribeEmbedderSpace(w.inner).EmbedderID,
			StoredDims:       expected,
			ActiveEmbedderID: DescribeEmbedderSpace(w.inner).EmbedderID,
			ActiveDims:       len(vec),
		}
		sink.set(mismatch)
		return nil, mismatch
	}
	return vec, nil
}

func (w *strictAwareEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	vecs, err := w.inner.EmbedBatch(ctx, texts)
	sink := embedErrorSinkFromContext(ctx)
	if err != nil {
		sink.set(err)
		return nil, err
	}
	if expected := expectedDimsFromContext(ctx); expected > 0 {
		for _, v := range vecs {
			if len(v) != expected {
				mismatch := &VectorSpaceMismatchError{
					StoredEmbedderID: DescribeEmbedderSpace(w.inner).EmbedderID,
					StoredDims:       expected,
					ActiveEmbedderID: DescribeEmbedderSpace(w.inner).EmbedderID,
					ActiveDims:       len(v),
				}
				sink.set(mismatch)
				return nil, mismatch
			}
		}
	}
	return vecs, nil
}

func (w *strictAwareEmbedder) EmbedMultimodal(ctx context.Context, text string, imageDataURIs []string) ([]float32, error) {
	mm, ok := w.inner.(interface {
		EmbedMultimodal(context.Context, string, []string) ([]float32, error)
	})
	if !ok {
		return w.Embed(ctx, text)
	}
	vec, err := mm.EmbedMultimodal(ctx, text, imageDataURIs)
	sink := embedErrorSinkFromContext(ctx)
	if err != nil {
		sink.set(err)
		return nil, err
	}
	if expected := expectedDimsFromContext(ctx); expected > 0 && len(vec) != expected {
		mismatch := &VectorSpaceMismatchError{
			StoredEmbedderID: DescribeEmbedderSpace(w.inner).EmbedderID,
			StoredDims:       expected,
			ActiveEmbedderID: DescribeEmbedderSpace(w.inner).EmbedderID,
			ActiveDims:       len(vec),
		}
		sink.set(mismatch)
		return nil, mismatch
	}
	return vec, nil
}

func (w *strictAwareEmbedder) Close() error {
	if c, ok := w.inner.(io.Closer); ok {
		return c.Close()
	}
	return nil
}

// DescribeEmbedderSpace returns a canonical space identifier and dimensionality for embedder.
func DescribeEmbedderSpace(embedder cortexdb.Embedder) VectorSpaceIdentity {
	embedder = unwrapEmbedder(embedder)
	if isNilEmbedder(embedder) {
		return VectorSpaceIdentity{}
	}
	dims := embedder.Dim()
	switch e := embedder.(type) {
	case *llm.SystemOneEmbedder:
		return VectorSpaceIdentity{
			EmbedderID: "system1:" + strings.TrimSpace(e.Model()),
			Dims:       dims,
		}
	case *llm.Embedder:
		return VectorSpaceIdentity{
			EmbedderID: "openai:" + strings.TrimSpace(e.Model()),
			Dims:       dims,
		}
	case *llm.LocalEmbedder, *llm.LazyLocalEmbedder:
		return VectorSpaceIdentity{
			EmbedderID: "local:" + llm.DefaultLocalModel,
			Dims:       dims,
		}
	default:
		if modeled, ok := embedder.(interface{ Model() string }); ok {
			if model := strings.TrimSpace(modeled.Model()); model != "" {
				return VectorSpaceIdentity{
					EmbedderID: model,
					Dims:       dims,
				}
			}
		}
		return VectorSpaceIdentity{
			EmbedderID: fmt.Sprintf("%T", embedder),
			Dims:       dims,
		}
	}
}

type sqlExecQuerier interface {
	ExecContext(ctx context.Context, query string, args ...any) (sql.Result, error)
	QueryRowContext(ctx context.Context, query string, args ...any) *sql.Row
}

func initVectorSpaceTable(ctx context.Context, db sqlExecQuerier) error {
	if db == nil {
		return nil
	}
	_, err := db.ExecContext(ctx, `
		CREATE TABLE IF NOT EXISTS memory_vector_space (
			id INTEGER PRIMARY KEY CHECK (id = 1),
			embedder_id TEXT NOT NULL,
			dims INTEGER NOT NULL,
			updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
		);
	`)
	return err
}

func readVectorSpace(ctx context.Context, db sqlExecQuerier) (VectorSpaceIdentity, bool, error) {
	if db == nil {
		return VectorSpaceIdentity{}, false, nil
	}
	var id VectorSpaceIdentity
	err := db.QueryRowContext(ctx, `SELECT embedder_id, dims FROM memory_vector_space WHERE id = 1`).Scan(&id.EmbedderID, &id.Dims)
	if err == nil {
		return id, true, nil
	}
	if errors.Is(err, sql.ErrNoRows) || isMissingVectorSpaceTableErr(err) {
		return VectorSpaceIdentity{}, false, nil
	}
	return VectorSpaceIdentity{}, false, fmt.Errorf("query memory_vector_space: %w", err)
}

// forceStampVectorSpace unconditionally writes the target vector space during migration/regeneration.
func forceStampVectorSpace(ctx context.Context, db sqlExecQuerier, embedder cortexdb.Embedder) error {
	if db == nil || isNilEmbedder(embedder) {
		return nil
	}
	space := DescribeEmbedderSpace(embedder)
	if space.EmbedderID == "" {
		return nil
	}
	if err := initVectorSpaceTable(ctx, db); err != nil {
		return fmt.Errorf("init memory_vector_space table: %w", err)
	}
	_, err := db.ExecContext(ctx, `
		INSERT INTO memory_vector_space (id, embedder_id, dims, updated_at)
		VALUES (1, ?, ?, CURRENT_TIMESTAMP)
		ON CONFLICT(id) DO UPDATE SET
			embedder_id = excluded.embedder_id,
			dims = excluded.dims,
			updated_at = CURRENT_TIMESTAMP
	`, space.EmbedderID, space.Dims)
	return err
}

// StampVectorSpace writes or updates the database's active vector-space stamp without ever
// overwriting a conflicting non-zero stamp from another embedder space or dimensionality.
func StampVectorSpace(ctx context.Context, rawDB *sql.DB, embedder cortexdb.Embedder) error {
	if rawDB == nil || isNilEmbedder(embedder) {
		return nil
	}
	space := DescribeEmbedderSpace(embedder)
	if space.EmbedderID == "" || space.Dims <= 0 {
		return nil
	}
	if err := initVectorSpaceTable(ctx, rawDB); err != nil {
		return fmt.Errorf("init memory_vector_space table: %w", err)
	}
	_, err := rawDB.ExecContext(ctx, `
		INSERT INTO memory_vector_space (id, embedder_id, dims, updated_at)
		VALUES (1, ?, ?, CURRENT_TIMESTAMP)
		ON CONFLICT(id) DO UPDATE SET
			dims = excluded.dims,
			updated_at = CURRENT_TIMESTAMP
		WHERE memory_vector_space.embedder_id = excluded.embedder_id
		  AND memory_vector_space.dims = 0
	`, space.EmbedderID, space.Dims)
	if err != nil {
		return fmt.Errorf("stamp memory_vector_space: %w", err)
	}

	stored, found, readErr := readVectorSpace(ctx, rawDB)
	if readErr != nil {
		return readErr
	}
	if !found {
		return errors.New("memory_vector_space stamp missing after insert")
	}
	if stored.EmbedderID != space.EmbedderID || (stored.Dims > 0 && stored.Dims != space.Dims) {
		return &VectorSpaceMismatchError{
			StoredEmbedderID: stored.EmbedderID,
			StoredDims:       stored.Dims,
			ActiveEmbedderID: space.EmbedderID,
			ActiveDims:       space.Dims,
		}
	}
	return nil
}

// CheckVectorSpaceCompatibility verifies that rawDB's stored vectors match embedder.
// Unstamped databases with existing vectors are treated as unknown provenance and require
// regeneration rather than guessing a default model identity.
func CheckVectorSpaceCompatibility(ctx context.Context, rawDB *sql.DB, embedder cortexdb.Embedder) error {
	_, err := checkVectorSpaceCompatibilityWithDims(ctx, rawDB, embedder)
	return err
}

func checkVectorSpaceCompatibilityWithDims(ctx context.Context, rawDB *sql.DB, embedder cortexdb.Embedder) (int, error) {
	if rawDB == nil || isNilEmbedder(embedder) {
		return 0, nil
	}
	active := DescribeEmbedderSpace(embedder)
	if active.EmbedderID == "" {
		return 0, nil
	}

	stored, found, err := readVectorSpace(ctx, rawDB)
	if err != nil {
		return 0, err
	}
	if found {
		if stored.EmbedderID != active.EmbedderID {
			return stored.Dims, &VectorSpaceMismatchError{
				StoredEmbedderID: stored.EmbedderID,
				StoredDims:       stored.Dims,
				ActiveEmbedderID: active.EmbedderID,
				ActiveDims:       active.Dims,
			}
		}
		if stored.Dims > 0 && active.Dims > 0 && stored.Dims != active.Dims {
			return stored.Dims, &VectorSpaceMismatchError{
				StoredEmbedderID: stored.EmbedderID,
				StoredDims:       stored.Dims,
				ActiveEmbedderID: active.EmbedderID,
				ActiveDims:       active.Dims,
			}
		}
		if stored.Dims == 0 && active.Dims > 0 {
			if stampErr := StampVectorSpace(ctx, rawDB, embedder); stampErr != nil {
				return 0, stampErr
			}
			return active.Dims, nil
		}
		return stored.Dims, nil
	}

	var sampleLen sql.NullInt64
	vecErr := rawDB.QueryRowContext(ctx,
		`SELECT length(vector) FROM messages WHERE session_id LIKE 'memory:%' AND vector IS NOT NULL LIMIT 1`,
	).Scan(&sampleLen)
	if vecErr != nil && !errors.Is(vecErr, sql.ErrNoRows) {
		if !isMissingVectorSpaceTableErr(vecErr) {
			return 0, fmt.Errorf("check existing memory vectors: %w", vecErr)
		}
	}
	if sampleLen.Valid {
		inferredDims := 0
		if sampleLen.Int64 > 4 && (sampleLen.Int64-4)%4 == 0 {
			inferredDims = int((sampleLen.Int64 - 4) / 4)
		}
		return inferredDims, &VectorSpaceMismatchError{
			StoredEmbedderID: "<unstamped_unknown>",
			StoredDims:       inferredDims,
			ActiveEmbedderID: active.EmbedderID,
			ActiveDims:       active.Dims,
		}
	}

	if active.Dims > 0 {
		if stampErr := StampVectorSpace(ctx, rawDB, embedder); stampErr != nil {
			return 0, stampErr
		}
		return active.Dims, nil
	}
	return 0, nil
}

func isMissingVectorSpaceTableErr(err error) bool {
	if err == nil {
		return false
	}
	return strings.Contains(strings.ToLower(err.Error()), "no such table")
}
