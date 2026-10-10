package llm

import (
	"context"
	"errors"
	"io"
	"sync"

	"bob/internal/config"

	"github.com/liliang-cn/cortexdb/v2/pkg/cortexdb"
)

// DefaultEmbeddingDim is the dimension of MiniLM-L12-v2 and standard local embedding models.
const DefaultEmbeddingDim = 384

// LazyLocalEmbedder wraps a local embedding model, delaying initialization until
// the first Embed or EmbedBatch call. Dim() can be called immediately without triggering initialization.
type LazyLocalEmbedder struct {
	cfg     *config.Config
	factory func() (cortexdb.Embedder, error)
	dim     int

	mu     sync.Mutex
	inst   cortexdb.Embedder
	initErr error
	closed bool
}

// NewLazyLocalEmbedder creates a LazyLocalEmbedder with the default local embedding engine.
func NewLazyLocalEmbedder(cfg *config.Config) *LazyLocalEmbedder {
	return NewLazyLocalEmbedderWithFactory(cfg, 0, func() (cortexdb.Embedder, error) {
		return NewLocalEmbedder(cfg)
	})
}

// NewLazyLocalEmbedderWithFactory allows injecting a custom embedder factory for testing.
func NewLazyLocalEmbedderWithFactory(cfg *config.Config, dim int, factory func() (cortexdb.Embedder, error)) *LazyLocalEmbedder {
	if dim <= 0 {
		dim = DefaultEmbeddingDim
	}
	return &LazyLocalEmbedder{
		cfg:     cfg,
		factory: factory,
		dim:     dim,
	}
}

// Dim returns the embedding dimensionality without initializing the underlying model.
func (l *LazyLocalEmbedder) Dim() int {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.inst != nil {
		return l.inst.Dim()
	}
	return l.dim
}

// getOrInit retrieves the initialized instance or creates it.
// If initialization fails due to context cancellation, the error is not cached permanently.
func (l *LazyLocalEmbedder) getOrInit(ctx context.Context) (cortexdb.Embedder, error) {
	l.mu.Lock()
	defer l.mu.Unlock()

	if l.closed {
		return nil, errors.New("embedder is closed")
	}
	if l.inst != nil {
		return l.inst, nil
	}
	if l.initErr != nil {
		return nil, l.initErr
	}

	if ctx != nil && ctx.Err() != nil {
		return nil, ctx.Err()
	}

	inst, err := l.factory()
	if err != nil {
		if ctx != nil && ctx.Err() != nil {
			return nil, ctx.Err()
		}
		l.initErr = err
		return nil, err
	}

	l.inst = inst
	return l.inst, nil
}

// Embed computes the embedding vector for the given text, initializing on first invocation.
func (l *LazyLocalEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	inst, err := l.getOrInit(ctx)
	if err != nil {
		return nil, err
	}
	return inst.Embed(ctx, text)
}

// EmbedBatch computes embedding vectors for multiple texts, initializing on first invocation.
func (l *LazyLocalEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	inst, err := l.getOrInit(ctx)
	if err != nil {
		return nil, err
	}
	return inst.EmbedBatch(ctx, texts)
}

// IsInitialized reports whether the underlying model has been loaded.
func (l *LazyLocalEmbedder) IsInitialized() bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.inst != nil
}

// Close closes the underlying embedder if initialized.
func (l *LazyLocalEmbedder) Close() error {
	l.mu.Lock()
	defer l.mu.Unlock()

	l.closed = true
	if l.inst != nil {
		if closer, ok := l.inst.(io.Closer); ok {
			return closer.Close()
		}
	}
	return nil
}
