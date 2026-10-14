package llm

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"bob/internal/config"

	"github.com/liliang-cn/cortexdb/v2/pkg/cortexdb"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type mockEmbedder struct {
	embedCount atomic.Int32
	closed     atomic.Bool
	dim        int
}

func (m *mockEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	m.embedCount.Add(1)
	return make([]float32, m.dim), nil
}

func (m *mockEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	m.embedCount.Add(int32(len(texts)))
	res := make([][]float32, len(texts))
	for i := range texts {
		res[i] = make([]float32, m.dim)
	}
	return res, nil
}

func (m *mockEmbedder) Dim() int {
	return m.dim
}

func (m *mockEmbedder) Close() error {
	m.closed.Store(true)
	return nil
}

func TestLazyLocalEmbedder_DimDoesNotInitialize(t *testing.T) {
	var factoryCalled atomic.Bool
	lazy := NewLazyLocalEmbedderWithFactory(&config.Config{}, 384, func() (cortexdb.Embedder, error) {
		factoryCalled.Store(true)
		return &mockEmbedder{dim: 384}, nil
	})

	assert.Equal(t, 384, lazy.Dim())
	assert.False(t, factoryCalled.Load())
	assert.False(t, lazy.IsInitialized())
}

func TestLazyLocalEmbedder_EmbedInitializesLazily(t *testing.T) {
	var factoryCalls atomic.Int32
	mock := &mockEmbedder{dim: 384}
	lazy := NewLazyLocalEmbedderWithFactory(&config.Config{}, 384, func() (cortexdb.Embedder, error) {
		factoryCalls.Add(1)
		return mock, nil
	})

	require.False(t, lazy.IsInitialized())

	vec, err := lazy.Embed(context.Background(), "hello world")
	require.NoError(t, err)
	assert.Len(t, vec, 384)
	assert.Equal(t, int32(1), factoryCalls.Load())
	assert.True(t, lazy.IsInitialized())

	// Subsequent call reuses same instance
	vec2, err := lazy.Embed(context.Background(), "another text")
	require.NoError(t, err)
	assert.Len(t, vec2, 384)
	assert.Equal(t, int32(1), factoryCalls.Load())
}

func TestLazyLocalEmbedder_ConcurrentInitialization(t *testing.T) {
	var factoryCalls atomic.Int32
	mock := &mockEmbedder{dim: 384}
	lazy := NewLazyLocalEmbedderWithFactory(&config.Config{}, 384, func() (cortexdb.Embedder, error) {
		factoryCalls.Add(1)
		time.Sleep(10 * time.Millisecond)
		return mock, nil
	})

	var wg sync.WaitGroup
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			v, err := lazy.Embed(context.Background(), "concurrent text")
			assert.NoError(t, err)
			assert.Len(t, v, 384)
		}()
	}
	wg.Wait()

	assert.Equal(t, int32(1), factoryCalls.Load())
	assert.True(t, lazy.IsInitialized())
}

func TestLazyLocalEmbedder_CloseWithoutInit(t *testing.T) {
	var factoryCalled atomic.Bool
	lazy := NewLazyLocalEmbedderWithFactory(&config.Config{}, 384, func() (cortexdb.Embedder, error) {
		factoryCalled.Store(true)
		return &mockEmbedder{dim: 384}, nil
	})

	err := lazy.Close()
	require.NoError(t, err)
	assert.False(t, factoryCalled.Load())

	// Embed after close should return error
	_, err = lazy.Embed(context.Background(), "test")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "closed")
	assert.False(t, factoryCalled.Load())
}

func TestLazyLocalEmbedder_CloseWithInit(t *testing.T) {
	mock := &mockEmbedder{dim: 384}
	lazy := NewLazyLocalEmbedderWithFactory(&config.Config{}, 384, func() (cortexdb.Embedder, error) {
		return mock, nil
	})

	_, err := lazy.Embed(context.Background(), "test")
	require.NoError(t, err)

	err = lazy.Close()
	require.NoError(t, err)
	assert.True(t, mock.closed.Load())
}

func TestLazyLocalEmbedder_ContextCancellationDoesNotPermanentlyFail(t *testing.T) {
	var attempts atomic.Int32
	mock := &mockEmbedder{dim: 384}

	canceledCtx, cancel := context.WithCancel(context.Background())
	lazy := NewLazyLocalEmbedderWithFactory(&config.Config{}, 384, func() (cortexdb.Embedder, error) {
		att := attempts.Add(1)
		if att == 1 {
			cancel()
			return nil, context.Canceled
		}
		return mock, nil
	})

	_, err := lazy.Embed(canceledCtx, "test")
	assert.Error(t, err)

	// Second call with fresh context should retry factory
	vec, err := lazy.Embed(context.Background(), "test")
	require.NoError(t, err)
	assert.Len(t, vec, 384)
	assert.Equal(t, int32(2), attempts.Load())
}

func TestLazyLocalEmbedder_DimensionMismatch(t *testing.T) {
	mock := &mockEmbedder{dim: 768}
	lazy := NewLazyLocalEmbedderWithFactory(&config.Config{}, 384, func() (cortexdb.Embedder, error) {
		return mock, nil
	})

	assert.Equal(t, 384, lazy.Dim())

	_, err := lazy.Embed(context.Background(), "test")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "underlying embedder dimension 768 does not match expected dimension 384")
	assert.True(t, mock.closed.Load(), "mismatched embedder should be closed")
	assert.Equal(t, 384, lazy.Dim(), "Dim() must remain immutable")
}
