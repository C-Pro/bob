package llm

import (
	"context"
	"math"
	"testing"

	"github.com/liliang-cn/cortexdb/v2/pkg/cortexdb"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Ensure LocalEmbedder implements cortexdb.Embedder interface at compile time.
var _ cortexdb.Embedder = (*LocalEmbedder)(nil)

func TestAggregateMultiWindowVectors(t *testing.T) {
	t.Run("empty slices", func(t *testing.T) {
		_, err := AggregateMultiWindowVectors(nil)
		assert.Error(t, err)

		_, err = AggregateMultiWindowVectors([][]float32{})
		assert.Error(t, err)
	})

	t.Run("single slice", func(t *testing.T) {
		vec := []float32{0.6, 0.8}
		res, err := AggregateMultiWindowVectors([][]float32{vec})
		require.NoError(t, err)
		assert.Equal(t, vec, res)
	})

	t.Run("multiple slices mean and normalized", func(t *testing.T) {
		s1 := []float32{1.0, 0.0, 0.0}
		s2 := []float32{0.0, 1.0, 0.0}
		res, err := AggregateMultiWindowVectors([][]float32{s1, s2})
		require.NoError(t, err)
		require.Len(t, res, 3)

		// Average is (0.5, 0.5, 0.0), norm is sqrt(0.5^2 + 0.5^2) = sqrt(0.5) = ~0.70710678
		// Normalized: (0.70710678, 0.70710678, 0.0)
		expectedVal := float32(1.0 / math.Sqrt(2))
		assert.InDelta(t, expectedVal, res[0], 1e-5)
		assert.InDelta(t, expectedVal, res[1], 1e-5)
		assert.InDelta(t, float32(0.0), res[2], 1e-5)

		// Check unit length L2 norm
		var normSq float32
		for _, v := range res {
			normSq += v * v
		}
		assert.InDelta(t, float32(1.0), normSq, 1e-5)
	})

	t.Run("inconsistent dimensions error", func(t *testing.T) {
		s1 := []float32{1.0, 0.0}
		s2 := []float32{1.0, 0.0, 0.0}
		_, err := AggregateMultiWindowVectors([][]float32{s1, s2})
		assert.ErrorContains(t, err, "inconsistent slice dimension")
	})
}

func TestLocalEmbedderBasic(t *testing.T) {
	emb := NewLocalEmbedderWithEngine(nil, "custom-model")
	assert.Equal(t, 384, emb.Dim())
	assert.Equal(t, "custom-model", emb.Model())

	// Test empty text
	_, err := emb.Embed(context.Background(), "")
	assert.ErrorContains(t, err, "empty text provided")

	// Test uninitialized engine
	_, err = emb.Embed(context.Background(), "hello world")
	assert.ErrorContains(t, err, "local embedding engine not initialized")

	// Test context cancellation
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err = emb.Embed(ctx, "hello world")
	assert.ErrorIs(t, err, context.Canceled)

	_, err = emb.EmbedBatch(ctx, []string{"hello"})
	assert.ErrorIs(t, err, context.Canceled)

	// EmbedBatch with empty texts
	vecs, err := emb.EmbedBatch(context.Background(), nil)
	assert.NoError(t, err)
	assert.Nil(t, vecs)

	// Close on nil engine
	assert.NoError(t, emb.Close())
}
