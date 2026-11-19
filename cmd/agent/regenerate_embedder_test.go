package main

import (
	"testing"

	"bob/internal/config"
	"bob/internal/llm"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestResolveRegenerateEmbedder_Precedence(t *testing.T) {
	t.Run("none returns error", func(t *testing.T) {
		cfg := &config.Config{
			EmbeddingModel:             "none",
			SystemOneBaseURL:           "http://127.0.0.1:8090",
			SystemOneEmbeddingsEnabled: true,
		}
		_, _, cleanup, err := resolveRegenerateEmbedder(cfg, nil)
		defer cleanup()
		require.Error(t, err)
	})

	t.Run("explicit embedding_model overrides system1", func(t *testing.T) {
		cfg := &config.Config{
			OpenAIAPIKey:               "test-key",
			EmbeddingModel:             "text-embedding-3-large",
			SystemOneBaseURL:           "http://127.0.0.1:8090",
			SystemOneEmbeddingsEnabled: true,
		}
		emb, name, cleanup, err := resolveRegenerateEmbedder(cfg, nil)
		defer cleanup()
		require.NoError(t, err)
		assert.Equal(t, "text-embedding-3-large", name)
		_, ok := emb.(*llm.Embedder)
		assert.True(t, ok)
	})

	t.Run("system1 enabled selects SystemOneEmbedder when embedding_model empty", func(t *testing.T) {
		cfg := &config.Config{
			SystemOneBaseURL:           "http://127.0.0.1:8090",
			SystemOneModel:             "bob-s1-0.8b",
			SystemOneEmbeddingDims:     128,
			SystemOneEmbeddingsEnabled: true,
		}
		emb, name, cleanup, err := resolveRegenerateEmbedder(cfg, nil)
		defer cleanup()
		require.NoError(t, err)
		assert.Equal(t, "bob-s1-0.8b", name)
		sysEmb, ok := emb.(*llm.SystemOneEmbedder)
		require.True(t, ok)
		assert.Equal(t, 128, sysEmb.Dim())
	})
}
