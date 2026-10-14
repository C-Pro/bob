package bootstrap

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"bob/internal/agentapi"
	"bob/internal/config"
	"bob/internal/llm"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type dummyFrontend struct{}

func (d *dummyFrontend) Bind(ctx context.Context, run agentapi.RunDescriptor) (agentapi.Bindings, error) {
	return agentapi.Bindings{}, nil
}

func (d *dummyFrontend) Deliver(ctx context.Context, c agentapi.Completion) error {
	return nil
}

func testConfig(t *testing.T) *config.Config {
	t.Helper()
	tempDir := t.TempDir()
	cfg := &config.Config{
		DataDir:                        tempDir,
		OpenAIModel:                    "test-model",
		OpenAIAPIKey:                   "test-key",
		TownhallToolMaxIterations:      10,
		DMToolMaxIterations:            20,
		KnowledgeMaxLoadedMemories:     5,
		KnowledgeMaxLoadedMemoryBytes:  1024,
		KnowledgeMaxLoadedSkills:       2,
		KnowledgeMaxLoadedSkillBytes:   2048,
		KnowledgeDefaultDiscoveryLimit: 3,
		KnowledgeMaxDiscoveryLimit:     5,
		SandboxEnabled:                 false,
		EmbeddingModel:                 "none",
	}
	return cfg
}

func TestNew_WithDefaults(t *testing.T) {
	cfg := testConfig(t)
	rt, err := New(cfg, Options{})
	require.NoError(t, err)
	require.NotNil(t, rt)

	assert.NotNil(t, rt.LLMClient)
	assert.NotNil(t, rt.MemoryManager)
	assert.NotNil(t, rt.StoreProvider)
	assert.NotNil(t, rt.ToolsRegistry)
	assert.Nil(t, rt.SandboxManager)
	assert.NotNil(t, rt.FSMEngine)
	assert.NotNil(t, rt.AgentEngine)
	assert.NotNil(t, rt.SessionLocker)

	require.NoError(t, rt.Close())
}

func TestNew_SandboxEnablement(t *testing.T) {
	t.Run("explicitly disabled via options", func(t *testing.T) {
		cfg := testConfig(t)
		cfg.SandboxEnabled = true
		f := false
		rt, err := New(cfg, Options{EnableSandbox: &f})
		require.NoError(t, err)
		assert.Nil(t, rt.SandboxManager)
		require.NoError(t, rt.Close())
	})

	t.Run("explicitly enabled via options", func(t *testing.T) {
		cfg := testConfig(t)
		cfg.SandboxEnabled = false
		tr := true
		rt, err := New(cfg, Options{EnableSandbox: &tr})
		require.NoError(t, err)
		assert.NotNil(t, rt.SandboxManager)
		require.NoError(t, rt.Close())
	})
}

func TestNew_EmbeddingModelSelection(t *testing.T) {
	t.Run("none produces nil embedder", func(t *testing.T) {
		cfg := testConfig(t)
		cfg.EmbeddingModel = "none"
		rt, err := New(cfg, Options{})
		require.NoError(t, err)
		assert.Nil(t, rt.MemoryManager.Embedder())
		require.NoError(t, rt.Close())
	})

	t.Run("empty string produces lazy local embedder", func(t *testing.T) {
		t.Setenv("BOB_MODELS_DIR", t.TempDir())
		cfg := testConfig(t)
		cfg.EmbeddingModel = ""
		rt, err := New(cfg, Options{})
		require.NoError(t, err)

		emb := rt.MemoryManager.Embedder()
		require.NotNil(t, emb)
		lazy, ok := emb.(*llm.LazyLocalEmbedder)
		require.True(t, ok)
		assert.False(t, lazy.IsInitialized(), "local embedder must not be initialized eagerly")
		assert.Equal(t, 384, lazy.Dim())

		require.NoError(t, rt.Close())
	})
}

func TestRegisterFrontend(t *testing.T) {
	cfg := testConfig(t)
	rt, err := New(cfg, Options{})
	require.NoError(t, err)

	fe := &dummyFrontend{}
	err = rt.RegisterFrontend("test-fe", fe)
	require.NoError(t, err)

	// Duplicate or re-registration is allowed and safe
	err = rt.RegisterFrontend("test-fe", fe)
	require.NoError(t, err)

	// Empty ID or nil frontend rejected
	assert.Error(t, rt.RegisterFrontend("", fe))
	assert.Error(t, rt.RegisterFrontend("valid", nil))

	require.NoError(t, rt.Close())
}

func TestClose_Idempotent(t *testing.T) {
	cfg := testConfig(t)
	tr := true
	rt, err := New(cfg, Options{EnableSandbox: &tr})
	require.NoError(t, err)

	err = rt.Close()
	require.NoError(t, err)

	// Second close must be safe and idempotent
	err = rt.Close()
	require.NoError(t, err)
}

func TestNew_ValidationErrors(t *testing.T) {
	_, err := New(nil, Options{})
	assert.Error(t, err)
}

func TestNew_CreatesRequiredDirectories(t *testing.T) {
	cfg := testConfig(t)
	dataDir := filepath.Join(cfg.DataDir, "subdata")
	cfg.DataDir = dataDir

	rt, err := New(cfg, Options{})
	require.NoError(t, err)
	require.NotNil(t, rt)

	_, err = os.Stat(dataDir)
	assert.NoError(t, err)

	require.NoError(t, rt.Close())
}
