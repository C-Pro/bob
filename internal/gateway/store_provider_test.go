package gateway

import (
	"context"
	"testing"

	"bob/internal/config"
	"bob/internal/fsm"
	"bob/internal/memory"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	_ "modernc.org/sqlite"
)

func TestMemoryStoreProvider_GetStore(t *testing.T) {
	tempDir := t.TempDir()
	cfg := &config.Config{
		DataDir: tempDir,
	}

	memMgr := memory.NewManager(cfg, nil)
	defer func() {
		_ = memMgr.Close()
	}()

	provider := NewMemoryStoreProvider(memMgr, tempDir)
	ctx := context.Background()

	// Townhall store
	thStore, err := provider.GetStore(ctx, "townhall", false)
	require.NoError(t, err)
	require.NotNil(t, thStore)

	activeRuns, err := thStore.ListActiveRuns(ctx)
	require.NoError(t, err)
	assert.Empty(t, activeRuns)

	// Repeated call returns cached store instance
	thStoreCached, err := provider.GetStore(ctx, "townhall", false)
	require.NoError(t, err)
	assert.Same(t, thStore, thStoreCached)

	// DM store
	dmStore, err := provider.GetStore(ctx, "user_123", true)
	require.NoError(t, err)
	require.NotNil(t, dmStore)

	// Repeated call returns cached DM store instance
	dmStoreCached, err := provider.GetStore(ctx, "user_123", true)
	require.NoError(t, err)
	assert.Same(t, dmStore, dmStoreCached)

	activeRuns, err = dmStore.ListActiveRuns(ctx)
	require.NoError(t, err)
	assert.Empty(t, activeRuns)
}

func TestMemoryStoreProvider_ActiveStores(t *testing.T) {
	tempDir := t.TempDir()
	cfg := &config.Config{
		DataDir: tempDir,
	}

	memMgr := memory.NewManager(cfg, nil)
	defer func() {
		_ = memMgr.Close()
	}()

	provider := NewMemoryStoreProvider(memMgr, tempDir)
	ctx := context.Background()

	// Open two stores
	_, err := provider.GetStore(ctx, "townhall", false)
	require.NoError(t, err)
	aliceStore, err := provider.GetStore(ctx, "user_alice", true)
	require.NoError(t, err)

	activeStores, err := provider.ActiveStores(ctx)
	require.NoError(t, err)
	assert.Len(t, activeStores, 2)

	// Create an active run in user_alice so it requires recovery/polling at startup.
	// townhall remains idle without active runs.
	err = aliceStore.CreateRun(ctx, &fsm.FSMRun{
		ID:           "run_alice_active",
		ChatID:       "user_alice",
		UserID:       "alice",
		IsDM:         true,
		FSMType:      fsm.FSMTypeToolLoop,
		CurrentState: fsm.StateLLMRequest,
		Status:       fsm.RunStatusRunning,
		ContextJSON:  "[]",
	})
	require.NoError(t, err)

	// Create a new provider pointing to the same dataDir without prior GetStore calls.
	// Only user_alice should be discovered because townhall has no active runs.
	memMgr2 := memory.NewManager(cfg, nil)
	defer func() {
		_ = memMgr2.Close()
	}()

	provider2 := NewMemoryStoreProvider(memMgr2, tempDir)
	discoveredStores, err := provider2.ActiveStores(ctx)
	require.NoError(t, err)
	assert.Len(t, discoveredStores, 1)
}

func TestMemoryStoreProvider_NilManager(t *testing.T) {
	provider := NewMemoryStoreProvider(nil, "")
	ctx := context.Background()

	store, err := provider.GetStore(ctx, "townhall", false)
	assert.Error(t, err)
	assert.Nil(t, store)

	stores, err := provider.ActiveStores(ctx)
	assert.NoError(t, err)
	assert.Nil(t, stores)
}
