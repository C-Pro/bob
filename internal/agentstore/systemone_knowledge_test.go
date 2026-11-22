package agentstore

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"

	"bob/internal/config"
	"bob/internal/knowledge"
	"bob/internal/memory"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type toggleFailEmbedder struct {
	model   string
	dims    int
	failNow atomic.Bool
}

func (t *toggleFailEmbedder) Model() string { return t.model }
func (t *toggleFailEmbedder) Dim() int      { return t.dims }
func (t *toggleFailEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	if t.failNow.Load() {
		return nil, errors.New("knowledge query embedder failure")
	}
	v := make([]float32, t.dims)
	v[0] = 1
	return v, nil
}
func (t *toggleFailEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i := range texts {
		v, err := t.Embed(ctx, texts[i])
		if err != nil {
			return nil, err
		}
		out[i] = v
	}
	return out, nil
}

func TestKnowledgeProvider_StampsOnWriteAndEnforcesStrictSearch(t *testing.T) {
	ctx := context.Background()
	dataDir := t.TempDir()
	emb := &toggleFailEmbedder{model: "system1:bob-s1-0.8b", dims: 128}
	cfg := &config.Config{
		DataDir:                  dataDir,
		SystemOneBaseURL:         "http://127.0.0.1:8090",
		SystemOneFallbackOnError: false,
	}

	memMgr := memory.NewManager(cfg, emb)
	defer func() { _ = memMgr.Close() }()
	provider := NewMemoryStoreProvider(memMgr, dataDir)

	// 1. Fresh DM: propose + approve + reconcile BEFORE any chat chunk/search must stamp DB properly.
	kStore, err := provider.GetKnowledgeStore(ctx, "dm_1", true)
	require.NoError(t, err)

	_, ver1, err := kStore.ProposeMemory(ctx, &knowledge.KnowledgeItem{
		ChatID: "dm_1",
		UserID: "u1",
	}, &knowledge.MemoryVersion{
		Type:       knowledge.MemoryTypePreference,
		Content:    "Prefers dark editor theme",
		Confidence: 0.9,
	})
	require.NoError(t, err)
	require.NoError(t, kStore.ApproveVersion(ctx, ver1.ID, "u1"))

	reconciled, err := provider.ReconcileChat(ctx, "dm_1", true, 10)
	require.NoError(t, err)
	assert.Equal(t, 1, reconciled)

	// DB vector space must now be stamped so compatible search succeeds without legacy mismatch!
	cortexDB, err := memMgr.GetDB(ctx, "dm_1", true)
	require.NoError(t, err)
	require.NoError(t, memory.CheckVectorSpaceCompatibility(ctx, cortexDB.SQL(), emb))

	searcher, err := provider.GetKnowledgeSearcher(ctx, "dm_1", true, "u1")
	require.NoError(t, err)

	res, err := searcher.SearchMemories(ctx, "unrelated query", nil, 5)
	require.NoError(t, err)
	assert.NotEmpty(t, res)

	// 2. Under strict mode (fallback_on_error=false), real CortexDB query embed failure must propagate!
	emb.failNow.Store(true)
	_, err = searcher.SearchMemories(ctx, "failing query", nil, 5)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "knowledge query embedder failure")

	// 3. Another embedder with mismatched model on the same DM DB must fail knowledge indexing writes!
	emb.failNow.Store(false)
	otherMgr := memory.NewManager(cfg, &toggleFailEmbedder{model: "other:384", dims: 384})
	defer func() { _ = otherMgr.Close() }()
	otherProvider := NewMemoryStoreProvider(otherMgr, dataDir)

	_, ver2, err := kStore.ProposeMemory(ctx, &knowledge.KnowledgeItem{
		ChatID: "dm_1",
		UserID: "u1",
	}, &knowledge.MemoryVersion{
		Type:       knowledge.MemoryTypePreference,
		Content:    "Mixed space item should fail",
		Confidence: 0.9,
	})
	require.NoError(t, err)
	require.NoError(t, kStore.ApproveVersion(ctx, ver2.ID, "u1"))

	otherIdx, err := otherProvider.GetKnowledgeIndexer(ctx, "dm_1", true)
	require.NoError(t, err)
	err = otherIdx.IndexVersion(ctx, ver2.ID)
	var mismatchErr *memory.VectorSpaceMismatchError
	require.ErrorAs(t, err, &mismatchErr)
}
