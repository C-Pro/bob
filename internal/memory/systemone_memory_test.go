package memory

import (
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"bob/internal/config"
	"bob/internal/llm"

	"github.com/liliang-cn/cortexdb/v2/pkg/cortexdb"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type mockEpisodeClassifier struct {
	fn    func(ctx context.Context, id, question string, options []llm.DecisionOptionSpec, state string, images []string) (*llm.DecisionResultEntry, *llm.DecisionResponse, error)
	calls atomic.Int32
}

func (m *mockEpisodeClassifier) DecideCustom(ctx context.Context, id, question string, options []llm.DecisionOptionSpec, state string, images []string) (*llm.DecisionResultEntry, *llm.DecisionResponse, error) {
	m.calls.Add(1)
	return m.fn(ctx, id, question, options, state, images)
}

func encodeTestVecBase64(v []float32) string {
	buf := make([]byte, len(v)*4)
	for i, val := range v {
		binary.LittleEndian.PutUint32(buf[i*4:(i+1)*4], math.Float32bits(val))
	}
	return base64.StdEncoding.EncodeToString(buf)
}

func makeUnitVector384(axis int) []float32 {
	v := make([]float32, 384)
	v[axis%384] = 1.0
	return v
}

type failingNullVectorEmbedder struct {
	dims int
}

func (f *failingNullVectorEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	return nil, errors.New("simulated embedder failure swallowed by cortexdb")
}

func (f *failingNullVectorEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	return nil, errors.New("simulated embedder batch failure")
}

func (f *failingNullVectorEmbedder) Dim() int { return f.dims }

type programmableEmbedder struct {
	model   string
	dimFn   func() int
	embedFn func(ctx context.Context, text string) ([]float32, error)
}

func (p *programmableEmbedder) Model() string { return p.model }
func (p *programmableEmbedder) Dim() int      { return p.dimFn() }
func (p *programmableEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	return p.embedFn(ctx, text)
}
func (p *programmableEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, t := range texts {
		v, err := p.embedFn(ctx, t)
		if err != nil {
			return nil, err
		}
		out[i] = v
	}
	return out, nil
}

func TestVectorSpaceGuard_SameDimDifferentModelAndMigration(t *testing.T) {
	ctx := context.Background()
	dataDir := t.TempDir()
	cfg := &config.Config{DataDir: dataDir}

	// 1. Create a legacy unstamped DB with 384d vectors (simulating pre-migration MiniLM-L12-v2).
	legacyEmb := &mockEmbedder{dim: 384, model: "local:" + llm.DefaultLocalModel}
	legacyMgr := NewManager(cfg, legacyEmb)
	db, err := legacyMgr.GetDB(ctx, "townhall", false)
	require.NoError(t, err)
	require.NoError(t, legacyMgr.IndexMessages(ctx, "townhall", false, []MessageToStore{
		{
			Seq:        1,
			Timestamp:  time.Now().Unix(),
			ChatID:     "townhall",
			UserID:     "alice",
			SenderName: "Alice",
			Role:       "user",
			Content:    "Alpha deploy rollback marker phrase",
		},
	}))
	// Strip the stamp to also cover unstamped legacy DB migration detection.
	_, err = db.SQL().ExecContext(ctx, `DELETE FROM memory_vector_space`)
	require.NoError(t, err)
	require.NoError(t, legacyMgr.Close())

	// Reopening unstamped DB even with local MiniLM must reject unknown provenance (no blind guessing).
	reopenLocal := NewManager(&config.Config{
		DataDir:                  dataDir,
		SystemOneBaseURL:         "http://127.0.0.1:8090",
		SystemOneFallbackOnError: false,
	}, legacyEmb)
	_, err = reopenLocal.Search(ctx, "rollback", "townhall", false, 5)
	var unstampedErr *VectorSpaceMismatchError
	require.ErrorAs(t, err, &unstampedErr)
	assert.Equal(t, "<unstamped_unknown>", unstampedErr.StoredEmbedderID)
	require.NoError(t, reopenLocal.Close())

	// 2. Reopen with SystemOneEmbedder (also 384d, so dim alone would NOT detect mismatch!).
	var queryEmbedCalls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		queryEmbedCalls.Add(1)
		var req map[string]any
		require.NoError(t, json.NewDecoder(r.Body).Decode(&req))
		inputs := req["input"].([]any)
		data := make([]map[string]any, len(inputs))
		for i := range inputs {
			data[i] = map[string]any{
				"object":    "embedding",
				"index":     i,
				"embedding": encodeTestVecBase64(makeUnitVector384(0)),
			}
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"object": "list", "model": "bob-s1-0.8b", "data": data})
	}))
	defer srv.Close()

	s1Client, err := llm.NewSystemOneClient(&config.Config{
		SystemOneBaseURL:       srv.URL,
		SystemOneModel:         "bob-s1-0.8b",
		SystemOneEmbeddingDims: 384,
	}, srv.Client())
	require.NoError(t, err)
	s1Emb, err := llm.NewSystemOneEmbedder(s1Client, 384, 0, true)
	require.NoError(t, err)

	// Non-strict mode: gates semantic search to lexical FTS5 without calling s1Emb!
	mgrPermissive := NewManager(&config.Config{
		DataDir:                  dataDir,
		SystemOneBaseURL:         srv.URL,
		SystemOneFallbackOnError: true,
	}, s1Emb)
	items, err := mgrPermissive.Search(ctx, "rollback marker", "townhall", false, 5)
	require.NoError(t, err)
	assert.NotEmpty(t, items)
	assert.Equal(t, int32(0), queryEmbedCalls.Load(), "mismatched vector space must gate semantic search to lexical")
	require.NoError(t, mgrPermissive.Close())

	// Strict mode: fails explicitly on Search, SearchEpisodes, and IndexMessages!
	mgrStrict := NewManager(&config.Config{
		DataDir:                  dataDir,
		SystemOneBaseURL:         srv.URL,
		SystemOneFallbackOnError: false,
	}, s1Emb)
	_, err = mgrStrict.Search(ctx, "rollback", "townhall", false, 5)
	var mismatchErr *VectorSpaceMismatchError
	require.ErrorAs(t, err, &mismatchErr)

	_, err = mgrStrict.SearchEpisodes(ctx, "rollback", "townhall", false, 5)
	require.ErrorAs(t, err, &mismatchErr)

	err = mgrStrict.IndexMessages(ctx, "townhall", false, []MessageToStore{
		{Seq: 2, Timestamp: time.Now().Unix(), ChatID: "townhall", UserID: "bob", Role: "user", Content: "new msg"},
	})
	require.ErrorAs(t, err, &mismatchErr)
	require.NoError(t, mgrStrict.Close())

	// 3. Migrate via RegenerateAllVectors and verify semantic search unlocks!
	report, err := RegenerateAllVectors(ctx, &config.Config{DataDir: dataDir}, s1Emb, 16)
	require.NoError(t, err)
	assert.Equal(t, 1, report.TotalDatabases)
	assert.Equal(t, 1, report.Reembedded)
	assert.Equal(t, 1, report.ReembeddedEpisodes)

	mgrPost := NewManager(&config.Config{
		DataDir:                  dataDir,
		SystemOneBaseURL:         srv.URL,
		SystemOneFallbackOnError: false,
	}, s1Emb)
	defer func() { _ = mgrPost.Close() }()

	queryEmbedCalls.Store(0)
	postItems, err := mgrPost.Search(ctx, "unrelated query terms", "townhall", false, 5)
	require.NoError(t, err)
	assert.NotEmpty(t, postItems, "vector search should return semantic hit once vector space is migrated")
	assert.Equal(t, int32(2), queryEmbedCalls.Load())
}

func TestStrictMode_DetectsNullVectorOnSwallowedEmbedderFailure(t *testing.T) {
	ctx := context.Background()
	cfg := &config.Config{
		DataDir:                  t.TempDir(),
		SystemOneBaseURL:         "http://127.0.0.1:8090",
		SystemOneFallbackOnError: false,
	}
	mgr := NewManager(cfg, &failingNullVectorEmbedder{dims: 384})
	defer func() { _ = mgr.Close() }()

	err := mgr.IndexMessages(ctx, "townhall", false, []MessageToStore{
		{Seq: 1, Timestamp: time.Now().Unix(), ChatID: "townhall", UserID: "u1", Role: "user", Content: "test strict null vector"},
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "simulated embedder failure swallowed by cortexdb")

	// Also verify silent nil-vector return (no explicit embed error, dimFn=0) is caught by verifyStoredVectorNonNull.
	silentCfg := &config.Config{
		DataDir:                  t.TempDir(),
		SystemOneBaseURL:         "http://127.0.0.1:8090",
		SystemOneFallbackOnError: false,
	}
	silentEmb := &programmableEmbedder{
		model: "silent-null",
		dimFn: func() int { return 0 },
		embedFn: func(ctx context.Context, text string) ([]float32, error) {
			return nil, nil
		},
	}
	silentMgr := NewManager(silentCfg, silentEmb)
	defer func() { _ = silentMgr.Close() }()

	err = silentMgr.IndexMessages(ctx, "townhall", false, []MessageToStore{
		{Seq: 1, Timestamp: time.Now().Unix(), ChatID: "townhall", UserID: "u1", Role: "user", Content: "test silent null vector"},
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "NULL vector")
}

func TestMultimodalMemory_IndexingSemanticFloorAndRegeneration(t *testing.T) {
	ctx := context.Background()

	var sawMultimodalRequest atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req map[string]any
		require.NoError(t, json.NewDecoder(r.Body).Decode(&req))
		inputs := req["input"].([]any)
		data := make([]map[string]any, len(inputs))
		for i, rawItem := range inputs {
			var vec []float32
			switch item := rawItem.(type) {
			case map[string]any:
				sawMultimodalRequest.Add(1)
				parts := item["content"].([]any)
				require.Len(t, parts, 9, "must cap at 8 images + 1 text part")
				// Axis 0: cosine = 1.0 against query [1, 0, ...] (>= 0.28)
				vec = makeUnitVector384(0)
			case string:
				switch {
				case item == "find matching picture":
					vec = makeUnitVector384(0)
				case strings.Contains(item, "Gamma"):
					// cosine = 0.40 against axis 0 (> 0.28 memorySemanticFloor)
					vec = make([]float32, 384)
					vec[0] = 0.40
					vec[1] = float32(math.Sqrt(1.0 - 0.40*0.40))
				case strings.Contains(item, "Beta"):
					// cosine = 0.20 against axis 0 (< 0.28 memorySemanticFloor)
					vec = make([]float32, 384)
					vec[0] = 0.20
					vec[1] = float32(math.Sqrt(1.0 - 0.20*0.20))
				default:
					vec = makeUnitVector384(2)
				}
			}
			data[i] = map[string]any{
				"object":    "embedding",
				"index":     i,
				"embedding": encodeTestVecBase64(vec),
			}
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"object": "list", "model": "bob-s1-0.8b", "data": data})
	}))
	defer srv.Close()

	cfg := &config.Config{
		DataDir:                  t.TempDir(),
		SystemOneBaseURL:         srv.URL,
		SystemOneModel:           "bob-s1-0.8b",
		SystemOneEmbeddingDims:   384,
		SystemOneFallbackOnError: false,
	}
	s1Client, err := llm.NewSystemOneClient(cfg, srv.Client())
	require.NoError(t, err)
	s1Emb, err := llm.NewSystemOneEmbedder(s1Client, 384, 0, false)
	require.NoError(t, err)

	mgr := NewManager(cfg, s1Emb)
	defer func() { _ = mgr.Close() }()

	// Build 10 distinct valid + 1 invalid image URI to test filtering, deduplication, and cap=8.
	imgs := []string{"https://invalid.example/pic.png"}
	for i := range 10 {
		imgs = append(imgs, "data:image/png;base64,"+base64.StdEncoding.EncodeToString([]byte(fmt.Sprintf("fakepng-%d", i))))
	}

	now := time.Now().Unix()
	require.NoError(t, mgr.IndexMessages(ctx, "townhall", false, []MessageToStore{
		{Seq: 1, Timestamp: now, ChatID: "townhall", UserID: "alice", SenderName: "Alice", Role: "user", Content: "Alpha visual artifact uploaded", Images: imgs},
	}))
	require.NoError(t, mgr.IndexMessages(ctx, "townhall", false, []MessageToStore{
		{Seq: 2, Timestamp: now + 1, ChatID: "townhall", UserID: "bob", SenderName: "Bob", Role: "user", Content: "Beta visual artifact uploaded"},
	}))
	require.NoError(t, mgr.IndexMessages(ctx, "townhall", false, []MessageToStore{
		{Seq: 3, Timestamp: now + 2, ChatID: "townhall", UserID: "carol", SenderName: "Carol", Role: "user", Content: "Gamma visual artifact uploaded"},
	}))
	assert.Equal(t, int32(1), sawMultimodalRequest.Load())

	// Query has ZERO lexical overlap with Alpha/Beta/Gamma chunks, testing CortexDB 0.28 floor directly.
	results, err := mgr.Search(ctx, "find matching picture", "townhall", false, 5)
	require.NoError(t, err)
	var matchedContents []string
	for _, item := range results {
		matchedContents = append(matchedContents, item.Content)
	}
	joined := strings.Join(matchedContents, "\n")
	assert.Contains(t, joined, "Alpha visual artifact uploaded", "cosine 1.0 >= 0.28 must be returned")
	assert.Contains(t, joined, "Gamma visual artifact uploaded", "cosine 0.40 >= 0.28 must be returned")
	assert.NotContains(t, joined, "Beta visual artifact uploaded", "cosine 0.20 < 0.28 must be filtered below semantic floor")

	db, err := mgr.GetDB(ctx, "townhall", false)
	require.NoError(t, err)
	// Same-space opaque multimodal row (no complete image_uris) + partial image_uris subset row
	_, err = db.SaveMemory(ctx, cortexdb.MemorySaveRequest{
		MemoryID: "mm_opaque",
		Role:     "user",
		Content:  "Large screenshot chunk without inline image_uris",
		Metadata: map[string]any{"has_images": true, "image_count": 1},
	})
	require.NoError(t, err)
	_, err = db.SaveMemory(ctx, cortexdb.MemorySaveRequest{
		MemoryID: "mm_partial",
		Role:     "user",
		Content:  "Partial image_uris chunk must NOT re-embed only 1 of 3 images",
		Metadata: map[string]any{
			"has_images":          true,
			"image_count":         3,
			"image_uris_complete": false,
			"image_uris":          []string{"data:image/png;base64," + base64.StdEncoding.EncodeToString([]byte("only-one"))},
		},
	})
	require.NoError(t, err)
	// Unreconstructable image row with NULL vector must NOT be sent to text-only EmbedBatch!
	_, err = db.SaveMemory(ctx, cortexdb.MemorySaveRequest{
		MemoryID: "mm_null_vec",
		Role:     "user",
		Content:  "Image chunk whose vector is NULL",
		Metadata: map[string]any{"has_images": true, "image_count": 1},
	})
	require.NoError(t, err)
	_, err = db.SQL().ExecContext(ctx, `UPDATE messages SET vector = NULL WHERE id = 'mm_null_vec'`)
	require.NoError(t, err)

	sentinelBytes, err := EncodeMemoryVector(makeUnitVector384(77))
	require.NoError(t, err)
	_, err = db.SQL().ExecContext(ctx, `UPDATE messages SET vector = ? WHERE id IN ('mm_opaque', 'mm_partial')`, sentinelBytes)
	require.NoError(t, err)

	sawMultimodalRequest.Store(0)
	report, err := mgr.RegenerateAllVectors(ctx, 16)
	require.NoError(t, err)
	assert.Equal(t, 1, report.ReembeddedMultimodal, "complete image_uris chunk reconstructed via EmbedMultimodal")
	assert.Equal(t, 2, report.SkippedMultimodal, "same-space opaque & partial image_uris rows preserve existing vectors")
	assert.Equal(t, 1, report.QuarantinedMultimodal, "NULL-vector unreconstructable image row quarantined instead of text-embedded")
	assert.Equal(t, int32(1), sawMultimodalRequest.Load())

	var preserved []byte
	require.NoError(t, db.SQL().QueryRowContext(ctx, `SELECT vector FROM messages WHERE id = 'mm_partial'`).Scan(&preserved))
	assert.Equal(t, sentinelBytes, preserved)

	// Cross-model migration must QUARANTINE opaque multimodal rows (vector = NULL, vector_quarantined = true)
	// rather than stamping mixed vector spaces as compatible!
	otherEmb := &mockEmbedder{dim: 128, model: "other-model-128d"}
	crossReport, err := RegenerateAllVectors(ctx, cfg, otherEmb, 16)
	require.NoError(t, err)
	assert.Equal(t, 4, crossReport.QuarantinedMultimodal, "cross-model/cross-dim migration must quarantine all unreconstructable multimodal rows (all 4 when new embedder lacks EmbedMultimodal)")

	var quarantinedVec []byte
	var quarantinedMeta string
	require.NoError(t, db.SQL().QueryRowContext(ctx, `SELECT vector, metadata FROM messages WHERE id = 'mm_opaque'`).Scan(&quarantinedVec, &quarantinedMeta))
	assert.Nil(t, quarantinedVec)
	assert.Contains(t, quarantinedMeta, `"vector_quarantined":true`)
}

func TestBilingualEpisodes_SystemOneAndRussianTagging(t *testing.T) {
	ctx := context.Background()
	now := time.Now().Unix()

	msgs := []MessageToStore{
		{Seq: 1, Timestamp: now, ChatID: "townhall", UserID: "u1", SenderName: "Иван", Role: "user", Content: "не работает API, баг в лог файле, нужен код"},
		{Seq: 2, Timestamp: now + 5, ChatID: "townhall", UserID: "u2", SenderName: "Alice", Role: "assistant", Content: "да, нашли баг и лог поправили, код готов"},
	}

	tags := extractTopicTags(msgs, maxEpisodeTopicTags)
	assert.NotContains(t, tags, "не")
	assert.NotContains(t, tags, "да")
	assert.NotContains(t, tags, "для")
	assert.Contains(t, tags, "api")
	assert.Contains(t, tags, "код")
	assert.Contains(t, tags, "баг")
	assert.Contains(t, tags, "лог")

	t.Run("System-1 4-way classification sets TaskStatus", func(t *testing.T) {
		mgr := NewManager(&config.Config{DataDir: t.TempDir(), EmbeddingModel: "none"}, nil)
		defer func() { _ = mgr.Close() }()

		cls := &mockEpisodeClassifier{
			fn: func(ctx context.Context, id, question string, options []llm.DecisionOptionSpec, state string, images []string) (*llm.DecisionResultEntry, *llm.DecisionResponse, error) {
				assert.Equal(t, "episode_task_status", id)
				assert.Len(t, options, 4)
				assert.Contains(t, state, "Иван")
				return &llm.DecisionResultEntry{ID: id, Choice: EpisodeStatusCompleted}, &llm.DecisionResponse{}, nil
			},
		}
		mgr.SetSystemOneConfig(cls, true, false)

		require.NoError(t, mgr.IndexMessages(ctx, "townhall", false, msgs))
		assert.Equal(t, int32(1), cls.calls.Load())

		eps, err := mgr.ListEpisodes(ctx, "townhall", false)
		require.NoError(t, err)
		require.Len(t, eps, 1)
		assert.Equal(t, EpisodeStatusCompleted, eps[0].TaskStatus)
	})

	t.Run("classifier failure falls back when fallback_on_error=true, errors when false", func(t *testing.T) {
		errCls := &mockEpisodeClassifier{
			fn: func(ctx context.Context, id, question string, options []llm.DecisionOptionSpec, state string, images []string) (*llm.DecisionResultEntry, *llm.DecisionResponse, error) {
				return nil, nil, errors.New("sidecar unavailable")
			},
		}

		mgrFallback := NewManager(&config.Config{DataDir: t.TempDir(), EmbeddingModel: "none"}, nil)
		defer func() { _ = mgrFallback.Close() }()
		mgrFallback.SetSystemOneConfig(errCls, true, true)
		require.NoError(t, mgrFallback.IndexMessages(ctx, "townhall", false, msgs))
		eps, err := mgrFallback.ListEpisodes(ctx, "townhall", false)
		require.NoError(t, err)
		require.Len(t, eps, 1)
		assert.Equal(t, EpisodeStatusDiscussion, eps[0].TaskStatus)

		mgrStrict := NewManager(&config.Config{DataDir: t.TempDir(), EmbeddingModel: "none"}, nil)
		defer func() { _ = mgrStrict.Close() }()
		mgrStrict.SetSystemOneConfig(errCls, true, false)
		err = mgrStrict.IndexMessages(ctx, "townhall", false, msgs)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "episode task_status classification failed")
	})

	t.Run("long episode preserves opening context and latest terminal message", func(t *testing.T) {
		longMsgs := make([]MessageToStore, 0, 60)
		longMsgs = append(longMsgs, MessageToStore{
			Seq: 1, Timestamp: now, ChatID: "townhall", UserID: "u1", SenderName: "Alice",
			Content: "opening task: working on database schema migration",
		})
		for i := 2; i <= 59; i++ {
			longMsgs = append(longMsgs, MessageToStore{
				Seq: int64(i), Timestamp: now + int64(i), ChatID: "townhall", UserID: "u1", SenderName: "Alice",
				Content: fmt.Sprintf("intermediate log step %d: %s", i, strings.Repeat("verbose output ", 18)),
			})
		}
		longMsgs = append(longMsgs, MessageToStore{
			Seq: 60, Timestamp: now + 60, ChatID: "townhall", UserID: "u2", SenderName: "Bob",
			Content: "FINAL OUTCOME: deployed and all verification tests pass cleanly",
		})

		mgr := NewManager(&config.Config{DataDir: t.TempDir(), EmbeddingModel: "none"}, nil)
		defer func() { _ = mgr.Close() }()
		cls := &mockEpisodeClassifier{
			fn: func(ctx context.Context, id, question string, options []llm.DecisionOptionSpec, state string, images []string) (*llm.DecisionResultEntry, *llm.DecisionResponse, error) {
				assert.LessOrEqual(t, len(state), 8192)
				assert.Contains(t, state, "opening task: working on database schema migration")
				assert.Contains(t, state, "FINAL OUTCOME: deployed and all verification tests pass cleanly")
				return &llm.DecisionResultEntry{ID: id, Choice: EpisodeStatusCompleted}, &llm.DecisionResponse{}, nil
			},
		}
		mgr.SetSystemOneConfig(cls, true, false)
		require.NoError(t, mgr.IndexMessages(ctx, "townhall", false, longMsgs))
	})
}

func TestStrictSearch_RealCortexDBPropagatesQueryAndEpisodeErrors(t *testing.T) {
	ctx := context.Background()
	var failQuery atomic.Bool
	var callNum atomic.Int32

	emb := &programmableEmbedder{
		model: "prog-384",
		dimFn: func() int { return 384 },
		embedFn: func(ctx context.Context, text string) ([]float32, error) {
			n := callNum.Add(1)
			if failQuery.Load() && n >= 2 {
				return nil, errors.New("episode query embedder outage")
			}
			if failQuery.Load() && strings.Contains(text, "fail-primary") {
				return nil, errors.New("primary query embedder outage")
			}
			return makeUnitVector384(0), nil
		},
	}

	mgr := NewManager(&config.Config{
		DataDir:                  t.TempDir(),
		SystemOneBaseURL:         "http://127.0.0.1:8090",
		SystemOneFallbackOnError: false,
	}, emb)
	defer func() { _ = mgr.Close() }()

	require.NoError(t, mgr.IndexMessages(ctx, "townhall", false, []MessageToStore{
		{Seq: 1, Timestamp: time.Now().Unix(), ChatID: "townhall", UserID: "u1", SenderName: "Alice", Role: "user", Content: "alpha searchable item"},
	}))

	// 1. Primary CortexDB query embed failure must propagate under strict mode (not swallowed by memory_api.go).
	callNum.Store(0)
	failQuery.Store(true)
	_, err := mgr.Search(ctx, "fail-primary", "townhall", false, 5)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "primary query embedder outage")

	// 2. Primary chunk search succeeds (call 1), second search (episode search, call 2) fails -> must propagate!
	callNum.Store(0)
	_, err = mgr.Search(ctx, "alpha", "townhall", false, 5)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "episode query embedder outage")
}

func TestUnknownDimsAndStampFailurePropagation(t *testing.T) {
	ctx := context.Background()
	dataDir := t.TempDir()

	// Seed a DB with stamped 384d space under openai:lazy-model.
	emb384 := &mockEmbedder{dim: 384, model: "openai:lazy-model"}
	seedMgr := NewManager(&config.Config{DataDir: dataDir}, emb384)
	require.NoError(t, seedMgr.IndexMessages(ctx, "townhall", false, []MessageToStore{
		{Seq: 1, Timestamp: 100, ChatID: "townhall", UserID: "u1", Content: "seed chunk"},
	}))
	require.NoError(t, seedMgr.Close())

	// Now reopen with an embedder whose Dim() initially returns 0, then unexpectedly produces 768 floats.
	lazyWrongDim := &programmableEmbedder{
		model: "openai:lazy-model",
		dimFn: func() int { return 0 },
		embedFn: func(ctx context.Context, text string) ([]float32, error) {
			return make([]float32, 768), nil
		},
	}
	mgr := NewManager(&config.Config{
		DataDir:                  dataDir,
		SystemOneBaseURL:         "http://127.0.0.1:8090",
		SystemOneFallbackOnError: true, // even when permissive, writing mismatched dims must fail!
	}, lazyWrongDim)
	defer func() { _ = mgr.Close() }()

	err := mgr.IndexMessages(ctx, "townhall", false, []MessageToStore{
		{Seq: 2, Timestamp: 200, ChatID: "townhall", UserID: "u1", Content: "mismatched 768 chunk"},
	})
	var mismatchErr *VectorSpaceMismatchError
	require.ErrorAs(t, err, &mismatchErr)
	assert.Equal(t, 384, mismatchErr.StoredDims)
	assert.Equal(t, 768, mismatchErr.ActiveDims)

	// Confirm original 384d stamp was NOT overwritten and watermark did NOT advance.
	db, err := mgr.GetDB(ctx, "townhall", false)
	require.NoError(t, err)
	space, ok, err := readVectorSpace(ctx, db.SQL())
	require.NoError(t, err)
	require.True(t, ok)
	assert.Equal(t, 384, space.Dims)
	wm, err := mgr.GetWatermark(ctx, "townhall", false)
	require.NoError(t, err)
	assert.Equal(t, int64(1), wm)

	// Stamp failure injection must abort IndexMessages before advancing watermark.
	freshDir := t.TempDir()
	embOk := &mockEmbedder{dim: 128, model: "custom:128"}
	failMgr := NewManager(&config.Config{DataDir: freshDir}, embOk)
	defer func() { _ = failMgr.Close() }()

	failDB, err := failMgr.GetDB(ctx, "townhall", false)
	require.NoError(t, err)
	_, err = failDB.SQL().ExecContext(ctx, `DELETE FROM memory_vector_space`)
	require.NoError(t, err)
	_, err = failDB.SQL().ExecContext(ctx, `CREATE TRIGGER reject_stamp BEFORE INSERT ON memory_vector_space BEGIN SELECT RAISE(ABORT, 'stamp write blocked'); END;`)
	require.NoError(t, err)

	err = failMgr.IndexMessages(ctx, "townhall", false, []MessageToStore{
		{Seq: 1, Timestamp: 100, ChatID: "townhall", UserID: "u1", Content: "should fail on stamp"},
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "stamp write blocked")
	wmFresh, err := failMgr.GetWatermark(ctx, "townhall", false)
	require.NoError(t, err)
	assert.Equal(t, int64(0), wmFresh)
}

func TestSystemOneMemory_LiveFloorValidation(t *testing.T) {
	liveURL := os.Getenv("BOB_S1_TEST_URL")
	if liveURL == "" {
		t.Skip("set BOB_S1_TEST_URL to run live System-1 memory similarity validation")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	cfg := &config.Config{
		DataDir:                  t.TempDir(),
		SystemOneBaseURL:         liveURL,
		SystemOneAPIKey:          os.Getenv("BOB_S1_TEST_API_KEY"),
		SystemOneModel:           config.DefaultSystemOneModel,
		SystemOneEmbeddingDims:   384,
		SystemOneEpisodesEnabled: true,
		SystemOneFallbackOnError: false,
	}
	client, err := llm.NewSystemOneClient(cfg)
	require.NoError(t, err)
	emb, err := llm.NewSystemOneEmbedder(client, 384, 0, false)
	require.NoError(t, err)

	mgr := NewManager(cfg, emb)
	defer func() { _ = mgr.Close() }()
	mgr.SetSystemOneConfig(client, true, false)

	require.NoError(t, mgr.IndexMessages(ctx, "townhall", false, []MessageToStore{
		{Seq: 1, Timestamp: time.Now().Unix(), ChatID: "townhall", UserID: "u1", SenderName: "Иван", Role: "user", Content: "Сборка Docker контейнера завершилась успешно"},
	}))
	items, err := mgr.Search(ctx, "Docker build status", "townhall", false, 5)
	require.NoError(t, err)
	assert.NotEmpty(t, items)
}
