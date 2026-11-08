package memory

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"
	"unicode/utf8"

	"bob/internal/config"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestEpisodeCreationAndMultiChunkLinking(t *testing.T) {
	tempDir := t.TempDir()
	cfg := &config.Config{
		DataDir:        tempDir,
		EmbeddingModel: "none",
	}
	mgr := NewManager(cfg, nil)
	defer func() { _ = mgr.Close() }()

	ctx := context.Background()
	baseTime := time.Date(2026, 11, 5, 12, 0, 0, 0, time.UTC).Unix()

	// Create 40 messages so IndexMessages splits them into 2 chunks (33 + 7) and 1 overarching Episode
	msgs := make([]MessageToStore, 0, 40)
	for i := 1; i <= 40; i++ {
		sender := "Alice"
		uid := "u_alice"
		content := fmt.Sprintf("Discussing postgres migration step %d for telemetry cluster.", i)
		if i%2 == 0 {
			sender = "Bob"
			uid = "u_bob"
		}
		if i == 40 {
			content = "Postgres migration for telemetry cluster is completed and verified."
		}
		msgs = append(msgs, MessageToStore{
			Seq:        int64(100 + i),
			Timestamp:  baseTime + int64(i*10),
			ChatID:     "townhall",
			UserID:     uid,
			SenderName: sender,
			Role:       "user",
			Content:    content,
		})
	}

	err := mgr.IndexMessages(ctx, "townhall", false, msgs)
	require.NoError(t, err)

	episodes, err := mgr.ListEpisodes(ctx, "townhall", false)
	require.NoError(t, err)
	require.Len(t, episodes, 1)

	ep := episodes[0]
	assert.Equal(t, "ep_townhall_101_140", ep.ID)
	assert.Equal(t, "townhall", ep.ChatID)
	assert.Equal(t, "townhall", ep.ChatType)
	assert.Equal(t, int64(101), ep.StartSeq)
	assert.Equal(t, int64(140), ep.EndSeq)
	assert.Equal(t, baseTime+10, ep.StartTime)
	assert.Equal(t, baseTime+400, ep.EndTime)
	assert.Equal(t, []string{"Alice", "Bob"}, ep.Participants)
	assert.Contains(t, ep.TopicTags, "postgres")
	assert.Contains(t, ep.TopicTags, "migration")
	assert.Contains(t, ep.TopicTags, "telemetry")
	assert.Equal(t, EpisodeStatusCompleted, ep.TaskStatus)
	assert.True(t, ep.ModelDerived)
	assert.Equal(t, []string{"chunk_townhall_101_133", "chunk_townhall_134_140"}, ep.SourceChunkIDs)
	assert.Contains(t, ep.Summary, "[Model-Derived Episode Summary | seq 101-140]")

	navEntry := ep.FormatNavigationEntry()
	assert.Contains(t, navEntry, "[Model-Derived Episode Summary | seq 101-140]")
	assert.Contains(t, navEntry, "Source Chunks: chunk_townhall_101_133, chunk_townhall_134_140")
	assert.Contains(t, navEntry, "Navigation summary only")
}

func TestSearchEpisodesAndRawChunkDrillDown(t *testing.T) {
	tempDir := t.TempDir()
	cfg := &config.Config{
		DataDir:        tempDir,
		EmbeddingModel: "none",
	}
	mgr := NewManager(cfg, nil)
	defer func() { _ = mgr.Close() }()

	ctx := context.Background()
	now := time.Now().Unix()

	batch1 := []MessageToStore{
		{
			Seq:        10,
			Timestamp:  now,
			ChatID:     "townhall",
			UserID:     "alice",
			SenderName: "Alice",
			Role:       "user",
			Content:    "Investigating raft consensus split-brain issue in staging.",
		},
		{
			Seq:        11,
			Timestamp:  now + 5,
			ChatID:     "townhall",
			UserID:     "charlie",
			SenderName: "Charlie",
			Role:       "user",
			Content:    "The raft split-brain issue is still unresolved and blocked on firewall rules.",
		},
	}
	require.NoError(t, mgr.IndexMessages(ctx, "townhall", false, batch1))

	batch2 := []MessageToStore{
		{
			Seq:        20,
			Timestamp:  now + 100,
			ChatID:     "townhall",
			UserID:     "dave",
			SenderName: "Dave",
			Role:       "user",
			Content:    "Upgraded grafana dashboards for latency percentiles.",
		},
		{
			Seq:        21,
			Timestamp:  now + 110,
			ChatID:     "townhall",
			UserID:     "dave",
			SenderName: "Dave",
			Role:       "user",
			Content:    "Grafana dashboard rollout completed.",
		},
	}
	require.NoError(t, mgr.IndexMessages(ctx, "townhall", false, batch2))

	// 1. Search episodes by topic keyword
	eps, err := mgr.SearchEpisodes(ctx, "raft", "townhall", false, 5)
	require.NoError(t, err)
	require.Len(t, eps, 1)
	assert.Equal(t, "ep_townhall_10_11", eps[0].ID)
	assert.Equal(t, EpisodeStatusUnresolved, eps[0].TaskStatus)

	// 2. Search episodes by episode ID or status
	epsByID, err := mgr.SearchEpisodes(ctx, "ep_townhall_20_21", "townhall", false, 5)
	require.NoError(t, err)
	require.NotEmpty(t, epsByID)
	assert.Equal(t, "ep_townhall_20_21", epsByID[0].ID)

	// 3. Manager.Search enriches chunk hits with EpisodeID and SourceChunkIDs
	searchItems, err := mgr.Search(ctx, "grafana", "townhall", false, 5)
	require.NoError(t, err)
	require.Len(t, searchItems, 1)
	assert.Equal(t, "chunk", searchItems[0].Kind)
	assert.Equal(t, "ep_townhall_20_21", searchItems[0].EpisodeID)
	assert.Equal(t, []string{"chunk_townhall_20_21"}, searchItems[0].SourceChunkIDs)

	// 4. Manager.Search returns standalone episode item when querying episode ID directly
	epItems, err := mgr.Search(ctx, "ep_townhall_10_11", "townhall", false, 5)
	require.NoError(t, err)
	require.NotEmpty(t, epItems)
	assert.Equal(t, "episode", epItems[0].Kind)
	assert.True(t, epItems[0].ModelDerived)
	assert.Contains(t, epItems[0].Content, "[Model-Derived Episode Summary | seq 10-11]")

	// 5. Drill down into raw chunks by chunk_id
	rawByChunk, err := mgr.GetRawChunks(ctx, "townhall", false, RawChunkQuery{
		ChunkID: "chunk_townhall_10_11",
	})
	require.NoError(t, err)
	require.Len(t, rawByChunk, 1)
	assert.Equal(t, "chunk_townhall_10_11", rawByChunk[0].ID)
	assert.Contains(t, rawByChunk[0].Content, "raft consensus split-brain")

	// 6. Drill down into raw chunks by episode_id
	rawByEp, err := mgr.GetRawChunks(ctx, "townhall", false, RawChunkQuery{
		ChunkID: "ep_townhall_20_21",
	})
	require.NoError(t, err)
	require.Len(t, rawByEp, 1)
	assert.Equal(t, "chunk_townhall_20_21", rawByEp[0].ID)
	assert.Contains(t, rawByEp[0].Content, "Grafana dashboard rollout completed")

	// 7. Drill down into raw chunks by sequence range [start_seq, end_seq]
	startSeq := int64(9)
	endSeq := int64(15)
	rawBySeq, err := mgr.GetRawChunks(ctx, "townhall", false, RawChunkQuery{
		StartSeq: &startSeq,
		EndSeq:   &endSeq,
	})
	require.NoError(t, err)
	require.Len(t, rawBySeq, 1)
	assert.Equal(t, "chunk_townhall_10_11", rawBySeq[0].ID)

	// 8. Validation errors on GetRawChunks
	_, err = mgr.GetRawChunks(ctx, "townhall", false, RawChunkQuery{})
	assert.ErrorContains(t, err, "chunk_id, start_seq, or end_seq is required")

	badStart := int64(30)
	badEnd := int64(10)
	_, err = mgr.GetRawChunks(ctx, "townhall", false, RawChunkQuery{
		StartSeq: &badStart,
		EndSeq:   &badEnd,
	})
	assert.ErrorContains(t, err, "cannot be greater than")
}

func TestEpisodePrivacyIsolation(t *testing.T) {
	tempDir := t.TempDir()
	cfg := &config.Config{
		DataDir:        tempDir,
		EmbeddingModel: "none",
	}
	mgr := NewManager(cfg, nil)
	defer func() { _ = mgr.Close() }()

	ctx := context.Background()
	now := time.Now().Unix()

	// Townhall episode
	require.NoError(t, mgr.IndexMessages(ctx, "townhall", false, []MessageToStore{
		{
			Seq:        1,
			Timestamp:  now,
			ChatID:     "townhall",
			UserID:     "alice",
			SenderName: "Alice",
			Role:       "user",
			Content:    "Public release schedule for release_v2 is next Monday.",
		},
	}))

	// User 1 DM episode
	require.NoError(t, mgr.IndexMessages(ctx, "dm_user1", true, []MessageToStore{
		{
			Seq:        50,
			Timestamp:  now + 1,
			ChatID:     "dm_user1",
			UserID:     "user1",
			SenderName: "Alice",
			Role:       "user",
			Content:    "Private DM token for release_v2 is alpha_secret_token_777.",
		},
	}))

	// User 2 DM episode
	require.NoError(t, mgr.IndexMessages(ctx, "dm_user2", true, []MessageToStore{
		{
			Seq:        80,
			Timestamp:  now + 2,
			ChatID:     "dm_user2",
			UserID:     "user2",
			SenderName: "Bob",
			Role:       "user",
			Content:    "Private DM token for release_v2 is beta_secret_token_999.",
		},
	}))

	// 1. Townhall cannot see DM episodes or raw chunks
	thEps, err := mgr.SearchEpisodes(ctx, "release_v2", "townhall", false, 10)
	require.NoError(t, err)
	require.Len(t, thEps, 1)
	assert.Equal(t, "townhall", thEps[0].ChatID)

	thRawLeak, err := mgr.GetRawChunks(ctx, "townhall", false, RawChunkQuery{
		ChunkID: "chunk_dm_user1_50_50",
	})
	require.NoError(t, err)
	assert.Empty(t, thRawLeak)

	// 2. User 1 DM sees User 1 DM + Townhall, but NEVER User 2 DM
	u1Eps, err := mgr.SearchEpisodes(ctx, "release_v2", "dm_user1", true, 10)
	require.NoError(t, err)
	require.Len(t, u1Eps, 2)
	for _, ep := range u1Eps {
		assert.NotEqual(t, "dm_user2", ep.ChatID)
		assert.NotContains(t, ep.Summary, "beta_secret_token_999")
	}

	u1RawLeak, err := mgr.GetRawChunks(ctx, "dm_user1", true, RawChunkQuery{
		ChunkID: "chunk_dm_user2_80_80",
	})
	require.NoError(t, err)
	assert.Empty(t, u1RawLeak)

	// User 1 can fetch their own DM chunk and public Townhall chunk
	u1OwnRaw, err := mgr.GetRawChunks(ctx, "dm_user1", true, RawChunkQuery{
		ChunkID: "chunk_dm_user1_50_50",
	})
	require.NoError(t, err)
	require.Len(t, u1OwnRaw, 1)
	assert.Contains(t, u1OwnRaw[0].Content, "alpha_secret_token_777")
}

type failableEmbedder struct {
	dim    int
	failOn func(text string) error
	vecFn  func(text string) []float32
}

func (f *failableEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	if f.failOn != nil {
		if err := f.failOn(text); err != nil {
			return nil, err
		}
	}
	if f.vecFn != nil {
		return f.vecFn(text), nil
	}
	out := make([]float32, f.dim)
	for i := range out {
		out[i] = 0.25
	}
	return out, nil
}

func (f *failableEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	res := make([][]float32, len(texts))
	for i, t := range texts {
		v, err := f.Embed(ctx, t)
		if err != nil {
			return nil, err
		}
		res[i] = v
	}
	return res, nil
}

func (f *failableEmbedder) Dim() int {
	return f.dim
}

func (f *failableEmbedder) Model() string {
	return "failable-mock"
}

func TestEpisodeHybridSearchAndEmbedderFallback(t *testing.T) {
	tempDir := t.TempDir()
	cfg := &config.Config{DataDir: tempDir}
	ctx := context.Background()
	now := time.Now().Unix()

	failQueries := false
	emb := &failableEmbedder{
		dim: 3,
		failOn: func(text string) error {
			if failQueries {
				return errors.New("simulated embedder outage")
			}
			return nil
		},
		vecFn: func(text string) []float32 {
			if strings.Contains(strings.ToLower(text), "kubernetes") || strings.Contains(strings.ToLower(text), "orchestration") {
				return []float32{0.95, 0.85, 0.1}
			}
			return []float32{0.1, 0.1, 0.95}
		},
	}

	mgr := NewManager(cfg, emb)
	defer func() { _ = mgr.Close() }()

	require.NoError(t, mgr.IndexMessages(ctx, "townhall", false, []MessageToStore{
		{
			Seq:        1,
			Timestamp:  now,
			ChatID:     "townhall",
			UserID:     "alice",
			SenderName: "Alice",
			Role:       "user",
			Content:    "Kubernetes pod autoscaling policy updated.",
		},
	}))

	// 1. Semantic/hybrid search succeeds when embedder is healthy
	eps, err := mgr.SearchEpisodes(ctx, "orchestration", "townhall", false, 5)
	require.NoError(t, err)
	require.Len(t, eps, 1)
	assert.Equal(t, "ep_townhall_1_1", eps[0].ID)

	// 2. When embedder fails at query time, SearchEpisodes falls back gracefully to lexical search
	failQueries = true
	lexEps, err := mgr.SearchEpisodes(ctx, "Kubernetes", "townhall", false, 5)
	require.NoError(t, err)
	require.Len(t, lexEps, 1)
	assert.Equal(t, "ep_townhall_1_1", lexEps[0].ID)
}

func TestEpisodeIndexingFailureAndIdempotentRetry(t *testing.T) {
	tempDir := t.TempDir()
	cfg := &config.Config{DataDir: tempDir, EmbeddingModel: "none"}
	ctx := context.Background()
	now := time.Now().Unix()

	mgr := NewManager(cfg, nil)
	defer func() { _ = mgr.Close() }()

	// Initialize DB and add a temporary trigger that aborts INSERT into memory_episodes
	// so raw chunk SaveMemory succeeds while saveEpisode fails on the first attempt.
	db, err := mgr.GetDB(ctx, "townhall", false)
	require.NoError(t, err)
	_, err = db.SQL().ExecContext(ctx, `
		CREATE TRIGGER fail_episode_insert_once
		BEFORE INSERT ON memory_episodes
		BEGIN
			SELECT RAISE(ABORT, 'simulated episode table failure');
		END;
	`)
	require.NoError(t, err)

	msgs := []MessageToStore{
		{
			Seq:        15,
			Timestamp:  now,
			ChatID:     "townhall",
			UserID:     "alice",
			SenderName: "Alice",
			Role:       "user",
			Content:    "Database failover drill completed.",
		},
	}

	// First attempt fails during episode persistence after raw chunk was saved
	err = mgr.IndexMessages(ctx, "townhall", false, msgs)
	require.Error(t, err)
	assert.ErrorContains(t, err, "failed to save episode")

	// Watermark MUST NOT advance when IndexMessages returns an error
	wm, err := mgr.GetWatermark(ctx, "townhall", false)
	require.NoError(t, err)
	assert.Equal(t, int64(0), wm)

	// Remove the temporary failure trigger and retry
	_, err = db.SQL().ExecContext(ctx, `DROP TRIGGER fail_episode_insert_once;`)
	require.NoError(t, err)

	// Retry succeeds idempotently without duplicating chunks or episodes
	require.NoError(t, mgr.IndexMessages(ctx, "townhall", false, msgs))

	wm, err = mgr.GetWatermark(ctx, "townhall", false)
	require.NoError(t, err)
	assert.Equal(t, int64(15), wm)

	eps, err := mgr.ListEpisodes(ctx, "townhall", false)
	require.NoError(t, err)
	require.Len(t, eps, 1)

	chunks, err := mgr.GetRawChunks(ctx, "townhall", false, RawChunkQuery{ChunkID: "ep_townhall_15_15"})
	require.NoError(t, err)
	require.Len(t, chunks, 1)
}

func TestEpisodeEmptyChunkLinksAndMalformedJSON(t *testing.T) {
	tempDir := t.TempDir()
	cfg := &config.Config{DataDir: tempDir, EmbeddingModel: "none"}
	mgr := NewManager(cfg, nil)
	defer func() { _ = mgr.Close() }()

	ctx := context.Background()
	now := time.Now().Unix()

	require.NoError(t, mgr.IndexMessages(ctx, "townhall", false, []MessageToStore{
		{
			Seq:        1,
			Timestamp:  now,
			ChatID:     "townhall",
			UserID:     "alice",
			SenderName: "Alice",
			Role:       "user",
			Content:    "Unrelated townhall chunk that must not leak on empty episode links.",
		},
	}))

	db, err := mgr.GetDB(ctx, "townhall", false)
	require.NoError(t, err)

	// 1. Insert an episode with empty source_chunk_ids_json ("[]")
	_, err = db.SQL().ExecContext(ctx, `
		INSERT INTO memory_episodes (
			id, chat_id, chat_type, thread_id, start_seq, end_seq,
			start_time, end_time, participants_json, topic_tags_json,
			task_status, summary, source_chunk_ids_json, model_derived, created_at
		) VALUES ('ep_empty_links', 'townhall', 'townhall', '', 50, 51, ?, ?, '["Alice"]', '[]', 'discussion', 'summary', '[]', 1, ?)
	`, now, now, now)
	require.NoError(t, err)

	emptyRes, err := mgr.GetRawChunks(ctx, "townhall", false, RawChunkQuery{ChunkID: "ep_empty_links"})
	require.NoError(t, err)
	assert.Empty(t, emptyRes, "episode with empty SourceChunkIDs must return no chunks, not unrelated chunks")

	// 2. Insert an episode with malformed source_chunk_ids_json
	_, err = db.SQL().ExecContext(ctx, `
		INSERT INTO memory_episodes (
			id, chat_id, chat_type, thread_id, start_seq, end_seq,
			start_time, end_time, participants_json, topic_tags_json,
			task_status, summary, source_chunk_ids_json, model_derived, created_at
		) VALUES ('ep_corrupt_json', 'townhall', 'townhall', '', 60, 61, ?, ?, '["Alice"]', '[]', 'discussion', 'summary', '{bad-json', 1, ?)
	`, now, now, now)
	require.NoError(t, err)

	_, err = mgr.GetRawChunks(ctx, "townhall", false, RawChunkQuery{ChunkID: "ep_corrupt_json"})
	require.Error(t, err)
	assert.ErrorContains(t, err, "decode source_chunk_ids_json")
}

func TestSanitizationCollidingChatIDsIsolation(t *testing.T) {
	tempDir := t.TempDir()
	cfg := &config.Config{DataDir: tempDir, EmbeddingModel: "none"}
	mgr := NewManager(cfg, nil)
	defer func() { _ = mgr.Close() }()

	ctx := context.Background()
	now := time.Now().Unix()

	// "user:alpha" and "user_alpha" both sanitize to "user_alpha" (sharing dm_user_alpha.db)
	chatA := "user:alpha"
	chatB := "user_alpha"
	assert.Equal(t, SanitizeChatID(chatA), SanitizeChatID(chatB))

	require.NoError(t, mgr.IndexMessages(ctx, chatA, true, []MessageToStore{
		{
			Seq:        1,
			Timestamp:  now,
			ChatID:     chatA,
			UserID:     "u_a",
			SenderName: "UserA",
			Role:       "user",
			Content:    "Shared keyword nebula with secret_from_colon_chat.",
		},
	}))

	require.NoError(t, mgr.IndexMessages(ctx, chatB, true, []MessageToStore{
		{
			Seq:        1,
			Timestamp:  now + 1,
			ChatID:     chatB,
			UserID:     "u_b",
			SenderName: "UserB",
			Role:       "user",
			Content:    "Shared keyword nebula with secret_from_underscore_chat.",
		},
	}))

	// 1. SearchEpisodes for chatA must only return chatA's episode
	epsA, err := mgr.SearchEpisodes(ctx, "nebula", chatA, true, 10)
	require.NoError(t, err)
	require.Len(t, epsA, 1)
	assert.Equal(t, chatA, epsA[0].ChatID)
	assert.NotContains(t, epsA[0].Summary, "secret_from_underscore_chat")

	// 2. Manager.Search for chatA must only return chatA's chunk/episode
	itemsA, err := mgr.Search(ctx, "nebula", chatA, true, 10)
	require.NoError(t, err)
	require.Len(t, itemsA, 1)
	assert.Equal(t, chatA, itemsA[0].ChatID)
	assert.NotContains(t, itemsA[0].Content, "secret_from_underscore_chat")

	// 3. GetRawChunks for chatA with chatB's episode ID must return empty
	epsB, err := mgr.ListEpisodes(ctx, chatB, true)
	require.NoError(t, err)
	require.Len(t, epsB, 1)

	crossRaw, err := mgr.GetRawChunks(ctx, chatA, true, RawChunkQuery{ChunkID: epsB[0].ID})
	require.NoError(t, err)
	assert.Empty(t, crossRaw)
}

func TestUTF8SafeEpisodeSummaryTruncation(t *testing.T) {
	now := time.Now().Unix()
	longUTF8 := strings.Repeat("Привет мир 🚀 日本語テスト ", 40)
	ep := buildEpisodeFromMessages("townhall", false, []MessageToStore{
		{
			Seq:        1,
			Timestamp:  now,
			ChatID:     "townhall",
			UserID:     "u1",
			SenderName: "Alice",
			Role:       "user",
			Content:    longUTF8,
		},
	}, []string{"chunk_townhall_1_1"})

	assert.True(t, utf8.ValidString(ep.Summary), "episode summary must remain valid UTF-8 after truncation")
	assert.True(t, utf8.ValidString(ep.FormatNavigationEntry()), "formatted navigation entry must remain valid UTF-8")
}

