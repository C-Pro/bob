package knowledge

import (
	"context"
	"errors"
	"path/filepath"
	"testing"
	"time"

	"github.com/liliang-cn/cortexdb/v2/pkg/cortexdb"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func setupTestVectorDB(t *testing.T) *cortexdb.DB {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "vector_test.db")
	db, err := cortexdb.Open(cortexdb.DefaultConfig(path))
	require.NoError(t, err)
	t.Cleanup(func() {
		_ = db.Close()
	})
	return db
}

type mockVectorDB struct {
	saveFn   func(ctx context.Context, req cortexdb.MemorySaveRequest) (*cortexdb.MemorySaveResponse, error)
	searchFn func(ctx context.Context, req cortexdb.MemorySearchRequest) (*cortexdb.MemorySearchResponse, error)
	deleteFn func(ctx context.Context, req cortexdb.MemoryDeleteRequest) (*cortexdb.MemoryDeleteResponse, error)
}

func (m *mockVectorDB) SaveMemory(ctx context.Context, req cortexdb.MemorySaveRequest) (*cortexdb.MemorySaveResponse, error) {
	if m.saveFn != nil {
		return m.saveFn(ctx, req)
	}
	return &cortexdb.MemorySaveResponse{}, nil
}

func (m *mockVectorDB) SearchMemory(ctx context.Context, req cortexdb.MemorySearchRequest) (*cortexdb.MemorySearchResponse, error) {
	if m.searchFn != nil {
		return m.searchFn(ctx, req)
	}
	return &cortexdb.MemorySearchResponse{}, nil
}

func (m *mockVectorDB) DeleteMemory(ctx context.Context, req cortexdb.MemoryDeleteRequest) (*cortexdb.MemoryDeleteResponse, error) {
	if m.deleteFn != nil {
		return m.deleteFn(ctx, req)
	}
	return &cortexdb.MemoryDeleteResponse{Deleted: true}, nil
}

func TestIndexer_IndexMemoryAndSkill(t *testing.T) {
	ctx := context.Background()
	kStore, rawDB := setupTestStore(t)
	defer func() { _ = rawDB.Close() }()

	vDB := setupTestVectorDB(t)
	indexer := NewIndexer(kStore, vDB)

	// 1. Propose and approve memory
	_, memVer, err := kStore.ProposeMemory(ctx, &KnowledgeItem{
		ChatID: "chat_idx_test",
		UserID: "user_idx_test",
	}, &MemoryVersion{
		Type:       MemoryTypePreference,
		Content:    "Prefers dark mode and concise logs",
		Confidence: 0.9,
	})
	require.NoError(t, err)
	require.NoError(t, kStore.ApproveVersion(ctx, memVer.ID, "user_idx_test"))

	// Index memory
	err = indexer.IndexVersion(ctx, memVer.ID)
	require.NoError(t, err)

	updatedMem, err := kStore.GetMemoryVersion(ctx, memVer.ID)
	require.NoError(t, err)
	assert.Equal(t, IndexStatusReady, updatedMem.IndexStatus)

	// Verify CortexDB search finds the indexed memory in dm_memories
	searchResp, err := vDB.SearchMemory(ctx, cortexdb.MemorySearchRequest{
		Query:         "dark mode",
		Scope:         cortexdb.MemoryScopeGlobal,
		Namespace:     NamespaceMemories,
		RetrievalMode: cortexdb.RetrievalModeLexical,
		TopK:          5,
	})
	require.NoError(t, err)
	require.NotNil(t, searchResp)
	require.NotEmpty(t, searchResp.Results)
	assert.Equal(t, memVer.ID, searchResp.Results[0].Memory.ID)

	// 2. Propose and approve skill
	_, skillVer, err := kStore.ProposeSkill(ctx, &KnowledgeItem{
		ChatID: "chat_idx_test",
		UserID: "user_idx_test",
	}, &SkillVersion{
		Name:                 "run_benchmarks",
		Description:          "Runs go benchmark tests",
		Triggers:             []string{"benchmark", "perf test"},
		Tags:                 []string{"performance", "testing"},
		InstructionsMarkdown: "# Run Benchmarks\n`go test -bench=. ./...`",
	})
	require.NoError(t, err)
	require.NoError(t, kStore.ApproveVersion(ctx, skillVer.ID, "user_idx_test"))

	// Index skill
	err = indexer.IndexVersion(ctx, skillVer.ID)
	require.NoError(t, err)

	updatedSkill, err := kStore.GetSkillVersion(ctx, skillVer.ID)
	require.NoError(t, err)
	assert.Equal(t, IndexStatusReady, updatedSkill.IndexStatus)

	// Verify CortexDB search finds the indexed skill in dm_skills
	searchSkillResp, err := vDB.SearchMemory(ctx, cortexdb.MemorySearchRequest{
		Query:         "benchmark",
		Scope:         cortexdb.MemoryScopeGlobal,
		Namespace:     NamespaceSkills,
		RetrievalMode: cortexdb.RetrievalModeLexical,
		TopK:          5,
	})
	require.NoError(t, err)
	require.NotNil(t, searchSkillResp)
	require.NotEmpty(t, searchSkillResp.Results)
	assert.Equal(t, skillVer.ID, searchSkillResp.Results[0].Memory.ID)

	// 3. Remove from index
	err = indexer.RemoveFromIndex(ctx, memVer.ID)
	require.NoError(t, err)
}

func TestIndexer_ReconcilePending(t *testing.T) {
	ctx := context.Background()
	kStore, rawDB := setupTestStore(t)
	defer func() { _ = rawDB.Close() }()

	vDB := setupTestVectorDB(t)
	indexer := NewIndexer(kStore, vDB)

	// Create 2 memories and 1 skill, approved, leaving index_status as pending
	_, mem1, err := kStore.ProposeMemory(ctx, &KnowledgeItem{
		ChatID: "chat_rec",
		UserID: "user_rec",
	}, &MemoryVersion{
		Type:    MemoryTypeFact,
		Content: "Server is located in us-east-1",
	})
	require.NoError(t, err)
	require.NoError(t, kStore.ApproveVersion(ctx, mem1.ID, "user_rec"))

	_, mem2, err := kStore.ProposeMemory(ctx, &KnowledgeItem{
		ChatID: "chat_rec",
		UserID: "user_rec",
	}, &MemoryVersion{
		Type:    MemoryTypeDecision,
		Content: "Use SQLite for state storage",
	})
	require.NoError(t, err)
	require.NoError(t, kStore.ApproveVersion(ctx, mem2.ID, "user_rec"))

	_, skill1, err := kStore.ProposeSkill(ctx, &KnowledgeItem{
		ChatID: "chat_rec",
		UserID: "user_rec",
	}, &SkillVersion{
		Name:                 "deploy",
		Description:          "Deploy to staging",
		InstructionsMarkdown: "make deploy",
	})
	require.NoError(t, err)
	require.NoError(t, kStore.ApproveVersion(ctx, skill1.ID, "user_rec"))

	// All 3 are pending index sync
	pending, err := kStore.GetPendingIndexVersions(ctx, 10)
	require.NoError(t, err)
	assert.Len(t, pending, 3)

	// Run ReconcilePending
	reconciled, err := indexer.ReconcilePending(ctx, 10)
	require.NoError(t, err)
	assert.Equal(t, 3, reconciled)

	// Pending queue is now empty
	pendingAfter, err := kStore.GetPendingIndexVersions(ctx, 10)
	require.NoError(t, err)
	assert.Empty(t, pendingAfter)

	// All 3 have status ready
	m1After, _ := kStore.GetMemoryVersion(ctx, mem1.ID)
	assert.Equal(t, IndexStatusReady, m1After.IndexStatus)
	m2After, _ := kStore.GetMemoryVersion(ctx, mem2.ID)
	assert.Equal(t, IndexStatusReady, m2After.IndexStatus)
	s1After, _ := kStore.GetSkillVersion(ctx, skill1.ID)
	assert.Equal(t, IndexStatusReady, s1After.IndexStatus)
}

func TestIndexer_FailureHandling(t *testing.T) {
	ctx := context.Background()
	kStore, rawDB := setupTestStore(t)
	defer func() { _ = rawDB.Close() }()

	mockV := &mockVectorDB{
		saveFn: func(ctx context.Context, req cortexdb.MemorySaveRequest) (*cortexdb.MemorySaveResponse, error) {
			return nil, errors.New("simulated vector db disk failure")
		},
	}
	indexer := NewIndexer(kStore, mockV)

	_, mem, err := kStore.ProposeMemory(ctx, &KnowledgeItem{
		ChatID: "chat_fail",
		UserID: "user_fail",
	}, &MemoryVersion{
		Type:    MemoryTypeFact,
		Content: "Fail test",
	})
	require.NoError(t, err)
	require.NoError(t, kStore.ApproveVersion(ctx, mem.ID, "user_fail"))

	err = indexer.IndexVersion(ctx, mem.ID)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "simulated vector db disk failure")

	verAfter, err := kStore.GetMemoryVersion(ctx, mem.ID)
	require.NoError(t, err)
	assert.Equal(t, IndexStatusError, verAfter.IndexStatus)
}

func TestIndexer_FormatContent(t *testing.T) {
	mem := &MemoryVersion{
		Type:    MemoryTypeDecision,
		Content: "Always run tests with -race",
	}
	assert.Equal(t, "[decision] Always run tests with -race", FormatMemoryContent(mem))

	skill := &SkillVersion{
		Name:                 "test_cmd",
		Description:          "runs tests",
		Triggers:             []string{"test", "check"},
		Tags:                 []string{"ci", "dev"},
		InstructionsMarkdown: "go test ./...",
	}
	formattedSkill := FormatSkillContent(skill)
	assert.Contains(t, formattedSkill, "Skill: test_cmd")
	assert.Contains(t, formattedSkill, "Description: runs tests")
	assert.Contains(t, formattedSkill, "Triggers: test, check")
	assert.Contains(t, formattedSkill, "Tags: ci, dev")
	assert.Contains(t, formattedSkill, "Instructions:\ngo test ./...")
}

func TestIndexer_PostSaveRaceResurrectPrevention(t *testing.T) {
	ctx := context.Background()
	kStore, rawDB := setupTestStore(t)
	defer func() { _ = rawDB.Close() }()

	deletedFromVector := false
	saveEntered := make(chan struct{})
	itemErased := make(chan struct{})

	mockV := &mockVectorDB{
		saveFn: func(ctx context.Context, req cortexdb.MemorySaveRequest) (*cortexdb.MemorySaveResponse, error) {
			close(saveEntered)
			<-itemErased
			return &cortexdb.MemorySaveResponse{}, nil
		},
		deleteFn: func(ctx context.Context, req cortexdb.MemoryDeleteRequest) (*cortexdb.MemoryDeleteResponse, error) {
			deletedFromVector = true
			return &cortexdb.MemoryDeleteResponse{Deleted: true}, nil
		},
	}

	indexer := NewIndexer(kStore, mockV)

	item, ver, err := kStore.ProposeMemory(ctx, &KnowledgeItem{
		ChatID: "chat_race",
		UserID: "user_race",
	}, &MemoryVersion{
		Type:    MemoryTypeFact,
		Content: "Durable secret",
	})
	require.NoError(t, err)
	require.NoError(t, kStore.ApproveVersion(ctx, ver.ID, "user_race"))

	errCh := make(chan error, 1)
	go func() {
		errCh <- indexer.IndexVersion(ctx, ver.ID)
	}()

	<-saveEntered
	// Item is concurrently forgotten while SaveMemory is in-flight
	require.NoError(t, kStore.ForgetItem(ctx, item.ID, "user_race"))
	close(itemErased)

	indexErr := <-errCh
	assert.Error(t, indexErr)
	assert.Contains(t, indexErr.Error(), "modified concurrently")
	assert.True(t, deletedFromVector, "entry should have been immediately deleted from vector DB after detecting race")
}

func TestIndexer_MetadataPreservation(t *testing.T) {
	ctx := context.Background()
	kStore, rawDB := setupTestStore(t)
	defer func() { _ = rawDB.Close() }()

	vDB := setupTestVectorDB(t)
	indexer := NewIndexer(kStore, vDB)

	_, skillVer, err := kStore.ProposeSkill(ctx, &KnowledgeItem{
		ChatID: "chat_meta",
		UserID: "user_meta",
	}, &SkillVersion{
		Name:                 "meta_skill",
		Description:          "meta desc",
		InstructionsMarkdown: "meta md",
	})
	require.NoError(t, err)
	require.NoError(t, kStore.ApproveVersion(ctx, skillVer.ID, "user_meta"))

	err = indexer.IndexVersion(ctx, skillVer.ID)
	require.NoError(t, err)

	resp, err := vDB.SearchMemory(ctx, cortexdb.MemorySearchRequest{
		Query:         "meta_skill",
		Scope:         cortexdb.MemoryScopeGlobal,
		Namespace:     NamespaceSkills,
		RetrievalMode: cortexdb.RetrievalModeLexical,
	})
	require.NoError(t, err)
	require.NotEmpty(t, resp.Results)
	meta := resp.Results[0].Memory.Metadata
	require.NotNil(t, meta)
	// While CortexDB sets "kind": "memory", "knowledge_kind" must be preserved as "skill"
	assert.Equal(t, string(KindSkill), meta["knowledge_kind"])
}

func TestIndexer_PostSaveCanceledContextCompensatingDeletion(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	kStore, rawDB := setupTestStore(t)
	defer func() { _ = rawDB.Close() }()

	deletedWithNonCanceledContext := false
	var item *KnowledgeItem
	mockV := &mockVectorDB{
		saveFn: func(saveCtx context.Context, req cortexdb.MemorySaveRequest) (*cortexdb.MemorySaveResponse, error) {
			// Concurrently forget item while save is in-flight so MarkVersionIndexed fails
			_ = kStore.ForgetItem(context.Background(), item.ID, "user_cancel")
			// Cancel parent ctx right after save completes
			cancel()
			return &cortexdb.MemorySaveResponse{}, nil
		},
		deleteFn: func(delCtx context.Context, req cortexdb.MemoryDeleteRequest) (*cortexdb.MemoryDeleteResponse, error) {
			if delCtx.Err() == nil {
				deletedWithNonCanceledContext = true
			}
			return &cortexdb.MemoryDeleteResponse{Deleted: true}, nil
		},
	}

	indexer := NewIndexer(kStore, mockV)

	var ver *MemoryVersion
	var err error
	item, ver, err = kStore.ProposeMemory(context.Background(), &KnowledgeItem{
		ChatID: "chat_cancel",
		UserID: "user_cancel",
	}, &MemoryVersion{
		Type:    MemoryTypeFact,
		Content: "Secret",
	})
	require.NoError(t, err)
	require.NoError(t, kStore.ApproveVersion(context.Background(), ver.ID, "user_cancel"))

	err = indexer.IndexVersion(ctx, ver.ID)
	assert.Error(t, err)
	assert.True(t, deletedWithNonCanceledContext, "compensating deletion must succeed even when request context was canceled")
}

func TestIndexer_ExpiredMemoryRejection(t *testing.T) {
	ctx := context.Background()
	kStore, rawDB := setupTestStore(t)
	defer func() { _ = rawDB.Close() }()

	deletedFromVector := false
	ttlDays := 1
	_, ver, err := kStore.ProposeMemory(ctx, &KnowledgeItem{
		ChatID: "chat_exp",
		UserID: "user_exp",
	}, &MemoryVersion{
		Type:    MemoryTypeFact,
		Content: "Expiring content",
		TTLDays: &ttlDays,
	})
	require.NoError(t, err)
	require.NoError(t, kStore.ApproveVersion(ctx, ver.ID, "user_exp"))

	// Initially before expiration
	currentTime := time.Unix(*ver.ExpiresAt-10, 0)

	mockV := &mockVectorDB{
		saveFn: func(ctx context.Context, req cortexdb.MemorySaveRequest) (*cortexdb.MemorySaveResponse, error) {
			// Memory expires while SaveMemory is in flight
			currentTime = time.Unix(*ver.ExpiresAt+10, 0)
			return &cortexdb.MemorySaveResponse{}, nil
		},
		deleteFn: func(ctx context.Context, req cortexdb.MemoryDeleteRequest) (*cortexdb.MemoryDeleteResponse, error) {
			deletedFromVector = true
			return &cortexdb.MemoryDeleteResponse{Deleted: true}, nil
		},
	}

	indexer := NewIndexer(kStore, mockV)
	indexer.now = func() time.Time {
		return currentTime
	}

	err = indexer.IndexVersion(ctx, ver.ID)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "modified concurrently")
	assert.True(t, deletedFromVector, "expired memory must be purged from vector database via compensating deletion")
}

