package knowledge

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync/atomic"
	"testing"
	"testing/synctest"
	"time"
	"unicode/utf8"

	"github.com/liliang-cn/cortexdb/v2/pkg/cortexdb"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSearcher_SearchMemories(t *testing.T) {
	ctx := context.Background()
	kStore, rawDB := setupTestStore(t)
	defer func() { _ = rawDB.Close() }()

	vDB := setupTestVectorDB(t)
	indexer := NewIndexer(kStore, vDB)

	session := SessionIdentity{
		ChatID: "chat_dm_search",
		UserID: "user_search",
	}

	searcher, err := NewSearcher(kStore, vDB, session, 10)
	require.NoError(t, err)

	// 1. Create and index valid preference memory
	_, memPref, err := kStore.ProposeMemory(ctx, &KnowledgeItem{
		ChatID: session.ChatID,
		UserID: session.UserID,
	}, &MemoryVersion{
		Type:       MemoryTypePreference,
		Content:    "Alice prefers concise markdown format with code examples",
		Confidence: 0.95,
	})
	require.NoError(t, err)
	require.NoError(t, kStore.ApproveVersion(ctx, memPref.ID, session.UserID))
	require.NoError(t, indexer.IndexVersion(ctx, memPref.ID))

	// 2. Create and index valid fact memory
	_, memFact, err := kStore.ProposeMemory(ctx, &KnowledgeItem{
		ChatID: session.ChatID,
		UserID: session.UserID,
	}, &MemoryVersion{
		Type:       MemoryTypeFact,
		Content:    "The production cluster runs on Kubernetes 1.30 in eu-central",
		Confidence: 0.99,
	})
	require.NoError(t, err)
	require.NoError(t, kStore.ApproveVersion(ctx, memFact.ID, session.UserID))
	require.NoError(t, indexer.IndexVersion(ctx, memFact.ID))

	// 3. Create memory belonging to another user (same chat, different user)
	_, memOtherUser, err := kStore.ProposeMemory(ctx, &KnowledgeItem{
		ChatID: session.ChatID,
		UserID: "other_user",
	}, &MemoryVersion{
		Type:    MemoryTypeFact,
		Content: "Other user confidential Kubernetes credentials",
	})
	require.NoError(t, err)
	require.NoError(t, kStore.ApproveVersion(ctx, memOtherUser.ID, "other_user"))
	require.NoError(t, indexer.IndexVersion(ctx, memOtherUser.ID))

	// 4. Create memory belonging to another chat
	_, memOtherChat, err := kStore.ProposeMemory(ctx, &KnowledgeItem{
		ChatID: "different_chat",
		UserID: session.UserID,
	}, &MemoryVersion{
		Type:    MemoryTypeFact,
		Content: "Different chat secret markdown notes",
	})
	require.NoError(t, err)
	require.NoError(t, kStore.ApproveVersion(ctx, memOtherChat.ID, session.UserID))
	require.NoError(t, indexer.IndexVersion(ctx, memOtherChat.ID))

	// Perform search for "markdown"
	results, err := searcher.SearchMemories(ctx, "markdown", nil, 5)
	require.NoError(t, err)
	require.Len(t, results, 1)
	assert.Equal(t, memPref.ItemID, results[0].ItemID)
	assert.Equal(t, memPref.ID, results[0].VersionID)
	assert.Equal(t, MemoryTypePreference, results[0].Type)
	assert.Equal(t, 0.95, results[0].Confidence)
	assert.Contains(t, results[0].Snippet, "Alice prefers concise markdown")

	// Search for "Kubernetes": MUST return memFact, MUST NOT return memOtherUser
	kubeResults, err := searcher.SearchMemories(ctx, "Kubernetes", nil, 5)
	require.NoError(t, err)
	require.Len(t, kubeResults, 1)
	assert.Equal(t, memFact.ItemID, kubeResults[0].ItemID)

	// Filter by memory type: search for "concise" but filter for MemoryTypeFact -> should return 0 results
	factOnlyResults, err := searcher.SearchMemories(ctx, "concise", []MemoryType{MemoryTypeFact}, 5)
	require.NoError(t, err)
	assert.Empty(t, factOnlyResults)

	// Filter by memory type: search for "concise" and filter for MemoryTypePreference -> returns memPref
	prefOnlyResults, err := searcher.SearchMemories(ctx, "concise", []MemoryType{MemoryTypePreference}, 5)
	require.NoError(t, err)
	require.Len(t, prefOnlyResults, 1)
	assert.Equal(t, memPref.ItemID, prefOnlyResults[0].ItemID)

	// Inactive / Forgotten memory filtering and index cleanup:
	// Forget memFact
	require.NoError(t, kStore.ForgetItem(ctx, memFact.ItemID, session.UserID))

	// Searching Kubernetes should now return nothing because memFact is forgotten
	afterForget, err := searcher.SearchMemories(ctx, "Kubernetes", nil, 5)
	require.NoError(t, err)
	assert.Empty(t, afterForget)
}

func TestSearcher_SearchSkills(t *testing.T) {
	ctx := context.Background()
	kStore, rawDB := setupTestStore(t)
	defer func() { _ = rawDB.Close() }()

	vDB := setupTestVectorDB(t)
	indexer := NewIndexer(kStore, vDB)

	session := SessionIdentity{
		ChatID: "chat_skill_search",
		UserID: "user_skill_search",
	}

	searcher, err := NewSearcher(kStore, vDB, session, 10)
	require.NoError(t, err)

	// 1. Create skill with description and tags
	_, skill1, err := kStore.ProposeSkill(ctx, &KnowledgeItem{
		ChatID: session.ChatID,
		UserID: session.UserID,
	}, &SkillVersion{
		Name:                 "deploy_app",
		Description:          "Deploy the containerized application to AWS ECS cluster",
		Triggers:             []string{"deploy", "ship it"},
		Tags:                 []string{"aws", "deploy", "prod"},
		InstructionsMarkdown: "# Deploy Instructions\n1. Run `make release`\n2. Run `aws ecs update-service`",
	})
	require.NoError(t, err)
	require.NoError(t, kStore.ApproveVersion(ctx, skill1.ID, session.UserID))
	require.NoError(t, indexer.IndexVersion(ctx, skill1.ID))

	// 2. Create skill without description (snippet falls back to InstructionsMarkdown)
	_, skill2, err := kStore.ProposeSkill(ctx, &KnowledgeItem{
		ChatID: session.ChatID,
		UserID: session.UserID,
	}, &SkillVersion{
		Name:                 "run_lint",
		Description:          "",
		Triggers:             []string{"lint", "check"},
		Tags:                 []string{"linter", "quality"},
		InstructionsMarkdown: "golangci-lint run --fast",
	})
	require.NoError(t, err)
	require.NoError(t, kStore.ApproveVersion(ctx, skill2.ID, session.UserID))
	require.NoError(t, indexer.IndexVersion(ctx, skill2.ID))

	// Search by keyword "containerized"
	results, err := searcher.SearchSkills(ctx, "containerized", nil, 5)
	require.NoError(t, err)
	require.Len(t, results, 1)
	assert.Equal(t, skill1.ItemID, results[0].ItemID)
	assert.Equal(t, "deploy_app", results[0].Name)
	assert.Equal(t, "Deploy the containerized application to AWS ECS cluster", results[0].Snippet)
	assert.Equal(t, []string{"aws", "deploy", "prod"}, results[0].Tags)

	// Search skill2: snippet falls back to instructions markdown
	lintResults, err := searcher.SearchSkills(ctx, "golangci", nil, 5)
	require.NoError(t, err)
	require.Len(t, lintResults, 1)
	assert.Equal(t, skill2.ItemID, lintResults[0].ItemID)
	assert.Equal(t, "golangci-lint run --fast", lintResults[0].Snippet)

	// Tag filtering: search "deploy" with tag "aws" (case-insensitive)
	tagResults, err := searcher.SearchSkills(ctx, "deploy", []string{"AWS"}, 5)
	require.NoError(t, err)
	require.Len(t, tagResults, 1)
	assert.Equal(t, skill1.ItemID, tagResults[0].ItemID)

	// Tag filtering: search "deploy" with tag "nonexistent" -> empty
	noTagResults, err := searcher.SearchSkills(ctx, "deploy", []string{"nonexistent"}, 5)
	require.NoError(t, err)
	assert.Empty(t, noTagResults)

	// Disable skill1: search should no longer return it
	require.NoError(t, kStore.DisableSkill(ctx, skill1.ItemID, session.UserID))
	afterDisable, err := searcher.SearchSkills(ctx, "containerized", nil, 5)
	require.NoError(t, err)
	assert.Empty(t, afterDisable)
}

func TestSearcher_LexicalFallback(t *testing.T) {
	ctx := context.Background()
	kStore, rawDB := setupTestStore(t)
	defer func() { _ = rawDB.Close() }()

	session := SessionIdentity{
		ChatID: "chat_lex",
		UserID: "user_lex",
	}

	_, mem, err := kStore.ProposeMemory(ctx, &KnowledgeItem{
		ChatID: session.ChatID,
		UserID: session.UserID,
	}, &MemoryVersion{
		Type:       MemoryTypeFact,
		Content:    "Fallback verification fact",
		Confidence: 1.0,
	})
	require.NoError(t, err)
	require.NoError(t, kStore.ApproveVersion(ctx, mem.ID, session.UserID))

	callCount := 0
	mockV := &mockVectorDB{
		searchFn: func(ctx context.Context, req cortexdb.MemorySearchRequest) (*cortexdb.MemorySearchResponse, error) {
			callCount++
			if req.RetrievalMode == cortexdb.RetrievalModeLexical {
				// Lexical fallback succeeded
				return &cortexdb.MemorySearchResponse{
					Results: []cortexdb.MemorySearchHit{
						{
							Memory: cortexdb.MemoryRecord{
								ID: mem.ID,
							},
							Score: 0.88,
						},
					},
				}, nil
			}
			// Vector search fails
			return nil, errors.New("embedding model timed out")
		},
	}

	searcher, err := NewSearcher(kStore, mockV, session, 5)
	require.NoError(t, err)

	results, err := searcher.SearchMemories(ctx, "verification", nil, 5)
	require.NoError(t, err)
	require.Len(t, results, 1)
	assert.Equal(t, mem.ItemID, results[0].ItemID)
	assert.Equal(t, 2, callCount, "should have called vector search first, then lexical fallback")
}

func TestSearcher_Validation(t *testing.T) {
	ctx := context.Background()
	kStore, rawDB := setupTestStore(t)
	defer func() { _ = rawDB.Close() }()

	// Missing chat ID or user ID
	_, err := NewSearcher(kStore, nil, SessionIdentity{ChatID: "", UserID: "u1"}, 5)
	assert.Error(t, err)

	_, err = NewSearcher(kStore, nil, SessionIdentity{ChatID: "c1", UserID: ""}, 5)
	assert.Error(t, err)

	searcher, err := NewSearcher(kStore, nil, SessionIdentity{ChatID: "c1", UserID: "u1"}, 5)
	require.NoError(t, err)

	// Blank query
	mems, err := searcher.SearchMemories(ctx, "   ", nil, 5)
	require.NoError(t, err)
	assert.Empty(t, mems)

	skills, err := searcher.SearchSkills(ctx, "   ", nil, 5)
	require.NoError(t, err)
	assert.Empty(t, skills)

	// Invalid memory type
	_, err = searcher.SearchMemories(ctx, "valid query", []MemoryType{"invalid_type"}, 5)
	assert.ErrorIs(t, err, ErrInvalidStatus)
}

func TestTruncateRunes(t *testing.T) {
	// Empty string
	assert.Equal(t, "", TruncateRunes("", 10))
	assert.Equal(t, "", TruncateRunes("hello", 0))
	assert.Equal(t, "", TruncateRunes("hello", -1))

	// Shorter than max
	assert.Equal(t, "short text", TruncateRunes("short text", 20))

	// Exactly max
	assert.Equal(t, "exact length", TruncateRunes("exact length", 12))

	// Longer than max (ASCII)
	longText := strings.Repeat("a", 200)
	truncated := TruncateRunes(longText, 150)
	assert.Equal(t, 150, utf8.RuneCountInString(truncated))
	assert.True(t, strings.HasSuffix(truncated, "..."))
	assert.Equal(t, strings.Repeat("a", 147)+"...", truncated)

	// Multibyte Unicode (Cyrillic + Emoji)
	unicodeText := "Привет, мир! 🚀 Привет, мир! 🚀 Привет, мир! 🚀"
	truncatedUnicode := TruncateRunes(unicodeText, 15)
	assert.Equal(t, 15, utf8.RuneCountInString(truncatedUnicode))
	assert.True(t, strings.HasSuffix(truncatedUnicode, "..."))

	// Short maxRunes <= 3
	assert.Equal(t, "abc", TruncateRunes("abcdef", 3))
	assert.Equal(t, "ab", TruncateRunes("abcdef", 2))
	assert.Equal(t, "a", TruncateRunes("abcdef", 1))
}

func TestSearcher_WrongKindCandidateNotDeleted(t *testing.T) {
	ctx := context.Background()
	kStore, rawDB := setupTestStore(t)
	defer func() { _ = rawDB.Close() }()

	session := SessionIdentity{
		ChatID: "c1",
		UserID: "u1",
	}

	var deletedIDs []string
	mockV := &mockVectorDB{
		searchFn: func(ctx context.Context, req cortexdb.MemorySearchRequest) (*cortexdb.MemorySearchResponse, error) {
			if req.Namespace == NamespaceMemories {
				return &cortexdb.MemorySearchResponse{
					Results: []cortexdb.MemorySearchHit{
						{Memory: cortexdb.MemoryRecord{ID: "skill_abc"}, Score: 0.9},
						{Memory: cortexdb.MemoryRecord{ID: "random_123"}, Score: 0.8},
						{Memory: cortexdb.MemoryRecord{ID: "mem_missing_id"}, Score: 0.7},
					},
				}, nil
			}
			return &cortexdb.MemorySearchResponse{
				Results: []cortexdb.MemorySearchHit{
					{Memory: cortexdb.MemoryRecord{ID: "mem_xyz"}, Score: 0.9},
					{Memory: cortexdb.MemoryRecord{ID: "random_456"}, Score: 0.8},
					{Memory: cortexdb.MemoryRecord{ID: "skill_missing_id"}, Score: 0.7},
				},
			}, nil
		},
		deleteFn: func(ctx context.Context, req cortexdb.MemoryDeleteRequest) (*cortexdb.MemoryDeleteResponse, error) {
			deletedIDs = append(deletedIDs, req.MemoryID)
			return &cortexdb.MemoryDeleteResponse{Deleted: true}, nil
		},
	}

	searcher, err := NewSearcher(kStore, mockV, session, 5)
	require.NoError(t, err)

	// Search memories
	memResults, err := searcher.SearchMemories(ctx, "test", nil, 5)
	require.NoError(t, err)
	assert.Empty(t, memResults)
	// Only mem_missing_id should be deleted, not skill_abc or random_123
	assert.Equal(t, []string{"mem_missing_id"}, deletedIDs)

	// Reset deletedIDs and search skills
	deletedIDs = nil
	skillResults, err := searcher.SearchSkills(ctx, "test", nil, 5)
	require.NoError(t, err)
	assert.Empty(t, skillResults)
	// Only skill_missing_id should be deleted, not mem_xyz or random_456
	assert.Equal(t, []string{"skill_missing_id"}, deletedIDs)
}

func TestSearcher_CleanupStaleIndicesContextCancellation(t *testing.T) {
	kStore, rawDB := setupTestStore(t)
	defer func() { _ = rawDB.Close() }()

	session := SessionIdentity{
		ChatID: "c1",
		UserID: "u1",
	}

	deleteCalled := false
	mockV := &mockVectorDB{
		searchFn: func(ctx context.Context, req cortexdb.MemorySearchRequest) (*cortexdb.MemorySearchResponse, error) {
			return &cortexdb.MemorySearchResponse{
				Results: []cortexdb.MemorySearchHit{
					{Memory: cortexdb.MemoryRecord{ID: "mem_missing_id"}, Score: 0.9},
				},
			}, nil
		},
		deleteFn: func(ctx context.Context, req cortexdb.MemoryDeleteRequest) (*cortexdb.MemoryDeleteResponse, error) {
			deleteCalled = true
			return &cortexdb.MemoryDeleteResponse{Deleted: true}, nil
		},
	}

	searcher, err := NewSearcher(kStore, mockV, session, 5)
	require.NoError(t, err)

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel context before search

	searcher.cleanupStaleIndices(ctx, []string{"mem_missing_id"})
	assert.False(t, deleteCalled, "delete should not be called if parent context is canceled")
}

func TestSearcher_AdaptiveTopK(t *testing.T) {
	ctx := context.Background()
	kStore, rawDB := setupTestStore(t)
	defer func() { _ = rawDB.Close() }()

	session := SessionIdentity{
		ChatID: "c1",
		UserID: "u1",
	}

	// Create 2 valid memories
	_, m1, err := kStore.ProposeMemory(ctx, &KnowledgeItem{ChatID: session.ChatID, UserID: session.UserID}, &MemoryVersion{
		Type:    MemoryTypeFact,
		Content: "Valid memory one",
	})
	require.NoError(t, err)
	require.NoError(t, kStore.ApproveVersion(ctx, m1.ID, session.UserID))

	_, m2, err := kStore.ProposeMemory(ctx, &KnowledgeItem{ChatID: session.ChatID, UserID: session.UserID}, &MemoryVersion{
		Type:    MemoryTypeFact,
		Content: "Valid memory two",
	})
	require.NoError(t, err)
	require.NoError(t, kStore.ApproveVersion(ctx, m2.ID, session.UserID))

	var requestedTopKs []int
	mockV := &mockVectorDB{
		searchFn: func(ctx context.Context, req cortexdb.MemorySearchRequest) (*cortexdb.MemorySearchResponse, error) {
			requestedTopKs = append(requestedTopKs, req.TopK)
			if req.TopK == 10 {
				// Return 10 other-user memory IDs that will get filtered out
				var hits []cortexdb.MemorySearchHit
				for i := 0; i < 10; i++ {
					hits = append(hits, cortexdb.MemorySearchHit{
						Memory: cortexdb.MemoryRecord{ID: fmt.Sprintf("mem_other_%d", i)},
						Score:  0.9,
					})
				}
				return &cortexdb.MemorySearchResponse{Results: hits}, nil
			}
			// TopK expanded to 20: return the 10 other plus m1 and m2
			var hits []cortexdb.MemorySearchHit
			for i := 0; i < 10; i++ {
				hits = append(hits, cortexdb.MemorySearchHit{
					Memory: cortexdb.MemoryRecord{ID: fmt.Sprintf("mem_other_%d", i)},
					Score:  0.9,
				})
			}
			hits = append(hits,
				cortexdb.MemorySearchHit{Memory: cortexdb.MemoryRecord{ID: m1.ID}, Score: 0.85},
				cortexdb.MemorySearchHit{Memory: cortexdb.MemoryRecord{ID: m2.ID}, Score: 0.80},
			)
			return &cortexdb.MemorySearchResponse{Results: hits}, nil
		},
	}

	searcher, err := NewSearcher(kStore, mockV, session, 5)
	require.NoError(t, err)

	results, err := searcher.SearchMemories(ctx, "test", nil, 2)
	require.NoError(t, err)
	require.Len(t, results, 2)
	assert.Equal(t, m1.ID, results[0].VersionID)
	assert.Equal(t, m2.ID, results[1].VersionID)
	assert.Equal(t, []int{10, 20}, requestedTopKs)
}

func TestSearcher_SearchVectorJoinedErrors(t *testing.T) {
	ctx := context.Background()
	kStore, rawDB := setupTestStore(t)
	defer func() { _ = rawDB.Close() }()

	session := SessionIdentity{
		ChatID: "c1",
		UserID: "u1",
	}

	errVector := errors.New("vector engine failure")
	errLexical := errors.New("lexical engine failure")

	mockV := &mockVectorDB{
		searchFn: func(ctx context.Context, req cortexdb.MemorySearchRequest) (*cortexdb.MemorySearchResponse, error) {
			if req.RetrievalMode == cortexdb.RetrievalModeLexical {
				return nil, errLexical
			}
			return nil, errVector
		},
	}

	searcher, err := NewSearcher(kStore, mockV, session, 5)
	require.NoError(t, err)

	_, err = searcher.SearchMemories(ctx, "error test", nil, 5)
	require.Error(t, err)
	assert.ErrorIs(t, err, errVector)
	assert.ErrorIs(t, err, errLexical)
}

func TestSearcher_CleanupStaleIndicesSharedTimeout(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		session := SessionIdentity{
			ChatID: "c1",
			UserID: "u1",
		}

		var callCount atomic.Int32
		mockV := &mockVectorDB{
			deleteFn: func(ctx context.Context, req cortexdb.MemoryDeleteRequest) (*cortexdb.MemoryDeleteResponse, error) {
				callCount.Add(1)
				select {
				case <-time.After(600 * time.Millisecond):
					return &cortexdb.MemoryDeleteResponse{Deleted: true}, nil
				case <-ctx.Done():
					return nil, ctx.Err()
				}
			},
		}

		searcher, err := NewSearcher(&Store{}, mockV, session, 5)
		require.NoError(t, err)

		start := time.Now()
		// Pass 10 IDs. With 600ms per delete, 10 deletes would take 6s without shared 2s timeout.
		searcher.cleanupStaleIndices(context.Background(), []string{"m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "m10"})
		elapsed := time.Since(start)

		assert.GreaterOrEqual(t, elapsed, 2*time.Second, "simulated time should advance past 2 seconds")
		assert.Less(t, callCount.Load(), int32(6), "should have aborted before processing all items")
	})
}
