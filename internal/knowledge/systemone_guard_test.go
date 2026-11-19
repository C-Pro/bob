package knowledge

import (
	"context"
	"errors"
	"testing"

	"github.com/liliang-cn/cortexdb/v2/pkg/cortexdb"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSearcher_VectorSpaceGuard(t *testing.T) {
	ctx := context.Background()
	kStore, db := setupTestStore(t)
	defer func() { _ = db.Close() }()

	var seenModes []string
	vecDB := &mockVectorDB{
		searchFn: func(ctx context.Context, req cortexdb.MemorySearchRequest) (*cortexdb.MemorySearchResponse, error) {
			seenModes = append(seenModes, req.RetrievalMode)
			if req.RetrievalMode == "" {
				t.Fatal("semantic auto search must not run when vector space guard fails")
			}
			return &cortexdb.MemorySearchResponse{}, nil
		},
	}

	searcher, err := NewSearcher(kStore, vecDB, SessionIdentity{ChatID: "c1", UserID: "u1"}, 5)
	require.NoError(t, err)

	mismatch := errors.New("vector space mismatch")

	// Permissive mode: gates to RetrievalModeLexical.
	searcher.SetVectorSpaceGuard(func(ctx context.Context) error { return mismatch }, true)
	_, err = searcher.SearchMemories(ctx, "test", nil, 5)
	require.NoError(t, err)
	assert.Equal(t, []string{cortexdb.RetrievalModeLexical}, seenModes)

	// Strict mode: returns the mismatch error directly.
	seenModes = nil
	searcher.SetVectorSpaceGuard(func(ctx context.Context) error { return mismatch }, false)
	_, err = searcher.SearchMemories(ctx, "test", nil, 5)
	require.ErrorIs(t, err, mismatch)
	assert.Empty(t, seenModes)
}
