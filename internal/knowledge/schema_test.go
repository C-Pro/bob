package knowledge

import (
	"context"
	"database/sql"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	_ "modernc.org/sqlite"
)

func TestEnsureKnowledgeSchema(t *testing.T) {
	ctx := context.Background()
	db, err := sql.Open("sqlite", ":memory:")
	require.NoError(t, err)
	defer func() { _ = db.Close() }()

	// First initialization
	err = EnsureKnowledgeSchema(ctx, db)
	require.NoError(t, err)

	// Idempotent re-execution
	err = EnsureKnowledgeSchema(ctx, db)
	require.NoError(t, err)

	// Verify tables exist
	var count int
	err = db.QueryRowContext(ctx, "SELECT count(*) FROM knowledge_items").Scan(&count)
	require.NoError(t, err)
	assert.Equal(t, 0, count)

	err = db.QueryRowContext(ctx, "SELECT count(*) FROM memory_versions").Scan(&count)
	require.NoError(t, err)
	assert.Equal(t, 0, count)

	err = db.QueryRowContext(ctx, "SELECT count(*) FROM skill_versions").Scan(&count)
	require.NoError(t, err)
	assert.Equal(t, 0, count)

	err = db.QueryRowContext(ctx, "SELECT count(*) FROM knowledge_tombstones").Scan(&count)
	require.NoError(t, err)
	assert.Equal(t, 0, count)

	// Nil DB check
	err = EnsureKnowledgeSchema(ctx, nil)
	assert.Error(t, err)
}
