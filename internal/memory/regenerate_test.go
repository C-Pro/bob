package memory

import (
	"context"
	"database/sql"
	"testing"
	"time"

	"bob/internal/config"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	_ "modernc.org/sqlite"
)

func TestEncodeMemoryVector(t *testing.T) {
	t.Run("empty or nil vector", func(t *testing.T) {
		b, err := EncodeMemoryVector(nil)
		assert.NoError(t, err)
		assert.Nil(t, b)

		b, err = EncodeMemoryVector([]float32{})
		assert.NoError(t, err)
		assert.Nil(t, b)
	})

	t.Run("valid vector roundtrip", func(t *testing.T) {
		vec := []float32{1.0, 2.5, -3.25}
		b, err := EncodeMemoryVector(vec)
		require.NoError(t, err)
		require.Len(t, b, 4+4*len(vec))

		// First 4 bytes length
		length := int(b[0]) | int(b[1])<<8 | int(b[2])<<16 | int(b[3])<<24
		assert.Equal(t, 3, length)
	})
}

func TestRegenerateAllVectors(t *testing.T) {
	tempDir := t.TempDir()
	cfg := &config.Config{
		DataDir: tempDir,
	}

	ctx := context.Background()
	now := time.Now().Unix()

	// 1. Setup townhall and a DM database with initial 3-dim mock embedder
	emb3 := &mockEmbedder{dim: 3, model: "mock-3d"}
	mgr := NewManager(cfg, emb3)

	thMsgs := []MessageToStore{
		{
			Seq:        1,
			Timestamp:  now,
			ChatID:     "townhall",
			UserID:     "user1",
			SenderName: "Alice",
			Role:       "user",
			Content:    "First townhall memory chunk.",
		},
		{
			Seq:        2,
			Timestamp:  now + 1,
			ChatID:     "townhall",
			UserID:     "user2",
			SenderName: "Bob",
			Role:       "user",
			Content:    "Second townhall memory chunk.",
		},
	}
	err := mgr.IndexMessages(ctx, "townhall", false, thMsgs)
	require.NoError(t, err)

	dmMsgs := []MessageToStore{
		{
			Seq:        1,
			Timestamp:  now,
			ChatID:     "dm_user1",
			UserID:     "user1",
			SenderName: "Alice",
			Role:       "user",
			Content:    "Private DM memory chunk.",
		},
	}
	err = mgr.IndexMessages(ctx, "dm_user1", true, dmMsgs)
	require.NoError(t, err)
	require.NoError(t, mgr.Close())

	// Verify original vector length (4 header + 3*4 = 16 bytes) in townhall.db
	thDB, err := sql.Open("sqlite", cfg.DBPath("townhall.db"))
	require.NoError(t, err)
	var rawVec []byte
	err = thDB.QueryRowContext(ctx, `SELECT vector FROM messages WHERE id LIKE 'chunk_%' LIMIT 1`).Scan(&rawVec)
	require.NoError(t, err)
	assert.Len(t, rawVec, 16)
	_ = thDB.Close()

	// 2. Regenerate with a new 5-dim embedder (simulating model switch)
	callCount := 0
	emb5 := &mockEmbedder{
		dim:   5,
		model: "mock-5d",
		fn: func(text string) []float32 {
			callCount++
			return []float32{0.5, 0.4, 0.3, 0.2, 0.1}
		},
	}

	report, err := RegenerateAllVectors(ctx, cfg, emb5, 2)
	require.NoError(t, err)
	assert.Equal(t, 2, report.TotalDatabases) // townhall.db and dm_dm_user1.db
	assert.Equal(t, 2, report.TotalChunks)    // 1 chunk in townhall + 1 chunk in DM
	assert.Equal(t, 2, report.Reembedded)
	assert.Equal(t, 0, report.Failed)
	assert.Empty(t, report.Errors)

	// 3. Verify new vectors have 5-dim size (4 header + 5*4 = 24 bytes)
	thDB, err = sql.Open("sqlite", cfg.DBPath("townhall.db"))
	require.NoError(t, err)
	rows, err := thDB.QueryContext(ctx, `SELECT vector FROM messages WHERE id LIKE 'chunk_%'`)
	require.NoError(t, err)
	for rows.Next() {
		var v []byte
		err := rows.Scan(&v)
		require.NoError(t, err)
		assert.Len(t, v, 24, "vector should now be 24 bytes (5 dimensions)")
	}
	_ = rows.Close()
	_ = thDB.Close()
}
