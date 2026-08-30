package memory

import (
	"context"
	"database/sql"
	"encoding/binary"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"os"
	"path/filepath"
	"strings"

	"bob/internal/config"
	"bob/internal/llm"

	"github.com/liliang-cn/cortexdb/v2/pkg/cortexdb"
	_ "modernc.org/sqlite"
)

// RegenerationReport summarizes the result of a vector regeneration pass.
type RegenerationReport struct {
	TotalDatabases int      `json:"totalDatabases"`
	TotalChunks    int      `json:"totalChunks"`
	Reembedded     int      `json:"reembedded"`
	Failed         int      `json:"failed"`
	Errors         []string `json:"errors,omitempty"`
}

// EncodeMemoryVector encodes a float32 vector into the 4-byte length prefixed little-endian
// format stored in cortexdb messages.vector.
func EncodeMemoryVector(vec []float32) ([]byte, error) {
	if len(vec) == 0 {
		return nil, nil
	}
	buf := make([]byte, 4+4*len(vec))
	binary.LittleEndian.PutUint32(buf[0:4], uint32(len(vec)))
	for i, v := range vec {
		binary.LittleEndian.PutUint32(buf[4+i*4:8+i*4], math.Float32bits(v))
	}
	return buf, nil
}

// RegenerateAllVectors scans DATA_DIR for townhall.db and all dm_*.db databases,
// fetches all stored memory records, generates vector embeddings using EmbedBatch,
// and updates the stored vectors.
func RegenerateAllVectors(ctx context.Context, cfg *config.Config, embedder cortexdb.Embedder, batchSize int) (*RegenerationReport, error) {
	if cfg == nil {
		return nil, errors.New("config is nil")
	}
	if embedder == nil {
		// Attempt to initialize local embedder as fallback
		localEmb, err := llm.NewLocalEmbedder(cfg)
		if err != nil {
			return nil, fmt.Errorf("no embedder provided and failed to initialize local embedder: %w", err)
		}
		embedder = localEmb
	}
	if batchSize <= 0 {
		batchSize = 16
	}

	report := &RegenerationReport{
		Errors: make([]string, 0),
	}

	entries, err := os.ReadDir(cfg.DataDir)
	if err != nil {
		return nil, fmt.Errorf("failed to read data directory %s: %w", cfg.DataDir, err)
	}

	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		if name == "townhall.db" || (strings.HasPrefix(name, "dm_") && strings.HasSuffix(name, ".db")) {
			dbPath := filepath.Join(cfg.DataDir, name)
			slog.Info("regenerating vectors for chat database", "db", name, "path", dbPath)
			if err := regenerateDatabaseVectors(ctx, dbPath, embedder, batchSize, report); err != nil {
				report.Errors = append(report.Errors, fmt.Sprintf("database %s: %v", name, err))
			}
			report.TotalDatabases++
		}
	}

	return report, nil
}

type memoryRecordToReembed struct {
	id      string
	content string
}

func regenerateDatabaseVectors(ctx context.Context, dbPath string, embedder cortexdb.Embedder, batchSize int, report *RegenerationReport) error {
	db, err := sql.Open("sqlite", dbPath)
	if err != nil {
		return fmt.Errorf("failed to open sqlite database: %w", err)
	}
	defer func() { _ = db.Close() }()

	// Query all memory chunks
	rows, err := db.QueryContext(ctx, `
		SELECT id, content FROM messages
		WHERE session_id LIKE 'memory:%'
		ORDER BY created_at ASC, id ASC
	`)
	if err != nil {
		return fmt.Errorf("failed to query memory rows: %w", err)
	}

	var records []memoryRecordToReembed
	for rows.Next() {
		var r memoryRecordToReembed
		if err := rows.Scan(&r.id, &r.content); err != nil {
			_ = rows.Close()
			return fmt.Errorf("failed to scan memory row: %w", err)
		}
		records = append(records, r)
	}
	if err := rows.Err(); err != nil {
		_ = rows.Close()
		return fmt.Errorf("rows iteration error: %w", err)
	}
	_ = rows.Close()

	if len(records) == 0 {
		slog.Debug("no memory chunks found in database", "db", dbPath)
		return nil
	}

	report.TotalChunks += len(records)
	slog.Info("processing memory chunks for vector regeneration", "db", dbPath, "count", len(records), "batchSize", batchSize)

	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		report.Failed += len(records)
		return fmt.Errorf("begin transaction: %w", err)
	}
	defer func() {
		_ = tx.Rollback()
	}()

	stmt, err := tx.PrepareContext(ctx, `UPDATE messages SET vector = ? WHERE id = ?`)
	if err != nil {
		report.Failed += len(records)
		return fmt.Errorf("prepare update statement: %w", err)
	}
	defer func() {
		_ = stmt.Close()
	}()

	dbReembedded := 0
	for start := 0; start < len(records); start += batchSize {
		if ctx.Err() != nil {
			report.Failed += len(records) - dbReembedded
			return ctx.Err()
		}

		end := start + batchSize
		if end > len(records) {
			end = len(records)
		}
		batch := records[start:end]

		texts := make([]string, len(batch))
		for i, r := range batch {
			texts[i] = r.content
		}

		vectors, err := embedder.EmbedBatch(ctx, texts)
		if err != nil {
			report.Failed += len(records) - dbReembedded
			errMsg := fmt.Sprintf("%s: EmbedBatch failed at offset %d: %v", filepath.Base(dbPath), start, err)
			slog.Error("batch embedding generation failed", "error", err, "db", dbPath, "offset", start)
			report.Errors = append(report.Errors, errMsg)
			return fmt.Errorf("batch embedding failed: %w", err)
		}

		if len(vectors) != len(batch) {
			report.Failed += len(records) - dbReembedded
			errMsg := fmt.Sprintf("%s: EmbedBatch returned %d vectors for %d texts at offset %d", filepath.Base(dbPath), len(vectors), len(batch), start)
			slog.Error("inconsistent vector count", "expected", len(batch), "got", len(vectors), "db", dbPath)
			report.Errors = append(report.Errors, errMsg)
			return errors.New("inconsistent vector count returned from embedder")
		}

		for i, r := range batch {
			vecBytes, err := EncodeMemoryVector(vectors[i])
			if err != nil {
				report.Failed += len(records) - dbReembedded
				errMsg := fmt.Sprintf("%s: encode vector for %s: %v", filepath.Base(dbPath), r.id, err)
				report.Errors = append(report.Errors, errMsg)
				return fmt.Errorf("encode vector failed: %w", err)
			}

			if _, err := stmt.ExecContext(ctx, vecBytes, r.id); err != nil {
				report.Failed += len(records) - dbReembedded
				errMsg := fmt.Sprintf("%s: update vector for %s: %v", filepath.Base(dbPath), r.id, err)
				report.Errors = append(report.Errors, errMsg)
				return fmt.Errorf("update vector failed: %w", err)
			}
			dbReembedded++
		}
	}

	if err := tx.Commit(); err != nil {
		report.Failed += len(records)
		return fmt.Errorf("commit transaction: %w", err)
	}

	report.Reembedded += dbReembedded
	return nil
}

// RegenerateAllVectors regenerates all memory chunk vectors across all managed chat databases.
func (m *Manager) RegenerateAllVectors(ctx context.Context, batchSize int) (*RegenerationReport, error) {
	return RegenerateAllVectors(ctx, m.cfg, m.embedder, batchSize)
}
