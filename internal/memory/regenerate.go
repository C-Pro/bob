package memory

import (
	"context"
	"database/sql"
	"encoding/binary"
	"encoding/json"
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
	TotalDatabases        int      `json:"totalDatabases"`
	TotalChunks           int      `json:"totalChunks"`
	Reembedded            int      `json:"reembedded"`
	ReembeddedEpisodes    int      `json:"reembeddedEpisodes,omitempty"`
	ReembeddedMultimodal  int      `json:"reembeddedMultimodal,omitempty"`
	SkippedMultimodal     int      `json:"skippedMultimodal,omitempty"`
	QuarantinedMultimodal int      `json:"quarantinedMultimodal,omitempty"`
	Failed                int      `json:"failed"`
	Errors                []string `json:"errors,omitempty"`
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
	if batchSize <= 0 || batchSize > llm.SystemOneMaxBatchSize {
		batchSize = llm.SystemOneMaxBatchSize
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
	id                string
	content           string
	rawMeta           string
	hasImages         bool
	imageURIsComplete bool
	imageURIs         []string
	existingOK        bool
}

type multimodalEmbedder interface {
	EmbedMultimodal(ctx context.Context, text string, imageDataURIs []string) ([]float32, error)
}

type strictMultimodalEmbedder interface {
	EmbedMultimodalStrict(ctx context.Context, text string, imageDataURIs []string) ([]float32, error)
}

func regenerateDatabaseVectors(ctx context.Context, dbPath string, embedder cortexdb.Embedder, batchSize int, report *RegenerationReport) error {
	db, err := sql.Open("sqlite", dbPath)
	if err != nil {
		return fmt.Errorf("failed to open sqlite database: %w", err)
	}
	defer func() { _ = db.Close() }()

	prevSpace, hasPrevSpace, err := readVectorSpace(ctx, db)
	if err != nil {
		return fmt.Errorf("read vector space before regeneration: %w", err)
	}
	targetSpace := DescribeEmbedderSpace(embedder)
	sameSpace := hasPrevSpace &&
		prevSpace.EmbedderID == targetSpace.EmbedderID &&
		prevSpace.Dims > 0 &&
		(targetSpace.Dims == 0 || prevSpace.Dims == targetSpace.Dims)

	// Query all memory records (chunks, episodes, dm_memories, dm_skills).
	rows, err := db.QueryContext(ctx, `
		SELECT id, content, COALESCE(metadata, ''), vector IS NOT NULL FROM messages
		WHERE session_id LIKE 'memory:%'
		ORDER BY created_at ASC, id ASC
	`)
	if err != nil {
		return fmt.Errorf("failed to query memory rows: %w", err)
	}

	var records []memoryRecordToReembed
	for rows.Next() {
		var r memoryRecordToReembed
		if err := rows.Scan(&r.id, &r.content, &r.rawMeta, &r.existingOK); err != nil {
			_ = rows.Close()
			return fmt.Errorf("failed to scan memory row: %w", err)
		}
		r.hasImages, r.imageURIsComplete, r.imageURIs = parseRegenerateMultimodalMetadata(r.rawMeta)
		records = append(records, r)
	}
	if err := rows.Err(); err != nil {
		_ = rows.Close()
		return fmt.Errorf("rows iteration error: %w", err)
	}
	_ = rows.Close()

	if len(records) == 0 {
		if embedder.Dim() > 0 {
			if err := forceStampVectorSpace(ctx, db, embedder); err != nil {
				return fmt.Errorf("stamp empty vector space: %w", err)
			}
		}
		slog.Debug("no memory chunks found in database", "db", dbPath)
		return nil
	}

	chunkCount := 0
	var textRecords []memoryRecordToReembed
	var multimodalRecords []memoryRecordToReembed
	for _, r := range records {
		if strings.HasPrefix(r.id, "chunk_") {
			chunkCount++
		}
		if r.hasImages {
			multimodalRecords = append(multimodalRecords, r)
		} else {
			textRecords = append(textRecords, r)
		}
	}
	report.TotalChunks += chunkCount
	slog.Info("processing memory records for vector regeneration",
		"db", dbPath,
		"count", len(records),
		"chunks", chunkCount,
		"multimodal", len(multimodalRecords),
		"batchSize", batchSize,
	)

	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		report.Failed += chunkCount
		return fmt.Errorf("begin transaction: %w", err)
	}
	defer func() {
		_ = tx.Rollback()
	}()

	stmt, err := tx.PrepareContext(ctx, `UPDATE messages SET vector = ? WHERE id = ?`)
	if err != nil {
		report.Failed += chunkCount
		return fmt.Errorf("prepare update statement: %w", err)
	}
	defer func() {
		_ = stmt.Close()
	}()

	quarantineStmt, err := tx.PrepareContext(ctx, `UPDATE messages SET vector = NULL, metadata = ? WHERE id = ?`)
	if err != nil {
		report.Failed += chunkCount
		return fmt.Errorf("prepare quarantine statement: %w", err)
	}
	defer func() {
		_ = quarantineStmt.Close()
	}()

	dbReembedded := 0
	dbReembeddedEpisodes := 0
	dbReembeddedMultimodal := 0
	dbSkippedMultimodal := 0
	dbQuarantinedMultimodal := 0

	strictMM, hasStrictMM := embedder.(strictMultimodalEmbedder)
	mmEmbedder, hasMultimodal := embedder.(multimodalEmbedder)
	for _, r := range multimodalRecords {
		if ctx.Err() != nil {
			report.Failed += chunkCount - dbReembedded
			return ctx.Err()
		}
		if r.imageURIsComplete && len(r.imageURIs) > 0 && (hasStrictMM || hasMultimodal) {
			var (
				vec   []float32
				mmErr error
			)
			if hasStrictMM {
				vec, mmErr = strictMM.EmbedMultimodalStrict(ctx, r.content, r.imageURIs)
			} else {
				vec, mmErr = mmEmbedder.EmbedMultimodal(ctx, r.content, r.imageURIs)
			}
			if mmErr != nil {
				report.Failed += chunkCount - dbReembedded
				errMsg := fmt.Sprintf("%s: EmbedMultimodal failed for %s: %v", filepath.Base(dbPath), r.id, mmErr)
				report.Errors = append(report.Errors, errMsg)
				return fmt.Errorf("multimodal embedding failed: %w", mmErr)
			}
			vecBytes, encErr := EncodeMemoryVector(vec)
			if encErr != nil {
				report.Failed += chunkCount - dbReembedded
				return fmt.Errorf("encode multimodal vector failed: %w", encErr)
			}
			if _, err := stmt.ExecContext(ctx, vecBytes, r.id); err != nil {
				report.Failed += chunkCount - dbReembedded
				return fmt.Errorf("update multimodal vector failed: %w", err)
			}
			dbReembeddedMultimodal++
			if strings.HasPrefix(r.id, "chunk_") {
				dbReembedded++
			}
			continue
		}

		// Preserve existing multimodal vectors ONLY when existing vector provenance already
		// matches the target space; otherwise quarantine vector to NULL rather than stamping
		// mixed dimensions/spaces or overwriting with text-only embeddings.
		if sameSpace && r.existingOK {
			slog.Warn("preserving existing same-space multimodal vector during regeneration because complete inline image_uris are unavailable",
				"db", dbPath,
				"record_id", r.id,
			)
			dbSkippedMultimodal++
			continue
		}

		quarantinedMeta := markMetadataQuarantined(r.rawMeta)
		if _, err := quarantineStmt.ExecContext(ctx, quarantinedMeta, r.id); err != nil {
			report.Failed += chunkCount - dbReembedded
			return fmt.Errorf("quarantine incompatible multimodal vector %s: %w", r.id, err)
		}
		slog.Warn("quarantined incompatible/unreconstructable multimodal vector during vector space migration",
			"db", dbPath,
			"record_id", r.id,
		)
		dbQuarantinedMultimodal++
	}

	for start := 0; start < len(textRecords); start += batchSize {
		if ctx.Err() != nil {
			report.Failed += chunkCount - dbReembedded
			return ctx.Err()
		}

		end := start + batchSize
		if end > len(textRecords) {
			end = len(textRecords)
		}
		batch := textRecords[start:end]

		texts := make([]string, len(batch))
		for i, r := range batch {
			texts[i] = r.content
		}

		vectors, err := embedder.EmbedBatch(ctx, texts)
		if err != nil {
			report.Failed += chunkCount - dbReembedded
			errMsg := fmt.Sprintf("%s: EmbedBatch failed at offset %d: %v", filepath.Base(dbPath), start, err)
			slog.Error("batch embedding generation failed", "error", err, "db", dbPath, "offset", start)
			report.Errors = append(report.Errors, errMsg)
			return fmt.Errorf("batch embedding failed: %w", err)
		}

		if len(vectors) != len(batch) {
			report.Failed += chunkCount - dbReembedded
			errMsg := fmt.Sprintf("%s: EmbedBatch returned %d vectors for %d texts at offset %d", filepath.Base(dbPath), len(vectors), len(batch), start)
			slog.Error("inconsistent vector count", "expected", len(batch), "got", len(vectors), "db", dbPath)
			report.Errors = append(report.Errors, errMsg)
			return errors.New("inconsistent vector count returned from embedder")
		}

		for i, r := range batch {
			vecBytes, err := EncodeMemoryVector(vectors[i])
			if err != nil {
				report.Failed += chunkCount - dbReembedded
				errMsg := fmt.Sprintf("%s: encode vector for %s: %v", filepath.Base(dbPath), r.id, err)
				report.Errors = append(report.Errors, errMsg)
				return fmt.Errorf("encode vector failed: %w", err)
			}

			if _, err := stmt.ExecContext(ctx, vecBytes, r.id); err != nil {
				report.Failed += chunkCount - dbReembedded
				errMsg := fmt.Sprintf("%s: update vector for %s: %v", filepath.Base(dbPath), r.id, err)
				report.Errors = append(report.Errors, errMsg)
				return fmt.Errorf("update vector failed: %w", err)
			}
			if strings.HasPrefix(r.id, "chunk_") {
				dbReembedded++
			} else if strings.HasPrefix(r.id, EpisodeIDPrefix) {
				dbReembeddedEpisodes++
			}
		}
	}

	if err := forceStampVectorSpace(ctx, tx, embedder); err != nil {
		report.Failed += chunkCount - dbReembedded
		return fmt.Errorf("stamp regenerated vector space: %w", err)
	}

	if err := tx.Commit(); err != nil {
		report.Failed += chunkCount
		return fmt.Errorf("commit transaction: %w", err)
	}

	report.Reembedded += dbReembedded
	report.ReembeddedEpisodes += dbReembeddedEpisodes
	report.ReembeddedMultimodal += dbReembeddedMultimodal
	report.SkippedMultimodal += dbSkippedMultimodal
	report.QuarantinedMultimodal += dbQuarantinedMultimodal
	return nil
}

func parseRegenerateMultimodalMetadata(rawMeta string) (hasImages bool, complete bool, imageURIs []string) {
	trimmed := strings.TrimSpace(rawMeta)
	if trimmed == "" || trimmed == "{}" {
		return false, false, nil
	}
	var meta struct {
		HasImages         bool     `json:"has_images"`
		ImageCount        int      `json:"image_count"`
		ImageURIsComplete bool     `json:"image_uris_complete"`
		ImageURIs         []string `json:"image_uris"`
	}
	if err := json.Unmarshal([]byte(trimmed), &meta); err != nil {
		return false, false, nil
	}
	var validURIs []string
	for _, u := range meta.ImageURIs {
		if llm.IsValidSystemOneImageDataURI(u) {
			validURIs = append(validURIs, strings.TrimSpace(u))
		}
	}
	hasImgs := meta.HasImages || len(validURIs) > 0
	isComplete := meta.ImageURIsComplete && len(validURIs) > 0 && (meta.ImageCount == 0 || len(validURIs) == meta.ImageCount)
	return hasImgs, isComplete, validURIs
}

func markMetadataQuarantined(rawMeta string) string {
	meta := make(map[string]any)
	trimmed := strings.TrimSpace(rawMeta)
	if trimmed != "" {
		_ = json.Unmarshal([]byte(trimmed), &meta)
	}
	meta["vector_quarantined"] = true
	b, err := json.Marshal(meta)
	if err != nil {
		return `{"vector_quarantined":true}`
	}
	return string(b)
}

// RegenerateAllVectors regenerates all memory chunk vectors across all managed chat databases.
func (m *Manager) RegenerateAllVectors(ctx context.Context, batchSize int) (*RegenerationReport, error) {
	return RegenerateAllVectors(ctx, m.cfg, m.embedder, batchSize)
}
