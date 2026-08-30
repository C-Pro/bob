package memory

import (
	"context"
	"database/sql"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"bob/internal/config"
	"bob/internal/llm"

	"github.com/liliang-cn/cortexdb/v2/pkg/cortexdb"
)

var nonAlphaNumRegex = regexp.MustCompile(`[^a-zA-Z0-9_\-]`)

// SanitizeChatID sanitizes a chat ID for safe use in file names.
func SanitizeChatID(chatID string) string {
	cleaned := nonAlphaNumRegex.ReplaceAllString(chatID, "_")
	if cleaned == "" {
		return "unknown"
	}
	return cleaned
}

// MemoryItem represents a retrieved conversation chunk with source and metadata.
type MemoryItem struct {
	ID        string  `json:"id"`
	Content   string  `json:"content"`
	Source    string  `json:"source"` // e.g. "[Townhall]" or "[Direct Message]"
	Score     float64 `json:"score"`
	StartSeq  int64   `json:"startSeq"`
	EndSeq    int64   `json:"endSeq"`
	StartTime int64   `json:"startTime"`
	EndTime   int64   `json:"endTime"`
	ChatID    string  `json:"chatId"`
}

// MessageToStore represents an individual message to be included in an indexed chunk.
type MessageToStore struct {
	Seq        int64
	Timestamp  int64
	ChatID     string
	UserID     string
	SenderName string
	Role       string
	Content    string
}

// Manager manages isolated SQLite sqvect databases per chat and watermark tracking.
type Manager struct {
	cfg      *config.Config
	embedder cortexdb.Embedder
	dbsMu    sync.RWMutex
	dbs      map[string]*cortexdb.DB
}

func isNilEmbedder(e cortexdb.Embedder) bool {
	if e == nil {
		return true
	}
	v := reflect.ValueOf(e)
	return v.Kind() == reflect.Pointer && v.IsNil()
}

// NewManager creates a new memory Store manager.
// If embedder is nil and cfg.EmbeddingModel is empty, a LocalEmbedder using go-embed is initialized.
// If cfg.EmbeddingModel is "none" or "disabled", embedder remains nil (pure FTS5 mode).
func NewManager(cfg *config.Config, embedder cortexdb.Embedder) *Manager {
	if isNilEmbedder(embedder) {
		embedder = nil
	}
	if embedder == nil && cfg != nil {
		modelSetting := strings.ToLower(strings.TrimSpace(cfg.EmbeddingModel))
		if modelSetting != "none" && modelSetting != "disabled" && modelSetting != "off" && modelSetting == "" {
			localEmb, err := llm.NewLocalEmbedder(cfg)
			if err != nil {
				slog.Warn("could not initialize local embedding engine, falling back to FTS5 lexical search", "error", err)
			} else {
				embedder = localEmb
			}
		}
	}
	return &Manager{
		cfg:      cfg,
		embedder: embedder,
		dbs:      make(map[string]*cortexdb.DB),
	}
}

func (m *Manager) dbKey(chatID string, isDM bool) (string, string) {
	if !isDM || chatID == "townhall" {
		return "townhall", filepath.Join(m.cfg.DataDir, "townhall.db")
	}
	sanitized := SanitizeChatID(chatID)
	return "dm_" + sanitized, filepath.Join(m.cfg.DataDir, fmt.Sprintf("dm_%s.db", sanitized))
}

// GetDB retrieves or opens the SQLite database for the specified chat context.
func (m *Manager) GetDB(ctx context.Context, chatID string, isDM bool) (*cortexdb.DB, error) {
	key, path := m.dbKey(chatID, isDM)

	m.dbsMu.RLock()
	db, ok := m.dbs[key]
	m.dbsMu.RUnlock()
	if ok {
		return db, nil
	}

	m.dbsMu.Lock()
	defer m.dbsMu.Unlock()

	// Double check
	if db, ok = m.dbs[key]; ok {
		return db, nil
	}

	if err := os.MkdirAll(m.cfg.DataDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create data dir: %w", err)
	}

	opts := make([]cortexdb.Option, 0, 1)
	if m.embedder != nil {
		opts = append(opts, cortexdb.WithEmbedder(m.embedder))
	}

	dbConfig := cortexdb.DefaultConfig(path)
	openedDB, err := cortexdb.Open(dbConfig, opts...)
	if err != nil {
		return nil, fmt.Errorf("failed to open database at %s: %w", path, err)
	}

	if err := m.initWatermarkTable(ctx, openedDB.SQL()); err != nil {
		_ = openedDB.Close()
		return nil, fmt.Errorf("failed to init watermark table at %s: %w", path, err)
	}

	m.dbs[key] = openedDB
	return openedDB, nil
}

func (m *Manager) initWatermarkTable(ctx context.Context, rawDB *sql.DB) error {
	schema := `
	CREATE TABLE IF NOT EXISTS watermark (
		id INTEGER PRIMARY KEY CHECK (id = 1),
		last_indexed_seq INTEGER NOT NULL DEFAULT 0,
		updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
	);
	INSERT OR IGNORE INTO watermark (id, last_indexed_seq) VALUES (1, 0);
	`
	_, err := rawDB.ExecContext(ctx, schema)
	return err
}

// GetWatermark returns the highest message sequence indexed for the given chat.
func (m *Manager) GetWatermark(ctx context.Context, chatID string, isDM bool) (int64, error) {
	db, err := m.GetDB(ctx, chatID, isDM)
	if err != nil {
		return 0, err
	}

	var seq int64
	row := db.SQL().QueryRowContext(ctx, `SELECT last_indexed_seq FROM watermark WHERE id = 1`)
	if err := row.Scan(&seq); err != nil {
		return 0, err
	}
	return seq, nil
}

// SetWatermark updates the watermark sequence for the given chat.
func (m *Manager) SetWatermark(ctx context.Context, chatID string, isDM bool, seq int64) error {
	db, err := m.GetDB(ctx, chatID, isDM)
	if err != nil {
		return err
	}

	_, err = db.SQL().ExecContext(ctx, `
		INSERT INTO watermark (id, last_indexed_seq, updated_at)
		VALUES (1, ?, CURRENT_TIMESTAMP)
		ON CONFLICT(id) DO UPDATE SET
			last_indexed_seq = excluded.last_indexed_seq,
			updated_at = CURRENT_TIMESTAMP
	`, seq)
	return err
}

const maxChunkSize = 33

// IndexMessages stores batches of messages as formatted conversation chunks into the chat's isolated database and updates the watermark.
func (m *Manager) IndexMessages(ctx context.Context, chatID string, isDM bool, msgs []MessageToStore) error {
	if len(msgs) == 0 {
		return nil
	}

	db, err := m.GetDB(ctx, chatID, isDM)
	if err != nil {
		return err
	}

	var maxSeq int64

	// Process messages in chunk batches
	for i := 0; i < len(msgs); i += maxChunkSize {
		end := i + maxChunkSize
		if end > len(msgs) {
			end = len(msgs)
		}
		chunkMsgs := msgs[i:end]

		startSeq := chunkMsgs[0].Seq
		endSeq := chunkMsgs[len(chunkMsgs)-1].Seq
		startTime := chunkMsgs[0].Timestamp
		endTime := chunkMsgs[len(chunkMsgs)-1].Timestamp

		var sb strings.Builder
		for idx, msg := range chunkMsgs {
			if msg.Seq > maxSeq {
				maxSeq = msg.Seq
			}
			if msg.Seq < startSeq {
				startSeq = msg.Seq
			}
			if msg.Seq > endSeq {
				endSeq = msg.Seq
			}
			if msg.Timestamp < startTime {
				startTime = msg.Timestamp
			}
			if msg.Timestamp > endTime {
				endTime = msg.Timestamp
			}

			if idx > 0 {
				sb.WriteString("\n")
			}
			timestampStr := time.Unix(msg.Timestamp, 0).UTC().Format("2006-01-02 15:04:05")
			name := msg.SenderName
			if name == "" {
				name = msg.UserID
			}
			if name == "" {
				name = "Unknown"
			}
			fmt.Fprintf(&sb, "[%s] %s: %s", timestampStr, name, msg.Content)
		}

		formattedContent := sb.String()
		chunkID := fmt.Sprintf("chunk_%s_%d_%d", SanitizeChatID(chatID), startSeq, endSeq)

		chatType := "townhall"
		if isDM {
			chatType = "dm"
		}

		metadata := map[string]any{
			"chat_id":    chatID,
			"chat_type":  chatType,
			"start_seq":  startSeq,
			"end_seq":    endSeq,
			"start_time": startTime,
			"end_time":   endTime,
			"msg_count":  len(chunkMsgs),
		}

		_, err := db.SaveMemory(ctx, cortexdb.MemorySaveRequest{
			MemoryID: chunkID,
			Role:     "user",
			Content:  formattedContent,
			Metadata: metadata,
		})
		if err != nil {
			return fmt.Errorf("failed to save memory chunk: %w", err)
		}
	}

	if maxSeq > 0 {
		curSeq, _ := m.GetWatermark(ctx, chatID, isDM)
		if maxSeq > curSeq {
			if err := m.SetWatermark(ctx, chatID, isDM, maxSeq); err != nil {
				slog.Warn("failed to update watermark", "chat_id", chatID, "seq", maxSeq, "error", err)
			}
		}
	}

	return nil
}

// Search performs hybrid or FTS5 search across the accessible databases.
// Townhall queries townhall.db only.
// DM queries both dm_<sanitized_chatID>.db and townhall.db, merging results.
func (m *Manager) Search(ctx context.Context, query string, chatID string, isDM bool, limit int) ([]MemoryItem, error) {
	if strings.TrimSpace(query) == "" {
		return nil, nil
	}
	if limit <= 0 {
		limit = 5
	}

	if !isDM || chatID == "townhall" {
		townhallDB, err := m.GetDB(ctx, "townhall", false)
		if err != nil {
			return nil, err
		}
		hits, err := m.searchDB(ctx, townhallDB, query, limit)
		if err != nil {
			return nil, err
		}
		return toMemoryItems(hits, "[Townhall]"), nil
	}

	// DM Context: Search private DM DB + public Townhall DB
	dmDB, err := m.GetDB(ctx, chatID, true)
	if err != nil {
		return nil, err
	}
	dmHits, err := m.searchDB(ctx, dmDB, query, limit)
	if err != nil {
		slog.Warn("search in DM db failed", "chat_id", chatID, "error", err)
	}

	townhallDB, err := m.GetDB(ctx, "townhall", false)
	var thHits []cortexdb.MemorySearchHit
	if err == nil {
		thHits, err = m.searchDB(ctx, townhallDB, query, limit)
		if err != nil {
			slog.Warn("search in Townhall db for DM context failed", "error", err)
		}
	}

	merged := make([]MemoryItem, 0, len(dmHits)+len(thHits))
	merged = append(merged, toMemoryItems(dmHits, "[Direct Message]")...)
	merged = append(merged, toMemoryItems(thHits, "[Townhall]")...)

	// Deduplicate by ID and sort by score descending
	seen := make(map[string]struct{}, len(merged))
	unique := make([]MemoryItem, 0, len(merged))
	for _, item := range merged {
		if _, exists := seen[item.ID]; exists {
			continue
		}
		seen[item.ID] = struct{}{}
		unique = append(unique, item)
	}

	sort.Slice(unique, func(i, j int) bool {
		return unique[i].Score > unique[j].Score
	})

	if len(unique) > limit {
		unique = unique[:limit]
	}

	return unique, nil
}

func (m *Manager) searchDB(ctx context.Context, db *cortexdb.DB, query string, limit int) ([]cortexdb.MemorySearchHit, error) {
	resp, err := db.SearchMemory(ctx, cortexdb.MemorySearchRequest{
		Query: query,
		TopK:  limit,
	})
	if err != nil {
		slog.Warn("memory search encountered error, falling back to lexical search", "query", query, "error", err)
		lexResp, lexErr := db.SearchMemory(ctx, cortexdb.MemorySearchRequest{
			Query:         query,
			RetrievalMode: cortexdb.RetrievalModeLexical,
			TopK:          limit,
		})
		if lexErr == nil && lexResp != nil {
			return lexResp.Results, nil
		}
		return nil, err
	}
	return resp.Results, nil
}

func toMemoryItems(hits []cortexdb.MemorySearchHit, sourceLabel string) []MemoryItem {
	items := make([]MemoryItem, len(hits))
	for i, hit := range hits {
		var startSeq, endSeq int64
		var startTime, endTime int64
		var chatID string

		if hit.Memory.Metadata != nil {
			startSeq = parseMetadataInt64(hit.Memory.Metadata["start_seq"])
			endSeq = parseMetadataInt64(hit.Memory.Metadata["end_seq"])
			startTime = parseMetadataInt64(hit.Memory.Metadata["start_time"])
			endTime = parseMetadataInt64(hit.Memory.Metadata["end_time"])

			// Backward compatibility if single message metadata was present
			if startSeq == 0 && endSeq == 0 {
				startSeq = parseMetadataInt64(hit.Memory.Metadata["seq"])
				endSeq = startSeq
			}
			if startTime == 0 && endTime == 0 {
				startTime = parseMetadataInt64(hit.Memory.Metadata["timestamp"])
				endTime = startTime
			}

			if cid, ok := hit.Memory.Metadata["chat_id"].(string); ok {
				chatID = cid
			}
		}

		items[i] = MemoryItem{
			ID:        hit.Memory.ID,
			Content:   hit.Memory.Content,
			Source:    sourceLabel,
			Score:     hit.Score,
			StartSeq:  startSeq,
			EndSeq:    endSeq,
			StartTime: startTime,
			EndTime:   endTime,
			ChatID:    chatID,
		}
	}
	return items
}

func parseMetadataInt64(val any) int64 {
	if val == nil {
		return 0
	}
	switch v := val.(type) {
	case float64:
		return int64(v)
	case int64:
		return v
	case int:
		return int64(v)
	case string:
		n, _ := strconv.ParseInt(v, 10, 64)
		return n
	default:
		return 0
	}
}

// Close closes all open database handles and embedder resources.
func (m *Manager) Close() error {
	m.dbsMu.Lock()
	defer m.dbsMu.Unlock()

	var firstErr error
	for key, db := range m.dbs {
		if err := db.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
		delete(m.dbs, key)
	}

	if closer, ok := m.embedder.(interface{ Close() error }); ok {
		if err := closer.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}

	return firstErr
}
