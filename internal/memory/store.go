package memory

import (
	"context"
	"database/sql"
	"errors"
	"flag"
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

// MemoryItem represents a retrieved conversation chunk or episode summary with source and metadata.
type MemoryItem struct {
	ID             string   `json:"id"`
	Kind           string   `json:"kind,omitempty"`
	Content        string   `json:"content"`
	Source         string   `json:"source"` // e.g. "[Townhall]" or "[Direct Message]"
	Score          float64  `json:"score"`
	StartSeq       int64    `json:"startSeq"`
	EndSeq         int64    `json:"endSeq"`
	StartTime      int64    `json:"startTime"`
	EndTime        int64    `json:"endTime"`
	ChatID         string   `json:"chatId"`
	EpisodeID      string   `json:"episodeId,omitempty"`
	SourceChunkIDs []string `json:"sourceChunkIds,omitempty"`
	Participants   []string `json:"participants,omitempty"`
	TopicTags      []string `json:"topicTags,omitempty"`
	TaskStatus     string   `json:"taskStatus,omitempty"`
	ModelDerived   bool     `json:"modelDerived,omitempty"`
}

const (
	maxChunkSize              = 33
	maxMetadataImageURIsBytes = 256 * 1024
)

// EpisodeClassifier abstracts System-1 episode task-status classification.
type EpisodeClassifier interface {
	DecideCustom(ctx context.Context, id, question string, options []llm.DecisionOptionSpec, state string, images []string) (*llm.DecisionResultEntry, *llm.DecisionResponse, error)
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
	Images     []string
}

// Manager manages isolated SQLite sqvect databases per chat and watermark tracking.
type Manager struct {
	cfg               *config.Config
	embedder          cortexdb.Embedder
	dbsMu             sync.RWMutex
	dbs               map[string]*cortexdb.DB
	chatLocksMu       sync.Mutex
	chatLocks         map[chatLockKey]*chatIndexLock
	s1Mu              sync.RWMutex
	episodeClassifier EpisodeClassifier
	episodesEnabled   bool
	fallbackOnError   bool
}

type chatLockKey struct {
	chatID string
	isDM   bool
}

type chatIndexLock struct {
	held chan struct{}
	refs int
}

func (m *Manager) acquireChatLock(ctx context.Context, chatID string, isDM bool) (func(), error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	key := chatLockKey{chatID: chatID, isDM: isDM}
	m.chatLocksMu.Lock()
	if m.chatLocks == nil {
		m.chatLocks = make(map[chatLockKey]*chatIndexLock)
	}
	lock := m.chatLocks[key]
	if lock == nil {
		lock = &chatIndexLock{held: make(chan struct{}, 1)}
		m.chatLocks[key] = lock
	}
	lock.refs++
	m.chatLocksMu.Unlock()

	releaseRef := func() {
		m.chatLocksMu.Lock()
		lock.refs--
		if lock.refs == 0 {
			delete(m.chatLocks, key)
		}
		m.chatLocksMu.Unlock()
	}
	select {
	case lock.held <- struct{}{}:
		if err := ctx.Err(); err != nil {
			<-lock.held
			releaseRef()
			return nil, err
		}
		return func() {
			<-lock.held
			releaseRef()
		}, nil
	case <-ctx.Done():
		releaseRef()
		return nil, ctx.Err()
	}
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
	fallback := true
	if cfg != nil && cfg.SystemOneEnabled() {
		fallback = cfg.SystemOneFallbackOnError
	}
	if embedder == nil && cfg != nil {
		modelSetting := strings.ToLower(strings.TrimSpace(cfg.EmbeddingModel))
		if modelSetting == "" {
			if flag.Lookup("test.v") != nil && os.Getenv("BOB_MODELS_DIR") == "" && !hasLocalModelInDir(cfg.DataDir) {
				return &Manager{
					cfg:             cfg,
					embedder:        nil,
					dbs:             make(map[string]*cortexdb.DB),
					fallbackOnError: fallback,
				}
			}
			embedder = llm.NewLazyLocalEmbedder(cfg)
		}
	}
	return &Manager{
		cfg:             cfg,
		embedder:        embedder,
		dbs:             make(map[string]*cortexdb.DB),
		fallbackOnError: fallback,
	}
}

// SetSystemOneConfig configures System-1 episode classification and strict fallback policy on the manager.
func (m *Manager) SetSystemOneConfig(classifier EpisodeClassifier, episodesEnabled, fallbackOnError bool) {
	if m == nil {
		return
	}
	m.s1Mu.Lock()
	defer m.s1Mu.Unlock()
	m.episodeClassifier = classifier
	m.episodesEnabled = episodesEnabled
	m.fallbackOnError = fallbackOnError
}

// SetFallbackOnError overrides the fallback-on-error policy used for strict evaluation mode.
func (m *Manager) SetFallbackOnError(fallback bool) {
	if m == nil {
		return
	}
	m.s1Mu.Lock()
	defer m.s1Mu.Unlock()
	m.fallbackOnError = fallback
}

// FallbackOnError returns whether non-fatal degradation to lexical/heuristic fallbacks is permitted.
func (m *Manager) FallbackOnError() bool {
	if m == nil {
		return true
	}
	m.s1Mu.RLock()
	defer m.s1Mu.RUnlock()
	return m.fallbackOnError
}

// PrepareVectorSearch checks vector-space compatibility for db and prepares a context + post-search
// verifier so strict mode (!FallbackOnError) never silently degrades inside CortexDB when embedding fails.
func (m *Manager) PrepareVectorSearch(ctx context.Context, db *cortexdb.DB) (searchCtx context.Context, postCheck func() error, lexicalOnly bool, err error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if m == nil || db == nil || isNilEmbedder(m.embedder) {
		return ctx, nil, false, nil
	}
	expectedDims, compatErr := checkVectorSpaceCompatibilityWithDims(ctx, db.SQL(), m.embedder)
	if compatErr != nil {
		if !m.FallbackOnError() {
			return nil, nil, false, compatErr
		}
		slog.Warn("vector space mismatch in database, gating semantic search to lexical FTS5", "error", compatErr)
		return ctx, nil, true, nil
	}
	strict := !m.FallbackOnError()
	sCtx, sink := withEmbedContext(ctx, expectedDims, strict)
	if !strict || sink == nil {
		return sCtx, nil, false, nil
	}
	return sCtx, sink.get, false, nil
}

// SaveMemoryWithGuard validates vector space compatibility, enforces runtime dimensionality via context,
// verifies non-null vector storage under strict mode, and establishes the DB vector-space stamp.
func (m *Manager) SaveMemoryWithGuard(ctx context.Context, db *cortexdb.DB, req cortexdb.MemorySaveRequest) error {
	if db == nil {
		return fmt.Errorf("database is nil")
	}
	if m == nil || isNilEmbedder(m.embedder) {
		_, err := db.SaveMemory(ctx, req)
		return err
	}

	expectedDims, compatErr := checkVectorSpaceCompatibilityWithDims(ctx, db.SQL(), m.embedder)
	if compatErr != nil {
		return compatErr
	}

	saveCtx, sink := withEmbedContext(ctx, expectedDims, true)
	_, err := db.SaveMemory(saveCtx, req)
	if err != nil {
		return err
	}

	if sinkErr := sink.get(); sinkErr != nil {
		var mismatch *VectorSpaceMismatchError
		if errors.As(sinkErr, &mismatch) || !m.FallbackOnError() {
			if _, delErr := db.DeleteMemory(context.WithoutCancel(ctx), cortexdb.MemoryDeleteRequest{
				MemoryID: req.MemoryID,
			}); delErr != nil {
				slog.Warn("failed to roll back guarded memory row after embed error", "memory_id", req.MemoryID, "error", delErr)
			}
			return sinkErr
		}
	}

	if !m.FallbackOnError() {
		if err := verifyStoredVectorNonNull(ctx, db.SQL(), req.MemoryID); err != nil {
			if _, delErr := db.DeleteMemory(context.WithoutCancel(ctx), cortexdb.MemoryDeleteRequest{
				MemoryID: req.MemoryID,
			}); delErr != nil {
				slog.Warn("failed to roll back guarded memory row with null vector", "memory_id", req.MemoryID, "error", delErr)
			}
			return err
		}
	}

	if m.embedder.Dim() > 0 {
		if err := StampVectorSpace(ctx, db.SQL(), m.embedder); err != nil {
			return err
		}
	}
	return nil
}

// HasLocalModelInDir reports whether the default local model weights exist in dataDir.
func HasLocalModelInDir(dataDir string) bool {
	return hasLocalModelInDir(dataDir)
}

func hasLocalModelInDir(dataDir string) bool {
	if dataDir == "" {
		return false
	}
	targetSafetensors := filepath.Join(dataDir, "models", llm.DefaultLocalModel, "model.safetensors")
	fi, err := os.Stat(targetSafetensors)
	return err == nil && fi.Size() > 0
}

// Embedder returns the active Embedder instance, or nil if running in lexical-only FTS5 mode.
func (m *Manager) Embedder() cortexdb.Embedder {
	return m.embedder
}

func (m *Manager) dbKey(chatID string, isDM bool) (string, string) {
	if !isDM || chatID == "townhall" {
		return "townhall", filepath.Join(m.cfg.DataDir, "townhall.db")
	}
	sanitized := SanitizeChatID(chatID)
	return "dm_" + sanitized, filepath.Join(m.cfg.DataDir, fmt.Sprintf("dm_%s.db", sanitized))
}

// ActiveDBs returns a map of database filenames to their active *sql.DB connection handles.
func (m *Manager) ActiveDBs() map[string]*sql.DB {
	m.dbsMu.RLock()
	defer m.dbsMu.RUnlock()

	active := make(map[string]*sql.DB, len(m.dbs))
	for key, db := range m.dbs {
		filename := key + ".db"
		active[filename] = db.SQL()
	}
	return active
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

	// Pre-initialize new chat databases with auto_vacuum=INCREMENTAL and WAL
	// before cortexdb creates its internal tables, so space reclamation works.
	if stat, err := os.Stat(path); os.IsNotExist(err) || (err == nil && stat.Size() == 0) {
		initDSN := fmt.Sprintf("file:%s?_auto_vacuum=INCREMENTAL&_pragma=journal_mode(WAL)&_pragma=foreign_keys(ON)&_pragma=busy_timeout(5000)", path)
		if preDB, err := sql.Open("sqlite", initDSN); err == nil {
			_ = preDB.Ping()
			_ = preDB.Close()
		}
	}

	opts := make([]cortexdb.Option, 0, 1)
	if m.embedder != nil {
		opts = append(opts, cortexdb.WithEmbedder(wrapEmbedder(m.embedder)))
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

	if err := m.initEpisodeTable(ctx, openedDB.SQL()); err != nil {
		_ = openedDB.Close()
		return nil, fmt.Errorf("failed to init episode table at %s: %w", path, err)
	}

	if err := initVectorSpaceTable(ctx, openedDB.SQL()); err != nil {
		_ = openedDB.Close()
		return nil, fmt.Errorf("failed to init vector space table at %s: %w", path, err)
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
			last_indexed_seq = MAX(excluded.last_indexed_seq, watermark.last_indexed_seq),
			updated_at = CASE WHEN excluded.last_indexed_seq >= watermark.last_indexed_seq
				THEN excluded.updated_at ELSE watermark.updated_at END
	`, seq)
	return err
}

// IndexMessages stores batches of messages as formatted conversation chunks into the chat's isolated database and updates the watermark.
func (m *Manager) IndexMessages(ctx context.Context, chatID string, isDM bool, msgs []MessageToStore) error {
	release, err := m.acquireChatLock(ctx, chatID, isDM)
	if err != nil {
		return err
	}
	defer release()

	if len(msgs) == 0 {
		return nil
	}

	db, err := m.GetDB(ctx, chatID, isDM)
	if err != nil {
		return err
	}
	if compatErr := CheckVectorSpaceCompatibility(ctx, db.SQL(), m.embedder); compatErr != nil {
		return fmt.Errorf("cannot index messages into %s: %w", chatID, compatErr)
	}

	var maxSeq int64
	chunkIDs := make([]string, 0, (len(msgs)+maxChunkSize-1)/maxChunkSize)

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
		chunkID := fmt.Sprintf("chunk_%s_%d_%d", uniqueChatIDSlug(chatID), startSeq, endSeq)
		chunkIDs = append(chunkIDs, chunkID)

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

		chunkImages := collectChunkImages(chunkMsgs, llm.SystemOneMaxImagesPerItem)
		if len(chunkImages) > 0 {
			metadata["has_images"] = true
			metadata["image_count"] = len(chunkImages)
			boundedURIs := boundMetadataImageURIs(chunkImages, maxMetadataImageURIsBytes)
			metadata["image_uris_complete"] = len(boundedURIs) == len(chunkImages)
			if len(boundedURIs) > 0 {
				metadata["image_uris"] = boundedURIs
			}
		}

		saveCtx := ctx
		if len(chunkImages) > 0 {
			saveCtx = llm.WithEmbeddingImages(ctx, chunkImages)
		}

		if err := m.SaveMemoryWithGuard(saveCtx, db, cortexdb.MemorySaveRequest{
			MemoryID: chunkID,
			Role:     "user",
			Content:  formattedContent,
			Metadata: metadata,
		}); err != nil {
			return fmt.Errorf("failed to save memory chunk: %w", err)
		}
	}

	if len(chunkIDs) > 0 {
		ep, err := m.buildEpisodeFromMessages(ctx, chatID, isDM, msgs, chunkIDs)
		if err != nil {
			return fmt.Errorf("failed to build episode for chat %s: %w", chatID, err)
		}
		if err := m.saveEpisode(ctx, db, ep); err != nil {
			return fmt.Errorf("failed to save episode for chat %s: %w", chatID, err)
		}
	}

	if maxSeq > 0 {
		curSeq, err := m.GetWatermark(ctx, chatID, isDM)
		if err != nil {
			return fmt.Errorf("failed to read watermark for chat %s: %w", chatID, err)
		}
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
		items, err := m.enrichChunksAndMergeEpisodes(ctx, townhallDB, "townhall", "[Townhall]", query, toMemoryItems(hits, "[Townhall]"), limit)
		if err != nil {
			return nil, err
		}
		if len(items) > limit {
			items = items[:limit]
		}
		return items, nil
	}

	// DM Context: Search private DM DB + public Townhall DB
	dmDB, err := m.GetDB(ctx, chatID, true)
	if err != nil {
		return nil, err
	}
	dmHits, err := m.searchDB(ctx, dmDB, query, limit)
	if err != nil {
		if !m.FallbackOnError() {
			return nil, err
		}
		slog.Warn("search in DM db failed", "chat_id", chatID, "error", err)
	}
	dmItems, err := m.enrichChunksAndMergeEpisodes(ctx, dmDB, chatID, "[Direct Message]", query, toMemoryItems(dmHits, "[Direct Message]"), limit)
	if err != nil {
		return nil, err
	}

	townhallDB, err := m.GetDB(ctx, "townhall", false)
	var thItems []MemoryItem
	if err == nil {
		thHits, thErr := m.searchDB(ctx, townhallDB, query, limit)
		if thErr != nil {
			if !m.FallbackOnError() {
				return nil, thErr
			}
			slog.Warn("search in Townhall db for DM context failed", "error", thErr)
		}
		var epErr error
		thItems, epErr = m.enrichChunksAndMergeEpisodes(ctx, townhallDB, "townhall", "[Townhall]", query, toMemoryItems(thHits, "[Townhall]"), limit)
		if epErr != nil {
			return nil, epErr
		}
	} else if !m.FallbackOnError() {
		return nil, err
	}

	merged := make([]MemoryItem, 0, len(dmItems)+len(thItems))
	merged = append(merged, dmItems...)
	merged = append(merged, thItems...)

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
	searchCtx, postCheck, lexicalOnly, compatErr := m.PrepareVectorSearch(ctx, db)
	if compatErr != nil {
		return nil, compatErr
	}
	if lexicalOnly {
		lexResp, lexErr := db.SearchMemory(ctx, cortexdb.MemorySearchRequest{
			Query:         query,
			RetrievalMode: cortexdb.RetrievalModeLexical,
			TopK:          limit,
		})
		if lexErr != nil {
			return nil, lexErr
		}
		if lexResp == nil {
			return nil, nil
		}
		return lexResp.Results, nil
	}

	resp, err := db.SearchMemory(searchCtx, cortexdb.MemorySearchRequest{
		Query: query,
		TopK:  limit,
	})
	if err == nil && postCheck != nil {
		err = postCheck()
	}
	if err != nil {
		if !m.FallbackOnError() {
			return nil, err
		}
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

func collectChunkImages(msgs []MessageToStore, limit int) []string {
	if limit <= 0 {
		return nil
	}
	seen := make(map[string]struct{})
	var out []string
	for _, msg := range msgs {
		for _, rawURI := range msg.Images {
			trimmed := strings.TrimSpace(rawURI)
			if !llm.IsValidSystemOneImageDataURI(trimmed) {
				continue
			}
			if _, exists := seen[trimmed]; exists {
				continue
			}
			seen[trimmed] = struct{}{}
			out = append(out, trimmed)
			if len(out) >= limit {
				return out
			}
		}
	}
	return out
}

func boundMetadataImageURIs(uris []string, maxTotalBytes int) []string {
	if len(uris) == 0 || maxTotalBytes <= 0 {
		return nil
	}
	var out []string
	total := 0
	for _, uri := range uris {
		if total+len(uri) > maxTotalBytes {
			break
		}
		total += len(uri)
		out = append(out, uri)
	}
	return out
}

func verifyStoredVectorNonNull(ctx context.Context, rawDB *sql.DB, id string) error {
	var hasVector bool
	err := rawDB.QueryRowContext(ctx, `SELECT vector IS NOT NULL FROM messages WHERE id = ?`, id).Scan(&hasVector)
	if err != nil {
		return fmt.Errorf("verify non-null vector for %s: %w", id, err)
	}
	if !hasVector {
		return fmt.Errorf("memory item %s stored with NULL vector under strict fallback_on_error=false mode", id)
	}
	return nil
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
