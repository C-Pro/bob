package gateway

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"bob/internal/fsm"
	"bob/internal/knowledge"
	"bob/internal/memory"
	"bob/internal/scheduler"
)

type storeEntry struct {
	mu     sync.Mutex
	store  *fsm.Store
	sStore *scheduler.Store
}

type knowledgeEntry struct {
	chatID        string
	initMu        sync.Mutex
	reconcileGate chan struct{}
	store         *knowledge.Store
	indexer       *knowledge.Indexer
}

// MemoryStoreProvider implements fsm.StoreProvider backed by memory.Manager and per-chat SQLite databases.
type MemoryStoreProvider struct {
	memMgr            *memory.Manager
	dataDir           string
	mu                sync.RWMutex
	storeEntries      map[string]*storeEntry
	knowledgeEntries  map[string]*knowledgeEntry
	maxDiscoveryLimit int
	initOnce          sync.Once
}

// NewMemoryStoreProvider creates a new MemoryStoreProvider.
func NewMemoryStoreProvider(memMgr *memory.Manager, dataDir string) *MemoryStoreProvider {
	return &MemoryStoreProvider{
		memMgr:            memMgr,
		dataDir:           dataDir,
		storeEntries:      make(map[string]*storeEntry),
		knowledgeEntries:  make(map[string]*knowledgeEntry),
		maxDiscoveryLimit: 10,
	}
}

// SetMaxDiscoveryLimit configures the maximum discovery limit for knowledge searchers.
func (p *MemoryStoreProvider) SetMaxDiscoveryLimit(limit int) {
	if p == nil {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if limit > 0 {
		p.maxDiscoveryLimit = limit
	}
}

func (p *MemoryStoreProvider) storeKey(chatID string, isDM bool) string {
	if !isDM || chatID == "townhall" {
		return "townhall"
	}
	return "dm_" + memory.SanitizeChatID(chatID)
}

func (p *MemoryStoreProvider) getOrCreateStoreEntry(chatID string, isDM bool) *storeEntry {
	key := p.storeKey(chatID, isDM)
	p.mu.Lock()
	defer p.mu.Unlock()
	entry, ok := p.storeEntries[key]
	if !ok {
		entry = &storeEntry{}
		p.storeEntries[key] = entry
	}
	return entry
}

func (p *MemoryStoreProvider) getOrCreateKnowledgeEntry(chatID string, isDM bool) *knowledgeEntry {
	key := p.storeKey(chatID, isDM)
	p.mu.Lock()
	defer p.mu.Unlock()
	entry, ok := p.knowledgeEntries[key]
	if !ok {
		entry = &knowledgeEntry{
			chatID:        chatID,
			reconcileGate: make(chan struct{}, 1),
		}
		p.knowledgeEntries[key] = entry
	}
	return entry
}

// GetStore retrieves or initializes the durable FSM store for the given chat context.
func (p *MemoryStoreProvider) GetStore(ctx context.Context, chatID string, isDM bool) (*fsm.Store, error) {
	if p.memMgr == nil {
		return nil, fmt.Errorf("memory manager is nil")
	}

	entry := p.getOrCreateStoreEntry(chatID, isDM)
	entry.mu.Lock()
	defer entry.mu.Unlock()

	if entry.store != nil {
		return entry.store, nil
	}

	cortexDB, err := p.memMgr.GetDB(ctx, chatID, isDM)
	if err != nil {
		return nil, fmt.Errorf("failed to get database for chat %s: %w", chatID, err)
	}

	rawDB := cortexDB.SQL()
	if err := fsm.EnsureDBSchema(ctx, rawDB); err != nil {
		return nil, fmt.Errorf("failed to ensure fsm schema for chat %s: %w", chatID, err)
	}
	if err := scheduler.EnsureScheduleSchema(ctx, rawDB); err != nil {
		return nil, fmt.Errorf("failed to ensure scheduler schema for chat %s: %w", chatID, err)
	}
	if isDM && chatID != "townhall" {
		if err := knowledge.EnsureKnowledgeSchema(ctx, rawDB); err != nil {
			return nil, fmt.Errorf("failed to ensure knowledge schema for chat %s: %w", chatID, err)
		}
	}

	entry.store = fsm.NewStore(rawDB)
	return entry.store, nil
}

// GetSchedulerStore retrieves or initializes the isolated scheduler store for the given chat context.
func (p *MemoryStoreProvider) GetSchedulerStore(ctx context.Context, chatID string, isDM bool) (*scheduler.Store, error) {
	if p.memMgr == nil {
		return nil, fmt.Errorf("memory manager is nil")
	}

	entry := p.getOrCreateStoreEntry(chatID, isDM)
	entry.mu.Lock()
	defer entry.mu.Unlock()

	if entry.sStore != nil {
		return entry.sStore, nil
	}

	cortexDB, err := p.memMgr.GetDB(ctx, chatID, isDM)
	if err != nil {
		return nil, fmt.Errorf("failed to get database for chat %s: %w", chatID, err)
	}

	rawDB := cortexDB.SQL()
	if err := scheduler.EnsureScheduleSchema(ctx, rawDB); err != nil {
		return nil, fmt.Errorf("failed to ensure scheduler schema for chat %s: %w", chatID, err)
	}

	entry.sStore = scheduler.NewStore(rawDB)
	return entry.sStore, nil
}

// GetKnowledgeStore retrieves or initializes the isolated knowledge store for the given chat context.
func (p *MemoryStoreProvider) GetKnowledgeStore(ctx context.Context, chatID string, isDM bool) (*knowledge.Store, error) {
	if !isDM || chatID == "townhall" {
		return nil, fmt.Errorf("knowledge storage is only available in direct messages")
	}
	if p.memMgr == nil {
		return nil, fmt.Errorf("memory manager is nil")
	}

	entry := p.getOrCreateKnowledgeEntry(chatID, isDM)
	entry.initMu.Lock()
	defer entry.initMu.Unlock()

	if entry.store != nil {
		return entry.store, nil
	}

	cortexDB, err := p.memMgr.GetDB(ctx, chatID, isDM)
	if err != nil {
		return nil, fmt.Errorf("failed to get database for chat %s: %w", chatID, err)
	}

	rawDB := cortexDB.SQL()
	if err := knowledge.EnsureKnowledgeSchema(ctx, rawDB); err != nil {
		return nil, fmt.Errorf("failed to ensure knowledge schema for chat %s: %w", chatID, err)
	}

	entry.store = knowledge.NewStore(rawDB)
	return entry.store, nil
}

// GetKnowledgeIndexer retrieves or initializes the knowledge indexer for the given chat context.
func (p *MemoryStoreProvider) GetKnowledgeIndexer(ctx context.Context, chatID string, isDM bool) (*knowledge.Indexer, error) {
	if !isDM || chatID == "townhall" {
		return nil, fmt.Errorf("knowledge indexer is only available in direct messages")
	}
	if p.memMgr == nil {
		return nil, fmt.Errorf("memory manager is nil")
	}

	entry := p.getOrCreateKnowledgeEntry(chatID, isDM)
	entry.initMu.Lock()
	defer entry.initMu.Unlock()

	if entry.indexer != nil {
		return entry.indexer, nil
	}

	cortexDB, err := p.memMgr.GetDB(ctx, chatID, isDM)
	if err != nil {
		return nil, fmt.Errorf("failed to get database for chat %s: %w", chatID, err)
	}

	if entry.store == nil {
		rawDB := cortexDB.SQL()
		if err := knowledge.EnsureKnowledgeSchema(ctx, rawDB); err != nil {
			return nil, fmt.Errorf("failed to ensure knowledge schema for chat %s: %w", chatID, err)
		}
		entry.store = knowledge.NewStore(rawDB)
	}

	entry.indexer = knowledge.NewIndexer(entry.store, cortexDB)
	return entry.indexer, nil
}

// ReconcileChat serializes reconciliation per chat, ensuring search-triggered, background,
// and startup passes for the same chat do not overlap while allowing different chats to proceed concurrently.
func (p *MemoryStoreProvider) ReconcileChat(ctx context.Context, chatID string, isDM bool, limit int) (int, error) {
	if !isDM || chatID == "townhall" {
		return 0, nil
	}

	entry := p.getOrCreateKnowledgeEntry(chatID, isDM)
	select {
	case entry.reconcileGate <- struct{}{}:
		defer func() { <-entry.reconcileGate }()
	case <-ctx.Done():
		return 0, ctx.Err()
	}

	idx, err := p.GetKnowledgeIndexer(ctx, chatID, isDM)
	if err != nil {
		return 0, err
	}

	return idx.ReconcilePending(ctx, limit)
}

// GetKnowledgeSearcher resolves a knowledge.Searcher for a chat context and user.
func (p *MemoryStoreProvider) GetKnowledgeSearcher(ctx context.Context, chatID string, isDM bool, userID string) (*knowledge.Searcher, error) {
	if !isDM || chatID == "townhall" {
		return nil, fmt.Errorf("knowledge search is only available in direct messages")
	}
	if p.memMgr == nil {
		return nil, fmt.Errorf("memory manager is nil")
	}

	// Trigger bounded index reconciliation serialized per chat via ReconcileChat
	if _, recErr := p.ReconcileChat(ctx, chatID, isDM, 100); recErr != nil {
		slog.Warn("failed to reconcile pending knowledge index records before search", "chat_id", chatID, "error", recErr)
	}

	kStore, err := p.GetKnowledgeStore(ctx, chatID, isDM)
	if err != nil {
		return nil, err
	}

	cortexDB, err := p.memMgr.GetDB(ctx, chatID, isDM)
	if err != nil {
		return nil, fmt.Errorf("failed to get database for chat %s: %w", chatID, err)
	}

	p.mu.RLock()
	maxLimit := p.maxDiscoveryLimit
	p.mu.RUnlock()
	if maxLimit <= 0 {
		maxLimit = 10
	}

	return knowledge.NewSearcher(kStore, cortexDB, knowledge.SessionIdentity{
		ChatID: chatID,
		UserID: userID,
	}, maxLimit)
}

func hasActiveRuns(dbPath string) (bool, error) {
	fi, err := os.Stat(dbPath)
	if err != nil {
		if os.IsNotExist(err) {
			return false, nil
		}
		return false, err
	}
	if fi.Size() == 0 {
		return false, nil
	}

	dsn := fmt.Sprintf("file:%s?mode=ro", dbPath)
	db, err := sql.Open("sqlite", dsn)
	if err != nil {
		return false, err
	}
	defer func() { _ = db.Close() }()

	var tableExists int
	err = db.QueryRow("SELECT 1 FROM sqlite_master WHERE type='table' AND name='fsm_runs'").Scan(&tableExists)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return false, nil
		}
		return false, err
	}

	var hasActive int
	err = db.QueryRow("SELECT 1 FROM fsm_runs WHERE status IN ('RUNNING', 'WAITING', 'PENDING') LIMIT 1").Scan(&hasActive)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return false, nil
		}
		return false, err
	}
	return true, nil
}

func hasActiveSchedules(dbPath string) (bool, error) {
	fi, err := os.Stat(dbPath)
	if err != nil {
		if os.IsNotExist(err) {
			return false, nil
		}
		return false, err
	}
	if fi.Size() == 0 {
		return false, nil
	}

	dsn := fmt.Sprintf("file:%s?mode=ro", dbPath)
	db, err := sql.Open("sqlite", dsn)
	if err != nil {
		return false, err
	}
	defer func() { _ = db.Close() }()

	var tableExists int
	err = db.QueryRow("SELECT 1 FROM sqlite_master WHERE type='table' AND name='schedules'").Scan(&tableExists)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return false, nil
		}
		return false, err
	}

	var hasActive int
	err = db.QueryRow("SELECT 1 FROM schedules WHERE status = 'ACTIVE' LIMIT 1").Scan(&hasActive)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return false, nil
		}
		return false, err
	}
	return true, nil
}

func hasPendingKnowledgeWork(dbPath string) (bool, error) {
	fi, err := os.Stat(dbPath)
	if err != nil {
		if os.IsNotExist(err) {
			return false, nil
		}
		return false, err
	}
	if fi.Size() == 0 {
		return false, nil
	}

	dsn := fmt.Sprintf("file:%s?mode=ro", dbPath)
	db, err := sql.Open("sqlite", dsn)
	if err != nil {
		return false, err
	}
	defer func() { _ = db.Close() }()

	var tableExists int
	err = db.QueryRow("SELECT 1 FROM sqlite_master WHERE type='table' AND name='pending_index_deletions'").Scan(&tableExists)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return false, nil
		}
		return false, err
	}

	// 1. Pending index deletions
	var hasDeletions int
	err = db.QueryRow("SELECT 1 FROM pending_index_deletions LIMIT 1").Scan(&hasDeletions)
	if err == nil {
		return true, nil
	} else if !errors.Is(err, sql.ErrNoRows) {
		return false, err
	}

	// 2. Unindexed active approved memories (excluding already expired memories)
	var hasMemories int
	err = db.QueryRow(`
		SELECT 1 FROM memory_versions mv
		JOIN knowledge_items ki ON ki.id = mv.item_id
		WHERE ki.status = 'active'
		  AND ki.active_version_id = mv.id
		  AND mv.status = 'approved'
		  AND mv.index_status IN ('pending', 'error')
		  AND (mv.expires_at IS NULL OR mv.expires_at > ?)
		LIMIT 1
	`, time.Now().Unix()).Scan(&hasMemories)
	if err == nil {
		return true, nil
	} else if !errors.Is(err, sql.ErrNoRows) {
		return false, err
	}

	// 3. Unindexed active approved skills
	var hasSkills int
	err = db.QueryRow(`
		SELECT 1 FROM skill_versions sv
		JOIN knowledge_items ki ON ki.id = sv.item_id
		WHERE ki.status = 'active'
		  AND ki.active_version_id = sv.id
		  AND sv.status = 'approved'
		  AND sv.index_status IN ('pending', 'error')
		LIMIT 1
	`).Scan(&hasSkills)
	if err == nil {
		return true, nil
	} else if !errors.Is(err, sql.ErrNoRows) {
		return false, err
	}

	return false, nil
}

func (p *MemoryStoreProvider) discoverStore(ctx context.Context, filePath, chatID string, isDM bool) {
	hasFSM, err := hasActiveRuns(filePath)
	if err != nil {
		slog.Warn("failed to inspect chat database for active FSM runs; opening for safety", "path", filePath, "error", err)
		hasFSM = true
	}
	if hasFSM {
		if _, err := p.GetStore(ctx, chatID, isDM); err != nil {
			slog.Warn("failed to open chat database for FSM recovery", "path", filePath, "error", err)
		}
	}

	hasSchedules, err := hasActiveSchedules(filePath)
	if err != nil {
		slog.Warn("failed to inspect chat database for active schedules; opening for safety", "path", filePath, "error", err)
		hasSchedules = true
	}
	if hasSchedules {
		if _, err := p.GetSchedulerStore(ctx, chatID, isDM); err != nil {
			slog.Warn("failed to open chat database for scheduler recovery", "path", filePath, "error", err)
		}
	}

	if isDM && chatID != "townhall" {
		hasWork, err := hasPendingKnowledgeWork(filePath)
		if err != nil {
			slog.Warn("failed to inspect chat database for pending knowledge work; opening for safety", "path", filePath, "error", err)
			hasWork = true
		}
		if hasWork {
			if _, err := p.GetKnowledgeIndexer(ctx, chatID, isDM); err != nil {
				slog.Warn("failed to open chat database for knowledge indexer recovery", "path", filePath, "error", err)
			}
		}
	}
}

func (p *MemoryStoreProvider) discoverStores(ctx context.Context) {
	if p.dataDir == "" {
		return
	}

	entries, err := os.ReadDir(p.dataDir)
	if err != nil {
		if !os.IsNotExist(err) {
			slog.Warn("failed to read data directory during initial active store discovery", "dataDir", p.dataDir, "error", err)
		}
		return
	}

	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		filePath := filepath.Join(p.dataDir, name)

		if name == "townhall.db" {
			p.discoverStore(ctx, filePath, "townhall", false)
		} else if strings.HasPrefix(name, "dm_") && strings.HasSuffix(name, ".db") {
			chatID := strings.TrimSuffix(strings.TrimPrefix(name, "dm_"), ".db")
			p.discoverStore(ctx, filePath, chatID, true)
		}
	}
}

// DiscoverKnowledgeTargets returns all DM chat IDs that currently have active knowledge indexers
// or unopened databases with pending index deletions or unindexed active versions.
func (p *MemoryStoreProvider) DiscoverKnowledgeTargets(ctx context.Context) []string {
	if p == nil {
		return nil
	}

	p.mu.RLock()
	knownTargets := make(map[string]struct{}, len(p.knowledgeEntries))
	for _, entry := range p.knowledgeEntries {
		if entry.chatID != "" && entry.chatID != "townhall" {
			knownTargets[entry.chatID] = struct{}{}
		}
	}
	dataDir := p.dataDir
	p.mu.RUnlock()

	if dataDir != "" {
		entries, err := os.ReadDir(dataDir)
		if err == nil {
			for _, entry := range entries {
				if entry.IsDir() {
					continue
				}
				name := entry.Name()
				if strings.HasPrefix(name, "dm_") && strings.HasSuffix(name, ".db") {
					rawChatID := strings.TrimSuffix(strings.TrimPrefix(name, "dm_"), ".db")
					if rawChatID == "" {
						continue
					}
					if _, alreadyKnown := knownTargets[rawChatID]; !alreadyKnown {
						filePath := filepath.Join(dataDir, name)
						hasWork, checkErr := hasPendingKnowledgeWork(filePath)
						if checkErr != nil {
							slog.Warn("failed to check pending knowledge work on dm db; targeting for safety", "path", filePath, "error", checkErr)
							hasWork = true
						}
						if hasWork {
							if _, openErr := p.GetKnowledgeIndexer(ctx, rawChatID, true); openErr == nil {
								knownTargets[rawChatID] = struct{}{}
							}
						}
					}
				}
			}
		}
	}

	targets := make([]string, 0, len(knownTargets))
	for cid := range knownTargets {
		targets = append(targets, cid)
	}
	return targets
}

// ActiveStores discovers and returns durable FSM stores for all existing or active chat databases.
func (p *MemoryStoreProvider) ActiveStores(ctx context.Context) ([]*fsm.Store, error) {
	if p.memMgr == nil {
		return nil, nil
	}

	p.initOnce.Do(func() {
		p.discoverStores(ctx)
	})

	p.mu.RLock()
	entries := make([]*storeEntry, 0, len(p.storeEntries))
	for _, entry := range p.storeEntries {
		entries = append(entries, entry)
	}
	p.mu.RUnlock()

	stores := make([]*fsm.Store, 0, len(entries))
	for _, entry := range entries {
		entry.mu.Lock()
		st := entry.store
		entry.mu.Unlock()
		if st != nil {
			stores = append(stores, st)
		}
	}

	return stores, nil
}

// ActiveSchedulerStores discovers and returns isolated scheduler stores for all active chat databases.
func (p *MemoryStoreProvider) ActiveSchedulerStores(ctx context.Context) ([]*scheduler.Store, error) {
	if p.memMgr == nil {
		return nil, nil
	}

	p.initOnce.Do(func() {
		p.discoverStores(ctx)
	})

	p.mu.RLock()
	entries := make([]*storeEntry, 0, len(p.storeEntries))
	for _, entry := range p.storeEntries {
		entries = append(entries, entry)
	}
	p.mu.RUnlock()

	stores := make([]*scheduler.Store, 0, len(entries))
	for _, entry := range entries {
		entry.mu.Lock()
		st := entry.sStore
		entry.mu.Unlock()
		if st != nil {
			stores = append(stores, st)
		}
	}

	return stores, nil
}
