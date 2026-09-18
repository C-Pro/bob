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

	"bob/internal/fsm"
	"bob/internal/memory"
)

// MemoryStoreProvider implements fsm.StoreProvider backed by memory.Manager and per-chat SQLite databases.
type MemoryStoreProvider struct {
	memMgr   *memory.Manager
	dataDir  string
	mu       sync.RWMutex
	stores   map[string]*fsm.Store
	initOnce sync.Once
}

// NewMemoryStoreProvider creates a new MemoryStoreProvider.
func NewMemoryStoreProvider(memMgr *memory.Manager, dataDir string) *MemoryStoreProvider {
	return &MemoryStoreProvider{
		memMgr:  memMgr,
		dataDir: dataDir,
		stores:  make(map[string]*fsm.Store),
	}
}

func (p *MemoryStoreProvider) storeKey(chatID string, isDM bool) string {
	if !isDM || chatID == "townhall" {
		return "townhall"
	}
	return "dm_" + memory.SanitizeChatID(chatID)
}

// GetStore retrieves or initializes the durable FSM store for the given chat context.
func (p *MemoryStoreProvider) GetStore(ctx context.Context, chatID string, isDM bool) (*fsm.Store, error) {
	if p.memMgr == nil {
		return nil, fmt.Errorf("memory manager is nil")
	}

	key := p.storeKey(chatID, isDM)

	p.mu.RLock()
	st, ok := p.stores[key]
	p.mu.RUnlock()
	if ok {
		return st, nil
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	if st, ok = p.stores[key]; ok {
		return st, nil
	}

	cortexDB, err := p.memMgr.GetDB(ctx, chatID, isDM)
	if err != nil {
		return nil, fmt.Errorf("failed to get database for chat %s: %w", chatID, err)
	}

	rawDB := cortexDB.SQL()
	if err := fsm.EnsureDBSchema(ctx, rawDB); err != nil {
		return nil, fmt.Errorf("failed to ensure fsm schema for chat %s: %w", chatID, err)
	}

	st = fsm.NewStore(rawDB)
	p.stores[key] = st
	return st, nil
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
			active, err := hasActiveRuns(filePath)
			if err != nil {
				slog.Warn("failed to inspect townhall.db for active runs; opening for safety", "error", err)
				active = true
			}
			if active {
				if _, err := p.GetStore(ctx, "townhall", false); err != nil {
					slog.Warn("failed to open townhall.db during initial active store discovery", "error", err)
				}
			}
		} else if strings.HasPrefix(name, "dm_") && strings.HasSuffix(name, ".db") {
			active, err := hasActiveRuns(filePath)
			if err != nil {
				slog.Warn("failed to inspect dm db for active runs; opening for safety", "name", name, "error", err)
				active = true
			}
			if active {
				chatID := strings.TrimSuffix(strings.TrimPrefix(name, "dm_"), ".db")
				if _, err := p.GetStore(ctx, chatID, true); err != nil {
					slog.Warn("failed to open dm db during initial active store discovery", "chatID", chatID, "error", err)
				}
			}
		}
	}
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
	defer p.mu.RUnlock()

	stores := make([]*fsm.Store, 0, len(p.stores))
	for _, st := range p.stores {
		stores = append(stores, st)
	}

	return stores, nil
}
