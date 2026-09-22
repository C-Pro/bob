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
	"bob/internal/scheduler"
)

// MemoryStoreProvider implements fsm.StoreProvider backed by memory.Manager and per-chat SQLite databases.
type MemoryStoreProvider struct {
	memMgr      *memory.Manager
	dataDir     string
	mu          sync.RWMutex
	stores      map[string]*fsm.Store
	schedStores map[string]*scheduler.Store
	initOnce    sync.Once
}

// NewMemoryStoreProvider creates a new MemoryStoreProvider.
func NewMemoryStoreProvider(memMgr *memory.Manager, dataDir string) *MemoryStoreProvider {
	return &MemoryStoreProvider{
		memMgr:      memMgr,
		dataDir:     dataDir,
		stores:      make(map[string]*fsm.Store),
		schedStores: make(map[string]*scheduler.Store),
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
	if err := scheduler.EnsureScheduleSchema(ctx, rawDB); err != nil {
		return nil, fmt.Errorf("failed to ensure scheduler schema for chat %s: %w", chatID, err)
	}

	st = fsm.NewStore(rawDB)
	p.stores[key] = st
	return st, nil
}

// GetSchedulerStore retrieves or initializes the isolated scheduler store for the given chat context.
func (p *MemoryStoreProvider) GetSchedulerStore(ctx context.Context, chatID string, isDM bool) (*scheduler.Store, error) {
	if p.memMgr == nil {
		return nil, fmt.Errorf("memory manager is nil")
	}

	key := p.storeKey(chatID, isDM)

	p.mu.RLock()
	st, ok := p.schedStores[key]
	p.mu.RUnlock()
	if ok {
		return st, nil
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	if st, ok = p.schedStores[key]; ok {
		return st, nil
	}

	cortexDB, err := p.memMgr.GetDB(ctx, chatID, isDM)
	if err != nil {
		return nil, fmt.Errorf("failed to get database for chat %s: %w", chatID, err)
	}

	rawDB := cortexDB.SQL()
	if err := scheduler.EnsureScheduleSchema(ctx, rawDB); err != nil {
		return nil, fmt.Errorf("failed to ensure scheduler schema for chat %s: %w", chatID, err)
	}

	st = scheduler.NewStore(rawDB)
	p.schedStores[key] = st
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

// ActiveSchedulerStores discovers and returns isolated scheduler stores for all active chat databases.
func (p *MemoryStoreProvider) ActiveSchedulerStores(ctx context.Context) ([]*scheduler.Store, error) {
	if p.memMgr == nil {
		return nil, nil
	}

	p.initOnce.Do(func() {
		p.discoverStores(ctx)
	})

	p.mu.RLock()
	defer p.mu.RUnlock()

	stores := make([]*scheduler.Store, 0, len(p.schedStores))
	for _, st := range p.schedStores {
		stores = append(stores, st)
	}

	return stores, nil
}
