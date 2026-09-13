package gateway

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"strings"

	"bob/internal/fsm"
	"bob/internal/memory"
	"bob/internal/store"
)

// MemoryStoreProvider implements fsm.StoreProvider backed by memory.Manager and per-chat SQLite databases.
type MemoryStoreProvider struct {
	memMgr  *memory.Manager
	dataDir string
}

// NewMemoryStoreProvider creates a new MemoryStoreProvider.
func NewMemoryStoreProvider(memMgr *memory.Manager, dataDir string) *MemoryStoreProvider {
	return &MemoryStoreProvider{
		memMgr:  memMgr,
		dataDir: dataDir,
	}
}

// GetStore retrieves or initializes the durable FSM store for the given chat context.
func (p *MemoryStoreProvider) GetStore(ctx context.Context, chatID string, isDM bool) (*fsm.Store, error) {
	if p.memMgr == nil {
		return nil, fmt.Errorf("memory manager is nil")
	}

	cortexDB, err := p.memMgr.GetDB(ctx, chatID, isDM)
	if err != nil {
		return nil, fmt.Errorf("failed to get database for chat %s: %w", chatID, err)
	}

	rawDB := cortexDB.SQL()
	if err := store.EnsureDBSchema(ctx, rawDB); err != nil {
		return nil, fmt.Errorf("failed to ensure fsm schema for chat %s: %w", chatID, err)
	}

	return fsm.NewStore(rawDB), nil
}

// ActiveStores discovers and returns durable FSM stores for all existing or active chat databases.
func (p *MemoryStoreProvider) ActiveStores(ctx context.Context) ([]*fsm.Store, error) {
	if p.memMgr == nil {
		return nil, nil
	}

	// Discover on-disk chat databases in dataDir to ensure they are loaded in memoryManager
	if p.dataDir != "" {
		entries, err := os.ReadDir(p.dataDir)
		if err != nil && !os.IsNotExist(err) {
			slog.Warn("failed to read data directory during active store discovery", "dataDir", p.dataDir, "error", err)
		} else if err == nil {
			for _, entry := range entries {
				if entry.IsDir() {
					continue
				}
				name := entry.Name()
				if name == "townhall.db" {
					if _, err := p.memMgr.GetDB(ctx, "townhall", false); err != nil {
						slog.Warn("failed to open townhall.db during active store discovery", "error", err)
					}
				} else if strings.HasPrefix(name, "dm_") && strings.HasSuffix(name, ".db") {
					chatID := strings.TrimSuffix(strings.TrimPrefix(name, "dm_"), ".db")
					if _, err := p.memMgr.GetDB(ctx, chatID, true); err != nil {
						slog.Warn("failed to open dm db during active store discovery", "chatID", chatID, "error", err)
					}
				}
			}
		}
	}

	activeDBs := p.memMgr.ActiveDBs()
	stores := make([]*fsm.Store, 0, len(activeDBs))
	for _, rawDB := range activeDBs {
		if err := store.EnsureDBSchema(ctx, rawDB); err != nil {
			return nil, fmt.Errorf("failed to ensure fsm schema for active database: %w", err)
		}
		stores = append(stores, fsm.NewStore(rawDB))
	}

	return stores, nil
}
