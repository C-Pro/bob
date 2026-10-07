package gateway

import (
	"bob/internal/agentstore"
	"bob/internal/memory"
)

// MemoryStoreProvider implements fsm.StoreProvider backed by memory.Manager and per-chat SQLite databases.
type MemoryStoreProvider = agentstore.MemoryStoreProvider

// NewMemoryStoreProvider creates a new MemoryStoreProvider.
func NewMemoryStoreProvider(memMgr *memory.Manager, dataDir string) *MemoryStoreProvider {
	return agentstore.NewMemoryStoreProvider(memMgr, dataDir)
}
