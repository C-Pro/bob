package gateway

import (
	"bob/internal/knowledge"
)

// KnowledgeWorkerConfig configures the background knowledge reconciliation engine.
type KnowledgeWorkerConfig = knowledge.KnowledgeWorkerConfig

// DefaultKnowledgeWorkerConfig returns default settings for the background knowledge worker.
func DefaultKnowledgeWorkerConfig() KnowledgeWorkerConfig {
	return knowledge.DefaultKnowledgeWorkerConfig()
}

// KnowledgeTargetProvider defines the provider contract needed by KnowledgeWorker.
type KnowledgeTargetProvider = knowledge.KnowledgeTargetProvider

// KnowledgeWorker periodically scans and reconciles pending index deletions and unindexed candidates
// across all discovered and active DM knowledge stores.
type KnowledgeWorker = knowledge.KnowledgeWorker

// NewKnowledgeWorker creates a new lifecycle-managed KnowledgeWorker.
func NewKnowledgeWorker(provider KnowledgeTargetProvider, cfg KnowledgeWorkerConfig) *KnowledgeWorker {
	return knowledge.NewKnowledgeWorker(provider, cfg)
}
