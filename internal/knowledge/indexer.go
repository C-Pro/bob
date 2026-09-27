package knowledge

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/liliang-cn/cortexdb/v2/pkg/cortexdb"
)

// Namespaces for isolated vector and FTS5 indexing.
const (
	NamespaceMemories = "dm_memories"
	NamespaceSkills   = "dm_skills"
)

// VectorDB defines the interface for vector and lexical storage required for knowledge indexing.
// *cortexdb.DB satisfies this interface.
type VectorDB interface {
	SaveMemory(ctx context.Context, req cortexdb.MemorySaveRequest) (*cortexdb.MemorySaveResponse, error)
	SearchMemory(ctx context.Context, req cortexdb.MemorySearchRequest) (*cortexdb.MemorySearchResponse, error)
	DeleteMemory(ctx context.Context, req cortexdb.MemoryDeleteRequest) (*cortexdb.MemoryDeleteResponse, error)
}

// Indexer coordinates synchronization between the authoritative relational Store and VectorDB.
type Indexer struct {
	store  *Store
	vector VectorDB
	now    func() time.Time
}

// NewIndexer creates a new knowledge Indexer.
func NewIndexer(store *Store, vector VectorDB) *Indexer {
	return &Indexer{
		store:  store,
		vector: vector,
		now:    time.Now,
	}
}

// IndexVersion indexes a specific version into CortexDB.
func (idx *Indexer) IndexVersion(ctx context.Context, versionID string) error {
	if idx.store == nil {
		return errors.New("store cannot be nil")
	}
	if idx.vector == nil {
		return errors.New("vector database is nil")
	}

	itemID, _, err := ParseVersionID(versionID)
	if err != nil {
		return err
	}

	item, err := idx.store.GetItem(ctx, itemID)
	if err != nil {
		return fmt.Errorf("failed to get item %s for indexing: %w", itemID, err)
	}

	switch item.Kind {
	case KindMemory:
		return idx.indexMemoryVersion(ctx, item, versionID)
	case KindSkill:
		return idx.indexSkillVersion(ctx, item, versionID)
	default:
		return fmt.Errorf("%w: unknown item kind %q", ErrInvalidStatus, item.Kind)
	}
}

func (idx *Indexer) indexMemoryVersion(ctx context.Context, item *KnowledgeItem, versionID string) error {
	ver, err := idx.store.GetMemoryVersion(ctx, versionID)
	if err != nil {
		return fmt.Errorf("failed to get memory version %s for indexing: %w", versionID, err)
	}

	// Verify eligibility for indexing
	if item.Status != ItemStatusActive || item.ActiveVersionID != versionID || ver.Status != VersionStatusApproved {
		return fmt.Errorf("%w: version %s is not active and approved", ErrInvalidStatus, versionID)
	}

	// Provenance consistency check
	if ver.ItemID != item.ID || ver.Provenance.ChatID != item.ChatID || ver.Provenance.UserID != item.UserID {
		return fmt.Errorf("%w: version %s provenance does not match item %s", ErrUnauthorized, versionID, item.ID)
	}

	nowUnix := idx.now().Unix()
	if ver.ExpiresAt != nil && nowUnix >= *ver.ExpiresAt {
		return fmt.Errorf("%w: version %s has already expired", ErrInvalidStatus, versionID)
	}

	ttlSeconds := 0
	if ver.ExpiresAt != nil {
		diff := *ver.ExpiresAt - nowUnix
		if diff > 0 {
			ttlSeconds = int(diff)
		}
	}

	_, err = idx.vector.SaveMemory(ctx, cortexdb.MemorySaveRequest{
		MemoryID:   ver.ID,
		Scope:      cortexdb.MemoryScopeGlobal,
		Namespace:  NamespaceMemories,
		Content:    ver.SearchableContent(),
		Metadata:   ver.IndexMetadata(),
		Importance: ver.Confidence,
		TTLSeconds: ttlSeconds,
	})
	if err != nil {
		statusErr := idx.store.UpdateIndexStatus(ctx, ver.ID, IndexStatusError)
		return errors.Join(fmt.Errorf("failed to save memory to vector database: %w", err), statusErr)
	}

	// Post-save verification: atomically mark ready only if item is still active and this version is active and unexpired
	if err := idx.store.MarkVersionIndexed(ctx, ver.ID, idx.now().Unix()); err != nil {
		// Concurrently modified, forgotten, or expired -> purge from CortexDB using independent context to survive cancellation
		cleanupCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 2*time.Second)
		defer cancel()
		deleteErr := idx.RemoveFromIndex(cleanupCtx, ver.ID)
		return errors.Join(
			fmt.Errorf("item was modified concurrently during indexing: %w", err),
			deleteErr,
		)
	}

	return nil
}

func (idx *Indexer) indexSkillVersion(ctx context.Context, item *KnowledgeItem, versionID string) error {
	ver, err := idx.store.GetSkillVersion(ctx, versionID)
	if err != nil {
		return fmt.Errorf("failed to get skill version %s for indexing: %w", versionID, err)
	}

	if item.Status != ItemStatusActive || item.ActiveVersionID != versionID || ver.Status != VersionStatusApproved {
		return fmt.Errorf("%w: version %s is not active and approved", ErrInvalidStatus, versionID)
	}

	// Provenance consistency check
	if ver.ItemID != item.ID || ver.Provenance.ChatID != item.ChatID || ver.Provenance.UserID != item.UserID {
		return fmt.Errorf("%w: version %s provenance does not match item %s", ErrUnauthorized, versionID, item.ID)
	}

	_, err = idx.vector.SaveMemory(ctx, cortexdb.MemorySaveRequest{
		MemoryID:  ver.ID,
		Scope:     cortexdb.MemoryScopeGlobal,
		Namespace: NamespaceSkills,
		Content:   ver.SearchableContent(),
		Metadata:  ver.IndexMetadata(),
	})
	if err != nil {
		statusErr := idx.store.UpdateIndexStatus(ctx, ver.ID, IndexStatusError)
		return errors.Join(fmt.Errorf("failed to save skill to vector database: %w", err), statusErr)
	}

	// Post-save verification
	if err := idx.store.MarkVersionIndexed(ctx, ver.ID); err != nil {
		cleanupCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 2*time.Second)
		defer cancel()
		deleteErr := idx.RemoveFromIndex(cleanupCtx, ver.ID)
		return errors.Join(
			fmt.Errorf("skill was modified concurrently during indexing: %w", err),
			deleteErr,
		)
	}

	return nil
}

// RemoveFromIndex best-effort removes an entry from the vector database by its version ID.
func (idx *Indexer) RemoveFromIndex(ctx context.Context, versionID string) error {
	if idx.vector == nil {
		return nil
	}
	_, err := idx.vector.DeleteMemory(ctx, cortexdb.MemoryDeleteRequest{
		MemoryID: versionID,
	})
	return err
}

// ReconcilePending processes unindexed or errored candidates from SQLite and indexes them into VectorDB.
// Candidates are prioritized with pending before error, ordered by creation time ascending.
func (idx *Indexer) ReconcilePending(ctx context.Context, limit int) (int, error) {
	if idx.store == nil || idx.vector == nil {
		return 0, nil
	}
	if limit <= 0 {
		limit = 50
	}

	candidates, err := idx.store.GetPendingIndexVersions(ctx, limit)
	if err != nil {
		return 0, fmt.Errorf("failed to get pending index versions: %w", err)
	}

	successCount := 0
	var allErrs []error
	for _, cand := range candidates {
		if ctx.Err() != nil {
			return successCount, errors.Join(append(allErrs, ctx.Err())...)
		}
		if err := idx.IndexVersion(ctx, cand.VersionID); err != nil {
			allErrs = append(allErrs, err)
			continue
		}
		successCount++
	}

	// Reconcile pending index deletions
	deletions, delErr := idx.store.GetPendingIndexDeletions(ctx, limit)
	if delErr != nil {
		allErrs = append(allErrs, fmt.Errorf("failed to get pending index deletions: %w", delErr))
	} else {
		for _, d := range deletions {
			if ctx.Err() != nil {
				return successCount, errors.Join(append(allErrs, ctx.Err())...)
			}
			if remErr := idx.RemoveFromIndex(ctx, d.VersionID); remErr != nil {
				allErrs = append(allErrs, remErr)
				continue
			}
			_ = idx.store.RemovePendingIndexDeletion(ctx, d.VersionID)
			successCount++
		}
	}

	return successCount, errors.Join(allErrs...)
}

// FormatMemoryContent builds the searchable representation for a structured memory version.
func FormatMemoryContent(ver *MemoryVersion) string {
	return ver.SearchableContent()
}

// FormatSkillContent builds the searchable representation for a skill version, combining metadata with procedural instructions.
func FormatSkillContent(ver *SkillVersion) string {
	return ver.SearchableContent()
}
