package knowledge

import (
	"context"
	"database/sql"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	_ "modernc.org/sqlite"
)

func setupTestStore(t *testing.T) (*Store, *sql.DB) {
	t.Helper()
	dbPath := filepath.Join(t.TempDir(), "knowledge_test.db")
	db, err := sql.Open("sqlite", "file:"+dbPath+"?_auto_vacuum=INCREMENTAL&_pragma=journal_mode(WAL)&_pragma=foreign_keys(ON)&_pragma=busy_timeout(5000)&_pragma=synchronous(NORMAL)")
	require.NoError(t, err)

	err = EnsureKnowledgeSchema(context.Background(), db)
	require.NoError(t, err)

	kStore := NewStore(db)
	return kStore, db
}

func TestStore_ProposeAndApproveMemory(t *testing.T) {
	ctx := context.Background()
	store, db := setupTestStore(t)
	defer func() { _ = db.Close() }()

	// 1. Propose brand new memory
	ttl := 30
	msgSeq := int64(42)
	item, ver, err := store.ProposeMemory(ctx, &KnowledgeItem{
		ChatID: "chat_dm_1",
		UserID: "user_alice",
	}, &MemoryVersion{
		Type:       MemoryTypePreference,
		Content:    "Alice prefers concise Go code",
		Confidence: 0.95,
		TTLDays:    &ttl,
		Provenance: Provenance{
			SourceMessageSeq: &msgSeq,
			FSMRunID:         "run_123",
		},
	})
	require.NoError(t, err)
	assert.NotEmpty(t, item.ID)
	assert.Equal(t, ItemStatusPending, item.Status)
	assert.Equal(t, "", item.ActiveVersionID)
	assert.Equal(t, 1, ver.Revision)
	assert.Equal(t, FormatVersionID(item.ID, 1), ver.ID)
	assert.Equal(t, VersionStatusProposed, ver.Status)
	assert.Equal(t, IndexStatusPending, ver.IndexStatus)
	assert.NotNil(t, ver.ExpiresAt)
	assert.NotEmpty(t, ver.Provenance.ContentHash)

	// Inactive before approval
	activeItems, err := store.ListItems(ctx, "chat_dm_1", KindMemory, "active")
	require.NoError(t, err)
	assert.Empty(t, activeItems)

	pendingItems, err := store.ListItems(ctx, "chat_dm_1", KindMemory, "pending")
	require.NoError(t, err)
	require.Len(t, pendingItems, 1)
	assert.Equal(t, item.ID, pendingItems[0].ID)

	// 2. Approve version 1
	err = store.ApproveVersion(ctx, ver.ID, "user_alice")
	require.NoError(t, err)

	approvedItem, err := store.GetItem(ctx, item.ID)
	require.NoError(t, err)
	assert.Equal(t, ItemStatusActive, approvedItem.Status)
	assert.Equal(t, ver.ID, approvedItem.ActiveVersionID)

	approvedVer, err := store.GetMemoryVersion(ctx, ver.ID)
	require.NoError(t, err)
	assert.Equal(t, VersionStatusApproved, approvedVer.Status)
	assert.Equal(t, "user_alice", approvedVer.Provenance.ReviewedBy)
	assert.NotNil(t, approvedVer.Provenance.ReviewedAt)

	// Now appears in active items
	activeItems, err = store.ListItems(ctx, "chat_dm_1", KindMemory, "active")
	require.NoError(t, err)
	require.Len(t, activeItems, 1)

	// 3. Propose revision to existing memory item
	_, ver2, err := store.ProposeMemory(ctx, nil, &MemoryVersion{
		ItemID:  item.ID,
		Type:    MemoryTypePreference,
		Content: "Alice prefers idiomatic Go with table-driven tests",
		Provenance: Provenance{
			UserID: "user_alice",
			ChatID: "chat_dm_1",
		},
	})
	require.NoError(t, err)
	assert.Equal(t, 2, ver2.Revision)
	assert.Equal(t, FormatVersionID(item.ID, 2), ver2.ID)
	assert.Equal(t, VersionStatusProposed, ver2.Status)

	// Active version remains version 1 while version 2 is proposed
	itemBeforeApprove, err := store.GetItem(ctx, item.ID)
	require.NoError(t, err)
	assert.Equal(t, ver.ID, itemBeforeApprove.ActiveVersionID)

	// 4. Approve revision 2 -> supersedes revision 1
	err = store.ApproveVersion(ctx, ver2.ID, "user_alice")
	require.NoError(t, err)

	itemAfterApprove, err := store.GetItem(ctx, item.ID)
	require.NoError(t, err)
	assert.Equal(t, ver2.ID, itemAfterApprove.ActiveVersionID)

	v1After, err := store.GetMemoryVersion(ctx, ver.ID)
	require.NoError(t, err)
	assert.Equal(t, VersionStatusSuperseded, v1After.Status)

	v2After, err := store.GetMemoryVersion(ctx, ver2.ID)
	require.NoError(t, err)
	assert.Equal(t, VersionStatusApproved, v2After.Status)

	// Cannot approve already approved version
	err = store.ApproveVersion(ctx, ver2.ID, "user_alice")
	assert.Error(t, err)
	assert.ErrorIs(t, err, ErrInvalidStatus)
}

func TestStore_ProposeAndRejectMemory(t *testing.T) {
	ctx := context.Background()
	store, db := setupTestStore(t)
	defer func() { _ = db.Close() }()

	item, ver, err := store.ProposeMemory(ctx, &KnowledgeItem{
		ChatID: "chat_1",
		UserID: "user_1",
	}, &MemoryVersion{
		Type:    MemoryTypeFact,
		Content: "The server is hosted on AWS",
	})
	require.NoError(t, err)

	err = store.RejectVersion(ctx, ver.ID, "user_1")
	require.NoError(t, err)

	rejectedVer, err := store.GetMemoryVersion(ctx, ver.ID)
	require.NoError(t, err)
	assert.Equal(t, VersionStatusRejected, rejectedVer.Status)
	assert.Equal(t, "user_1", rejectedVer.Provenance.ReviewedBy)

	rejectedItems, err := store.ListItems(ctx, "chat_1", KindMemory, "rejected")
	require.NoError(t, err)
	require.Len(t, rejectedItems, 1)
	assert.Equal(t, item.ID, rejectedItems[0].ID)
}

func TestStore_SkillLifecycle(t *testing.T) {
	ctx := context.Background()
	store, db := setupTestStore(t)
	defer func() { _ = db.Close() }()

	// 1. Propose skill
	item, ver, err := store.ProposeSkill(ctx, &KnowledgeItem{
		ChatID: "chat_dm_2",
		UserID: "user_bob",
	}, &SkillVersion{
		Name:                 "run_tests",
		Description:          "Executes test suite with race detector",
		Triggers:             []string{"run tests", "verify build"},
		Tags:                 []string{"test", "ci"},
		InstructionsMarkdown: "# Run Tests\n`go test -race ./...`",
	})
	require.NoError(t, err)
	assert.Equal(t, ItemStatusPending, item.Status)
	assert.Equal(t, 1, ver.Revision)

	// 2. Approve skill
	err = store.ApproveVersion(ctx, ver.ID, "user_bob")
	require.NoError(t, err)

	appItem, err := store.GetItem(ctx, item.ID)
	require.NoError(t, err)
	assert.Equal(t, ItemStatusActive, appItem.Status)
	assert.Equal(t, ver.ID, appItem.ActiveVersionID)

	appVer, err := store.GetSkillVersion(ctx, ver.ID)
	require.NoError(t, err)
	assert.Equal(t, VersionStatusApproved, appVer.Status)
	assert.Equal(t, []string{"run tests", "verify build"}, appVer.Triggers)
	assert.Equal(t, []string{"test", "ci"}, appVer.Tags)

	// Generic GetVersion
	genVer, err := store.GetVersion(ctx, ver.ID)
	require.NoError(t, err)
	assert.Equal(t, KindSkill, genVer.GetKind())
	assert.Equal(t, ver.ID, genVer.GetID())

	// 3. Disable skill
	err = store.DisableSkill(ctx, item.ID, "user_bob")
	require.NoError(t, err)
	disItem, err := store.GetItem(ctx, item.ID)
	require.NoError(t, err)
	assert.Equal(t, ItemStatusDisabled, disItem.Status)

	disabledList, err := store.ListItems(ctx, "chat_dm_2", KindSkill, "disabled")
	require.NoError(t, err)
	require.Len(t, disabledList, 1)

	// 4. Enable skill
	err = store.EnableSkill(ctx, item.ID, "user_bob")
	require.NoError(t, err)
	enItem, err := store.GetItem(ctx, item.ID)
	require.NoError(t, err)
	assert.Equal(t, ItemStatusActive, enItem.Status)

	// 5. Archive skill
	err = store.ArchiveSkill(ctx, item.ID, "user_bob")
	require.NoError(t, err)
	arcItem, err := store.GetItem(ctx, item.ID)
	require.NoError(t, err)
	assert.Equal(t, ItemStatusArchived, arcItem.Status)

	arcVer, err := store.GetSkillVersion(ctx, ver.ID)
	require.NoError(t, err)
	assert.Equal(t, VersionStatusArchived, arcVer.Status)

	// 6. Delete skill -> tombstone and erased versions
	err = store.DeleteSkill(ctx, item.ID, "user_bob")
	require.NoError(t, err)

	delItem, err := store.GetItem(ctx, item.ID)
	require.NoError(t, err)
	assert.Equal(t, ItemStatusForgotten, delItem.Status)
	assert.Empty(t, delItem.ActiveVersionID)

	// Versions are deleted
	_, err = store.GetSkillVersion(ctx, ver.ID)
	assert.ErrorIs(t, err, ErrNotFound)

	// Tombstone is recorded
	ts, err := store.GetTombstone(ctx, item.ID)
	require.NoError(t, err)
	assert.Equal(t, item.ID, ts.ItemID)
	assert.Equal(t, KindSkill, ts.Kind)
	assert.Equal(t, "user_bob", ts.DeletedBy)
}

func TestStore_ForgetItem(t *testing.T) {
	ctx := context.Background()
	store, db := setupTestStore(t)
	defer func() { _ = db.Close() }()

	item, ver, err := store.ProposeMemory(ctx, &KnowledgeItem{
		ChatID: "chat_dm_forget",
		UserID: "user_carol",
	}, &MemoryVersion{
		Type:    MemoryTypeDecision,
		Content: "Use PostgreSQL for metadata",
	})
	require.NoError(t, err)
	require.NoError(t, store.ApproveVersion(ctx, ver.ID, "user_carol"))

	// Forgetting item
	err = store.ForgetItem(ctx, item.ID, "user_carol")
	require.NoError(t, err)

	// Item status is forgotten
	fItem, err := store.GetItem(ctx, item.ID)
	require.NoError(t, err)
	assert.Equal(t, ItemStatusForgotten, fItem.Status)
	assert.Empty(t, fItem.ActiveVersionID)

	// Versions are completely erased
	_, err = store.GetMemoryVersion(ctx, ver.ID)
	assert.ErrorIs(t, err, ErrNotFound)

	versions, err := store.ListMemoryVersions(ctx, item.ID)
	require.NoError(t, err)
	assert.Empty(t, versions)

	// Tombstone is present
	ts, err := store.GetTombstone(ctx, item.ID)
	require.NoError(t, err)
	assert.Equal(t, item.ID, ts.ItemID)
	assert.Equal(t, KindMemory, ts.Kind)
	assert.Equal(t, "user_carol", ts.DeletedBy)
}

func TestStore_IndexCandidates(t *testing.T) {
	ctx := context.Background()
	store, db := setupTestStore(t)
	defer func() { _ = db.Close() }()

	_, ver1, err := store.ProposeMemory(ctx, &KnowledgeItem{
		ChatID: "chat_idx",
		UserID: "user_idx",
	}, &MemoryVersion{
		Type:    MemoryTypeFact,
		Content: "Fact 1",
	})
	require.NoError(t, err)

	_, ver2, err := store.ProposeSkill(ctx, &KnowledgeItem{
		ChatID: "chat_idx",
		UserID: "user_idx",
	}, &SkillVersion{
		Name:                 "skill1",
		Description:          "desc",
		InstructionsMarkdown: "md",
	})
	require.NoError(t, err)

	// Neither is approved yet
	cands, err := store.GetPendingIndexVersions(ctx, 10)
	require.NoError(t, err)
	assert.Empty(t, cands)

	// Approve both
	require.NoError(t, store.ApproveVersion(ctx, ver1.ID, "user_idx"))
	require.NoError(t, store.ApproveVersion(ctx, ver2.ID, "user_idx"))

	cands, err = store.GetPendingIndexVersions(ctx, 10)
	require.NoError(t, err)
	assert.Len(t, cands, 2)

	// Update index status to ready
	err = store.UpdateIndexStatus(ctx, ver1.ID, IndexStatusReady)
	require.NoError(t, err)

	candsAfter, err := store.GetPendingIndexVersions(ctx, 10)
	require.NoError(t, err)
	assert.Len(t, candsAfter, 1)
	assert.Equal(t, ver2.ID, candsAfter[0].VersionID)
}

func TestStore_MarkVersionIndexed_Expiration(t *testing.T) {
	ctx := context.Background()
	store, db := setupTestStore(t)
	defer func() { _ = db.Close() }()

	ttlDays := 1
	_, ver, err := store.ProposeMemory(ctx, &KnowledgeItem{
		ChatID: "chat_exp",
		UserID: "user_exp",
	}, &MemoryVersion{
		Type:    MemoryTypeFact,
		Content: "Expiring fact",
		TTLDays: &ttlDays,
	})
	require.NoError(t, err)
	require.NoError(t, store.ApproveVersion(ctx, ver.ID, "user_exp"))

	// 1. Mark indexed with now < expiresAt -> success
	nowUnix := ver.Provenance.CreatedAt + 100
	require.NoError(t, store.MarkVersionIndexed(ctx, ver.ID, nowUnix))

	// 2. Mark indexed with now >= expiresAt -> fails
	futureUnix := *ver.ExpiresAt + 1
	err = store.MarkVersionIndexed(ctx, ver.ID, futureUnix)
	require.Error(t, err)
	assert.ErrorIs(t, err, ErrInvalidStatus)
}

func TestStore_Authorization(t *testing.T) {
	ctx := context.Background()
	store, db := setupTestStore(t)
	defer func() { _ = db.Close() }()

	// Create memory owned by user_owner
	memItem, memVer, err := store.ProposeMemory(ctx, &KnowledgeItem{
		ChatID: "chat_auth",
		UserID: "user_owner",
	}, &MemoryVersion{
		Type:    MemoryTypeFact,
		Content: "Secret fact",
	})
	require.NoError(t, err)

	// Create skill owned by user_owner
	skillItem, skillVer, err := store.ProposeSkill(ctx, &KnowledgeItem{
		ChatID: "chat_auth",
		UserID: "user_owner",
	}, &SkillVersion{
		Name:                 "test_skill",
		Description:          "desc",
		InstructionsMarkdown: "# instructions",
	})
	require.NoError(t, err)

	// 1. Propose memory revision with unauthorized user
	_, _, err = store.ProposeMemory(ctx, nil, &MemoryVersion{
		ItemID:  memItem.ID,
		Type:    MemoryTypeFact,
		Content: "Tampered fact",
		Provenance: Provenance{
			UserID: "user_intruder",
		},
	})
	assert.ErrorIs(t, err, ErrUnauthorized)

	// 2. Propose skill revision with unauthorized user
	_, _, err = store.ProposeSkill(ctx, nil, &SkillVersion{
		ItemID:               skillItem.ID,
		Name:                 "tampered_skill",
		InstructionsMarkdown: "# tampered",
		Provenance: Provenance{
			UserID: "user_intruder",
		},
	})
	assert.ErrorIs(t, err, ErrUnauthorized)

	// 3. Approve with unauthorized user
	err = store.ApproveVersion(ctx, memVer.ID, "user_intruder")
	assert.ErrorIs(t, err, ErrUnauthorized)

	err = store.ApproveVersion(ctx, skillVer.ID, "user_intruder")
	assert.ErrorIs(t, err, ErrUnauthorized)

	// 4. Reject with unauthorized user
	err = store.RejectVersion(ctx, memVer.ID, "user_intruder")
	assert.ErrorIs(t, err, ErrUnauthorized)

	// Approve both as owner
	require.NoError(t, store.ApproveVersion(ctx, memVer.ID, "user_owner"))
	require.NoError(t, store.ApproveVersion(ctx, skillVer.ID, "user_owner"))

	// 5. Disable skill with unauthorized user
	err = store.DisableSkill(ctx, skillItem.ID, "user_intruder")
	assert.ErrorIs(t, err, ErrUnauthorized)

	// 6. Enable skill with unauthorized user
	require.NoError(t, store.DisableSkill(ctx, skillItem.ID, "user_owner"))
	err = store.EnableSkill(ctx, skillItem.ID, "user_intruder")
	assert.ErrorIs(t, err, ErrUnauthorized)

	// 7. Archive skill with unauthorized user
	err = store.ArchiveSkill(ctx, skillItem.ID, "user_intruder")
	assert.ErrorIs(t, err, ErrUnauthorized)

	// 8. Forget memory with unauthorized user
	err = store.ForgetItem(ctx, memItem.ID, "user_intruder")
	assert.ErrorIs(t, err, ErrUnauthorized)

	// 9. Delete skill with unauthorized user
	err = store.DeleteSkill(ctx, skillItem.ID, "user_intruder")
	assert.ErrorIs(t, err, ErrUnauthorized)
}

func TestStore_InvalidLifecycleTransitions(t *testing.T) {
	ctx := context.Background()
	store, db := setupTestStore(t)
	defer func() { _ = db.Close() }()

	skillItem, skillVer, err := store.ProposeSkill(ctx, &KnowledgeItem{
		ChatID: "chat_life",
		UserID: "user_life",
	}, &SkillVersion{
		Name:                 "life_skill",
		Description:          "desc",
		InstructionsMarkdown: "# instructions",
	})
	require.NoError(t, err)

	// Cannot disable a pending skill
	err = store.DisableSkill(ctx, skillItem.ID, "user_life")
	assert.ErrorIs(t, err, ErrInvalidStatus)

	// Cannot enable a pending skill
	err = store.EnableSkill(ctx, skillItem.ID, "user_life")
	assert.ErrorIs(t, err, ErrInvalidStatus)

	// Approve skill
	require.NoError(t, store.ApproveVersion(ctx, skillVer.ID, "user_life"))

	// Cannot enable an already active skill
	err = store.EnableSkill(ctx, skillItem.ID, "user_life")
	assert.ErrorIs(t, err, ErrInvalidStatus)

	// Disable skill
	require.NoError(t, store.DisableSkill(ctx, skillItem.ID, "user_life"))

	// Cannot disable an already disabled skill
	err = store.DisableSkill(ctx, skillItem.ID, "user_life")
	assert.ErrorIs(t, err, ErrInvalidStatus)

	// Archive skill
	require.NoError(t, store.ArchiveSkill(ctx, skillItem.ID, "user_life"))

	// Cannot archive an already archived skill
	err = store.ArchiveSkill(ctx, skillItem.ID, "user_life")
	assert.ErrorIs(t, err, ErrInvalidStatus)

	// Cannot propose revision to archived skill
	_, _, err = store.ProposeSkill(ctx, nil, &SkillVersion{
		ItemID:               skillItem.ID,
		Name:                 "new_rev",
		InstructionsMarkdown: "# new instructions",
	})
	assert.ErrorIs(t, err, ErrInvalidStatus)

	// Delete skill
	require.NoError(t, store.DeleteSkill(ctx, skillItem.ID, "user_life"))

	// Cannot delete an already deleted skill
	err = store.DeleteSkill(ctx, skillItem.ID, "user_life")
	assert.ErrorIs(t, err, ErrInvalidStatus)

	// Cannot propose revision to deleted skill
	_, _, err = store.ProposeSkill(ctx, nil, &SkillVersion{
		ItemID:               skillItem.ID,
		Name:                 "rev_after_delete",
		InstructionsMarkdown: "# instructions",
	})
	assert.ErrorIs(t, err, ErrInvalidStatus)

	// Memory forgotten transitions
	memItem, memVer, err := store.ProposeMemory(ctx, &KnowledgeItem{
		ChatID: "chat_life",
		UserID: "user_life",
	}, &MemoryVersion{
		Type:    MemoryTypePreference,
		Content: "Coffee with milk",
	})
	require.NoError(t, err)
	require.NoError(t, store.ApproveVersion(ctx, memVer.ID, "user_life"))

	require.NoError(t, store.ForgetItem(ctx, memItem.ID, "user_life"))

	// Cannot forget already forgotten memory
	err = store.ForgetItem(ctx, memItem.ID, "user_life")
	assert.ErrorIs(t, err, ErrInvalidStatus)

	// Cannot propose revision to forgotten memory
	_, _, err = store.ProposeMemory(ctx, nil, &MemoryVersion{
		ItemID:  memItem.ID,
		Type:    MemoryTypePreference,
		Content: "Black coffee",
	})
	assert.ErrorIs(t, err, ErrInvalidStatus)

	// Cannot approve or reject version on forgotten memory
	err = store.ApproveVersion(ctx, memVer.ID, "user_life")
	assert.ErrorIs(t, err, ErrInvalidStatus)

	err = store.RejectVersion(ctx, memVer.ID, "user_life")
	assert.ErrorIs(t, err, ErrInvalidStatus)
}

func TestStore_Validation(t *testing.T) {
	ctx := context.Background()
	store, db := setupTestStore(t)
	defer func() { _ = db.Close() }()

	// Memory confidence bounds
	_, _, err := store.ProposeMemory(ctx, &KnowledgeItem{
		ChatID: "chat_v",
		UserID: "user_v",
	}, &MemoryVersion{
		Type:       MemoryTypeFact,
		Content:    "fact",
		Confidence: -0.1,
	})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "confidence must be between 0.0 and 1.0")

	_, _, err = store.ProposeMemory(ctx, &KnowledgeItem{
		ChatID: "chat_v",
		UserID: "user_v",
	}, &MemoryVersion{
		Type:       MemoryTypeFact,
		Content:    "fact",
		Confidence: 1.05,
	})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "confidence must be between 0.0 and 1.0")

	// Memory TTL <= 0
	invalidTTL := 0
	_, _, err = store.ProposeMemory(ctx, &KnowledgeItem{
		ChatID: "chat_v",
		UserID: "user_v",
	}, &MemoryVersion{
		Type:    MemoryTypeFact,
		Content: "fact",
		TTLDays: &invalidTTL,
	})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "ttl_days must be positive")

	negativeTTL := -5
	_, _, err = store.ProposeMemory(ctx, &KnowledgeItem{
		ChatID: "chat_v",
		UserID: "user_v",
	}, &MemoryVersion{
		Type:    MemoryTypeFact,
		Content: "fact",
		TTLDays: &negativeTTL,
	})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "ttl_days must be positive")

	// Memory empty content
	_, _, err = store.ProposeMemory(ctx, &KnowledgeItem{
		ChatID: "chat_v",
		UserID: "user_v",
	}, &MemoryVersion{
		Type:    MemoryTypeFact,
		Content: "   ",
	})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "memory content cannot be empty")

	// Memory invalid type
	_, _, err = store.ProposeMemory(ctx, &KnowledgeItem{
		ChatID: "chat_v",
		UserID: "user_v",
	}, &MemoryVersion{
		Type:    "unsupported_type",
		Content: "fact",
	})
	assert.ErrorIs(t, err, ErrInvalidStatus)

	// Skill empty name
	_, _, err = store.ProposeSkill(ctx, &KnowledgeItem{
		ChatID: "chat_v",
		UserID: "user_v",
	}, &SkillVersion{
		Name:                 "   ",
		InstructionsMarkdown: "# instructions",
	})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "skill name cannot be empty")

	// Skill empty instructions
	_, _, err = store.ProposeSkill(ctx, &KnowledgeItem{
		ChatID: "chat_v",
		UserID: "user_v",
	}, &SkillVersion{
		Name:                 "skill_name",
		InstructionsMarkdown: "   ",
	})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "skill instructions_markdown cannot be empty")

	// UpdateIndexStatus invalid status
	err = store.UpdateIndexStatus(ctx, "mem_xyz@1", "invalid_status")
	assert.ErrorIs(t, err, ErrInvalidStatus)

	// UpdateIndexStatus nonexistent version
	err = store.UpdateIndexStatus(ctx, "mem_0000000000000000@1", IndexStatusReady)
	assert.ErrorIs(t, err, ErrNotFound)

	// ListItems unknown filter
	_, err = store.ListItems(ctx, "chat_v", KindMemory, "nonexistent_filter")
	assert.ErrorIs(t, err, ErrInvalidStatus)

	// Memory invalid custom ID prefix
	_, _, err = store.ProposeMemory(ctx, &KnowledgeItem{
		ID:     "custom_id_without_prefix",
		ChatID: "chat_v",
		UserID: "user_v",
	}, &MemoryVersion{
		Type:    MemoryTypeFact,
		Content: "valid fact",
	})
	assert.ErrorIs(t, err, ErrInvalidStatus)

	// Skill invalid custom ID prefix
	_, _, err = store.ProposeSkill(ctx, &KnowledgeItem{
		ID:     "custom_id_without_prefix",
		ChatID: "chat_v",
		UserID: "user_v",
	}, &SkillVersion{
		Name:                 "skill_name",
		InstructionsMarkdown: "valid instruction",
	})
	assert.ErrorIs(t, err, ErrInvalidStatus)
}

func TestStore_ListFilteringAndExpiry(t *testing.T) {
	ctx := context.Background()
	store, db := setupTestStore(t)
	defer func() { _ = db.Close() }()

	// Propose and approve active non-expired memory
	validTTL := 10
	_, verValid, err := store.ProposeMemory(ctx, &KnowledgeItem{
		ChatID: "chat_filter",
		UserID: "user_filter",
	}, &MemoryVersion{
		Type:    MemoryTypeFact,
		Content: "Valid fact",
		TTLDays: &validTTL,
	})
	require.NoError(t, err)
	require.NoError(t, store.ApproveVersion(ctx, verValid.ID, "user_filter"))

	// Propose and approve memory, then artificially expire it in the DB
	_, verExpired, err := store.ProposeMemory(ctx, &KnowledgeItem{
		ChatID: "chat_filter",
		UserID: "user_filter",
	}, &MemoryVersion{
		Type:    MemoryTypeFact,
		Content: "Expired fact",
		TTLDays: &validTTL,
	})
	require.NoError(t, err)
	require.NoError(t, store.ApproveVersion(ctx, verExpired.ID, "user_filter"))

	// Artificially expire verExpired
	pastTime := int64(1000)
	_, err = db.ExecContext(ctx, "UPDATE memory_versions SET expires_at = ? WHERE id = ?", pastTime, verExpired.ID)
	require.NoError(t, err)

	// List active memories: should contain verValid but NOT verExpired
	activeMems, err := store.ListItems(ctx, "chat_filter", KindMemory, "active")
	require.NoError(t, err)
	require.Len(t, activeMems, 1)
	assert.Equal(t, verValid.ItemID, activeMems[0].ID)

	// List expired memories: should contain verExpired
	expiredMems, err := store.ListItems(ctx, "chat_filter", KindMemory, "expired")
	require.NoError(t, err)
	require.Len(t, expiredMems, 1)
	assert.Equal(t, verExpired.ItemID, expiredMems[0].ID)

	// Pending index versions: should exclude expired memory
	cands, err := store.GetPendingIndexVersions(ctx, 50)
	require.NoError(t, err)
	require.Len(t, cands, 1)
	assert.Equal(t, verValid.ID, cands[0].VersionID)

	// Propose, approve, then disable a skill: should not be in pending index versions
	skillItem, skillVer, err := store.ProposeSkill(ctx, &KnowledgeItem{
		ChatID: "chat_filter",
		UserID: "user_filter",
	}, &SkillVersion{
		Name:                 "skill_temp",
		Description:          "desc",
		InstructionsMarkdown: "content",
	})
	require.NoError(t, err)
	require.NoError(t, store.ApproveVersion(ctx, skillVer.ID, "user_filter"))

	cands, err = store.GetPendingIndexVersions(ctx, 50)
	require.NoError(t, err)
	assert.Len(t, cands, 2)

	// Disable the skill
	require.NoError(t, store.DisableSkill(ctx, skillItem.ID, "user_filter"))

	// Should now be excluded because skill is not active
	cands, err = store.GetPendingIndexVersions(ctx, 50)
	require.NoError(t, err)
	require.Len(t, cands, 1)
	assert.Equal(t, verValid.ID, cands[0].VersionID)
}

func TestStore_ActorValidation_RejectsEmptyAndWhitespace(t *testing.T) {
	ctx := context.Background()
	store, rawDB := setupTestStore(t)
	defer func() { _ = rawDB.Close() }()

	// Propose memory with empty/whitespace actor
	_, _, err := store.ProposeMemory(ctx, &KnowledgeItem{
		ChatID: "chat_1",
		UserID: "   ",
	}, &MemoryVersion{
		Type:    MemoryTypeFact,
		Content: "Valid content",
	})
	assert.ErrorIs(t, err, ErrUnauthorized)

	// Propose skill with empty actor
	_, _, err = store.ProposeSkill(ctx, &KnowledgeItem{
		ChatID: "chat_1",
		UserID: "",
	}, &SkillVersion{
		Name:                 "skill_1",
		InstructionsMarkdown: "# instructions",
	})
	assert.ErrorIs(t, err, ErrUnauthorized)

	// Valid creation
	memItem, memVer, err := store.ProposeMemory(ctx, &KnowledgeItem{
		ChatID: "chat_1",
		UserID: "user_valid",
	}, &MemoryVersion{
		Type:    MemoryTypeFact,
		Content: "Valid content",
	})
	require.NoError(t, err)

	// Approve with empty/whitespace actor
	assert.ErrorIs(t, store.ApproveVersion(ctx, memVer.ID, ""), ErrUnauthorized)
	assert.ErrorIs(t, store.ApproveVersion(ctx, memVer.ID, "   "), ErrUnauthorized)

	// Reject with empty/whitespace actor
	assert.ErrorIs(t, store.RejectVersion(ctx, memVer.ID, ""), ErrUnauthorized)

	// Approve validly
	require.NoError(t, store.ApproveVersion(ctx, memVer.ID, "user_valid"))

	// Propose revision with empty actor in provenance
	_, _, err = store.ProposeMemory(ctx, nil, &MemoryVersion{
		ItemID:  memItem.ID,
		Type:    MemoryTypeFact,
		Content: "Revision without user",
		Provenance: Provenance{
			UserID: "  ",
			ChatID: "chat_1",
		},
	})
	assert.ErrorIs(t, err, ErrUnauthorized)

	// Propose valid skill
	skillItem, skillVer, err := store.ProposeSkill(ctx, &KnowledgeItem{
		ChatID: "chat_1",
		UserID: "user_valid",
	}, &SkillVersion{
		Name:                 "skill_1",
		InstructionsMarkdown: "# instructions",
	})
	require.NoError(t, err)
	require.NoError(t, store.ApproveVersion(ctx, skillVer.ID, "user_valid"))

	// Disable with empty actor
	assert.ErrorIs(t, store.DisableSkill(ctx, skillItem.ID, ""), ErrUnauthorized)
	assert.ErrorIs(t, store.DisableSkill(ctx, skillItem.ID, "  "), ErrUnauthorized)

	// Enable with empty actor
	require.NoError(t, store.DisableSkill(ctx, skillItem.ID, "user_valid"))
	assert.ErrorIs(t, store.EnableSkill(ctx, skillItem.ID, ""), ErrUnauthorized)

	// Archive with empty actor
	assert.ErrorIs(t, store.ArchiveSkill(ctx, skillItem.ID, " "), ErrUnauthorized)

	// Forget with empty actor
	assert.ErrorIs(t, store.ForgetItem(ctx, memItem.ID, ""), ErrUnauthorized)

	// Delete with empty actor
	assert.ErrorIs(t, store.DeleteSkill(ctx, skillItem.ID, "  "), ErrUnauthorized)
}

func TestStore_RejectionLifecycle(t *testing.T) {
	ctx := context.Background()
	store, rawDB := setupTestStore(t)
	defer func() { _ = rawDB.Close() }()

	// 1. Propose memory item -> reject sole proposal -> item status becomes 'rejected'
	item1, ver1, err := store.ProposeMemory(ctx, &KnowledgeItem{
		ChatID: "chat_rej",
		UserID: "user_rej",
	}, &MemoryVersion{
		Type:    MemoryTypePreference,
		Content: "Likes apples",
	})
	require.NoError(t, err)
	assert.Equal(t, ItemStatusPending, item1.Status)

	require.NoError(t, store.RejectVersion(ctx, ver1.ID, "user_rej"))

	item1After, err := store.GetItem(ctx, item1.ID)
	require.NoError(t, err)
	assert.Equal(t, ItemStatusRejected, item1After.Status, "rejecting sole proposed version must mark item as rejected")

	// 2. Propose revision on rejected item (no active version) -> item transitions back to pending
	_, ver1Rev2, err := store.ProposeMemory(ctx, nil, &MemoryVersion{
		ItemID:  item1.ID,
		Type:    MemoryTypePreference,
		Content: "Likes green apples",
		Provenance: Provenance{
			UserID: "user_rej",
			ChatID: "chat_rej",
		},
	})
	require.NoError(t, err)
	assert.Equal(t, 2, ver1Rev2.Revision)

	item1Reproposed, err := store.GetItem(ctx, item1.ID)
	require.NoError(t, err)
	assert.Equal(t, ItemStatusPending, item1Reproposed.Status, "proposing revision on rejected item must transition item back to pending")

	// 3. Propose two revisions on pending item: rejecting one leaves item pending
	_, ver1Rev3, err := store.ProposeMemory(ctx, nil, &MemoryVersion{
		ItemID:  item1.ID,
		Type:    MemoryTypePreference,
		Content: "Likes honeycrisp apples",
		Provenance: Provenance{
			UserID: "user_rej",
			ChatID: "chat_rej",
		},
	})
	require.NoError(t, err)

	require.NoError(t, store.RejectVersion(ctx, ver1Rev2.ID, "user_rej"))
	itemWithOneRemaining, err := store.GetItem(ctx, item1.ID)
	require.NoError(t, err)
	assert.Equal(t, ItemStatusPending, itemWithOneRemaining.Status, "item must remain pending while another proposal exists")

	// Approve remaining proposal -> item becomes active
	require.NoError(t, store.ApproveVersion(ctx, ver1Rev3.ID, "user_rej"))
	itemActive, err := store.GetItem(ctx, item1.ID)
	require.NoError(t, err)
	assert.Equal(t, ItemStatusActive, itemActive.Status)

	// 4. Propose revision to active item -> reject revision -> item remains active
	_, ver1Rev4, err := store.ProposeMemory(ctx, nil, &MemoryVersion{
		ItemID:  item1.ID,
		Type:    MemoryTypePreference,
		Content: "Dislikes sour apples",
		Provenance: Provenance{
			UserID: "user_rej",
			ChatID: "chat_rej",
		},
	})
	require.NoError(t, err)

	require.NoError(t, store.RejectVersion(ctx, ver1Rev4.ID, "user_rej"))
	itemStillActive, err := store.GetItem(ctx, item1.ID)
	require.NoError(t, err)
	assert.Equal(t, ItemStatusActive, itemStillActive.Status, "rejecting revision of active item must not change active item status")
	assert.Equal(t, ver1Rev3.ID, itemStillActive.ActiveVersionID)
}

func TestStore_ApproveGuards_AgainstArchivedOrForgotten(t *testing.T) {
	ctx := context.Background()
	store, rawDB := setupTestStore(t)
	defer func() { _ = rawDB.Close() }()

	skillItem, skillVer1, err := store.ProposeSkill(ctx, &KnowledgeItem{
		ChatID: "chat_guard",
		UserID: "user_guard",
	}, &SkillVersion{
		Name:                 "skill_guard",
		InstructionsMarkdown: "# instructions",
	})
	require.NoError(t, err)
	require.NoError(t, store.ApproveVersion(ctx, skillVer1.ID, "user_guard"))

	// Propose revision 2
	_, skillVer2, err := store.ProposeSkill(ctx, nil, &SkillVersion{
		ItemID:               skillItem.ID,
		Name:                 "skill_guard",
		InstructionsMarkdown: "# updated instructions",
		Provenance: Provenance{
			UserID: "user_guard",
			ChatID: "chat_guard",
		},
	})
	require.NoError(t, err)

	// Archive the skill while revision 2 is pending
	require.NoError(t, store.ArchiveSkill(ctx, skillItem.ID, "user_guard"))

	// Attempting to approve revision 2 must be rejected
	err = store.ApproveVersion(ctx, skillVer2.ID, "user_guard")
	assert.ErrorIs(t, err, ErrInvalidStatus, "cannot approve revision for archived skill")

	// Verify skill item remains archived
	itemAfter, err := store.GetItem(ctx, skillItem.ID)
	require.NoError(t, err)
	assert.Equal(t, ItemStatusArchived, itemAfter.Status)
}

func TestStore_AtomicIndexDeletionQueues(t *testing.T) {
	ctx := context.Background()
	store, rawDB := setupTestStore(t)
	defer func() { _ = rawDB.Close() }()

	// 1. Approve revision superseding previous active version
	memItem, memVer1, err := store.ProposeMemory(ctx, &KnowledgeItem{
		ChatID: "chat_del_queue",
		UserID: "user_del_queue",
	}, &MemoryVersion{
		Type:    MemoryTypeFact,
		Content: "Fact v1",
	})
	require.NoError(t, err)
	require.NoError(t, store.ApproveVersion(ctx, memVer1.ID, "user_del_queue"))

	// Propose and approve v2
	_, memVer2, err := store.ProposeMemory(ctx, nil, &MemoryVersion{
		ItemID:  memItem.ID,
		Type:    MemoryTypeFact,
		Content: "Fact v2",
		Provenance: Provenance{
			UserID: "user_del_queue",
			ChatID: "chat_del_queue",
		},
	})
	require.NoError(t, err)
	require.NoError(t, store.ApproveVersion(ctx, memVer2.ID, "user_del_queue"))

	// memVer1 must be atomically queued in pending_index_deletions
	dels, err := store.GetPendingIndexDeletions(ctx, 50)
	require.NoError(t, err)
	require.Len(t, dels, 1)
	assert.Equal(t, memVer1.ID, dels[0].VersionID)
	assert.Equal(t, NamespaceMemories, dels[0].Namespace)

	// Clean queue
	require.NoError(t, store.RemovePendingIndexDeletion(ctx, memVer1.ID))

	// 2. DisableSkill enqueues active version
	skillItem, skillVer1, err := store.ProposeSkill(ctx, &KnowledgeItem{
		ChatID: "chat_del_queue",
		UserID: "user_del_queue",
	}, &SkillVersion{
		Name:                 "skill_del_test",
		InstructionsMarkdown: "# instructions",
	})
	require.NoError(t, err)
	require.NoError(t, store.ApproveVersion(ctx, skillVer1.ID, "user_del_queue"))

	require.NoError(t, store.DisableSkill(ctx, skillItem.ID, "user_del_queue"))
	dels, err = store.GetPendingIndexDeletions(ctx, 50)
	require.NoError(t, err)
	require.Len(t, dels, 1)
	assert.Equal(t, skillVer1.ID, dels[0].VersionID)
	assert.Equal(t, NamespaceSkills, dels[0].Namespace)

	// 3. EnableSkill removes from pending_index_deletions and marks index_status pending
	require.NoError(t, store.EnableSkill(ctx, skillItem.ID, "user_del_queue"))
	dels, err = store.GetPendingIndexDeletions(ctx, 50)
	require.NoError(t, err)
	assert.Empty(t, dels, "EnableSkill must cancel pending index deletion")

	sv, err := store.GetSkillVersion(ctx, skillVer1.ID)
	require.NoError(t, err)
	assert.Equal(t, IndexStatusPending, sv.IndexStatus, "EnableSkill must mark active version index_status as pending")

	// 4. ArchiveSkill enqueues active version
	require.NoError(t, store.ArchiveSkill(ctx, skillItem.ID, "user_del_queue"))
	dels, err = store.GetPendingIndexDeletions(ctx, 50)
	require.NoError(t, err)
	require.Len(t, dels, 1)
	assert.Equal(t, skillVer1.ID, dels[0].VersionID)

	// Clean queue
	require.NoError(t, store.RemovePendingIndexDeletion(ctx, skillVer1.ID))

	// 5. ForgetItem enqueues all versions
	require.NoError(t, store.ForgetItem(ctx, memItem.ID, "user_del_queue"))
	dels, err = store.GetPendingIndexDeletions(ctx, 50)
	require.NoError(t, err)
	// memItem has memVer1 and memVer2
	require.Len(t, dels, 2)
}

func TestStore_AtomicIndexDeletionQueues_RollbackOnQueueFailure(t *testing.T) {
	ctx := context.Background()
	store, rawDB := setupTestStore(t)
	defer func() { _ = rawDB.Close() }()

	skillItem, skillVer, err := store.ProposeSkill(ctx, &KnowledgeItem{
		ChatID: "chat_rollback",
		UserID: "user_rollback",
	}, &SkillVersion{
		Name:                 "skill_rb_test",
		InstructionsMarkdown: "# instructions",
	})
	require.NoError(t, err)
	require.NoError(t, store.ApproveVersion(ctx, skillVer.ID, "user_rollback"))

	// Attach abort trigger on pending_index_deletions
	_, err = rawDB.Exec(`
		CREATE TRIGGER fail_pending_insert
		BEFORE INSERT ON pending_index_deletions
		BEGIN
			SELECT RAISE(ABORT, 'forced pending index deletion failure');
		END;
	`)
	require.NoError(t, err)

	// Attempt DisableSkill: should fail due to trigger and rollback completely
	err = store.DisableSkill(ctx, skillItem.ID, "user_rollback")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "forced pending index deletion failure")

	// State must still be active
	item, err := store.GetItem(ctx, skillItem.ID)
	require.NoError(t, err)
	assert.Equal(t, ItemStatusActive, item.Status, "DisableSkill state must be rolled back on queue error")

	// Drop trigger and verify DisableSkill succeeds
	_, err = rawDB.Exec("DROP TRIGGER fail_pending_insert")
	require.NoError(t, err)
	require.NoError(t, store.DisableSkill(ctx, skillItem.ID, "user_rollback"))
}
