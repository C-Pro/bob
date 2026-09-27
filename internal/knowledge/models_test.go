package knowledge

import (
	"crypto/sha256"
	"encoding/hex"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestGenerateItemID(t *testing.T) {
	memID1, err := GenerateItemID(KindMemory)
	require.NoError(t, err)
	memID2, err := GenerateItemID(KindMemory)
	require.NoError(t, err)
	assert.True(t, strings.HasPrefix(memID1, "mem_"), "memory ID must start with mem_")
	assert.True(t, strings.HasPrefix(memID2, "mem_"), "memory ID must start with mem_")
	assert.NotEqual(t, memID1, memID2, "generated memory IDs must be distinct")

	skillID1, err := GenerateItemID(KindSkill)
	require.NoError(t, err)
	skillID2, err := GenerateItemID(KindSkill)
	require.NoError(t, err)
	assert.True(t, strings.HasPrefix(skillID1, "skill_"), "skill ID must start with skill_")
	assert.True(t, strings.HasPrefix(skillID2, "skill_"), "skill ID must start with skill_")
	assert.NotEqual(t, skillID1, skillID2, "generated skill IDs must be distinct")

	_, err = GenerateItemID("invalid_kind")
	assert.Error(t, err, "invalid item kind should return error")
}

func TestFormatAndParseVersionID(t *testing.T) {
	formatted := FormatVersionID("mem_12345", 2)
	assert.Equal(t, "mem_12345@2", formatted)

	itemID, rev, err := ParseVersionID("mem_12345@2")
	require.NoError(t, err)
	assert.Equal(t, "mem_12345", itemID)
	assert.Equal(t, 2, rev)

	// Invalid cases
	_, _, err = ParseVersionID("mem_12345")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "must be formatted as <item_id>@<revision>")

	_, _, err = ParseVersionID("mem_12345@0")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "must be a positive integer")

	_, _, err = ParseVersionID("mem_12345@-1")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "must be a positive integer")

	_, _, err = ParseVersionID("mem_12345@abc")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "must be a positive integer")

	_, _, err = ParseVersionID("@2")
	assert.Error(t, err)

	_, _, err = ParseVersionID("mem_12345@")
	assert.Error(t, err)
}

func TestParseItemOrVersionRef(t *testing.T) {
	// Item ref without revision
	itemID, rev, hasRev, err := ParseItemOrVersionRef("mem_abcdef")
	require.NoError(t, err)
	assert.Equal(t, "mem_abcdef", itemID)
	assert.Equal(t, 0, rev)
	assert.False(t, hasRev)

	// Version ref with revision
	itemID, rev, hasRev, err = ParseItemOrVersionRef("mem_abcdef@3")
	require.NoError(t, err)
	assert.Equal(t, "mem_abcdef", itemID)
	assert.Equal(t, 3, rev)
	assert.True(t, hasRev)

	// Whitespace tolerance
	itemID, rev, hasRev, err = ParseItemOrVersionRef("  skill_xyz@1  ")
	require.NoError(t, err)
	assert.Equal(t, "skill_xyz", itemID)
	assert.Equal(t, 1, rev)
	assert.True(t, hasRev)

	// Empty string
	_, _, _, err = ParseItemOrVersionRef("   ")
	assert.Error(t, err)

	// Invalid version suffix
	_, _, _, err = ParseItemOrVersionRef("skill_xyz@0")
	assert.Error(t, err)
}

func TestComputeContentHash(t *testing.T) {
	content := "User prefers dark mode and short answers"
	expectedSum := sha256.Sum256([]byte(content))
	expectedHex := hex.EncodeToString(expectedSum[:])

	hash := ComputeContentHash(content)
	assert.Equal(t, expectedHex, hash)
	assert.Len(t, hash, 64)
}

func TestEnumValidations(t *testing.T) {
	// MemoryType
	assert.True(t, IsValidMemoryType("preference"))
	assert.True(t, IsValidMemoryType("fact"))
	assert.True(t, IsValidMemoryType("decision"))
	assert.True(t, IsValidMemoryType("ongoing_task"))
	assert.False(t, IsValidMemoryType("custom"))
	assert.False(t, IsValidMemoryType(""))

	// ItemStatus
	assert.True(t, IsValidItemStatus("pending"))
	assert.True(t, IsValidItemStatus("active"))
	assert.True(t, IsValidItemStatus("disabled"))
	assert.True(t, IsValidItemStatus("archived"))
	assert.True(t, IsValidItemStatus("forgotten"))
	assert.False(t, IsValidItemStatus("unknown"))

	// VersionStatus
	assert.True(t, IsValidVersionStatus("proposed"))
	assert.True(t, IsValidVersionStatus("approved"))
	assert.True(t, IsValidVersionStatus("rejected"))
	assert.True(t, IsValidVersionStatus("superseded"))
	assert.True(t, IsValidVersionStatus("expired"))
	assert.True(t, IsValidVersionStatus("archived"))
	assert.False(t, IsValidVersionStatus("active"))

	// IndexStatus
	assert.True(t, IsValidIndexStatus("pending"))
	assert.True(t, IsValidIndexStatus("ready"))
	assert.True(t, IsValidIndexStatus("error"))
	assert.False(t, IsValidIndexStatus("indexing"))
}

func TestVersion_Methods(t *testing.T) {
	// Nil checks
	var nilMem *MemoryVersion
	assert.Equal(t, "", nilMem.SearchableContent())
	assert.Empty(t, nilMem.IndexMetadata())

	var nilSkill *SkillVersion
	assert.Equal(t, "", nilSkill.SearchableContent())
	assert.Empty(t, nilSkill.IndexMetadata())

	// Memory methods
	mem := &MemoryVersion{
		ID:         "mem_123@1",
		ItemID:     "mem_123",
		Type:       MemoryTypeFact,
		Content:    "Go 1.24 is installed",
		Confidence: 0.98,
		Provenance: Provenance{
			ChatID: "chat_1",
			UserID: "user_1",
		},
	}
	assert.Equal(t, "[fact] Go 1.24 is installed", mem.SearchableContent())
	memMeta := mem.IndexMetadata()
	assert.Equal(t, "mem_123", memMeta["item_id"])
	assert.Equal(t, "mem_123@1", memMeta["version_id"])
	assert.Equal(t, "memory", memMeta["kind"])
	assert.Equal(t, "memory", memMeta["knowledge_kind"])
	assert.Equal(t, "chat_1", memMeta["chat_id"])
	assert.Equal(t, "user_1", memMeta["user_id"])
	assert.Equal(t, "fact", memMeta["type"])
	assert.Equal(t, 0.98, memMeta["confidence"])

	// Skill methods
	skill := &SkillVersion{
		ID:                   "skill_abc@1",
		ItemID:               "skill_abc",
		Name:                 "run_tests",
		Description:          "runs tests with race detector",
		Triggers:             []string{"test", "check"},
		Tags:                 []string{"ci", "unit"},
		InstructionsMarkdown: "go test -race ./...",
		Provenance: Provenance{
			ChatID: "chat_2",
			UserID: "user_2",
		},
	}
	skillContent := skill.SearchableContent()
	assert.Contains(t, skillContent, "Skill: run_tests")
	assert.Contains(t, skillContent, "Description: runs tests with race detector")
	assert.Contains(t, skillContent, "Triggers: test, check")
	assert.Contains(t, skillContent, "Tags: ci, unit")
	assert.Contains(t, skillContent, "Instructions:\ngo test -race ./...")

	skillMeta := skill.IndexMetadata()
	assert.Equal(t, "skill_abc", skillMeta["item_id"])
	assert.Equal(t, "skill_abc@1", skillMeta["version_id"])
	assert.Equal(t, "skill", skillMeta["kind"])
	assert.Equal(t, "skill", skillMeta["knowledge_kind"])
	assert.Equal(t, "chat_2", skillMeta["chat_id"])
	assert.Equal(t, "user_2", skillMeta["user_id"])
	assert.Equal(t, "run_tests", skillMeta["name"])
	assert.Equal(t, "runs tests with race detector", skillMeta["description"])
	assert.Equal(t, []string{"test", "check"}, skillMeta["triggers"])
	assert.Equal(t, []string{"ci", "unit"}, skillMeta["tags"])
}
