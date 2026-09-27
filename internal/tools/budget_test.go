package tools

import (
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestKnowledgeBudgetTracker_MemoryLimits(t *testing.T) {
	tracker := NewKnowledgeBudgetTracker(KnowledgeBudgetLimits{
		MaxLoadedMemories:    3,
		MaxLoadedMemoryBytes: 100,
	})

	// 1. Charge memory within budget
	require.NoError(t, tracker.ChargeMemory("mem_1", "mem_1@1", "hello world"))
	memCount, memBytes, _, _ := tracker.Stats()
	assert.Equal(t, 1, memCount)
	assert.Equal(t, 11, memBytes)

	// 2. Duplicate charge is idempotent
	require.NoError(t, tracker.ChargeMemory("mem_1", "mem_1@1", "different content ignored"))
	memCount, memBytes, _, _ = tracker.Stats()
	assert.Equal(t, 1, memCount)
	assert.Equal(t, 11, memBytes)

	// 3. Second and third memories succeed
	require.NoError(t, tracker.ChargeMemory("mem_2", "mem_2@1", "second memory"))
	require.NoError(t, tracker.ChargeMemory("mem_3", "mem_3@1", "third memory"))
	memCount, _, _, _ = tracker.Stats()
	assert.Equal(t, 3, memCount)

	// 4. Fourth memory exceeds count limit
	err := tracker.ChargeMemory("mem_4", "mem_4@1", "fourth")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "count limit 3 reached")

	// 5. Byte limit enforcement
	byteTracker := NewKnowledgeBudgetTracker(KnowledgeBudgetLimits{
		MaxLoadedMemories:    10,
		MaxLoadedMemoryBytes: 20,
	})
	require.NoError(t, byteTracker.ChargeMemory("mem_1", "mem_1@1", "1234567890")) // 10 bytes
	err = byteTracker.ChargeMemory("mem_2", "mem_2@1", "123456789012")             // 12 bytes -> 22 > 20
	require.Error(t, err)
	assert.Contains(t, err.Error(), "byte limit 20 bytes exceeded")
}

func TestKnowledgeBudgetTracker_SkillLimits(t *testing.T) {
	tracker := NewKnowledgeBudgetTracker(KnowledgeBudgetLimits{
		MaxLoadedSkills:     2,
		MaxLoadedSkillBytes: 50,
	})

	// 1. Charge skills
	require.NoError(t, tracker.ChargeSkill("skill_1", "skill_1@1", "instructions one"))
	_, _, skillCount, skillBytes := tracker.Stats()
	assert.Equal(t, 1, skillCount)
	assert.Equal(t, 16, skillBytes)

	// 2. Duplicate charge is idempotent
	require.NoError(t, tracker.ChargeSkill("skill_1", "skill_1@1", "ignored"))
	_, _, skillCount, skillBytes = tracker.Stats()
	assert.Equal(t, 1, skillCount)
	assert.Equal(t, 16, skillBytes)

	// 3. Second skill succeeds
	require.NoError(t, tracker.ChargeSkill("skill_2", "skill_2@1", "instructions two"))

	// 4. Third skill exceeds count limit
	err := tracker.ChargeSkill("skill_3", "skill_3@1", "three")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "count limit 2 reached")

	// 5. Byte limit enforcement
	byteTracker := NewKnowledgeBudgetTracker(KnowledgeBudgetLimits{
		MaxLoadedSkills:     5,
		MaxLoadedSkillBytes: 30,
	})
	require.NoError(t, byteTracker.ChargeSkill("skill_1", "skill_1@1", strings.Repeat("a", 20)))
	err = byteTracker.ChargeSkill("skill_2", "skill_2@1", strings.Repeat("b", 15))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "byte limit 30 bytes exceeded")
}

func TestKnowledgeBudgetTracker_CachingAndRestoration(t *testing.T) {
	tracker := NewKnowledgeBudgetTracker(DefaultKnowledgeBudgetLimits())

	// Charge and set cached result
	require.NoError(t, tracker.ChargeMemory("mem_a", "mem_a@1", "content a"))
	tracker.SetCachedResult("mem_a", `{"item_id":"mem_a","type":"fact"}`)

	res, ok := tracker.GetCachedResult("mem_a")
	assert.True(t, ok)
	assert.Equal(t, `{"item_id":"mem_a","type":"fact"}`, res)

	vID, content, bytes, ok := tracker.GetCached("mem_a")
	assert.True(t, ok)
	assert.Equal(t, "mem_a@1", vID)
	assert.Equal(t, "content a", content)
	assert.Equal(t, 9, bytes)

	// RestoreItem from crash recovery
	require.NoError(t, tracker.RestoreItem("skill", "skill_restored", "skill_restored@1", "proc instructions", 17, `{"item_id":"skill_restored"}`))
	_, _, skillCount, skillBytes := tracker.Stats()
	assert.Equal(t, 1, skillCount)
	assert.Equal(t, 17, skillBytes)

	res, ok = tracker.GetCachedResult("skill_restored")
	assert.True(t, ok)
	assert.Equal(t, `{"item_id":"skill_restored"}`, res)

	// Attempting to charge restored item is a no-op
	require.NoError(t, tracker.ChargeSkill("skill_restored", "skill_restored@1", "proc instructions"))
	_, _, skillCount, skillBytes = tracker.Stats()
	assert.Equal(t, 1, skillCount)
	assert.Equal(t, 17, skillBytes)
}

func TestKnowledgeBudgetTracker_RestoreValidation(t *testing.T) {
	tracker := NewKnowledgeBudgetTracker(DefaultKnowledgeBudgetLimits())

	// Unknown kind
	err := tracker.RestoreItem("unknown", "item_1", "item_1@1", "content", 7)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "invalid item kind")

	// Empty item ID
	err = tracker.RestoreItem("memory", "", "item_1@1", "content", 7)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "empty item_id")

	// Empty version ID
	err = tracker.RestoreItem("memory", "mem_1", "", "content", 7)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "empty version_id")

	// Empty content
	err = tracker.RestoreItem("memory", "mem_1", "mem_1@1", "", 0)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "empty content")

	// Memory content_bytes mismatch
	err = tracker.RestoreItem("memory", "mem_1", "mem_1@1", "actual content", 999)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "content_bytes mismatch")

	// Skill content_bytes less than instructions length
	err = tracker.RestoreItem("skill", "skill_1", "skill_1@1", "actual instructions", 5)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "content_bytes mismatch")
}

func TestKnowledgeBudgetTracker_Concurrency(t *testing.T) {
	tracker := NewKnowledgeBudgetTracker(KnowledgeBudgetLimits{
		MaxLoadedMemories:    100,
		MaxLoadedMemoryBytes: 100000,
		MaxLoadedSkills:      100,
		MaxLoadedSkillBytes:  100000,
	})

	var wg sync.WaitGroup
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			itemID := "mem_concurrent"
			_, _, _ = tracker.ChargeOrGetMemory(itemID, itemID+"@1", "shared concurrent memory", `{"cached":true}`)
			_, _ = tracker.GetCachedResult(itemID)
			_, _, _, _ = tracker.GetCached(itemID)
		}(i)
	}
	wg.Wait()

	memCount, _, _, _ := tracker.Stats()
	assert.Equal(t, 1, memCount, "idempotent concurrent charges of same item must count once")
}
