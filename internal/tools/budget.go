package tools

import (
	"errors"
	"fmt"
	"strings"
	"sync"
)

// KnowledgeBudgetLimits defines the maximum allowable loaded items and bytes.
type KnowledgeBudgetLimits struct {
	MaxLoadedMemories    int
	MaxLoadedMemoryBytes int
	MaxLoadedSkills      int
	MaxLoadedSkillBytes  int
}

// DefaultKnowledgeBudgetLimits returns default budget limits matching configuration defaults.
func DefaultKnowledgeBudgetLimits() KnowledgeBudgetLimits {
	return KnowledgeBudgetLimits{
		MaxLoadedMemories:    8,
		MaxLoadedMemoryBytes: 16 * 1024, // 16 KiB
		MaxLoadedSkills:      3,
		MaxLoadedSkillBytes:  24 * 1024, // 24 KiB
	}
}

type cachedItem struct {
	ItemID       string
	VersionID    string
	Kind         string // "memory" or "skill"
	Content      string
	ContentBytes int
	ResultJSON   string
}

// KnowledgeBudgetTracker provides thread-safe accounting and caching of loaded memories and skills per request/run.
type KnowledgeBudgetTracker struct {
	mu     sync.Mutex
	limits KnowledgeBudgetLimits

	loadedMemoriesCount int
	loadedMemoriesBytes int
	loadedSkillsCount   int
	loadedSkillsBytes   int

	loadedItems map[string]cachedItem // itemID -> cachedItem
}

// NewKnowledgeBudgetTracker constructs an initialized KnowledgeBudgetTracker with the given limits.
func NewKnowledgeBudgetTracker(limits KnowledgeBudgetLimits) *KnowledgeBudgetTracker {
	if limits.MaxLoadedMemories <= 0 {
		limits.MaxLoadedMemories = 8
	}
	if limits.MaxLoadedMemoryBytes <= 0 {
		limits.MaxLoadedMemoryBytes = 16 * 1024
	}
	if limits.MaxLoadedSkills <= 0 {
		limits.MaxLoadedSkills = 3
	}
	if limits.MaxLoadedSkillBytes <= 0 {
		limits.MaxLoadedSkillBytes = 24 * 1024
	}
	return &KnowledgeBudgetTracker{
		limits:      limits,
		loadedItems: make(map[string]cachedItem),
	}
}

// GetCached returns a previously loaded item by its stable item ID, or false if not loaded.
func (t *KnowledgeBudgetTracker) GetCached(itemID string) (versionID, content string, bytes int, ok bool) {
	if t == nil {
		return "", "", 0, false
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	item, found := t.loadedItems[itemID]
	if !found {
		return "", "", 0, false
	}
	return item.VersionID, item.Content, item.ContentBytes, true
}

// SetCachedResult stores the rendered result JSON string for an already loaded item.
func (t *KnowledgeBudgetTracker) SetCachedResult(itemID, resultJSON string) {
	if t == nil {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if item, ok := t.loadedItems[itemID]; ok {
		item.ResultJSON = resultJSON
		t.loadedItems[itemID] = item
	}
}

// GetCachedResult returns the cached result JSON if the item was loaded and has a cached result.
func (t *KnowledgeBudgetTracker) GetCachedResult(itemID string) (string, bool) {
	if t == nil {
		return "", false
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	item, ok := t.loadedItems[itemID]
	if !ok || item.ResultJSON == "" {
		return "", false
	}
	return item.ResultJSON, true
}

// ChargeMemory attempts to charge the budget for loading a memory.
// If the item was already loaded, it returns nil without charging additional budget.
// If charging would exceed the memory count or byte limit, an error is returned and no state is modified.
func (t *KnowledgeBudgetTracker) ChargeMemory(itemID, versionID, content string) error {
	if t == nil {
		return nil
	}
	t.mu.Lock()
	defer t.mu.Unlock()

	if _, exists := t.loadedItems[itemID]; exists {
		return nil
	}

	contentBytes := len([]byte(content))
	if t.loadedMemoriesCount+1 > t.limits.MaxLoadedMemories {
		return fmt.Errorf("memory load budget exceeded: count limit %d reached (%d loaded)", t.limits.MaxLoadedMemories, t.loadedMemoriesCount)
	}
	if t.loadedMemoriesBytes+contentBytes > t.limits.MaxLoadedMemoryBytes {
		return fmt.Errorf("memory load budget exceeded: byte limit %d bytes exceeded (%d already loaded + %d requested = %d bytes)",
			t.limits.MaxLoadedMemoryBytes, t.loadedMemoriesBytes, contentBytes, t.loadedMemoriesBytes+contentBytes)
	}

	t.loadedMemoriesCount++
	t.loadedMemoriesBytes += contentBytes
	t.loadedItems[itemID] = cachedItem{
		ItemID:       itemID,
		VersionID:    versionID,
		Kind:         "memory",
		Content:      content,
		ContentBytes: contentBytes,
	}
	return nil
}

// ChargeSkill attempts to charge the budget for loading a skill.
// If the item was already loaded, it returns nil without charging additional budget.
// If charging would exceed the skill count or byte limit, an error is returned and no state is modified.
func (t *KnowledgeBudgetTracker) ChargeSkill(itemID, versionID, content string) error {
	if t == nil {
		return nil
	}
	t.mu.Lock()
	defer t.mu.Unlock()

	if _, exists := t.loadedItems[itemID]; exists {
		return nil
	}

	contentBytes := len([]byte(content))
	if t.loadedSkillsCount+1 > t.limits.MaxLoadedSkills {
		return fmt.Errorf("skill load budget exceeded: count limit %d reached (%d loaded)", t.limits.MaxLoadedSkills, t.loadedSkillsCount)
	}
	if t.loadedSkillsBytes+contentBytes > t.limits.MaxLoadedSkillBytes {
		return fmt.Errorf("skill load budget exceeded: byte limit %d bytes exceeded (%d already loaded + %d requested = %d bytes)",
			t.limits.MaxLoadedSkillBytes, t.loadedSkillsBytes, contentBytes, t.loadedSkillsBytes+contentBytes)
	}

	t.loadedSkillsCount++
	t.loadedSkillsBytes += contentBytes
	t.loadedItems[itemID] = cachedItem{
		ItemID:       itemID,
		VersionID:    versionID,
		Kind:         "skill",
		Content:      content,
		ContentBytes: contentBytes,
	}
	return nil
}

// ChargeOrGetMemory atomically checks if the memory is already cached.
// If already cached, it returns (cachedResult, true, nil).
// If not loaded and within budget limits, it reserves the budget, caches the item with resultJSON,
// and returns (resultJSON, false, nil).
// If budget limits are exceeded, it returns ("", false, err).
func (t *KnowledgeBudgetTracker) ChargeOrGetMemory(itemID, versionID, content, resultJSON string) (cachedResult string, alreadyLoaded bool, err error) {
	if t == nil {
		return resultJSON, false, nil
	}
	t.mu.Lock()
	defer t.mu.Unlock()

	if item, exists := t.loadedItems[itemID]; exists {
		if item.ResultJSON != "" {
			return item.ResultJSON, true, nil
		}
		item.ResultJSON = resultJSON
		t.loadedItems[itemID] = item
		return resultJSON, true, nil
	}

	contentBytes := len([]byte(content))
	if t.loadedMemoriesCount+1 > t.limits.MaxLoadedMemories {
		return "", false, fmt.Errorf("memory load budget exceeded: count limit %d reached (%d loaded)", t.limits.MaxLoadedMemories, t.loadedMemoriesCount)
	}
	if t.loadedMemoriesBytes+contentBytes > t.limits.MaxLoadedMemoryBytes {
		return "", false, fmt.Errorf("memory load budget exceeded: byte limit %d bytes exceeded (%d already loaded + %d requested = %d bytes)",
			t.limits.MaxLoadedMemoryBytes, t.loadedMemoriesBytes, contentBytes, t.loadedMemoriesBytes+contentBytes)
	}

	t.loadedMemoriesCount++
	t.loadedMemoriesBytes += contentBytes
	t.loadedItems[itemID] = cachedItem{
		ItemID:       itemID,
		VersionID:    versionID,
		Kind:         "memory",
		Content:      content,
		ContentBytes: contentBytes,
		ResultJSON:   resultJSON,
	}
	return resultJSON, false, nil
}

// ChargeOrGetSkill atomically checks if the skill is already cached.
// If already cached, it returns (cachedResult, true, nil).
// If not loaded and within budget limits, it reserves the budget, caches the item with resultJSON,
// and returns (resultJSON, false, nil).
// If budget limits are exceeded, it returns ("", false, err).
func (t *KnowledgeBudgetTracker) ChargeOrGetSkill(itemID, versionID, instructionsMarkdown string, totalContentBytes int, resultJSON string) (cachedResult string, alreadyLoaded bool, err error) {
	if t == nil {
		return resultJSON, false, nil
	}
	t.mu.Lock()
	defer t.mu.Unlock()

	if item, exists := t.loadedItems[itemID]; exists {
		if item.ResultJSON != "" {
			return item.ResultJSON, true, nil
		}
		item.ResultJSON = resultJSON
		t.loadedItems[itemID] = item
		return resultJSON, true, nil
	}

	if totalContentBytes <= 0 {
		totalContentBytes = len([]byte(instructionsMarkdown))
	}

	if t.loadedSkillsCount+1 > t.limits.MaxLoadedSkills {
		return "", false, fmt.Errorf("skill load budget exceeded: count limit %d reached (%d loaded)", t.limits.MaxLoadedSkills, t.loadedSkillsCount)
	}
	if t.loadedSkillsBytes+totalContentBytes > t.limits.MaxLoadedSkillBytes {
		return "", false, fmt.Errorf("skill load budget exceeded: byte limit %d bytes exceeded (%d already loaded + %d requested = %d bytes)",
			t.limits.MaxLoadedSkillBytes, t.loadedSkillsBytes, totalContentBytes, t.loadedSkillsBytes+totalContentBytes)
	}

	t.loadedSkillsCount++
	t.loadedSkillsBytes += totalContentBytes
	t.loadedItems[itemID] = cachedItem{
		ItemID:       itemID,
		VersionID:    versionID,
		Kind:         "skill",
		Content:      instructionsMarkdown,
		ContentBytes: totalContentBytes,
		ResultJSON:   resultJSON,
	}
	return resultJSON, false, nil
}

// RestoreItem restores a previously loaded item into the tracker during crash recovery / run resumption.
// Returns an error if the item kind is unknown, identifiers are empty, or byte length mismatches actual UTF-8 length.
func (t *KnowledgeBudgetTracker) RestoreItem(kind, itemID, versionID, content string, contentBytes int, resultJSON ...string) error {
	if t == nil {
		return errors.New("knowledge budget tracker is nil")
	}
	t.mu.Lock()
	defer t.mu.Unlock()

	if kind != "memory" && kind != "skill" {
		return fmt.Errorf("invalid item kind %q for restored knowledge item %s", kind, itemID)
	}
	if strings.TrimSpace(itemID) == "" {
		return errors.New("empty item_id in restored knowledge item")
	}
	if strings.TrimSpace(versionID) == "" {
		return fmt.Errorf("empty version_id for restored knowledge item %s", itemID)
	}
	actualBytes := len([]byte(content))
	if actualBytes == 0 {
		return fmt.Errorf("empty content for restored knowledge item %s", itemID)
	}
	if kind == "memory" && contentBytes != actualBytes {
		return fmt.Errorf("content_bytes mismatch for restored knowledge item %s: recorded %d, actual %d", itemID, contentBytes, actualBytes)
	}
	if kind == "skill" && contentBytes < actualBytes {
		return fmt.Errorf("content_bytes mismatch for restored knowledge item %s: recorded %d, minimum %d", itemID, contentBytes, actualBytes)
	}

	if _, exists := t.loadedItems[itemID]; exists {
		return nil
	}

	switch kind {
	case "memory":
		t.loadedMemoriesCount++
		t.loadedMemoriesBytes += contentBytes
	case "skill":
		t.loadedSkillsCount++
		t.loadedSkillsBytes += contentBytes
	}

	resJSON := ""
	if len(resultJSON) > 0 {
		resJSON = resultJSON[0]
	}

	t.loadedItems[itemID] = cachedItem{
		ItemID:       itemID,
		VersionID:    versionID,
		Kind:         kind,
		Content:      content,
		ContentBytes: contentBytes,
		ResultJSON:   resJSON,
	}
	return nil
}

// Stats returns current loaded counts and bytes.
func (t *KnowledgeBudgetTracker) Stats() (memCount, memBytes, skillCount, skillBytes int) {
	if t == nil {
		return 0, 0, 0, 0
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.loadedMemoriesCount, t.loadedMemoriesBytes, t.loadedSkillsCount, t.loadedSkillsBytes
}
