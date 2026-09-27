package tools

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"
	"strings"
	"testing"

	"bob/internal/knowledge"

	openai "github.com/sashabaranov/go-openai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	_ "modernc.org/sqlite"
)

type mockKnowledgeStoreProvider struct {
	dir    string
	stores map[string]*knowledge.Store
}

func newMockKnowledgeStoreProvider(t *testing.T) *mockKnowledgeStoreProvider {
	t.Helper()
	return &mockKnowledgeStoreProvider{
		dir:    t.TempDir(),
		stores: make(map[string]*knowledge.Store),
	}
}

func (m *mockKnowledgeStoreProvider) GetKnowledgeStore(ctx context.Context, chatID string, isDM bool) (*knowledge.Store, error) {
	key := chatID
	if st, ok := m.stores[key]; ok {
		return st, nil
	}
	dbPath := filepath.Join(m.dir, key+".db")
	db, err := sql.Open("sqlite", "file:"+dbPath+"?_pragma=foreign_keys(ON)&_pragma=journal_mode(WAL)")
	if err != nil {
		return nil, err
	}
	if err := knowledge.EnsureKnowledgeSchema(ctx, db); err != nil {
		return nil, err
	}
	st := knowledge.NewStore(db)
	m.stores[key] = st
	return st, nil
}

type mockKnowledgeSearcherProvider struct {
	searcher *knowledge.Searcher
	err      error
}

func (m *mockKnowledgeSearcherProvider) GetKnowledgeSearcher(ctx context.Context, chatID string, isDM bool, userID string) (*knowledge.Searcher, error) {
	if m.err != nil {
		return nil, m.err
	}
	return m.searcher, nil
}

func setupKnowledgeTestRegistry(t *testing.T) (*Registry, *mockKnowledgeStoreProvider) {
	t.Helper()
	kProv := newMockKnowledgeStoreProvider(t)
	reg := NewRegistry(nil, nil)
	reg.SetKnowledgeStoreProvider(kProv)
	return reg, kProv
}

func toolNames(tools []openai.Tool) []string {
	var names []string
	for _, t := range tools {
		if t.Function != nil {
			names = append(names, t.Function.Name)
		}
	}
	return names
}

func containsTool(names []string, target string) bool {
	for _, n := range names {
		if n == target {
			return true
		}
	}
	return false
}

func TestToolDefinitionsForSession_KnowledgeFiltering(t *testing.T) {
	reg, _ := setupKnowledgeTestRegistry(t)

	knowledgeToolList := []string{
		"propose_memory", "discover_memories", "load_memory",
		"propose_skill", "discover_skills", "load_skill",
	}

	// 1. Townhall chat (IsDM = false) -> No knowledge tools
	thSession := NewChatSessionContext("townhall", "user1", false)
	thTools := toolNames(reg.ToolDefinitionsForSession(thSession))
	for _, kt := range knowledgeToolList {
		assert.False(t, containsTool(thTools, kt), "Townhall must not have %s", kt)
	}

	// 2. Scheduled DM (IsDM = true, IsScheduled = true) -> No knowledge tools
	schedSession := NewChatSessionContext("dm_chat", "user1", true)
	schedSession.IsScheduled = true
	schedTools := toolNames(reg.ToolDefinitionsForSession(schedSession))
	for _, kt := range knowledgeToolList {
		assert.False(t, containsTool(schedTools, kt), "Scheduled run must not have %s", kt)
	}

	// 3. Interactive DM (IsDM = true, IsScheduled = false) -> All knowledge tools present
	dmSession := NewChatSessionContext("dm_chat", "user1", true)
	dmTools := toolNames(reg.ToolDefinitionsForSession(dmSession))
	for _, kt := range knowledgeToolList {
		assert.True(t, containsTool(dmTools, kt), "Interactive DM must have %s", kt)
	}
}

func TestKnowledgeTools_GatingAndProvenance(t *testing.T) {
	reg, _ := setupKnowledgeTestRegistry(t)
	ctx := context.Background()

	validSeq := int64(42)

	// 1. Context without session
	_, err := reg.Execute(ctx, "propose_memory", `{"type":"fact","content":"valid"}`)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "chat session context missing")

	// 2. Townhall rejection
	thCtx := WithChatSession(ctx, ChatSessionContext{
		ChatID:           "townhall",
		UserID:           "u1",
		IsDM:             false,
		SourceMessageSeq: &validSeq,
		FSMRunID:         "run_1",
	})
	_, err = reg.Execute(thCtx, "propose_memory", `{"type":"fact","content":"valid"}`)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "direct messages")

	_, err = reg.Execute(thCtx, "discover_memories", `{"query":"valid"}`)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "direct messages")

	_, err = reg.Execute(thCtx, "load_memory", `{"memory_id":"mem_1"}`)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "direct messages")

	// 3. Scheduled task rejection
	schedCtx := WithChatSession(ctx, ChatSessionContext{
		ChatID:           "dm_1",
		UserID:           "u1",
		IsDM:             true,
		IsScheduled:      true,
		SourceMessageSeq: &validSeq,
		FSMRunID:         "run_1",
	})
	_, err = reg.Execute(schedCtx, "propose_memory", `{"type":"fact","content":"valid"}`)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "scheduled tasks")

	_, err = reg.Execute(schedCtx, "propose_skill", `{"name":"test","description":"d","triggers":["t"],"instructions_markdown":"i"}`)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "scheduled tasks")

	_, err = reg.Execute(schedCtx, "discover_memories", `{"query":"valid"}`)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "scheduled tasks")

	_, err = reg.Execute(schedCtx, "load_memory", `{"memory_id":"mem_1"}`)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "scheduled tasks")

	// 4. Missing SourceMessageSeq (nil or <= 0)
	noSeqCtx := WithChatSession(ctx, ChatSessionContext{
		ChatID:   "dm_1",
		UserID:   "u1",
		IsDM:     true,
		FSMRunID: "run_1",
	})
	_, err = reg.Execute(noSeqCtx, "propose_memory", `{"type":"fact","content":"valid"}`)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "user message provenance")

	zeroSeq := int64(0)
	zeroSeqCtx := WithChatSession(ctx, ChatSessionContext{
		ChatID:           "dm_1",
		UserID:           "u1",
		IsDM:             true,
		SourceMessageSeq: &zeroSeq,
		FSMRunID:         "run_1",
	})
	_, err = reg.Execute(zeroSeqCtx, "propose_skill", `{"name":"test","description":"d","triggers":["t"],"instructions_markdown":"i"}`)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "user message provenance")

	// 5. Missing FSMRunID
	noRunCtx := WithChatSession(ctx, ChatSessionContext{
		ChatID:           "dm_1",
		UserID:           "u1",
		IsDM:             true,
		SourceMessageSeq: &validSeq,
	})
	_, err = reg.Execute(noRunCtx, "propose_memory", `{"type":"fact","content":"valid"}`)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "FSM run ID")
}

func TestKnowledgeTools_ArgumentValidation(t *testing.T) {
	reg, _ := setupKnowledgeTestRegistry(t)
	validSeq := int64(10)
	ctx := WithChatSession(context.Background(), ChatSessionContext{
		ChatID:           "dm_chat",
		UserID:           "alice",
		IsDM:             true,
		SourceMessageSeq: &validSeq,
		FSMRunID:         "run_test",
		KnowledgeBudget:  NewKnowledgeBudgetTracker(DefaultKnowledgeBudgetLimits()),
	})

	// propose_memory validation
	_, err := reg.Execute(ctx, "propose_memory", `{"type":"unknown","content":"valid"}`)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "invalid memory type")

	_, err = reg.Execute(ctx, "propose_memory", `{"type":"fact","content":"   "}`)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "content cannot be empty")

	_, err = reg.Execute(ctx, "propose_memory", `{"type":"fact","content":"valid","confidence":1.5}`)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "confidence must be between 0.0 and 1.0")

	_, err = reg.Execute(ctx, "propose_memory", `{"type":"fact","content":"valid","ttl_days":0}`)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "ttl_days must be positive")

	_, err = reg.Execute(ctx, "propose_memory", `{"type":"fact","content":"valid","revises_memory_id":"invalid_id"}`)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "invalid revises_memory_id")

	// propose_skill validation
	_, err = reg.Execute(ctx, "propose_skill", `{"name":"Invalid-Name!","description":"desc","triggers":["t"],"tags":["tag"],"instructions_markdown":"i"}`)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "invalid skill name")

	_, err = reg.Execute(ctx, "propose_skill", `{"name":"valid_name","description":"","triggers":["t"],"tags":["tag"],"instructions_markdown":"i"}`)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "description cannot be empty")

	_, err = reg.Execute(ctx, "propose_skill", `{"name":"valid_name","description":"desc","triggers":[],"tags":["tag"],"instructions_markdown":"i"}`)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "triggers cannot be empty")

	_, err = reg.Execute(ctx, "propose_skill", `{"name":"valid_name","description":"desc","triggers":["t"],"tags":[],"instructions_markdown":"i"}`)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "tags cannot be empty")

	_, err = reg.Execute(ctx, "propose_skill", `{"name":"valid_name","description":"desc","triggers":["t"],"tags":["tag"],"instructions_markdown":""}`)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "instructions_markdown cannot be empty")

	_, err = reg.Execute(ctx, "propose_skill", `{"name":"valid_name","description":"desc","triggers":["t"],"tags":["tag"],"instructions_markdown":"i","revises_skill_id":"invalid_id"}`)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "invalid revises_skill_id")

	// load_memory validation
	_, err = reg.Execute(ctx, "load_memory", `{"memory_id":""}`)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "memory_id cannot be empty")

	_, err = reg.Execute(ctx, "load_memory", `{"memory_id":"skill_123"}`)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "must start with 'mem_'")

	// load_skill validation
	_, err = reg.Execute(ctx, "load_skill", `{"skill_id":""}`)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "skill_id cannot be empty")

	_, err = reg.Execute(ctx, "load_skill", `{"skill_id":"mem_123"}`)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "must start with 'skill_'")
}

func TestKnowledgeTools_ProposeAndLoadLifecycle(t *testing.T) {
	reg, prov := setupKnowledgeTestRegistry(t)
	validSeq := int64(10)
	sessionCtx := ChatSessionContext{
		ChatID:           "dm_alice",
		UserID:           "alice",
		IsDM:             true,
		SourceMessageSeq: &validSeq,
		FSMRunID:         "run_alice_1",
		KnowledgeBudget:  NewKnowledgeBudgetTracker(DefaultKnowledgeBudgetLimits()),
	}
	ctx := WithChatSession(context.Background(), sessionCtx)

	// 1. Propose memory
	res, err := reg.Execute(ctx, "propose_memory", `{
		"type": "fact",
		"content": "Bob is a companion agent",
		"confidence": 0.95
	}`)
	require.NoError(t, err)
	var propMem struct {
		ItemID    string `json:"item_id"`
		VersionID string `json:"version_id"`
		Revision  int    `json:"revision"`
		Status    string `json:"status"`
	}
	require.NoError(t, json.Unmarshal([]byte(res), &propMem))
	assert.True(t, strings.HasPrefix(propMem.ItemID, "mem_"))
	assert.Equal(t, 1, propMem.Revision)
	assert.Equal(t, "proposed", propMem.Status)

	// 2. Attempting to load unapproved memory must fail
	_, err = reg.Execute(ctx, "load_memory", fmt.Sprintf(`{"memory_id":%q}`, propMem.ItemID))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "not active")

	// 3. Approve memory directly via store
	kStore, err := prov.GetKnowledgeStore(ctx, "dm_alice", true)
	require.NoError(t, err)
	require.NoError(t, kStore.ApproveVersion(ctx, propMem.VersionID, "alice"))

	// 4. Load memory succeeds and charges budget
	loadRes, err := reg.Execute(ctx, "load_memory", fmt.Sprintf(`{"memory_id":%q}`, propMem.ItemID))
	require.NoError(t, err)
	var loadedMem struct {
		ItemID       string `json:"item_id"`
		VersionID    string `json:"version_id"`
		Type         string `json:"type"`
		Content      string `json:"content"`
		ContentBytes int    `json:"content_bytes"`
	}
	require.NoError(t, json.Unmarshal([]byte(loadRes), &loadedMem))
	assert.Equal(t, propMem.ItemID, loadedMem.ItemID)
	assert.Equal(t, propMem.VersionID, loadedMem.VersionID)
	assert.Equal(t, "fact", loadedMem.Type)
	assert.Equal(t, "Bob is a companion agent", loadedMem.Content)
	assert.Equal(t, len([]byte(loadedMem.Content)), loadedMem.ContentBytes)

	memCount, memBytes, _, _ := sessionCtx.KnowledgeBudget.Stats()
	assert.Equal(t, 1, memCount)
	assert.Equal(t, loadedMem.ContentBytes, memBytes)

	// 5. Repeated load is idempotent and returns cached result without extra budget charge
	loadRes2, err := reg.Execute(ctx, "load_memory", fmt.Sprintf(`{"memory_id":%q}`, propMem.ItemID))
	require.NoError(t, err)
	assert.Equal(t, loadRes, loadRes2)
	memCount2, memBytes2, _, _ := sessionCtx.KnowledgeBudget.Stats()
	assert.Equal(t, 1, memCount2)
	assert.Equal(t, memBytes, memBytes2)

	// 6. Propose skill
	res, err = reg.Execute(ctx, "propose_skill", `{
		"name": "greet_user",
		"description": "Friendly greeting routine",
		"triggers": ["hello", "hi"],
		"tags": ["social"],
		"instructions_markdown": "# Greet\nSay hello politely."
	}`)
	require.NoError(t, err)
	var propSkill struct {
		ItemID    string `json:"item_id"`
		VersionID string `json:"version_id"`
		Revision  int    `json:"revision"`
		Status    string `json:"status"`
	}
	require.NoError(t, json.Unmarshal([]byte(res), &propSkill))
	assert.True(t, strings.HasPrefix(propSkill.ItemID, "skill_"))
	assert.Equal(t, 1, propSkill.Revision)
	assert.Equal(t, "proposed", propSkill.Status)

	// 7. Approve skill and load
	require.NoError(t, kStore.ApproveVersion(ctx, propSkill.VersionID, "alice"))
	loadSkillRes, err := reg.Execute(ctx, "load_skill", fmt.Sprintf(`{"skill_id":%q}`, propSkill.ItemID))
	require.NoError(t, err)
	var loadedSkill struct {
		ItemID               string `json:"item_id"`
		VersionID            string `json:"version_id"`
		Name                 string `json:"name"`
		Description          string `json:"description"`
		InstructionsMarkdown string `json:"instructions_markdown"`
		ContentBytes         int    `json:"content_bytes"`
	}
	require.NoError(t, json.Unmarshal([]byte(loadSkillRes), &loadedSkill))
	assert.Equal(t, "greet_user", loadedSkill.Name)
	assert.Equal(t, "# Greet\nSay hello politely.", loadedSkill.InstructionsMarkdown)

	_, _, skillCount, skillBytes := sessionCtx.KnowledgeBudget.Stats()
	assert.Equal(t, 1, skillCount)
	assert.Equal(t, loadedSkill.ContentBytes, skillBytes)

	// 8. Repeated skill load is idempotent
	loadSkillRes2, err := reg.Execute(ctx, "load_skill", fmt.Sprintf(`{"skill_id":%q}`, propSkill.ItemID))
	require.NoError(t, err)
	assert.Equal(t, loadSkillRes, loadSkillRes2)
	_, _, skillCount2, skillBytes2 := sessionCtx.KnowledgeBudget.Stats()
	assert.Equal(t, 1, skillCount2)
	assert.Equal(t, skillBytes, skillBytes2)
}

func TestKnowledgeTools_BudgetExceeded(t *testing.T) {
	reg, prov := setupKnowledgeTestRegistry(t)
	validSeq := int64(1)
	tracker := NewKnowledgeBudgetTracker(KnowledgeBudgetLimits{
		MaxLoadedMemories:    1,
		MaxLoadedMemoryBytes: 100,
	})
	sessionCtx := ChatSessionContext{
		ChatID:           "dm_budget",
		UserID:           "alice",
		IsDM:             true,
		SourceMessageSeq: &validSeq,
		FSMRunID:         "run_b1",
		KnowledgeBudget:  tracker,
	}
	ctx := WithChatSession(context.Background(), sessionCtx)

	kStore, err := prov.GetKnowledgeStore(ctx, "dm_budget", true)
	require.NoError(t, err)

	// Create and approve 2 memories
	_, ver1, err := kStore.ProposeMemory(ctx, &knowledge.KnowledgeItem{
		ChatID: "dm_budget",
		UserID: "alice",
	}, &knowledge.MemoryVersion{
		Type:       knowledge.MemoryTypeFact,
		Content:    "First memory",
		Confidence: 1.0,
		Provenance: knowledge.Provenance{ChatID: "dm_budget", UserID: "alice", SourceMessageSeq: &validSeq, FSMRunID: "run_b1"},
	})
	require.NoError(t, err)
	require.NoError(t, kStore.ApproveVersion(ctx, ver1.ID, "alice"))

	_, ver2, err := kStore.ProposeMemory(ctx, &knowledge.KnowledgeItem{
		ChatID: "dm_budget",
		UserID: "alice",
	}, &knowledge.MemoryVersion{
		Type:       knowledge.MemoryTypeFact,
		Content:    "Second memory",
		Confidence: 1.0,
		Provenance: knowledge.Provenance{ChatID: "dm_budget", UserID: "alice", SourceMessageSeq: &validSeq, FSMRunID: "run_b1"},
	})
	require.NoError(t, err)
	require.NoError(t, kStore.ApproveVersion(ctx, ver2.ID, "alice"))

	// First load succeeds
	_, err = reg.Execute(ctx, "load_memory", fmt.Sprintf(`{"memory_id":%q}`, ver1.ItemID))
	require.NoError(t, err)

	// Second load exceeds memory limit of 1
	_, err = reg.Execute(ctx, "load_memory", fmt.Sprintf(`{"memory_id":%q}`, ver2.ItemID))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "budget exceeded")
}

func TestKnowledgeTools_DiscoveryEmptyAndBoundLimits(t *testing.T) {
	reg, _ := setupKnowledgeTestRegistry(t)
	validSeq := int64(1)
	ctx := WithChatSession(context.Background(), ChatSessionContext{
		ChatID:           "dm_disc",
		UserID:           "alice",
		IsDM:             true,
		SourceMessageSeq: &validSeq,
		FSMRunID:         "run_d",
	})

	// Empty query returns empty list without calling provider
	res, err := reg.Execute(ctx, "discover_memories", `{"query":"   "}`)
	require.NoError(t, err)
	assert.Equal(t, `{"memories":[],"count":0}`, res)

	res, err = reg.Execute(ctx, "discover_skills", `{"query":""}`)
	require.NoError(t, err)
	assert.Equal(t, `{"skills":[],"count":0}`, res)
}

func TestKnowledgeTools_DiscoveryWithProvider(t *testing.T) {
	reg, prov := setupKnowledgeTestRegistry(t)
	validSeq := int64(1)
	ctx := WithChatSession(context.Background(), ChatSessionContext{
		ChatID:           "dm_disc_prov",
		UserID:           "alice",
		IsDM:             true,
		SourceMessageSeq: &validSeq,
		FSMRunID:         "run_dp",
	})

	// 1. Without searcher provider configured
	_, err := reg.Execute(ctx, "discover_memories", `{"query":"some query"}`)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "searcher provider is not configured")

	_, err = reg.Execute(ctx, "discover_skills", `{"query":"some query"}`)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "searcher provider is not configured")

	// 2. With mock searcher provider returning an error
	errProv := errors.New("searcher lookup failed")
	reg.SetKnowledgeSearcherProvider(&mockKnowledgeSearcherProvider{err: errProv})
	_, err = reg.Execute(ctx, "discover_memories", `{"query":"some query"}`)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "searcher lookup failed")

	// 3. With mock searcher provider returning a valid Searcher
	kStore, err := prov.GetKnowledgeStore(ctx, "dm_disc_prov", true)
	require.NoError(t, err)
	searcher, err := knowledge.NewSearcher(kStore, nil, knowledge.SessionIdentity{
		ChatID: "dm_disc_prov",
		UserID: "alice",
	}, 10)
	require.NoError(t, err)

	reg.SetKnowledgeSearcherProvider(&mockKnowledgeSearcherProvider{searcher: searcher})
	res, err := reg.Execute(ctx, "discover_memories", `{"query":"kubernetes","types":["fact"],"limit":3}`)
	require.NoError(t, err)
	assert.Equal(t, `{"count":0,"memories":[]}`, res)

	res, err = reg.Execute(ctx, "discover_skills", `{"query":"deploy","tags":["devops"],"limit":3}`)
	require.NoError(t, err)
	assert.Equal(t, `{"count":0,"skills":[]}`, res)
}

func TestKnowledgeTools_SecurityAndBounds(t *testing.T) {
	reg, prov := setupKnowledgeTestRegistry(t)
	validSeq := int64(1)
	baseSession := ChatSessionContext{
		ChatID:           "dm_alice",
		UserID:           "alice",
		IsDM:             true,
		SourceMessageSeq: &validSeq,
		FSMRunID:         "run_sec_1",
		KnowledgeBudget:  NewKnowledgeBudgetTracker(DefaultKnowledgeBudgetLimits()),
	}
	ctx := WithChatSession(context.Background(), baseSession)

	t.Run("proposal responses contain actionable commands", func(t *testing.T) {
		memRes, err := reg.Execute(ctx, "propose_memory", `{"type":"fact","content":"Apples are fruits"}`)
		require.NoError(t, err)
		assert.Contains(t, memRes, "/memory approve mem_")

		skillRes, err := reg.Execute(ctx, "propose_skill", `{
			"name": "sample_skill",
			"description": "sample description",
			"triggers": ["sample"],
			"tags": ["testing"],
			"instructions_markdown": "echo sample"
		}`)
		require.NoError(t, err)
		assert.Contains(t, skillRes, "/skill approve skill_")
	})

	t.Run("propose_memory bounds ttl_days <= 3650", func(t *testing.T) {
		_, err := reg.Execute(ctx, "propose_memory", `{"type":"fact","content":"text","ttl_days":5000}`)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "between 1 and 3650 days")
	})

	t.Run("propose_skill bounds description <= 500 characters", func(t *testing.T) {
		longDesc := strings.Repeat("a", 501)
		_, err := reg.Execute(ctx, "propose_skill", fmt.Sprintf(`{
			"name": "long_desc_skill",
			"description": %q,
			"triggers": ["sample"],
			"tags": ["test"],
			"instructions_markdown": "echo"
		}`, longDesc))
		require.Error(t, err)
		assert.Contains(t, err.Error(), "description exceeds maximum length")
	})

	t.Run("propose_skill bounds triggers <= 20 and trigger length <= 100", func(t *testing.T) {
		var tooManyTriggers []string
		for i := 0; i < 21; i++ {
			tooManyTriggers = append(tooManyTriggers, fmt.Sprintf("t%d", i))
		}
		trigJSON, _ := json.Marshal(tooManyTriggers)
		_, err := reg.Execute(ctx, "propose_skill", fmt.Sprintf(`{
			"name": "many_trig_skill",
			"description": "desc",
			"triggers": %s,
			"tags": ["test"],
			"instructions_markdown": "echo"
		}`, string(trigJSON)))
		require.Error(t, err)
		assert.Contains(t, err.Error(), "maximum 20 triggers allowed")

		longTrig := strings.Repeat("x", 101)
		_, err = reg.Execute(ctx, "propose_skill", fmt.Sprintf(`{
			"name": "long_trig_skill",
			"description": "desc",
			"triggers": [%q],
			"tags": ["test"],
			"instructions_markdown": "echo"
		}`, longTrig))
		require.Error(t, err)
		assert.Contains(t, err.Error(), "exceeds maximum length of 100 characters")
	})

	t.Run("propose_skill bounds tags <= 10 and tag length <= 50", func(t *testing.T) {
		var tooManyTags []string
		for i := 0; i < 11; i++ {
			tooManyTags = append(tooManyTags, fmt.Sprintf("tag%d", i))
		}
		tagsJSON, _ := json.Marshal(tooManyTags)
		_, err := reg.Execute(ctx, "propose_skill", fmt.Sprintf(`{
			"name": "many_tags_skill",
			"description": "desc",
			"triggers": ["trig"],
			"tags": %s,
			"instructions_markdown": "echo"
		}`, string(tagsJSON)))
		require.Error(t, err)
		assert.Contains(t, err.Error(), "maximum 10 tags allowed")

		longTag := strings.Repeat("y", 51)
		_, err = reg.Execute(ctx, "propose_skill", fmt.Sprintf(`{
			"name": "long_tag_skill",
			"description": "desc",
			"triggers": ["trig"],
			"tags": [%q],
			"instructions_markdown": "echo"
		}`, longTag))
		require.Error(t, err)
		assert.Contains(t, err.Error(), "exceeds maximum length of 50 characters")
	})

	t.Run("fail-closed on missing user_id or chat_id or nil budget", func(t *testing.T) {
		noUserCtx := WithChatSession(context.Background(), ChatSessionContext{
			ChatID:           "dm_alice",
			UserID:           "",
			IsDM:             true,
			SourceMessageSeq: &validSeq,
			FSMRunID:         "run_1",
			KnowledgeBudget:  NewKnowledgeBudgetTracker(DefaultKnowledgeBudgetLimits()),
		})
		_, err := reg.Execute(noUserCtx, "discover_memories", `{"query":"test"}`)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "missing user_id or chat_id")

		_, err = reg.Execute(noUserCtx, "load_memory", `{"memory_id":"mem_123"}`)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "missing user_id or chat_id")

		_, err = reg.Execute(noUserCtx, "load_skill", `{"skill_id":"skill_123"}`)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "missing user_id or chat_id")

		noBudgetCtx := WithChatSession(context.Background(), ChatSessionContext{
			ChatID:           "dm_alice",
			UserID:           "alice",
			IsDM:             true,
			SourceMessageSeq: &validSeq,
			FSMRunID:         "run_1",
			KnowledgeBudget:  nil,
		})
		_, err = reg.Execute(noBudgetCtx, "load_memory", `{"memory_id":"mem_123"}`)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "knowledge budget tracker is not initialized")

		_, err = reg.Execute(noBudgetCtx, "load_skill", `{"skill_id":"skill_123"}`)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "knowledge budget tracker is not initialized")
	})

	t.Run("cross-user revision rejected", func(t *testing.T) {
		// Alice proposes memory
		res, err := reg.Execute(ctx, "propose_memory", `{"type":"fact","content":"Secret fact"}`)
		require.NoError(t, err)
		var m struct{ ItemID string `json:"item_id"` }
		require.NoError(t, json.Unmarshal([]byte(res), &m))

		// Bob tries to revise Alice's memory in his DM
		bobSession := ChatSessionContext{
			ChatID:           "dm_bob",
			UserID:           "bob",
			IsDM:             true,
			SourceMessageSeq: &validSeq,
			FSMRunID:         "run_bob_1",
			KnowledgeBudget:  NewKnowledgeBudgetTracker(DefaultKnowledgeBudgetLimits()),
		}
		bobCtx := WithChatSession(context.Background(), bobSession)
		_, err = reg.Execute(bobCtx, "propose_memory", fmt.Sprintf(`{"type":"fact","content":"Hijacked fact","revises_memory_id":%q}`, m.ItemID))
		require.Error(t, err)
		assert.Contains(t, err.Error(), "not found")
	})

	t.Run("cross-user load rejected", func(t *testing.T) {
		// Alice proposes and approves a memory
		kStore, err := prov.GetKnowledgeStore(ctx, "dm_alice", true)
		require.NoError(t, err)
		res, err := reg.Execute(ctx, "propose_memory", `{"type":"fact","content":"Alice private data"}`)
		require.NoError(t, err)
		var m struct {
			ItemID    string `json:"item_id"`
			VersionID string `json:"version_id"`
		}
		require.NoError(t, json.Unmarshal([]byte(res), &m))
		require.NoError(t, kStore.ApproveVersion(ctx, m.VersionID, "alice"))

		// Bob cannot load it even if he knows the item ID
		bobSession := ChatSessionContext{
			ChatID:           "dm_bob",
			UserID:           "bob",
			IsDM:             true,
			SourceMessageSeq: &validSeq,
			FSMRunID:         "run_bob_1",
			KnowledgeBudget:  NewKnowledgeBudgetTracker(DefaultKnowledgeBudgetLimits()),
		}
		bobCtx := WithChatSession(context.Background(), bobSession)
		_, err = reg.Execute(bobCtx, "load_memory", fmt.Sprintf(`{"memory_id":%q}`, m.ItemID))
		require.Error(t, err)
		assert.Contains(t, err.Error(), "not found")
	})

	t.Run("cached load rejects mismatched requested version", func(t *testing.T) {
		kStore, err := prov.GetKnowledgeStore(ctx, "dm_alice", true)
		require.NoError(t, err)

		// 1. Memory cached load with mismatched version
		res, err := reg.Execute(ctx, "propose_memory", `{"type":"fact","content":"Version check content"}`)
		require.NoError(t, err)
		var m struct {
			ItemID    string `json:"item_id"`
			VersionID string `json:"version_id"`
		}
		require.NoError(t, json.Unmarshal([]byte(res), &m))
		require.NoError(t, kStore.ApproveVersion(ctx, m.VersionID, "alice"))

		// First load populates cache
		_, err = reg.Execute(ctx, "load_memory", fmt.Sprintf(`{"memory_id":%q}`, m.ItemID))
		require.NoError(t, err)

		// Same item with invalid/nonexistent version suffix must NOT return cached result
		_, err = reg.Execute(ctx, "load_memory", fmt.Sprintf(`{"memory_id":"%s@999"}`, m.ItemID))
		require.Error(t, err)
		assert.Contains(t, err.Error(), "not the active version")

		// 2. Skill cached load with mismatched version
		skillRes, err := reg.Execute(ctx, "propose_skill", `{
			"name": "ver_check_skill",
			"description": "desc",
			"triggers": ["check"],
			"tags": ["testing"],
			"instructions_markdown": "echo check"
		}`)
		require.NoError(t, err)
		var sk struct {
			ItemID    string `json:"item_id"`
			VersionID string `json:"version_id"`
		}
		require.NoError(t, json.Unmarshal([]byte(skillRes), &sk))
		require.NoError(t, kStore.ApproveVersion(ctx, sk.VersionID, "alice"))

		// First load populates cache
		_, err = reg.Execute(ctx, "load_skill", fmt.Sprintf(`{"skill_id":%q}`, sk.ItemID))
		require.NoError(t, err)

		// Same skill with invalid version suffix must NOT return cached result
		_, err = reg.Execute(ctx, "load_skill", fmt.Sprintf(`{"skill_id":"%s@999"}`, sk.ItemID))
		require.Error(t, err)
		assert.Contains(t, err.Error(), "not the active version")
	})
}
