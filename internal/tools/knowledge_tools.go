package tools

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"
	"unicode/utf8"

	"bob/internal/knowledge"

	openai "github.com/sashabaranov/go-openai"
)

const (
	KnowledgeDefaultDiscoveryLimit = 5
	KnowledgeMaxDiscoveryLimit     = 10
)

// KnowledgeStoreProvider resolves a knowledge.Store for a chat context.
type KnowledgeStoreProvider interface {
	GetKnowledgeStore(ctx context.Context, chatID string, isDM bool) (*knowledge.Store, error)
}

// KnowledgeSearcherProvider resolves a knowledge.Searcher for a chat context and user.
type KnowledgeSearcherProvider interface {
	GetKnowledgeSearcher(ctx context.Context, chatID string, isDM bool, userID string) (*knowledge.Searcher, error)
}

// ProposeMemoryArgs defines arguments for the propose_memory tool.
type ProposeMemoryArgs struct {
	Type            string   `json:"type"`
	Content         string   `json:"content"`
	Confidence      *float64 `json:"confidence,omitempty"`
	TTLDays         *int     `json:"ttl_days,omitempty"`
	RevisesMemoryID string   `json:"revises_memory_id,omitempty"`
}

// DiscoverMemoriesArgs defines arguments for the discover_memories tool.
type DiscoverMemoriesArgs struct {
	Query string   `json:"query"`
	Types []string `json:"types,omitempty"`
	Limit int      `json:"limit,omitempty"`
}

// LoadMemoryArgs defines arguments for the load_memory tool.
type LoadMemoryArgs struct {
	MemoryID string `json:"memory_id"`
}

// ProposeSkillArgs defines arguments for the propose_skill tool.
type ProposeSkillArgs struct {
	Name                 string   `json:"name"`
	Description          string   `json:"description"`
	Triggers             []string `json:"triggers"`
	Tags                 []string `json:"tags"`
	InstructionsMarkdown string   `json:"instructions_markdown"`
	RevisesSkillID       string   `json:"revises_skill_id,omitempty"`
}

// DiscoverSkillsArgs defines arguments for the discover_skills tool.
type DiscoverSkillsArgs struct {
	Query string   `json:"query"`
	Tags  []string `json:"tags,omitempty"`
	Limit int      `json:"limit,omitempty"`
}

// LoadSkillArgs defines arguments for the load_skill tool.
type LoadSkillArgs struct {
	SkillID string `json:"skill_id"`
}

var (
	proposeMemoryToolSchema = map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"type": map[string]interface{}{
				"type":        "string",
				"enum":        []string{"preference", "fact", "decision", "ongoing_task"},
				"description": "Classification of the memory: 'preference' (user likes/dislikes/habits), 'fact' (stable knowledge about user, project, environment), 'decision' (architectural or operational decision), or 'ongoing_task' (active goals, todos, or in-flight state).",
			},
			"content": map[string]interface{}{
				"type":        "string",
				"description": "Clear, concise, self-contained statement to remember. Maximum 4000 bytes.",
			},
			"confidence": map[string]interface{}{
				"type":        "number",
				"description": "Confidence score from 0.0 to 1.0 (defaults to 1.0 if omitted).",
			},
			"ttl_days": map[string]interface{}{
				"type":        "integer",
				"description": "Optional time-to-live in days (1-3650 days). Use for temporary facts, decisions, or ongoing tasks.",
			},
			"revises_memory_id": map[string]interface{}{
				"type":        "string",
				"description": "Optional ID of an existing memory (e.g. 'mem_...') that this new revision updates or supersedes.",
			},
		},
		"required": []string{"type", "content"},
	}

	discoverMemoriesToolSchema = map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"query": map[string]interface{}{
				"type":        "string",
				"description": "Semantic query describing facts, preferences, decisions, or tasks to discover.",
			},
			"types": map[string]interface{}{
				"type": "array",
				"items": map[string]interface{}{
					"type": "string",
					"enum": []string{"preference", "fact", "decision", "ongoing_task"},
				},
				"description": "Optional filter restricting discovery to specific memory types.",
			},
			"limit": map[string]interface{}{
				"type":        "integer",
				"description": "Maximum number of memory summaries to return (1-10, default 5).",
			},
		},
		"required": []string{"query"},
	}

	loadMemoryToolSchema = map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"memory_id": map[string]interface{}{
				"type":        "string",
				"description": "The stable memory identifier (e.g. 'mem_abc123') discovered via discover_memories.",
			},
		},
		"required": []string{"memory_id"},
	}

	proposeSkillToolSchema = map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"name": map[string]interface{}{
				"type":        "string",
				"description": "Short identifier for the procedural skill (lowercase alphanumeric and underscores, e.g. 'deploy_preview', 'run_benchmark').",
			},
			"description": map[string]interface{}{
				"type":        "string",
				"description": "High-level summary of what the skill accomplishes and when it should be invoked (maximum 500 characters).",
			},
			"triggers": map[string]interface{}{
				"type": "array",
				"items": map[string]interface{}{
					"type": "string",
				},
				"description": "Keywords or trigger phrases indicating when this skill should be used (1-20 phrases, max 100 characters each).",
			},
			"tags": map[string]interface{}{
				"type": "array",
				"items": map[string]interface{}{
					"type": "string",
				},
				"description": "Category tags for search and categorization (1-10 tags, max 50 characters each, e.g. ['deployment', 'git']).",
			},
			"instructions_markdown": map[string]interface{}{
				"type":        "string",
				"description": "Procedural instructions in markdown format detailing step-by-step actions and tool invocations (maximum 24KB).",
			},
			"revises_skill_id": map[string]interface{}{
				"type":        "string",
				"description": "Optional ID of an existing skill (e.g. 'skill_...') that this new revision updates or supersedes.",
			},
		},
		"required": []string{"name", "description", "triggers", "tags", "instructions_markdown"},
	}

	discoverSkillsToolSchema = map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"query": map[string]interface{}{
				"type":        "string",
				"description": "Semantic query describing tasks, procedures, or workflows to discover relevant skills.",
			},
			"tags": map[string]interface{}{
				"type": "array",
				"items": map[string]interface{}{
					"type": "string",
				},
				"description": "Optional filter restricting discovery to skills having specific tags.",
			},
			"limit": map[string]interface{}{
				"type":        "integer",
				"description": "Maximum number of skill summaries to return (1-10, default 5).",
			},
		},
		"required": []string{"query"},
	}

	loadSkillToolSchema = map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"skill_id": map[string]interface{}{
				"type":        "string",
				"description": "The stable skill identifier (e.g. 'skill_abc123') discovered via discover_skills.",
			},
		},
		"required": []string{"skill_id"},
	}
)

func isValidSkillName(name string) bool {
	if len(name) == 0 || len(name) > 64 {
		return false
	}
	for _, r := range name {
		if (r < 'a' || r > 'z') && (r < '0' || r > '9') && r != '_' {
			return false
		}
	}
	return true
}

func (r *Registry) knowledgeToolDefinitions() []openai.Tool {
	return []openai.Tool{
		{
			Type: openai.ToolTypeFunction,
			Function: &openai.FunctionDefinition{
				Name:        "propose_memory",
				Description: "Propose a structured memory (preference, fact, decision, ongoing task) to be stored in long-term memory for this DM. Requires user confirmation before activation.",
				Parameters:  proposeMemoryToolSchema,
			},
		},
		{
			Type: openai.ToolTypeFunction,
			Function: &openai.FunctionDefinition{
				Name:        "discover_memories",
				Description: "Discover memories relevant to a query. Returns lightweight summaries with item IDs, versions, and match scores without loading full content into context budget.",
				Parameters:  discoverMemoriesToolSchema,
			},
		},
		{
			Type: openai.ToolTypeFunction,
			Function: &openai.FunctionDefinition{
				Name:        "load_memory",
				Description: "Load the full active approved content of a structured memory by ID into context. Subject to memory budget limits.",
				Parameters:  loadMemoryToolSchema,
			},
		},
		{
			Type: openai.ToolTypeFunction,
			Function: &openai.FunctionDefinition{
				Name:        "propose_skill",
				Description: "Propose a reusable procedural skill with triggers and step-by-step instructions. Requires user confirmation before activation.",
				Parameters:  proposeSkillToolSchema,
			},
		},
		{
			Type: openai.ToolTypeFunction,
			Function: &openai.FunctionDefinition{
				Name:        "discover_skills",
				Description: "Discover skills relevant to a query or tags. Returns lightweight summaries with skill IDs, names, descriptions, triggers, and tags without loading full instructions.",
				Parameters:  discoverSkillsToolSchema,
			},
		},
		{
			Type: openai.ToolTypeFunction,
			Function: &openai.FunctionDefinition{
				Name:        "load_skill",
				Description: "Load the full procedural instructions of an active approved skill by ID into context. Subject to skill budget limits.",
				Parameters:  loadSkillToolSchema,
			},
		},
	}
}

func (r *Registry) executeProposeMemory(ctx context.Context, argsJSON string) (string, error) {
	session, ok := ChatSessionFromContext(ctx)
	if !ok {
		return "", errors.New("chat session context missing")
	}
	if !session.IsDM {
		return "", errors.New("knowledge tools are only available in direct messages")
	}
	if session.IsScheduled {
		return "", errors.New("knowledge proposals cannot be created from scheduled tasks")
	}
	if session.SourceMessageSeq == nil || *session.SourceMessageSeq <= 0 {
		return "", errors.New("knowledge proposals require direct interactive user message provenance")
	}
	if strings.TrimSpace(session.FSMRunID) == "" {
		return "", errors.New("knowledge proposals require an active FSM run ID")
	}
	if strings.TrimSpace(session.UserID) == "" || strings.TrimSpace(session.ChatID) == "" {
		return "", errors.New("missing user_id or chat_id in session context")
	}

	var args ProposeMemoryArgs
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
		return "", fmt.Errorf("failed to parse propose_memory arguments: %w", err)
	}

	memType := knowledge.MemoryType(strings.TrimSpace(args.Type))
	if !knowledge.IsValidMemoryType(memType) {
		return "", fmt.Errorf("invalid memory type %q: must be one of 'preference', 'fact', 'decision', 'ongoing_task'", args.Type)
	}
	content := strings.TrimSpace(args.Content)
	if content == "" {
		return "", errors.New("memory content cannot be empty")
	}
	if len(content) > 4000 {
		return "", errors.New("memory content exceeds maximum length of 4000 bytes")
	}

	confidence := 1.0
	if args.Confidence != nil {
		confidence = *args.Confidence
		if confidence < 0.0 || confidence > 1.0 {
			return "", fmt.Errorf("confidence must be between 0.0 and 1.0, got %f", confidence)
		}
	}

	if args.TTLDays != nil && (*args.TTLDays < 1 || *args.TTLDays > 3650) {
		return "", fmt.Errorf("ttl_days must be positive (between 1 and 3650 days), got %d", *args.TTLDays)
	}

	var targetItemID string
	if strings.TrimSpace(args.RevisesMemoryID) != "" {
		itemID, _, _, err := knowledge.ParseItemOrVersionRef(args.RevisesMemoryID)
		if err != nil || !strings.HasPrefix(itemID, "mem_") {
			return "", fmt.Errorf("invalid revises_memory_id %q: must be a valid memory reference (e.g. 'mem_...')", args.RevisesMemoryID)
		}
		targetItemID = itemID
	}

	if r.knowledgeStoreProvider == nil {
		return "", errors.New("knowledge store provider is not configured")
	}

	store, err := r.knowledgeStoreProvider.GetKnowledgeStore(ctx, session.ChatID, session.IsDM)
	if err != nil {
		return "", fmt.Errorf("failed to get knowledge store: %w", err)
	}

	if targetItemID != "" {
		existing, err := store.GetItem(ctx, targetItemID)
		if err != nil {
			return "", fmt.Errorf("revises_memory_id target %s not found: %w", targetItemID, err)
		}
		if existing.ChatID != session.ChatID || existing.UserID != session.UserID {
			return "", fmt.Errorf("revises_memory_id target %s not found or access denied", targetItemID)
		}
	}

	now := time.Now().Unix()
	provenance := knowledge.Provenance{
		ChatID:           session.ChatID,
		UserID:           session.UserID,
		SourceMessageSeq: session.SourceMessageSeq,
		FSMRunID:         session.FSMRunID,
		ContentHash:      knowledge.ComputeContentHash(content),
		CreatedAt:        now,
	}

	ver := &knowledge.MemoryVersion{
		ItemID:     targetItemID,
		Type:       memType,
		Content:    content,
		Confidence: confidence,
		TTLDays:    args.TTLDays,
		Provenance: provenance,
	}
	var item *knowledge.KnowledgeItem
	if targetItemID != "" {
		item = &knowledge.KnowledgeItem{ID: targetItemID, ChatID: session.ChatID, UserID: session.UserID}
	} else {
		item = &knowledge.KnowledgeItem{ChatID: session.ChatID, UserID: session.UserID}
	}

	createdItem, createdVer, err := store.ProposeMemory(ctx, item, ver)
	if err != nil {
		return "", fmt.Errorf("failed to propose memory: %w", err)
	}

	resp := map[string]interface{}{
		"item_id":           createdItem.ID,
		"version_id":        createdVer.ID,
		"revision":          createdVer.Revision,
		"status":            string(createdVer.Status),
		"approval_command":  fmt.Sprintf("/memory approve %s", createdVer.ID),
		"rejection_command": fmt.Sprintf("/memory reject %s", createdVer.ID),
		"message":           fmt.Sprintf("Memory proposed successfully (revision %d). Waiting for user approval (run '/memory approve %s').", createdVer.Revision, createdVer.ID),
	}
	out, err := json.Marshal(resp)
	if err != nil {
		return "", fmt.Errorf("failed to serialize propose_memory response: %w", err)
	}
	return string(out), nil
}

func (r *Registry) executeDiscoverMemories(ctx context.Context, argsJSON string) (string, error) {
	session, ok := ChatSessionFromContext(ctx)
	if !ok {
		return "", errors.New("chat session context missing")
	}
	if !session.IsDM {
		return "", errors.New("knowledge tools are only available in direct messages")
	}
	if session.IsScheduled {
		return "", errors.New("knowledge tools are not available in scheduled tasks")
	}
	if strings.TrimSpace(session.UserID) == "" || strings.TrimSpace(session.ChatID) == "" {
		return "", errors.New("missing user_id or chat_id in session context")
	}

	var args DiscoverMemoriesArgs
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
		return "", fmt.Errorf("failed to parse discover_memories arguments: %w", err)
	}

	query := strings.TrimSpace(args.Query)
	if query == "" {
		return `{"memories":[],"count":0}`, nil
	}

	defaultLimit := r.knowledgeDefaultDiscoveryLimit
	if defaultLimit <= 0 {
		defaultLimit = KnowledgeDefaultDiscoveryLimit
	}
	maxLimit := r.knowledgeMaxDiscoveryLimit
	if maxLimit <= 0 {
		maxLimit = KnowledgeMaxDiscoveryLimit
	}
	limit := args.Limit
	if limit <= 0 {
		limit = defaultLimit
	}
	if limit > maxLimit {
		limit = maxLimit
	}

	var memTypes []knowledge.MemoryType
	for _, t := range args.Types {
		mt := knowledge.MemoryType(strings.TrimSpace(t))
		if !knowledge.IsValidMemoryType(mt) {
			return "", fmt.Errorf("invalid memory type %q in types filter", t)
		}
		memTypes = append(memTypes, mt)
	}

	if r.knowledgeSearcherProvider == nil {
		return "", errors.New("knowledge searcher provider is not configured")
	}

	searcher, err := r.knowledgeSearcherProvider.GetKnowledgeSearcher(ctx, session.ChatID, session.IsDM, session.UserID)
	if err != nil {
		return "", fmt.Errorf("failed to get knowledge searcher: %w", err)
	}

	results, err := searcher.SearchMemories(ctx, query, memTypes, limit)
	if err != nil {
		return "", fmt.Errorf("memory discovery failed: %w", err)
	}

	resp := map[string]interface{}{
		"memories": results,
		"count":    len(results),
	}
	out, err := json.Marshal(resp)
	if err != nil {
		return "", fmt.Errorf("failed to serialize discover_memories response: %w", err)
	}
	return string(out), nil
}

func (r *Registry) executeLoadMemory(ctx context.Context, argsJSON string) (string, error) {
	session, ok := ChatSessionFromContext(ctx)
	if !ok {
		return "", errors.New("chat session context missing")
	}
	if !session.IsDM {
		return "", errors.New("knowledge tools are only available in direct messages")
	}
	if session.IsScheduled {
		return "", errors.New("knowledge tools are not available in scheduled tasks")
	}
	if strings.TrimSpace(session.UserID) == "" || strings.TrimSpace(session.ChatID) == "" {
		return "", errors.New("missing user_id or chat_id in session context")
	}
	if session.KnowledgeBudget == nil {
		return "", errors.New("knowledge budget tracker is not initialized")
	}

	var args LoadMemoryArgs
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
		return "", fmt.Errorf("failed to parse load_memory arguments: %w", err)
	}

	memID := strings.TrimSpace(args.MemoryID)
	if memID == "" {
		return "", errors.New("memory_id cannot be empty")
	}
	itemID, rev, hasRev, err := knowledge.ParseItemOrVersionRef(memID)
	if err != nil || !strings.HasPrefix(itemID, "mem_") {
		return "", fmt.Errorf("invalid memory_id %q: must start with 'mem_'", memID)
	}

	// Check cache in session knowledge budget
	if cached, ok := session.KnowledgeBudget.GetCachedResult(itemID); ok {
		if hasRev {
			cachedVerID, _, _, ok := session.KnowledgeBudget.GetCached(itemID)
			if ok && cachedVerID != knowledge.FormatVersionID(itemID, rev) {
				return "", fmt.Errorf("version %s is not the active version for memory %s", memID, itemID)
			}
		}
		return cached, nil
	}

	if r.knowledgeStoreProvider == nil {
		return "", errors.New("knowledge store provider is not configured")
	}

	store, err := r.knowledgeStoreProvider.GetKnowledgeStore(ctx, session.ChatID, session.IsDM)
	if err != nil {
		return "", fmt.Errorf("failed to get knowledge store: %w", err)
	}

	item, err := store.GetItem(ctx, itemID)
	if err != nil {
		return "", fmt.Errorf("memory %s not found: %w", itemID, err)
	}
	if item.Kind != knowledge.KindMemory {
		return "", fmt.Errorf("item %s is not a memory", itemID)
	}
	if item.Status != knowledge.ItemStatusActive {
		return "", fmt.Errorf("memory %s is not active (status: %s)", itemID, item.Status)
	}
	if item.ChatID != session.ChatID || item.UserID != session.UserID {
		return "", fmt.Errorf("memory %s not found", itemID)
	}
	if item.ActiveVersionID == "" {
		return "", fmt.Errorf("memory %s has no active version", itemID)
	}

	if hasRev {
		expectedVerID := knowledge.FormatVersionID(item.ID, rev)
		if expectedVerID != item.ActiveVersionID {
			return "", fmt.Errorf("version %s is not the active version for memory %s", memID, itemID)
		}
	}

	ver, err := store.GetMemoryVersion(ctx, item.ActiveVersionID)
	if err != nil {
		return "", fmt.Errorf("failed to get active version for memory %s: %w", itemID, err)
	}
	if ver.ItemID != item.ID {
		return "", fmt.Errorf("memory version %s item mismatch: expected %s, got %s", ver.ID, item.ID, ver.ItemID)
	}
	if ver.ID != item.ActiveVersionID {
		return "", fmt.Errorf("memory version %s is not active version %s", ver.ID, item.ActiveVersionID)
	}
	if ver.Provenance.ChatID != session.ChatID || ver.Provenance.UserID != session.UserID {
		return "", fmt.Errorf("memory %s provenance mismatch with session", itemID)
	}
	if ver.Status != knowledge.VersionStatusApproved {
		return "", fmt.Errorf("memory version %s is not approved (status: %s)", ver.ID, ver.Status)
	}
	if ver.ExpiresAt != nil && time.Now().Unix() >= *ver.ExpiresAt {
		return "", fmt.Errorf("memory %s has expired", itemID)
	}

	resp := map[string]interface{}{
		"item_id":       item.ID,
		"version_id":    ver.ID,
		"type":          string(ver.Type),
		"content":       ver.Content,
		"content_bytes": len([]byte(ver.Content)),
	}
	out, err := json.Marshal(resp)
	if err != nil {
		return "", fmt.Errorf("failed to serialize load_memory response: %w", err)
	}
	outStr := string(out)

	resultJSON, alreadyLoaded, err := session.KnowledgeBudget.ChargeOrGetMemory(item.ID, ver.ID, ver.Content, outStr)
	if err != nil {
		return "", fmt.Errorf("budget exceeded: %w", err)
	}
	if alreadyLoaded {
		return resultJSON, nil
	}
	return outStr, nil
}

func (r *Registry) executeProposeSkill(ctx context.Context, argsJSON string) (string, error) {
	session, ok := ChatSessionFromContext(ctx)
	if !ok {
		return "", errors.New("chat session context missing")
	}
	if !session.IsDM {
		return "", errors.New("knowledge tools are only available in direct messages")
	}
	if session.IsScheduled {
		return "", errors.New("knowledge proposals cannot be created from scheduled tasks")
	}
	if session.SourceMessageSeq == nil || *session.SourceMessageSeq <= 0 {
		return "", errors.New("knowledge proposals require direct interactive user message provenance")
	}
	if strings.TrimSpace(session.FSMRunID) == "" {
		return "", errors.New("knowledge proposals require an active FSM run ID")
	}
	if strings.TrimSpace(session.UserID) == "" || strings.TrimSpace(session.ChatID) == "" {
		return "", errors.New("missing user_id or chat_id in session context")
	}

	var args ProposeSkillArgs
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
		return "", fmt.Errorf("failed to parse propose_skill arguments: %w", err)
	}

	name := strings.TrimSpace(args.Name)
	if !isValidSkillName(name) {
		return "", fmt.Errorf("invalid skill name %q: must consist of 1-64 lowercase alphanumeric characters and underscores", args.Name)
	}

	desc := strings.TrimSpace(args.Description)
	if desc == "" {
		return "", errors.New("skill description cannot be empty")
	}
	if utf8.RuneCountInString(desc) > 500 {
		return "", fmt.Errorf("skill description exceeds maximum length of 500 characters (got %d)", utf8.RuneCountInString(desc))
	}

	var triggers []string
	for _, trig := range args.Triggers {
		t := strings.TrimSpace(trig)
		if t != "" {
			if utf8.RuneCountInString(t) > 100 {
				return "", fmt.Errorf("trigger %q exceeds maximum length of 100 characters", t)
			}
			triggers = append(triggers, t)
		}
	}
	if len(triggers) == 0 {
		return "", errors.New("skill triggers cannot be empty: at least one trigger phrase is required")
	}
	if len(triggers) > 20 {
		return "", fmt.Errorf("too many triggers: maximum 20 triggers allowed (got %d)", len(triggers))
	}

	var tags []string
	for _, tag := range args.Tags {
		tg := strings.TrimSpace(tag)
		if tg != "" {
			if utf8.RuneCountInString(tg) > 50 {
				return "", fmt.Errorf("tag %q exceeds maximum length of 50 characters", tg)
			}
			tags = append(tags, tg)
		}
	}
	if len(tags) == 0 {
		return "", errors.New("skill tags cannot be empty: at least one category tag is required")
	}
	if len(tags) > 10 {
		return "", fmt.Errorf("too many tags: maximum 10 tags allowed (got %d)", len(tags))
	}

	instructions := strings.TrimSpace(args.InstructionsMarkdown)
	if instructions == "" {
		return "", errors.New("skill instructions_markdown cannot be empty")
	}
	if len(instructions) > 24*1024 {
		return "", errors.New("skill instructions_markdown exceeds maximum allowable size (24KB)")
	}

	var targetItemID string
	if strings.TrimSpace(args.RevisesSkillID) != "" {
		itemID, _, _, err := knowledge.ParseItemOrVersionRef(args.RevisesSkillID)
		if err != nil || !strings.HasPrefix(itemID, "skill_") {
			return "", fmt.Errorf("invalid revises_skill_id %q: must be a valid skill reference (e.g. 'skill_...')", args.RevisesSkillID)
		}
		targetItemID = itemID
	}

	if r.knowledgeStoreProvider == nil {
		return "", errors.New("knowledge store provider is not configured")
	}

	store, err := r.knowledgeStoreProvider.GetKnowledgeStore(ctx, session.ChatID, session.IsDM)
	if err != nil {
		return "", fmt.Errorf("failed to get knowledge store: %w", err)
	}

	if targetItemID != "" {
		existing, err := store.GetItem(ctx, targetItemID)
		if err != nil {
			return "", fmt.Errorf("revises_skill_id target %s not found: %w", targetItemID, err)
		}
		if existing.ChatID != session.ChatID || existing.UserID != session.UserID {
			return "", fmt.Errorf("revises_skill_id target %s not found or access denied", targetItemID)
		}
	}

	now := time.Now().Unix()
	provenance := knowledge.Provenance{
		ChatID:           session.ChatID,
		UserID:           session.UserID,
		SourceMessageSeq: session.SourceMessageSeq,
		FSMRunID:         session.FSMRunID,
		ContentHash:      knowledge.ComputeContentHash(instructions),
		CreatedAt:        now,
	}

	ver := &knowledge.SkillVersion{
		ItemID:               targetItemID,
		Name:                 name,
		Description:          desc,
		Triggers:             triggers,
		Tags:                 tags,
		InstructionsMarkdown: instructions,
		Provenance:           provenance,
	}
	var item *knowledge.KnowledgeItem
	if targetItemID != "" {
		item = &knowledge.KnowledgeItem{ID: targetItemID, ChatID: session.ChatID, UserID: session.UserID}
	} else {
		item = &knowledge.KnowledgeItem{ChatID: session.ChatID, UserID: session.UserID}
	}

	createdItem, createdVer, err := store.ProposeSkill(ctx, item, ver)
	if err != nil {
		return "", fmt.Errorf("failed to propose skill: %w", err)
	}

	resp := map[string]interface{}{
		"item_id":           createdItem.ID,
		"version_id":        createdVer.ID,
		"revision":          createdVer.Revision,
		"status":            string(createdVer.Status),
		"approval_command":  fmt.Sprintf("/skill approve %s", createdVer.ID),
		"rejection_command": fmt.Sprintf("/skill reject %s", createdVer.ID),
		"message":           fmt.Sprintf("Skill proposed successfully (revision %d). Waiting for user approval (run '/skill approve %s').", createdVer.Revision, createdVer.ID),
	}
	out, err := json.Marshal(resp)
	if err != nil {
		return "", fmt.Errorf("failed to serialize propose_skill response: %w", err)
	}
	return string(out), nil
}

func (r *Registry) executeDiscoverSkills(ctx context.Context, argsJSON string) (string, error) {
	session, ok := ChatSessionFromContext(ctx)
	if !ok {
		return "", errors.New("chat session context missing")
	}
	if !session.IsDM {
		return "", errors.New("knowledge tools are only available in direct messages")
	}
	if session.IsScheduled {
		return "", errors.New("knowledge tools are not available in scheduled tasks")
	}
	if strings.TrimSpace(session.UserID) == "" || strings.TrimSpace(session.ChatID) == "" {
		return "", errors.New("missing user_id or chat_id in session context")
	}

	var args DiscoverSkillsArgs
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
		return "", fmt.Errorf("failed to parse discover_skills arguments: %w", err)
	}

	query := strings.TrimSpace(args.Query)
	if query == "" {
		return `{"skills":[],"count":0}`, nil
	}

	defaultLimit := r.knowledgeDefaultDiscoveryLimit
	if defaultLimit <= 0 {
		defaultLimit = KnowledgeDefaultDiscoveryLimit
	}
	maxLimit := r.knowledgeMaxDiscoveryLimit
	if maxLimit <= 0 {
		maxLimit = KnowledgeMaxDiscoveryLimit
	}
	limit := args.Limit
	if limit <= 0 {
		limit = defaultLimit
	}
	if limit > maxLimit {
		limit = maxLimit
	}

	if r.knowledgeSearcherProvider == nil {
		return "", errors.New("knowledge searcher provider is not configured")
	}

	searcher, err := r.knowledgeSearcherProvider.GetKnowledgeSearcher(ctx, session.ChatID, session.IsDM, session.UserID)
	if err != nil {
		return "", fmt.Errorf("failed to get knowledge searcher: %w", err)
	}

	results, err := searcher.SearchSkills(ctx, query, args.Tags, limit)
	if err != nil {
		return "", fmt.Errorf("skill discovery failed: %w", err)
	}

	resp := map[string]interface{}{
		"skills": results,
		"count":  len(results),
	}
	out, err := json.Marshal(resp)
	if err != nil {
		return "", fmt.Errorf("failed to serialize discover_skills response: %w", err)
	}
	return string(out), nil
}

func (r *Registry) executeLoadSkill(ctx context.Context, argsJSON string) (string, error) {
	session, ok := ChatSessionFromContext(ctx)
	if !ok {
		return "", errors.New("chat session context missing")
	}
	if !session.IsDM {
		return "", errors.New("knowledge tools are only available in direct messages")
	}
	if session.IsScheduled {
		return "", errors.New("knowledge tools are not available in scheduled tasks")
	}
	if strings.TrimSpace(session.UserID) == "" || strings.TrimSpace(session.ChatID) == "" {
		return "", errors.New("missing user_id or chat_id in session context")
	}
	if session.KnowledgeBudget == nil {
		return "", errors.New("knowledge budget tracker is not initialized")
	}

	var args LoadSkillArgs
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
		return "", fmt.Errorf("failed to parse load_skill arguments: %w", err)
	}

	skillID := strings.TrimSpace(args.SkillID)
	if skillID == "" {
		return "", errors.New("skill_id cannot be empty")
	}
	itemID, rev, hasRev, err := knowledge.ParseItemOrVersionRef(skillID)
	if err != nil || !strings.HasPrefix(itemID, "skill_") {
		return "", fmt.Errorf("invalid skill_id %q: must start with 'skill_'", skillID)
	}

	// Check cache in session knowledge budget
	if cached, ok := session.KnowledgeBudget.GetCachedResult(itemID); ok {
		if hasRev {
			cachedVerID, _, _, ok := session.KnowledgeBudget.GetCached(itemID)
			if ok && cachedVerID != knowledge.FormatVersionID(itemID, rev) {
				return "", fmt.Errorf("version %s is not the active version for skill %s", skillID, itemID)
			}
		}
		return cached, nil
	}

	if r.knowledgeStoreProvider == nil {
		return "", errors.New("knowledge store provider is not configured")
	}

	store, err := r.knowledgeStoreProvider.GetKnowledgeStore(ctx, session.ChatID, session.IsDM)
	if err != nil {
		return "", fmt.Errorf("failed to get knowledge store: %w", err)
	}

	item, err := store.GetItem(ctx, itemID)
	if err != nil {
		return "", fmt.Errorf("skill %s not found: %w", itemID, err)
	}
	if item.Kind != knowledge.KindSkill {
		return "", fmt.Errorf("item %s is not a skill", itemID)
	}
	if item.Status != knowledge.ItemStatusActive {
		return "", fmt.Errorf("skill %s is not active (status: %s)", itemID, item.Status)
	}
	if item.ChatID != session.ChatID || item.UserID != session.UserID {
		return "", fmt.Errorf("skill %s not found", itemID)
	}
	if item.ActiveVersionID == "" {
		return "", fmt.Errorf("skill %s has no active version", itemID)
	}

	if hasRev {
		expectedVerID := knowledge.FormatVersionID(item.ID, rev)
		if expectedVerID != item.ActiveVersionID {
			return "", fmt.Errorf("version %s is not the active version for skill %s", skillID, itemID)
		}
	}

	ver, err := store.GetSkillVersion(ctx, item.ActiveVersionID)
	if err != nil {
		return "", fmt.Errorf("failed to get active version for skill %s: %w", itemID, err)
	}
	if ver.ItemID != item.ID {
		return "", fmt.Errorf("skill version %s item mismatch: expected %s, got %s", ver.ID, item.ID, ver.ItemID)
	}
	if ver.ID != item.ActiveVersionID {
		return "", fmt.Errorf("skill version %s is not active version %s", ver.ID, item.ActiveVersionID)
	}
	if ver.Provenance.ChatID != session.ChatID || ver.Provenance.UserID != session.UserID {
		return "", fmt.Errorf("skill %s provenance mismatch with session", itemID)
	}
	if ver.Status != knowledge.VersionStatusApproved {
		return "", fmt.Errorf("skill version %s is not approved (status: %s)", ver.ID, ver.Status)
	}

	totalBytes := len([]byte(ver.Name)) + len([]byte(ver.Description)) + len([]byte(ver.InstructionsMarkdown))

	resp := map[string]interface{}{
		"item_id":               item.ID,
		"version_id":            ver.ID,
		"name":                  ver.Name,
		"description":           ver.Description,
		"instructions_markdown": ver.InstructionsMarkdown,
		"content_bytes":         totalBytes,
	}
	out, err := json.Marshal(resp)
	if err != nil {
		return "", fmt.Errorf("failed to serialize load_skill response: %w", err)
	}
	outStr := string(out)

	resultJSON, alreadyLoaded, err := session.KnowledgeBudget.ChargeOrGetSkill(item.ID, ver.ID, ver.InstructionsMarkdown, totalBytes, outStr)
	if err != nil {
		return "", fmt.Errorf("budget exceeded: %w", err)
	}
	if alreadyLoaded {
		return resultJSON, nil
	}
	return outStr, nil
}
