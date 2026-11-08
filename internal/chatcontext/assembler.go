package chatcontext

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	"bob/internal/knowledge"
	"bob/internal/memory"

	openai "github.com/sashabaranov/go-openai"
)

const (
	DefaultSystemPromptBudgetBytes      = 32 * 1024  // 32 KiB
	DefaultStructuredMemoryBudgetBytes  = 16 * 1024  // 16 KiB
	DefaultProceduralSkillBudgetBytes   = 24 * 1024  // 24 KiB
	DefaultHistoricalContextBudgetBytes = 16 * 1024  // 16 KiB
	DefaultRecentTurnsBudgetBytes       = 128 * 1024 // 128 KiB
	EstimatedImageBytes                 = 4096       // Approximately 1024 tokens, independent of encoded image size.
)

// SourceBudgets defines per-source byte budgets enforced during context assembly.
type SourceBudgets struct {
	SystemPromptBytes      int `json:"system_prompt_bytes"`
	StructuredMemoryBytes  int `json:"structured_memory_bytes"`
	ProceduralSkillBytes   int `json:"procedural_skill_bytes"`
	HistoricalContextBytes int `json:"historical_context_bytes"`
	RecentTurnsBytes       int `json:"recent_turns_bytes"`
}

// DefaultSourceBudgets returns standard default byte budgets for all context sources.
func DefaultSourceBudgets() SourceBudgets {
	return SourceBudgets{
		SystemPromptBytes:      DefaultSystemPromptBudgetBytes,
		StructuredMemoryBytes:  DefaultStructuredMemoryBudgetBytes,
		ProceduralSkillBytes:   DefaultProceduralSkillBudgetBytes,
		HistoricalContextBytes: DefaultHistoricalContextBudgetBytes,
		RecentTurnsBytes:       DefaultRecentTurnsBudgetBytes,
	}
}

func (b SourceBudgets) withDefaults() SourceBudgets {
	def := DefaultSourceBudgets()
	if b.SystemPromptBytes <= 0 {
		b.SystemPromptBytes = def.SystemPromptBytes
	}
	if b.StructuredMemoryBytes <= 0 {
		b.StructuredMemoryBytes = def.StructuredMemoryBytes
	}
	if b.ProceduralSkillBytes <= 0 {
		b.ProceduralSkillBytes = def.ProceduralSkillBytes
	}
	if b.HistoricalContextBytes <= 0 {
		b.HistoricalContextBytes = def.HistoricalContextBytes
	}
	if b.RecentTurnsBytes <= 0 {
		b.RecentTurnsBytes = def.RecentTurnsBytes
	}
	return b
}

// AssemblyInput encapsulates all candidate context sources for a single conversational turn.
type AssemblyInput struct {
	ChatID             string                    `json:"chat_id,omitempty"`
	SystemPrompt       string                    `json:"system_prompt"`
	Memories           []knowledge.MemoryVersion `json:"memories,omitempty"`
	Skills             []knowledge.SkillVersion  `json:"skills,omitempty"`
	Episodes           []memory.Episode          `json:"episodes,omitempty"`
	HistoricalItems    []memory.MemoryItem       `json:"historical_items,omitempty"`
	RecentEntries      []Entry                   `json:"recent_entries,omitempty"`
	ContinuationPrompt string                    `json:"continuation_prompt,omitempty"`
	Now                time.Time                 `json:"-"`
}

// ManifestItemUsage records the version identity and byte footprint of an injected knowledge item.
type ManifestItemUsage struct {
	ItemID    string `json:"item_id"`
	VersionID string `json:"version_id"`
	Bytes     int    `json:"bytes"`
}

// ManifestHistoricalSpan records an injected episode or raw conversation chunk and its sequence span.
type ManifestHistoricalSpan struct {
	ID        string `json:"id"`
	EpisodeID string `json:"episode_id,omitempty"`
	StartSeq  int64  `json:"start_seq,omitempty"`
	EndSeq    int64  `json:"end_seq,omitempty"`
	IsEpisode bool   `json:"is_episode,omitempty"`
	Bytes     int    `json:"bytes"`
}

// ManifestRingBufferSpan records the sequence range and entry counts of the retained ring buffer.
type ManifestRingBufferSpan struct {
	MinSeq       int64 `json:"min_seq"`
	MaxSeq       int64 `json:"max_seq"`
	EntryCount   int   `json:"entry_count"`
	DroppedCount int   `json:"dropped_count,omitempty"`
}

// SourceUsage tracks byte and estimated token usage for a single context source.
type SourceUsage struct {
	Bytes             int  `json:"bytes"`
	BudgetBytes       int  `json:"budget_bytes"`
	EstimatedTokens   int  `json:"estimated_tokens"`
	Count             int  `json:"count,omitempty"`
	OmittedCount      int  `json:"omitted_count,omitempty"`
	DeduplicatedCount int  `json:"deduplicated_count,omitempty"`
	Truncated         bool `json:"truncated,omitempty"`
}

// SourceUsageSummary aggregates per-source usage metrics across all assembled horizons.
type SourceUsageSummary struct {
	SystemPrompt       SourceUsage `json:"system_prompt"`
	StructuredMemories SourceUsage `json:"structured_memories"`
	ProceduralSkills   SourceUsage `json:"procedural_skills"`
	HistoricalContext  SourceUsage `json:"historical_context"`
	RecentTurns        SourceUsage `json:"recent_turns"`
	TotalBytes         int         `json:"total_bytes"`
	TotalTokens        int         `json:"total_estimated_tokens"`
}

// AssemblyManifest provides an inspectable provenance record of everything assembled into a turn's context.
type AssemblyManifest struct {
	MemoryVersions        []string                 `json:"memory_versions,omitempty"`
	Memories              []ManifestItemUsage      `json:"memories,omitempty"`
	OmittedMemoryVersions []string                 `json:"omitted_memory_versions,omitempty"`
	SkillVersions         []string                 `json:"skill_versions,omitempty"`
	Skills                []ManifestItemUsage      `json:"skills,omitempty"`
	OmittedSkillVersions  []string                 `json:"omitted_skill_versions,omitempty"`
	EpisodeIDs            []string                 `json:"episode_ids,omitempty"`
	Episodes              []ManifestHistoricalSpan `json:"episodes,omitempty"`
	ChunkIDs              []string                 `json:"chunk_ids,omitempty"`
	Chunks                []ManifestHistoricalSpan `json:"chunks,omitempty"`
	DeduplicatedChunkIDs  []string                 `json:"deduplicated_chunk_ids,omitempty"`
	OmittedHistoricalIDs  []string                 `json:"omitted_historical_ids,omitempty"`
	RingBuffer            ManifestRingBufferSpan   `json:"ring_buffer"`
	Budgets               SourceBudgets            `json:"budgets"`
	Usage                 SourceUsageSummary       `json:"usage"`
}

// JSON serializes the AssemblyManifest to compact JSON.
func (m AssemblyManifest) JSON() (string, error) {
	b, err := json.Marshal(m)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// AssemblyResult contains the assembled system prompt, conversation messages, and provenance manifest.
type AssemblyResult struct {
	SystemPrompt string                         `json:"system_prompt"`
	Messages     []openai.ChatCompletionMessage `json:"messages"`
	Manifest     AssemblyManifest               `json:"manifest"`
	ManifestJSON json.RawMessage                `json:"manifest_json,omitempty"`
}

// Assembler deterministically constructs multi-source LLM context under explicit per-source byte budgets.
type Assembler struct {
	budgets SourceBudgets
}

// NewAssembler creates a new Assembler with the specified per-source byte budgets.
// Any zero or negative budget fields are populated from DefaultSourceBudgets().
func NewAssembler(budgets ...SourceBudgets) *Assembler {
	var b SourceBudgets
	if len(budgets) > 0 {
		b = budgets[0]
	}
	return &Assembler{
		budgets: b.withDefaults(),
	}
}

// Budgets returns the effective per-source byte budgets configured on the Assembler.
func (a *Assembler) Budgets() SourceBudgets {
	if a == nil {
		return DefaultSourceBudgets()
	}
	return a.budgets.withDefaults()
}

// Assemble constructs the turn's initial LLM context from all 5 sources, enforcing per-source
// byte budgets, deduplicating historical chunks overlapping with retained ring-buffer sequence
// numbers, and producing an AssemblyManifest.
func (a *Assembler) Assemble(input AssemblyInput) AssemblyResult {
	budgets := a.Budgets()
	manifest := AssemblyManifest{
		Budgets: budgets,
		Usage: SourceUsageSummary{
			SystemPrompt:       SourceUsage{BudgetBytes: budgets.SystemPromptBytes},
			StructuredMemories: SourceUsage{BudgetBytes: budgets.StructuredMemoryBytes},
			ProceduralSkills:   SourceUsage{BudgetBytes: budgets.ProceduralSkillBytes},
			HistoricalContext:  SourceUsage{BudgetBytes: budgets.HistoricalContextBytes},
			RecentTurns:        SourceUsage{BudgetBytes: budgets.RecentTurnsBytes},
		},
	}

	nowUnix := time.Now().UTC().Unix()
	if !input.Now.IsZero() {
		nowUnix = input.Now.UTC().Unix()
	}

	// 1. Recent conversation ring buffer (retained suffix + continuation prompt within RecentTurnsBytes)
	needsContinuation := len(input.RecentEntries) > 0 && input.RecentEntries[len(input.RecentEntries)-1].Role == openai.ChatMessageRoleAssistant
	var continuationText string
	var continuationTruncated bool
	entryBudget := budgets.RecentTurnsBytes
	if needsContinuation {
		continuationText = strings.TrimSpace(input.ContinuationPrompt)
		if continuationText == "" {
			continuationText = "Please continue."
		}
		maxContBytes := budgets.RecentTurnsBytes / 2
		if len(continuationText) > maxContBytes {
			continuationText = truncateUTF8Bytes(continuationText, maxContBytes)
			continuationTruncated = true
		}
		entryBudget = budgets.RecentTurnsBytes - len(continuationText)
	}

	retainedEntries, recentBytes, droppedEntries, recentTruncated := selectRecentEntriesWithinBudget(input.RecentEntries, entryBudget)
	if needsContinuation && len(retainedEntries) > 0 && retainedEntries[len(retainedEntries)-1].Role == openai.ChatMessageRoleAssistant && continuationText != "" {
		recentBytes += len(continuationText)
	}
	manifest.RingBuffer.EntryCount = len(retainedEntries)
	manifest.RingBuffer.DroppedCount = droppedEntries
	manifest.Usage.RecentTurns.Bytes = recentBytes
	manifest.Usage.RecentTurns.EstimatedTokens = estimateTokens(recentBytes)
	manifest.Usage.RecentTurns.Count = len(retainedEntries)
	manifest.Usage.RecentTurns.OmittedCount = droppedEntries
	manifest.Usage.RecentTurns.Truncated = recentTruncated || continuationTruncated || droppedEntries > 0

	bufferedSeqSet := make(map[int64]bool, len(retainedEntries))
	var ringMinSeq, ringMaxSeq int64
	for _, e := range retainedEntries {
		if e.Seq > 0 {
			bufferedSeqSet[e.Seq] = true
			if ringMinSeq == 0 || e.Seq < ringMinSeq {
				ringMinSeq = e.Seq
			}
			if e.Seq > ringMaxSeq {
				ringMaxSeq = e.Seq
			}
		}
	}
	manifest.RingBuffer.MinSeq = ringMinSeq
	manifest.RingBuffer.MaxSeq = ringMaxSeq

	// 2. Pinned system prompt (within SystemPromptBytes)
	basePrompt := strings.TrimSpace(input.SystemPrompt)
	sysTruncated := false
	if len(basePrompt) > budgets.SystemPromptBytes {
		basePrompt = truncateUTF8Bytes(basePrompt, budgets.SystemPromptBytes)
		sysTruncated = true
	}
	sysBytes := len(basePrompt)
	manifest.Usage.SystemPrompt.Bytes = sysBytes
	manifest.Usage.SystemPrompt.EstimatedTokens = estimateTokens(sysBytes)
	if sysBytes > 0 {
		manifest.Usage.SystemPrompt.Count = 1
	}
	manifest.Usage.SystemPrompt.Truncated = sysTruncated
	promptLenSoFar := sysBytes

	// 3. Approved structured memories (within StructuredMemoryBytes, including section header and separators)
	var memoryLines []string
	seenMemories := make(map[string]bool)
	memUsedBytes := 0
	for _, m := range input.Memories {
		if m.Status != knowledge.VersionStatusApproved {
			continue
		}
		if m.ExpiresAt != nil && nowUnix >= *m.ExpiresAt {
			continue
		}
		content := strings.TrimSpace(m.Content)
		if content == "" {
			continue
		}
		verID, itemID := resolveMemoryIdentity(m)
		dedupKey := verID
		if itemID != "" {
			dedupKey = itemID
		}
		if seenMemories[dedupKey] || seenMemories[verID] {
			continue
		}
		seenMemories[dedupKey] = true
		seenMemories[verID] = true

		memType := string(m.Type)
		if memType == "" {
			memType = string(knowledge.MemoryTypeFact)
		}
		rendered := fmt.Sprintf("- [%s] (%s): %s", verID, memType, content)
		overhead := 1 // "\n" between lines
		if len(memoryLines) == 0 {
			overhead = len("[Approved Structured Memories]\n")
			if promptLenSoFar > 0 {
				overhead += 2 // leading "\n\n"
			}
		}
		costBytes := overhead + len(rendered)
		if memUsedBytes+costBytes > budgets.StructuredMemoryBytes {
			manifest.OmittedMemoryVersions = append(manifest.OmittedMemoryVersions, verID)
			manifest.Usage.StructuredMemories.OmittedCount++
			manifest.Usage.StructuredMemories.Truncated = true
			continue
		}

		memUsedBytes += costBytes
		promptLenSoFar += costBytes
		manifest.MemoryVersions = append(manifest.MemoryVersions, verID)
		manifest.Memories = append(manifest.Memories, ManifestItemUsage{
			ItemID:    itemID,
			VersionID: verID,
			Bytes:     costBytes,
		})
		memoryLines = append(memoryLines, rendered)
	}
	manifest.Usage.StructuredMemories.Bytes = memUsedBytes
	manifest.Usage.StructuredMemories.EstimatedTokens = estimateTokens(memUsedBytes)
	manifest.Usage.StructuredMemories.Count = len(manifest.Memories)

	// 4. Approved procedural skills (within ProceduralSkillBytes, including section header and separators)
	var skillBlocks []string
	seenSkills := make(map[string]bool)
	skillUsedBytes := 0
	for _, s := range input.Skills {
		if s.Status != knowledge.VersionStatusApproved {
			continue
		}
		instructions := strings.TrimSpace(s.InstructionsMarkdown)
		if instructions == "" {
			continue
		}
		verID, itemID := resolveSkillIdentity(s)
		dedupKey := verID
		if itemID != "" {
			dedupKey = itemID
		}
		if seenSkills[dedupKey] || seenSkills[verID] {
			continue
		}
		seenSkills[dedupKey] = true
		seenSkills[verID] = true

		name := strings.TrimSpace(s.Name)
		if name == "" {
			name = itemID
		}
		desc := strings.TrimSpace(s.Description)
		rendered := formatSkillBlock(verID, name, desc, instructions)
		overhead := 2 // "\n\n" between skill blocks
		if len(skillBlocks) == 0 {
			overhead = len("[Approved Procedural Skills]\n")
			if promptLenSoFar > 0 {
				overhead += 2 // leading "\n\n"
			}
		}
		costBytes := overhead + len(rendered)
		if skillUsedBytes+costBytes > budgets.ProceduralSkillBytes {
			manifest.OmittedSkillVersions = append(manifest.OmittedSkillVersions, verID)
			manifest.Usage.ProceduralSkills.OmittedCount++
			manifest.Usage.ProceduralSkills.Truncated = true
			continue
		}

		skillUsedBytes += costBytes
		promptLenSoFar += costBytes
		manifest.SkillVersions = append(manifest.SkillVersions, verID)
		manifest.Skills = append(manifest.Skills, ManifestItemUsage{
			ItemID:    itemID,
			VersionID: verID,
			Bytes:     costBytes,
		})
		skillBlocks = append(skillBlocks, rendered)
	}
	manifest.Usage.ProceduralSkills.Bytes = skillUsedBytes
	manifest.Usage.ProceduralSkills.EstimatedTokens = estimateTokens(skillUsedBytes)
	manifest.Usage.ProceduralSkills.Count = len(manifest.Skills)

	// 5. Episodic & historical passages (within HistoricalContextBytes, deduplicated against same-chat ring buffer)
	candidates := collectHistoricalCandidates(input.Episodes, input.HistoricalItems)
	var historicalBlocks []string
	seenHistorical := make(map[string]bool)
	histUsedBytes := 0
	targetChatID := strings.TrimSpace(input.ChatID)
	for _, item := range candidates {
		id := strings.TrimPrefix(strings.TrimSpace(item.ID), memory.EpisodeIDPrefix)
		if id == "" {
			id = strings.TrimSpace(item.EpisodeID)
		}
		text := strings.TrimSpace(item.Content)
		if id == "" || text == "" {
			continue
		}
		itemChatID := strings.TrimSpace(item.ChatID)
		dedupKey := id
		if itemChatID != "" {
			dedupKey = itemChatID + "\x00" + id
		}
		if seenHistorical[dedupKey] {
			continue
		}
		seenHistorical[dedupKey] = true

		isEpisode := item.Kind == "episode" || strings.HasPrefix(id, "ep_") || strings.HasPrefix(item.ID, memory.EpisodeIDPrefix)
		startSeq, endSeq := resolveHistoricalSeqRange(id, item.StartSeq, item.EndSeq)

		if !isEpisode && overlapsWithRingBuffer(id, itemChatID, targetChatID, startSeq, endSeq, bufferedSeqSet) {
			manifest.DeduplicatedChunkIDs = append(manifest.DeduplicatedChunkIDs, id)
			manifest.Usage.HistoricalContext.DeduplicatedCount++
			continue
		}

		rendered := formatHistoricalBlock(id, item, isEpisode, startSeq, endSeq, text)
		overhead := 2 // "\n\n" between historical blocks
		if len(historicalBlocks) == 0 {
			overhead = len("[Episodic & Historical Context]\n")
			if promptLenSoFar > 0 {
				overhead += 2 // leading "\n\n"
			}
		}
		costBytes := overhead + len(rendered)
		if histUsedBytes+costBytes > budgets.HistoricalContextBytes {
			manifest.OmittedHistoricalIDs = append(manifest.OmittedHistoricalIDs, id)
			manifest.Usage.HistoricalContext.OmittedCount++
			manifest.Usage.HistoricalContext.Truncated = true
			continue
		}

		histUsedBytes += costBytes
		promptLenSoFar += costBytes
		span := ManifestHistoricalSpan{
			ID:        id,
			EpisodeID: item.EpisodeID,
			StartSeq:  startSeq,
			EndSeq:    endSeq,
			IsEpisode: isEpisode,
			Bytes:     costBytes,
		}
		if isEpisode {
			if span.EpisodeID == "" {
				span.EpisodeID = id
			}
			manifest.EpisodeIDs = append(manifest.EpisodeIDs, id)
			manifest.Episodes = append(manifest.Episodes, span)
		} else {
			manifest.ChunkIDs = append(manifest.ChunkIDs, id)
			manifest.Chunks = append(manifest.Chunks, span)
		}
		historicalBlocks = append(historicalBlocks, rendered)
	}
	manifest.Usage.HistoricalContext.Bytes = histUsedBytes
	manifest.Usage.HistoricalContext.EstimatedTokens = estimateTokens(histUsedBytes)
	manifest.Usage.HistoricalContext.Count = len(manifest.Episodes) + len(manifest.Chunks)

	// Combine sections into assembled SystemPrompt
	var promptBuilder strings.Builder
	promptBuilder.WriteString(basePrompt)
	if len(memoryLines) > 0 {
		if promptBuilder.Len() > 0 {
			promptBuilder.WriteString("\n\n")
		}
		promptBuilder.WriteString("[Approved Structured Memories]\n")
		promptBuilder.WriteString(strings.Join(memoryLines, "\n"))
	}
	if len(skillBlocks) > 0 {
		if promptBuilder.Len() > 0 {
			promptBuilder.WriteString("\n\n")
		}
		promptBuilder.WriteString("[Approved Procedural Skills]\n")
		promptBuilder.WriteString(strings.Join(skillBlocks, "\n\n"))
	}
	if len(historicalBlocks) > 0 {
		if promptBuilder.Len() > 0 {
			promptBuilder.WriteString("\n\n")
		}
		promptBuilder.WriteString("[Episodic & Historical Context]\n")
		promptBuilder.WriteString(strings.Join(historicalBlocks, "\n\n"))
	}
	assembledSystemPrompt := promptBuilder.String()

	// Build conversation messages from retained ring-buffer entries
	msgs := make([]openai.ChatCompletionMessage, 0, len(retainedEntries)+1)
	for _, e := range retainedEntries {
		msgs = append(msgs, entryToLLMMessage(e))
	}
	if len(msgs) > 0 && msgs[len(msgs)-1].Role == openai.ChatMessageRoleAssistant && continuationText != "" {
		msgs = append(msgs, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: continuationText,
		})
	}

	manifest.Usage.TotalBytes = manifest.Usage.SystemPrompt.Bytes +
		manifest.Usage.StructuredMemories.Bytes +
		manifest.Usage.ProceduralSkills.Bytes +
		manifest.Usage.HistoricalContext.Bytes +
		manifest.Usage.RecentTurns.Bytes
	manifest.Usage.TotalTokens = estimateTokens(manifest.Usage.TotalBytes)

	manifestJSON, _ := json.Marshal(manifest)

	return AssemblyResult{
		SystemPrompt: assembledSystemPrompt,
		Messages:     msgs,
		Manifest:     manifest,
		ManifestJSON: manifestJSON,
	}
}

// Assemble builds the context for chatID using the chat's buffered entries and the provided AssemblyInput.
func (m *Manager) Assemble(chatID string, input AssemblyInput, budgets ...SourceBudgets) AssemblyResult {
	if input.ChatID == "" {
		input.ChatID = chatID
	}
	if len(input.RecentEntries) == 0 && m != nil {
		rb := m.GetOrCreate(chatID)
		input.RecentEntries = rb.Entries()
	}
	return NewAssembler(budgets...).Assemble(input)
}

func selectRecentEntriesWithinBudget(entries []Entry, budgetBytes int) (retained []Entry, usedBytes int, dropped int, truncated bool) {
	if len(entries) == 0 {
		return nil, 0, 0, false
	}
	if budgetBytes <= 0 {
		budgetBytes = DefaultRecentTurnsBudgetBytes
	}

	var reversed []Entry
	for i := len(entries) - 1; i >= 0; i-- {
		e := entries[i]
		if i != len(entries)-1 {
			e.Images = nil
		}
		cost := entryFormattedBytes(e)
		if len(reversed) == 0 && cost > budgetBytes {
			// Ensure the latest turn is always retained by truncating its text content to fit budgetBytes.
			e = truncateEntryToFit(e, budgetBytes)
			reversed = append(reversed, e)
			usedBytes = entryFormattedBytes(e)
			dropped = i
			truncated = true
			break
		}
		if usedBytes+cost > budgetBytes {
			dropped = i + 1
			break
		}
		reversed = append(reversed, e)
		usedBytes += cost
	}

	retained = make([]Entry, len(reversed))
	for i := range reversed {
		retained[i] = reversed[len(reversed)-1-i]
	}
	return retained, usedBytes, dropped, truncated
}

func entryFormattedBytes(e Entry) int {
	msg := entryToLLMMessage(e)
	if len(msg.MultiContent) > 0 {
		total := 0
		for _, part := range msg.MultiContent {
			total += len(part.Text)
			if part.ImageURL != nil {
				total += EstimatedImageBytes
			}
		}
		return total
	}
	return len(msg.Content)
}

func truncateEntryToFit(e Entry, budgetBytes int) Entry {
	role := e.Role
	if role == "" {
		role = openai.ChatMessageRoleUser
	}
	prefixLen := 0
	senderName := strings.TrimSpace(e.SenderName)
	if role == openai.ChatMessageRoleUser && senderName != "" && senderName != e.SenderID {
		prefixLen = len(senderName) + 2 // "<SenderName>: "
	}
	imageBytes := 0
	if role == openai.ChatMessageRoleUser {
		imageBytes = len(e.Images) * EstimatedImageBytes
	}
	if imageBytes > budgetBytes {
		e.Images = nil
		imageBytes = 0
	}
	avail := budgetBytes - imageBytes - prefixLen
	if avail < 0 {
		e.SenderName = ""
		avail = budgetBytes - imageBytes
	}
	e.Content = truncateUTF8Bytes(e.Content, avail)
	return e
}

func truncateUTF8Bytes(s string, maxBytes int) string {
	if maxBytes <= 0 {
		return ""
	}
	if len(s) <= maxBytes {
		return s
	}
	cut := s[:maxBytes]
	for len(cut) > 0 && !utf8.ValidString(cut) {
		cut = cut[:len(cut)-1]
	}
	return cut
}

func estimateTokens(byteCount int) int {
	if byteCount <= 0 {
		return 0
	}
	return (byteCount + 3) / 4
}

func resolveMemoryIdentity(m knowledge.MemoryVersion) (versionID, itemID string) {
	versionID = strings.TrimSpace(m.ID)
	itemID = strings.TrimSpace(m.ItemID)
	if versionID == "" && itemID != "" {
		rev := m.Revision
		if rev <= 0 {
			rev = 1
		}
		versionID = knowledge.FormatVersionID(itemID, rev)
	}
	if itemID == "" && versionID != "" {
		if parsedItem, _, _, err := knowledge.ParseItemOrVersionRef(versionID); err == nil {
			itemID = parsedItem
		}
	}
	if versionID != "" && !strings.Contains(versionID, "@") && itemID != "" {
		rev := m.Revision
		if rev <= 0 {
			rev = 1
		}
		versionID = knowledge.FormatVersionID(itemID, rev)
	}
	return versionID, itemID
}

func resolveSkillIdentity(s knowledge.SkillVersion) (versionID, itemID string) {
	versionID = strings.TrimSpace(s.ID)
	itemID = strings.TrimSpace(s.ItemID)
	if versionID == "" && itemID != "" {
		rev := s.Revision
		if rev <= 0 {
			rev = 1
		}
		versionID = knowledge.FormatVersionID(itemID, rev)
	}
	if itemID == "" && versionID != "" {
		if parsedItem, _, _, err := knowledge.ParseItemOrVersionRef(versionID); err == nil {
			itemID = parsedItem
		}
	}
	if versionID != "" && !strings.Contains(versionID, "@") && itemID != "" {
		rev := s.Revision
		if rev <= 0 {
			rev = 1
		}
		versionID = knowledge.FormatVersionID(itemID, rev)
	}
	return versionID, itemID
}

func formatSkillBlock(versionID, name, description, instructions string) string {
	var sb strings.Builder
	fmt.Fprintf(&sb, "### Skill [%s] %s\n", versionID, name)
	if description != "" {
		fmt.Fprintf(&sb, "Description: %s\n", description)
	}
	sb.WriteString(instructions)
	return sb.String()
}

func collectHistoricalCandidates(episodes []memory.Episode, items []memory.MemoryItem) []memory.MemoryItem {
	out := make([]memory.MemoryItem, 0, len(episodes)+len(items))
	for _, ep := range episodes {
		id := strings.TrimSpace(ep.ID)
		if id == "" {
			continue
		}
		out = append(out, memory.MemoryItem{
			ID:             id,
			Kind:           "episode",
			Content:        ep.FormatNavigationEntry(),
			ChatID:         ep.ChatID,
			EpisodeID:      id,
			StartSeq:       ep.StartSeq,
			EndSeq:         ep.EndSeq,
			TaskStatus:     ep.TaskStatus,
			TopicTags:      ep.TopicTags,
			Participants:   ep.Participants,
			SourceChunkIDs: ep.SourceChunkIDs,
			ModelDerived:   ep.ModelDerived,
		})
	}
	out = append(out, items...)
	return out
}

func resolveHistoricalSeqRange(id string, startSeq, endSeq int64) (int64, int64) {
	if startSeq > 0 && endSeq >= startSeq {
		return startSeq, endSeq
	}
	if startSeq > 0 && endSeq == 0 {
		return startSeq, startSeq
	}
	// Fallback: parse trailing "_<start>_<end>" from chunk_<chat>_<start>_<end> or ep_<chat>_<start>_<end>
	parts := strings.Split(id, "_")
	if len(parts) >= 3 {
		s, err1 := strconv.ParseInt(parts[len(parts)-2], 10, 64)
		e, err2 := strconv.ParseInt(parts[len(parts)-1], 10, 64)
		if err1 == nil && err2 == nil && s > 0 && e >= s {
			return s, e
		}
	}
	return startSeq, endSeq
}

func overlapsWithRingBuffer(id, itemChatID, targetChatID string, startSeq, endSeq int64, bufferedSeqSet map[int64]bool) bool {
	if startSeq <= 0 || endSeq < startSeq || len(bufferedSeqSet) == 0 {
		return false
	}
	if targetChatID != "" {
		if itemChatID != "" && itemChatID != targetChatID {
			return false
		}
		if itemChatID == "" {
			parts := strings.Split(id, "_")
			if len(parts) >= 4 && parts[0] == "chunk" {
				inferredChat := strings.Join(parts[1:len(parts)-2], "_")
				if inferredChat != "" && inferredChat != memory.SanitizeChatID(targetChatID) {
					return false
				}
			}
		}
	}
	for seq := range bufferedSeqSet {
		if seq >= startSeq && seq <= endSeq {
			return true
		}
	}
	return false
}

func formatHistoricalBlock(id string, item memory.MemoryItem, isEpisode bool, startSeq, endSeq int64, text string) string {
	if isEpisode {
		if strings.HasPrefix(text, "[Model-Derived Episode Summary") {
			return text
		}
		ep := memory.Episode{
			ID:             id,
			StartSeq:       startSeq,
			EndSeq:         endSeq,
			TaskStatus:     item.TaskStatus,
			Participants:   item.Participants,
			TopicTags:      item.TopicTags,
			SourceChunkIDs: item.SourceChunkIDs,
			Summary:        text,
		}
		return ep.FormatNavigationEntry()
	}
	if strings.HasPrefix(text, "[Raw Conversation Chunk") {
		return text
	}
	header := fmt.Sprintf("[Raw Conversation Chunk | chunk_id: %s", id)
	if startSeq > 0 && endSeq >= startSeq {
		header += fmt.Sprintf(" | seq %d-%d", startSeq, endSeq)
	}
	if item.EpisodeID != "" {
		header += fmt.Sprintf(" | episode_id: %s", item.EpisodeID)
	}
	header += "]"
	return header + "\n" + text
}
