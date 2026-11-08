package memory

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"hash/fnv"
	"sort"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"

	"github.com/liliang-cn/cortexdb/v2/pkg/cortexdb"
)

const (
	// NamespaceEpisodes is the CortexDB namespace used to index episodic navigation summaries.
	NamespaceEpisodes = "episode"
	// EpisodeIDPrefix is prepended to episode IDs stored in CortexDB messages.
	EpisodeIDPrefix = "episode:"
	// maxEpisodeTopicTags caps the number of normalized topic tags extracted per episode.
	maxEpisodeTopicTags = 8
	// maxEpisodeSummaryBytes caps the length of the heuristic episode summary text.
	maxEpisodeSummaryBytes = 600
)

// Episode TaskStatus constants.
const (
	EpisodeStatusCompleted  = "completed"
	EpisodeStatusOngoing    = "ongoing"
	EpisodeStatusUnresolved = "unresolved"
	EpisodeStatusDiscussion = "discussion"
)

// Episode represents a coherent segment of evicted conversation history indexed for long-horizon navigation.
// Episode summaries are navigation indexes (never approved user facts) and always link back to raw chunks and sequence ranges.
type Episode struct {
	ID             string   `json:"id"`
	ChatID         string   `json:"chat_id"`
	ChatType       string   `json:"chat_type"`
	ThreadID       string   `json:"thread_id,omitempty"`
	StartSeq       int64    `json:"start_seq"`
	EndSeq         int64    `json:"end_seq"`
	StartTime      int64    `json:"start_time"`
	EndTime        int64    `json:"end_time"`
	Participants   []string `json:"participants"`
	TopicTags      []string `json:"topic_tags"`
	TaskStatus     string   `json:"task_status"`
	Summary        string   `json:"summary"`
	SourceChunkIDs []string `json:"source_chunk_ids"`
	ModelDerived   bool     `json:"model_derived"`
	CreatedAt      int64    `json:"created_at"`
}

// RawChunkQuery specifies filters for retrieving raw conversation chunks by ID or sequence range.
type RawChunkQuery struct {
	ChunkID  string
	StartSeq *int64
	EndSeq   *int64
	Limit    int
}

// FormatHeader returns the mandatory trust/navigation header for an episode summary.
func (e Episode) FormatHeader() string {
	return fmt.Sprintf("[Model-Derived Episode Summary | seq %d-%d]", e.StartSeq, e.EndSeq)
}

// FormatNavigationEntry renders the episode with its mandatory trust label, metadata, and raw chunk pointers.
func (e Episode) FormatNavigationEntry() string {
	var b strings.Builder
	b.WriteString(e.FormatHeader())
	b.WriteString("\n")
	if len(e.Participants) > 0 {
		fmt.Fprintf(&b, "Participants: %s\n", strings.Join(e.Participants, ", "))
	}
	if len(e.TopicTags) > 0 {
		fmt.Fprintf(&b, "Topics: %s\n", strings.Join(e.TopicTags, ", "))
	}
	if e.TaskStatus != "" {
		fmt.Fprintf(&b, "Status: %s\n", e.TaskStatus)
	}
	if len(e.SourceChunkIDs) > 0 {
		fmt.Fprintf(&b, "Source Chunks: %s\n", strings.Join(e.SourceChunkIDs, ", "))
	}
	summaryText := strings.TrimSpace(e.Summary)
	header := e.FormatHeader()
	if strings.HasPrefix(summaryText, header) {
		summaryText = strings.TrimSpace(strings.TrimPrefix(summaryText, header))
	}
	if summaryText != "" {
		fmt.Fprintf(&b, "Summary: %s\n", summaryText)
	}
	b.WriteString("(Note: Navigation summary only — if this episode is relevant and you need exact details, inspect its raw chunks via recall_memory with start_seq/end_seq or chunk_id.)")
	return b.String()
}

// ToMemoryItem converts an Episode into a navigation MemoryItem.
func (e Episode) ToMemoryItem(sourceLabel string, score float64) MemoryItem {
	if score <= 0 {
		score = 0.5
	}
	return MemoryItem{
		ID:             EpisodeIDPrefix + e.ID,
		Kind:           "episode",
		Content:        e.FormatNavigationEntry(),
		Source:         sourceLabel,
		Score:          score,
		StartSeq:       e.StartSeq,
		EndSeq:         e.EndSeq,
		StartTime:      e.StartTime,
		EndTime:        e.EndTime,
		ChatID:         e.ChatID,
		EpisodeID:      e.ID,
		SourceChunkIDs: e.SourceChunkIDs,
		Participants:   e.Participants,
		TopicTags:      e.TopicTags,
		TaskStatus:     e.TaskStatus,
		ModelDerived:   true,
	}
}

func (m *Manager) enrichChunksAndMergeEpisodes(ctx context.Context, db *cortexdb.DB, chatID, sourceLabel, query string, chunkItems []MemoryItem, limit int) []MemoryItem {
	allEps, _ := queryEpisodesFromSQL(ctx, db.SQL(), chatID)
	epByChunkID := make(map[string]Episode, len(allEps))
	for _, ep := range allEps {
		if ep.ChatID != chatID {
			continue
		}
		for _, cid := range ep.SourceChunkIDs {
			epByChunkID[cid] = ep
		}
	}

	presentChunks := make(map[string]struct{}, len(chunkItems))
	enriched := make([]MemoryItem, 0, len(chunkItems))
	for _, item := range chunkItems {
		if item.ChatID != "" && item.ChatID != chatID {
			continue
		}
		if item.Kind == "" {
			item.Kind = "chunk"
		}
		presentChunks[item.ID] = struct{}{}
		if ep, ok := epByChunkID[item.ID]; ok {
			item.EpisodeID = ep.ID
			item.SourceChunkIDs = ep.SourceChunkIDs
			item.Participants = ep.Participants
			item.TopicTags = ep.TopicTags
			item.TaskStatus = ep.TaskStatus
		} else {
			for _, ep := range allEps {
				if ep.ChatID == chatID && item.StartSeq >= ep.StartSeq && item.EndSeq <= ep.EndSeq {
					item.EpisodeID = ep.ID
					item.SourceChunkIDs = ep.SourceChunkIDs
					item.Participants = ep.Participants
					item.TopicTags = ep.TopicTags
					item.TaskStatus = ep.TaskStatus
					break
				}
			}
		}
		enriched = append(enriched, item)
	}

	matchedEps, err := m.searchEpisodesInDB(ctx, db, chatID, query, limit)
	if err != nil || len(matchedEps) == 0 {
		return enriched
	}

	qLower := strings.ToLower(strings.TrimSpace(query))
	for _, ep := range matchedEps {
		if ep.ChatID != chatID {
			continue
		}
		allChunksPresent := len(ep.SourceChunkIDs) > 0
		for _, cid := range ep.SourceChunkIDs {
			if _, ok := presentChunks[cid]; !ok {
				allChunksPresent = false
				break
			}
		}
		if allChunksPresent && !strings.Contains(qLower, strings.ToLower(ep.ID)) {
			continue
		}
		score := 0.75
		if strings.Contains(qLower, strings.ToLower(ep.ID)) {
			score = 1.05
		}
		enriched = append(enriched, ep.ToMemoryItem(sourceLabel, score))
	}

	sort.SliceStable(enriched, func(i, j int) bool {
		return enriched[i].Score > enriched[j].Score
	})

	return enriched
}

func (m *Manager) initEpisodeTable(ctx context.Context, rawDB *sql.DB) error {
	schema := `
	CREATE TABLE IF NOT EXISTS memory_episodes (
		id TEXT PRIMARY KEY,
		chat_id TEXT NOT NULL,
		chat_type TEXT NOT NULL,
		thread_id TEXT NOT NULL DEFAULT '',
		start_seq INTEGER NOT NULL,
		end_seq INTEGER NOT NULL,
		start_time INTEGER NOT NULL,
		end_time INTEGER NOT NULL,
		participants_json TEXT NOT NULL,
		topic_tags_json TEXT NOT NULL,
		task_status TEXT NOT NULL,
		summary TEXT NOT NULL,
		source_chunk_ids_json TEXT NOT NULL,
		model_derived INTEGER NOT NULL DEFAULT 1,
		created_at INTEGER NOT NULL
	);
	CREATE INDEX IF NOT EXISTS idx_memory_episodes_seq
		ON memory_episodes (chat_id, start_seq ASC, end_seq ASC);
	`
	_, err := rawDB.ExecContext(ctx, schema)
	return err
}

// uniqueChatIDSlug returns SanitizeChatID(chatID) when chatID contains only safe characters,
// or appends a deterministic 8-hex-character FNV-1a hash when sanitization lossily replaced characters.
func uniqueChatIDSlug(chatID string) string {
	sanitized := SanitizeChatID(chatID)
	if sanitized == chatID {
		return sanitized
	}
	h := fnv.New32a()
	_, _ = h.Write([]byte(chatID))
	return fmt.Sprintf("%s_%08x", sanitized, h.Sum32())
}

// buildEpisodeFromMessages constructs an Episode record from a batch of evicted messages and their raw chunk IDs.
func buildEpisodeFromMessages(chatID string, isDM bool, msgs []MessageToStore, chunkIDs []string) Episode {
	chatType := "townhall"
	if isDM && chatID != "townhall" {
		chatType = "dm"
	}

	startSeq := msgs[0].Seq
	endSeq := msgs[len(msgs)-1].Seq
	startTime := msgs[0].Timestamp
	endTime := msgs[len(msgs)-1].Timestamp

	participantSet := make(map[string]struct{})
	var participants []string

	for _, msg := range msgs {
		if msg.Seq < startSeq {
			startSeq = msg.Seq
		}
		if msg.Seq > endSeq {
			endSeq = msg.Seq
		}
		if msg.Timestamp < startTime {
			startTime = msg.Timestamp
		}
		if msg.Timestamp > endTime {
			endTime = msg.Timestamp
		}

		name := strings.TrimSpace(msg.SenderName)
		if name == "" {
			name = strings.TrimSpace(msg.UserID)
		}
		if name == "" {
			name = "Unknown"
		}
		if _, exists := participantSet[name]; !exists {
			participantSet[name] = struct{}{}
			participants = append(participants, name)
		}
	}

	topicTags := extractTopicTags(msgs, maxEpisodeTopicTags)
	taskStatus := inferEpisodeTaskStatus(msgs)

	epID := fmt.Sprintf("ep_%s_%d_%d", uniqueChatIDSlug(chatID), startSeq, endSeq)
	header := fmt.Sprintf("[Model-Derived Episode Summary | seq %d-%d]", startSeq, endSeq)
	bodySummary := buildEpisodeSummaryText(msgs, participants, topicTags, taskStatus)
	fullSummary := header + " " + bodySummary

	return Episode{
		ID:             epID,
		ChatID:         chatID,
		ChatType:       chatType,
		ThreadID:       "",
		StartSeq:       startSeq,
		EndSeq:         endSeq,
		StartTime:      startTime,
		EndTime:        endTime,
		Participants:   participants,
		TopicTags:      topicTags,
		TaskStatus:     taskStatus,
		Summary:        fullSummary,
		SourceChunkIDs: chunkIDs,
		ModelDerived:   true,
		CreatedAt:      time.Now().Unix(),
	}
}

func buildEpisodeSummaryText(msgs []MessageToStore, participants, topicTags []string, taskStatus string) string {
	if len(msgs) == 0 {
		return ""
	}
	firstSnippet := compactSnippet(msgs[0].Content, 180)
	var b strings.Builder
	fmt.Fprintf(&b, "Conversation (%d messages, status: %s) involving %s.", len(msgs), taskStatus, strings.Join(participants, ", "))
	if len(topicTags) > 0 {
		fmt.Fprintf(&b, " Key topics: %s.", strings.Join(topicTags, ", "))
	}
	if firstSnippet != "" {
		fmt.Fprintf(&b, " Started with: %q.", firstSnippet)
	}
	if len(msgs) > 1 {
		lastSnippet := compactSnippet(msgs[len(msgs)-1].Content, 180)
		if lastSnippet != "" && lastSnippet != firstSnippet {
			fmt.Fprintf(&b, " Latest: %q.", lastSnippet)
		}
	}
	out := b.String()
	return truncateUTF8Bytes(out, maxEpisodeSummaryBytes)
}

func compactSnippet(s string, maxLen int) string {
	cleaned := strings.Join(strings.Fields(strings.TrimSpace(s)), " ")
	return truncateUTF8Bytes(cleaned, maxLen)
}

func truncateUTF8Bytes(s string, maxBytes int) string {
	if maxBytes <= 0 {
		return ""
	}
	if len(s) <= maxBytes {
		return s
	}
	cut := maxBytes
	for cut > 0 && !utf8.RuneStart(s[cut]) {
		cut--
	}
	return s[:cut] + "..."
}

// TODO(next-track): This hardcoded English stop-word and keyword heuristic is brittle and does not
// work across languages. Immediately after finalizing track context-assembly-compaction_20261105,
// research and replace extractTopicTags and inferEpisodeTaskStatus with a simple, fast, small
// multilingual ML model running locally on CPU.
var episodeStopWords = map[string]struct{}{
	"the": {}, "and": {}, "for": {}, "that": {}, "this": {}, "with": {}, "from": {},
	"have": {}, "has": {}, "had": {}, "not": {}, "are": {}, "was": {}, "were": {},
	"will": {}, "would": {}, "could": {}, "should": {}, "can": {}, "about": {},
	"into": {}, "your": {}, "you": {}, "our": {}, "their": {}, "they": {}, "them": {},
	"what": {}, "when": {}, "where": {}, "which": {}, "who": {}, "whom": {}, "whose": {},
	"how": {}, "why": {}, "just": {}, "like": {}, "some": {}, "more": {}, "than": {},
	"then": {}, "also": {}, "very": {}, "here": {}, "there": {}, "been": {}, "being": {},
	"over": {}, "under": {}, "after": {}, "before": {}, "between": {}, "through": {},
	"please": {}, "thanks": {}, "thank": {}, "hello": {}, "hey": {}, "yes": {}, "okay": {},
}

func extractTopicTags(msgs []MessageToStore, maxTags int) []string {
	freq := make(map[string]int)
	firstPos := make(map[string]int)
	pos := 0

	for _, m := range msgs {
		tokens := tokenizeForTags(m.Content)
		for _, tok := range tokens {
			pos++
			if _, stop := episodeStopWords[tok]; stop {
				continue
			}
			if _, seen := firstPos[tok]; !seen {
				firstPos[tok] = pos
			}
			freq[tok]++
		}
	}

	type scoredTag struct {
		tag   string
		count int
		pos   int
	}
	candidates := make([]scoredTag, 0, len(freq))
	for tag, count := range freq {
		candidates = append(candidates, scoredTag{
			tag:   tag,
			count: count,
			pos:   firstPos[tag],
		})
	}

	sort.Slice(candidates, func(i, j int) bool {
		if candidates[i].count != candidates[j].count {
			return candidates[i].count > candidates[j].count
		}
		return candidates[i].pos < candidates[j].pos
	})

	if len(candidates) > maxTags {
		candidates = candidates[:maxTags]
	}

	tags := make([]string, len(candidates))
	for i, c := range candidates {
		tags[i] = c.tag
	}
	return tags
}

func tokenizeForTags(s string) []string {
	var tokens []string
	var cur strings.Builder

	flush := func() {
		if cur.Len() == 0 {
			return
		}
		tok := strings.ToLower(strings.Trim(cur.String(), "_-."))
		cur.Reset()
		if len(tok) < 4 || len(tok) > 36 {
			return
		}
		// Skip pure numbers
		allDigits := true
		for _, r := range tok {
			if !unicode.IsDigit(r) {
				allDigits = false
				break
			}
		}
		if !allDigits {
			tokens = append(tokens, tok)
		}
	}

	for _, r := range s {
		if unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_' || r == '-' {
			cur.WriteRune(r)
		} else {
			flush()
		}
	}
	flush()
	return tokens
}

// TODO(next-track): Replace keyword-based task status inference with the lightweight local CPU
// classification model planned immediately after track context-assembly-compaction_20261105.
func inferEpisodeTaskStatus(msgs []MessageToStore) string {
	if len(msgs) == 0 {
		return EpisodeStatusDiscussion
	}
	// Inspect messages from newest to oldest to determine latest status
	for i := len(msgs) - 1; i >= 0; i-- {
		lower := strings.ToLower(msgs[i].Content)
		if containsAnyKeyword(lower, "failed", "error:", "panic:", "unresolved", "blocked", "broken", "exception") {
			return EpisodeStatusUnresolved
		}
		if containsAnyKeyword(lower, "completed", "resolved", "fixed", "deployed", "merged", "finished", "succeeded") {
			return EpisodeStatusCompleted
		}
		if containsAnyKeyword(lower, "in progress", "working on", "investigating", "next step", "todo", "ongoing") {
			return EpisodeStatusOngoing
		}
	}
	return EpisodeStatusDiscussion
}

func containsAnyKeyword(lower string, keywords ...string) bool {
	for _, kw := range keywords {
		if strings.Contains(lower, kw) {
			return true
		}
	}
	return false
}

// saveEpisode persists an Episode row in memory_episodes and indexes it in CortexDB under NamespaceEpisodes.
func (m *Manager) saveEpisode(ctx context.Context, db *cortexdb.DB, ep Episode) error {
	participantsJSON, err := json.Marshal(ep.Participants)
	if err != nil {
		return fmt.Errorf("marshal episode participants: %w", err)
	}
	topicTagsJSON, err := json.Marshal(ep.TopicTags)
	if err != nil {
		return fmt.Errorf("marshal episode topic tags: %w", err)
	}
	sourceChunksJSON, err := json.Marshal(ep.SourceChunkIDs)
	if err != nil {
		return fmt.Errorf("marshal episode source chunk IDs: %w", err)
	}
	modelDerivedInt := 0
	if ep.ModelDerived {
		modelDerivedInt = 1
	}
	if ep.CreatedAt == 0 {
		ep.CreatedAt = time.Now().Unix()
	}

	_, err = db.SQL().ExecContext(ctx, `
		INSERT INTO memory_episodes (
			id, chat_id, chat_type, thread_id, start_seq, end_seq,
			start_time, end_time, participants_json, topic_tags_json,
			task_status, summary, source_chunk_ids_json, model_derived, created_at
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
		ON CONFLICT(id) DO UPDATE SET
			start_seq = excluded.start_seq,
			end_seq = excluded.end_seq,
			start_time = excluded.start_time,
			end_time = excluded.end_time,
			participants_json = excluded.participants_json,
			topic_tags_json = excluded.topic_tags_json,
			task_status = excluded.task_status,
			summary = excluded.summary,
			source_chunk_ids_json = excluded.source_chunk_ids_json,
			model_derived = excluded.model_derived
	`,
		ep.ID, ep.ChatID, ep.ChatType, ep.ThreadID, ep.StartSeq, ep.EndSeq,
		ep.StartTime, ep.EndTime, string(participantsJSON), string(topicTagsJSON),
		ep.TaskStatus, ep.Summary, string(sourceChunksJSON), modelDerivedInt, ep.CreatedAt,
	)
	if err != nil {
		return fmt.Errorf("failed to persist memory_episode %s: %w", ep.ID, err)
	}

	metadata := map[string]any{
		"kind":             "episode",
		"episode_id":       ep.ID,
		"chat_id":          ep.ChatID,
		"chat_type":        ep.ChatType,
		"start_seq":        ep.StartSeq,
		"end_seq":          ep.EndSeq,
		"start_time":       ep.StartTime,
		"end_time":         ep.EndTime,
		"task_status":      ep.TaskStatus,
		"participants":     ep.Participants,
		"topic_tags":       ep.TopicTags,
		"source_chunk_ids": ep.SourceChunkIDs,
		"model_derived":    ep.ModelDerived,
	}

	_, err = db.SaveMemory(ctx, cortexdb.MemorySaveRequest{
		MemoryID:  EpisodeIDPrefix + ep.ID,
		Scope:     cortexdb.MemoryScopeGlobal,
		Namespace: NamespaceEpisodes,
		Role:      "system",
		Content:   ep.FormatNavigationEntry(),
		Metadata:  metadata,
	})
	if err != nil {
		return fmt.Errorf("failed to index episode %s in cortexdb: %w", ep.ID, err)
	}
	return nil
}

// ListEpisodes returns all persisted Episode records for the given chat ordered by start_seq ASC.
func (m *Manager) ListEpisodes(ctx context.Context, chatID string, isDM bool) ([]Episode, error) {
	if !isDM || chatID == "townhall" {
		chatID = "townhall"
		isDM = false
	}
	db, err := m.GetDB(ctx, chatID, isDM)
	if err != nil {
		return nil, err
	}
	return queryEpisodesFromSQL(ctx, db.SQL(), chatID)
}

func queryEpisodesFromSQL(ctx context.Context, rawDB *sql.DB, chatID string) ([]Episode, error) {
	rows, err := rawDB.QueryContext(ctx, `
		SELECT id, chat_id, chat_type, thread_id, start_seq, end_seq,
		       start_time, end_time, participants_json, topic_tags_json,
		       task_status, summary, source_chunk_ids_json, model_derived, created_at
		FROM memory_episodes
		WHERE chat_id = ?
		ORDER BY start_seq ASC, end_seq ASC
	`, chatID)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()

	var out []Episode
	for rows.Next() {
		ep, err := scanEpisodeRow(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, ep)
	}
	return out, rows.Err()
}

func getEpisodeByID(ctx context.Context, rawDB *sql.DB, epID string) (*Episode, error) {
	epID = strings.TrimPrefix(epID, EpisodeIDPrefix)
	row := rawDB.QueryRowContext(ctx, `
		SELECT id, chat_id, chat_type, thread_id, start_seq, end_seq,
		       start_time, end_time, participants_json, topic_tags_json,
		       task_status, summary, source_chunk_ids_json, model_derived, created_at
		FROM memory_episodes
		WHERE id = ?
	`, epID)
	ep, err := scanEpisodeRow(row)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return nil, nil
		}
		return nil, err
	}
	return &ep, nil
}

type episodeRowScanner interface {
	Scan(dest ...any) error
}

func scanEpisodeRow(scanner episodeRowScanner) (Episode, error) {
	var ep Episode
	var participantsJSON, topicTagsJSON, sourceChunksJSON string
	var modelDerivedInt int
	err := scanner.Scan(
		&ep.ID, &ep.ChatID, &ep.ChatType, &ep.ThreadID,
		&ep.StartSeq, &ep.EndSeq, &ep.StartTime, &ep.EndTime,
		&participantsJSON, &topicTagsJSON, &ep.TaskStatus,
		&ep.Summary, &sourceChunksJSON, &modelDerivedInt, &ep.CreatedAt,
	)
	if err != nil {
		return Episode{}, err
	}
	if err := json.Unmarshal([]byte(participantsJSON), &ep.Participants); err != nil {
		return Episode{}, fmt.Errorf("decode participants_json for episode %s: %w", ep.ID, err)
	}
	if err := json.Unmarshal([]byte(topicTagsJSON), &ep.TopicTags); err != nil {
		return Episode{}, fmt.Errorf("decode topic_tags_json for episode %s: %w", ep.ID, err)
	}
	if err := json.Unmarshal([]byte(sourceChunksJSON), &ep.SourceChunkIDs); err != nil {
		return Episode{}, fmt.Errorf("decode source_chunk_ids_json for episode %s: %w", ep.ID, err)
	}
	ep.ModelDerived = modelDerivedInt != 0
	return ep, nil
}

// SearchEpisodes searches the episodic navigation index across accessible databases for the given chat context.
// Townhall queries townhall.db only. DM queries both the private DM database and townhall.db.
func (m *Manager) SearchEpisodes(ctx context.Context, query string, chatID string, isDM bool, limit int) ([]Episode, error) {
	query = strings.TrimSpace(query)
	if query == "" {
		return nil, nil
	}
	if limit <= 0 {
		limit = 5
	}

	if !isDM || chatID == "townhall" {
		thDB, err := m.GetDB(ctx, "townhall", false)
		if err != nil {
			return nil, err
		}
		return m.searchEpisodesInDB(ctx, thDB, "townhall", query, limit)
	}

	dmDB, err := m.GetDB(ctx, chatID, true)
	if err != nil {
		return nil, err
	}
	dmEps, _ := m.searchEpisodesInDB(ctx, dmDB, chatID, query, limit)

	thDB, err := m.GetDB(ctx, "townhall", false)
	var thEps []Episode
	if err == nil {
		thEps, _ = m.searchEpisodesInDB(ctx, thDB, "townhall", query, limit)
	}

	seen := make(map[string]struct{}, len(dmEps)+len(thEps))
	merged := make([]Episode, 0, len(dmEps)+len(thEps))
	for _, ep := range append(dmEps, thEps...) {
		if _, ok := seen[ep.ID]; ok {
			continue
		}
		seen[ep.ID] = struct{}{}
		merged = append(merged, ep)
		if len(merged) >= limit {
			break
		}
	}
	return merged, nil
}

func (m *Manager) searchEpisodesInDB(ctx context.Context, db *cortexdb.DB, chatID, query string, limit int) ([]Episode, error) {
	qTrim := strings.TrimSpace(query)
	qLower := strings.ToLower(qTrim)
	if strings.HasPrefix(qLower, "ep_") || strings.HasPrefix(qLower, EpisodeIDPrefix) {
		if ep, err := getEpisodeByID(ctx, db.SQL(), qTrim); err == nil && ep != nil && ep.ChatID == chatID {
			return []Episode{*ep}, nil
		}
	}

	req := cortexdb.MemorySearchRequest{
		Query:     query,
		Scope:     cortexdb.MemoryScopeGlobal,
		Namespace: NamespaceEpisodes,
		TopK:      limit,
	}
	var hits []cortexdb.MemorySearchHit
	resp, err := db.SearchMemory(ctx, req)
	if err != nil {
		req.RetrievalMode = cortexdb.RetrievalModeLexical
		if lexResp, lexErr := db.SearchMemory(ctx, req); lexErr == nil && lexResp != nil {
			hits = lexResp.Results
		}
	} else if resp != nil {
		hits = resp.Results
	}

	seen := make(map[string]struct{})
	var results []Episode
	for _, h := range hits {
		epID := strings.TrimPrefix(h.Memory.ID, EpisodeIDPrefix)
		if _, exists := seen[epID]; exists {
			continue
		}
		ep, err := getEpisodeByID(ctx, db.SQL(), epID)
		if err == nil && ep != nil && ep.ChatID == chatID {
			seen[ep.ID] = struct{}{}
			results = append(results, *ep)
		}
	}

	if len(results) < limit {
		// Supplement with SQL substring match over summary, topic_tags_json, participants_json, and task_status
		allEps, err := queryEpisodesFromSQL(ctx, db.SQL(), chatID)
		if err == nil {
			qLower := strings.ToLower(query)
			for _, ep := range allEps {
				if ep.ChatID != chatID {
					continue
				}
				if _, exists := seen[ep.ID]; exists {
					continue
				}
				if matchesEpisode(ep, qLower) {
					seen[ep.ID] = struct{}{}
					results = append(results, ep)
					if len(results) >= limit {
						break
					}
				}
			}
		}
	}

	return results, nil
}

func matchesEpisode(ep Episode, qLower string) bool {
	if strings.Contains(strings.ToLower(ep.ID), qLower) ||
		strings.Contains(strings.ToLower(ep.Summary), qLower) ||
		strings.Contains(strings.ToLower(ep.TaskStatus), qLower) {
		return true
	}
	for _, p := range ep.Participants {
		if strings.Contains(strings.ToLower(p), qLower) {
			return true
		}
	}
	for _, tag := range ep.TopicTags {
		if strings.Contains(strings.ToLower(tag), qLower) {
			return true
		}
	}
	for _, cid := range ep.SourceChunkIDs {
		if strings.Contains(strings.ToLower(cid), qLower) {
			return true
		}
	}
	return false
}

// GetRawChunks retrieves raw conversation chunks by chunk_id (or episode_id) or overlapping [start_seq, end_seq].
// Townhall context strictly queries townhall.db only. DM context queries the user's private DM DB and (if needed) townhall.db.
func (m *Manager) GetRawChunks(ctx context.Context, chatID string, isDM bool, q RawChunkQuery) ([]MemoryItem, error) {
	limit := q.Limit
	if limit <= 0 {
		limit = 5
	}
	if limit > 20 {
		limit = 20
	}

	q.ChunkID = strings.TrimSpace(q.ChunkID)
	if q.ChunkID == "" && q.StartSeq == nil && q.EndSeq == nil {
		return nil, errors.New("chunk_id, start_seq, or end_seq is required")
	}
	if q.StartSeq != nil && q.EndSeq != nil && *q.StartSeq > *q.EndSeq {
		return nil, fmt.Errorf("start_seq (%d) cannot be greater than end_seq (%d)", *q.StartSeq, *q.EndSeq)
	}

	if !isDM || chatID == "townhall" {
		thDB, err := m.GetDB(ctx, "townhall", false)
		if err != nil {
			return nil, err
		}
		return queryRawChunksInDB(ctx, thDB.SQL(), "townhall", "[Townhall]", q, limit)
	}

	// DM Context: search private DM DB first, then Townhall DB if limit not reached
	dmDB, err := m.GetDB(ctx, chatID, true)
	if err != nil {
		return nil, err
	}
	dmItems, err := queryRawChunksInDB(ctx, dmDB.SQL(), chatID, "[Direct Message]", q, limit)
	if err != nil {
		return nil, err
	}
	if len(dmItems) >= limit || (q.ChunkID != "" && len(dmItems) > 0) {
		return dmItems, nil
	}

	thDB, err := m.GetDB(ctx, "townhall", false)
	if err != nil {
		return dmItems, nil
	}
	thItems, err := queryRawChunksInDB(ctx, thDB.SQL(), "townhall", "[Townhall]", q, limit-len(dmItems))
	if err != nil {
		return nil, err
	}
	return append(dmItems, thItems...), nil
}

func queryRawChunksInDB(ctx context.Context, rawDB *sql.DB, expectedChatID, sourceLabel string, q RawChunkQuery, limit int) ([]MemoryItem, error) {
	targetChunkIDs := make(map[string]struct{})
	if q.ChunkID != "" {
		if strings.HasPrefix(q.ChunkID, "ep_") || strings.HasPrefix(q.ChunkID, EpisodeIDPrefix) {
			ep, err := getEpisodeByID(ctx, rawDB, q.ChunkID)
			if err != nil {
				return nil, err
			}
			if ep == nil || ep.ChatID != expectedChatID || len(ep.SourceChunkIDs) == 0 {
				return nil, nil
			}
			for _, cid := range ep.SourceChunkIDs {
				cid = strings.TrimSpace(cid)
				if cid != "" {
					targetChunkIDs[cid] = struct{}{}
				}
			}
			if len(targetChunkIDs) == 0 {
				return nil, nil
			}
		} else {
			targetChunkIDs[q.ChunkID] = struct{}{}
		}
	}

	rows, err := rawDB.QueryContext(ctx, `
		SELECT id, content, metadata
		FROM messages
		WHERE session_id = 'memory:global:default' AND id LIKE 'chunk_%'
		ORDER BY created_at ASC, id ASC
	`)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()

	var items []MemoryItem
	for rows.Next() {
		var id, content, metadataJSON string
		if err := rows.Scan(&id, &content, &metadataJSON); err != nil {
			return nil, err
		}

		var meta map[string]any
		_ = json.Unmarshal([]byte(metadataJSON), &meta)

		startSeq := parseMetadataInt64(meta["start_seq"])
		endSeq := parseMetadataInt64(meta["end_seq"])
		startTime := parseMetadataInt64(meta["start_time"])
		endTime := parseMetadataInt64(meta["end_time"])
		cid, _ := meta["chat_id"].(string)
		if cid == "" {
			cid = expectedChatID
		}
		if cid != expectedChatID {
			continue
		}

		if len(targetChunkIDs) > 0 {
			if _, ok := targetChunkIDs[id]; !ok {
				continue
			}
		}

		if q.StartSeq != nil && endSeq < *q.StartSeq {
			continue
		}
		if q.EndSeq != nil && startSeq > *q.EndSeq {
			continue
		}

		// Filter individual lines in the chunk when a narrower [StartSeq, EndSeq] range was requested
		items = append(items, MemoryItem{
			ID:        id,
			Kind:      "chunk",
			Content:   content,
			Source:    sourceLabel,
			Score:     1.0,
			StartSeq:  startSeq,
			EndSeq:    endSeq,
			StartTime: startTime,
			EndTime:   endTime,
			ChatID:    cid,
		})
		if len(items) >= limit {
			break
		}
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}

	sort.Slice(items, func(i, j int) bool {
		return items[i].StartSeq < items[j].StartSeq
	})
	return items, nil
}
