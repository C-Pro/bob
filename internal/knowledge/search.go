package knowledge

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/liliang-cn/cortexdb/v2/pkg/cortexdb"
)

// SessionIdentity captures the required tenant isolation parameters for a search.
type SessionIdentity struct {
	ChatID string
	UserID string
}

// MemorySearchResult represents a validated, summarized memory match.
type MemorySearchResult struct {
	ItemID     string     `json:"item_id"`
	VersionID  string     `json:"version_id"`
	Snippet    string     `json:"snippet"`
	Type       MemoryType `json:"type"`
	Confidence float64    `json:"confidence"`
	Score      float64    `json:"score"`
}

// SkillSearchResult represents a validated, summarized skill match.
type SkillSearchResult struct {
	ItemID    string   `json:"item_id"`
	VersionID string   `json:"version_id"`
	Name      string   `json:"name"`
	Snippet   string   `json:"snippet"`
	Triggers  []string `json:"triggers,omitempty"`
	Tags      []string `json:"tags,omitempty"`
	Score     float64  `json:"score"`
}

// Searcher coordinates vector and lexical candidate discovery with strict relational revalidation.
type Searcher struct {
	store    *Store
	vector   VectorDB
	session  SessionIdentity
	now      func() time.Time
	maxLimit int
}

// NewSearcher creates a new Searcher bound to a session identity.
func NewSearcher(store *Store, vector VectorDB, session SessionIdentity, maxLimit int) (*Searcher, error) {
	if store == nil {
		return nil, errors.New("store cannot be nil")
	}
	session.ChatID = strings.TrimSpace(session.ChatID)
	session.UserID = strings.TrimSpace(session.UserID)
	if session.ChatID == "" {
		return nil, errors.New("chat_id is required in session identity")
	}
	if session.UserID == "" {
		return nil, errors.New("user_id is required in session identity")
	}
	if maxLimit <= 0 {
		maxLimit = 10
	}
	return &Searcher{
		store:    store,
		vector:   vector,
		session:  session,
		now:      time.Now,
		maxLimit: maxLimit,
	}, nil
}

// SearchMemories searches memories in the dm_memories namespace and validates results against the relational store.
func (s *Searcher) SearchMemories(ctx context.Context, query string, types []MemoryType, limit int) ([]MemorySearchResult, error) {
	if strings.TrimSpace(query) == "" {
		return []MemorySearchResult{}, nil
	}

	// Validate memory types filter
	typeSet := make(map[MemoryType]struct{}, len(types))
	for _, t := range types {
		if !IsValidMemoryType(t) {
			return nil, fmt.Errorf("%w: invalid memory type filter %q", ErrInvalidStatus, t)
		}
		typeSet[t] = struct{}{}
	}

	limit = s.normalizeLimit(limit)
	if s.vector == nil {
		return []MemorySearchResult{}, nil
	}

	topK := s.initialTopK(limit)
	maxTopK := max(50, limit*4)
	const maxAttempts = 2

	var finalResults []MemorySearchResult
	seenVersions := make(map[string]struct{})
	var staleIDs []string
	defer func() {
		s.cleanupStaleIndices(ctx, staleIDs)
	}()

	for attempt := 0; attempt < maxAttempts; attempt++ {
		hits, err := s.searchVector(ctx, query, NamespaceMemories, topK)
		if err != nil {
			return nil, err
		}
		if len(hits) == 0 {
			break
		}

		// Collect candidate version IDs matching memory prefix, preserving rank order
		var candidateIDs []string
		hitScores := make(map[string]float64)
		for _, hit := range hits {
			vID := hit.Memory.ID
			if _, seen := seenVersions[vID]; !seen {
				seenVersions[vID] = struct{}{}
				// If returned ID is not a memory ID, discard without deleting (avoid deleting valid skill records)
				if !strings.HasPrefix(vID, "mem_") {
					continue
				}
				candidateIDs = append(candidateIDs, vID)
				hitScores[vID] = hit.Score
			}
		}

		if len(candidateIDs) > 0 {
			candidates, err := s.store.GetSearchCandidates(ctx, KindMemory, candidateIDs)
			if err != nil {
				return nil, fmt.Errorf("search revalidation failed: %w", err)
			}

			nowUnix := s.now().Unix()
			for _, vID := range candidateIDs {
				cand, exists := candidates[vID]
				if !exists {
					// Double check if truly deleted in SQLite before scheduling index cleanup
					if _, err := s.store.GetMemoryVersion(ctx, vID); errors.Is(err, ErrNotFound) {
						staleIDs = append(staleIDs, vID)
					}
					continue
				}

				if cand.Item.Kind != KindMemory {
					continue
				}

				// Session tenant isolation check: discard without deleting if belonging to another user/chat
				if cand.Item.ChatID != s.session.ChatID || cand.Item.UserID != s.session.UserID {
					continue
				}

				// Active state check
				if cand.Item.Status != ItemStatusActive {
					staleIDs = append(staleIDs, vID)
					continue
				}

				// Active version alignment and provenance check
				if cand.Memory == nil || cand.Item.ActiveVersionID != cand.Memory.ID || cand.Memory.Status != VersionStatusApproved {
					staleIDs = append(staleIDs, vID)
					continue
				}
				if cand.Memory.Provenance.ChatID != cand.Item.ChatID || cand.Memory.Provenance.UserID != cand.Item.UserID {
					continue
				}

				// Expiration check
				if cand.Memory.ExpiresAt != nil && nowUnix >= *cand.Memory.ExpiresAt {
					staleIDs = append(staleIDs, vID)
					continue
				}

				// Optional type set check
				if len(typeSet) > 0 {
					if _, match := typeSet[cand.Memory.Type]; !match {
						continue
					}
				}

				finalResults = append(finalResults, MemorySearchResult{
					ItemID:     cand.Item.ID,
					VersionID:  cand.Memory.ID,
					Snippet:    TruncateRunes(cand.Memory.Content, 150),
					Type:       cand.Memory.Type,
					Confidence: cand.Memory.Confidence,
					Score:      hitScores[vID],
				})

				if len(finalResults) >= limit {
					return finalResults, nil
				}
			}
		}

		if len(hits) < topK || topK >= maxTopK {
			break
		}
		topK = min(topK*2, maxTopK)
	}

	return finalResults, nil
}

// SearchSkills searches procedural skills in the dm_skills namespace and validates results against the relational store.
func (s *Searcher) SearchSkills(ctx context.Context, query string, tags []string, limit int) ([]SkillSearchResult, error) {
	if strings.TrimSpace(query) == "" {
		return []SkillSearchResult{}, nil
	}

	limit = s.normalizeLimit(limit)
	if s.vector == nil {
		return []SkillSearchResult{}, nil
	}

	// Normalize requested tags (lowercase, trimmed)
	normTags := make([]string, 0, len(tags))
	for _, t := range tags {
		tr := strings.ToLower(strings.TrimSpace(t))
		if tr != "" {
			normTags = append(normTags, tr)
		}
	}

	topK := s.initialTopK(limit)
	maxTopK := max(50, limit*4)
	const maxAttempts = 2

	var finalResults []SkillSearchResult
	seenVersions := make(map[string]struct{})
	var staleIDs []string
	defer func() {
		s.cleanupStaleIndices(ctx, staleIDs)
	}()

	for attempt := 0; attempt < maxAttempts; attempt++ {
		hits, err := s.searchVector(ctx, query, NamespaceSkills, topK)
		if err != nil {
			return nil, err
		}
		if len(hits) == 0 {
			break
		}

		var candidateIDs []string
		hitScores := make(map[string]float64)
		for _, hit := range hits {
			vID := hit.Memory.ID
			if _, seen := seenVersions[vID]; !seen {
				seenVersions[vID] = struct{}{}
				// If returned ID is not a skill ID, discard without deleting
				if !strings.HasPrefix(vID, "skill_") {
					continue
				}
				candidateIDs = append(candidateIDs, vID)
				hitScores[vID] = hit.Score
			}
		}

		if len(candidateIDs) > 0 {
			candidates, err := s.store.GetSearchCandidates(ctx, KindSkill, candidateIDs)
			if err != nil {
				return nil, fmt.Errorf("search revalidation failed: %w", err)
			}

			for _, vID := range candidateIDs {
				cand, exists := candidates[vID]
				if !exists {
					if _, err := s.store.GetSkillVersion(ctx, vID); errors.Is(err, ErrNotFound) {
						staleIDs = append(staleIDs, vID)
					}
					continue
				}

				if cand.Item.Kind != KindSkill {
					continue
				}

				// Session tenant isolation check
				if cand.Item.ChatID != s.session.ChatID || cand.Item.UserID != s.session.UserID {
					continue
				}

				// Active state check
				if cand.Item.Status != ItemStatusActive {
					staleIDs = append(staleIDs, vID)
					continue
				}

				// Active version alignment and provenance check
				if cand.Skill == nil || cand.Item.ActiveVersionID != cand.Skill.ID || cand.Skill.Status != VersionStatusApproved {
					staleIDs = append(staleIDs, vID)
					continue
				}
				if cand.Skill.Provenance.ChatID != cand.Item.ChatID || cand.Skill.Provenance.UserID != cand.Item.UserID {
					continue
				}

				// Optional tag filter: any requested tag matches
				if len(normTags) > 0 && !hasMatchingTag(cand.Skill.Tags, normTags) {
					continue
				}

				snippetText := cand.Skill.Description
				if strings.TrimSpace(snippetText) == "" {
					snippetText = cand.Skill.InstructionsMarkdown
				}

				finalResults = append(finalResults, SkillSearchResult{
					ItemID:    cand.Item.ID,
					VersionID: cand.Skill.ID,
					Name:      TruncateRunes(cand.Skill.Name, 100),
					Snippet:   TruncateRunes(snippetText, 150),
					Triggers:  boundStringSlice(cand.Skill.Triggers, 5, 50),
					Tags:      boundStringSlice(cand.Skill.Tags, 5, 50),
					Score:     hitScores[vID],
				})

				if len(finalResults) >= limit {
					return finalResults, nil
				}
			}
		}

		if len(hits) < topK || topK >= maxTopK {
			break
		}
		topK = min(topK*2, maxTopK)
	}

	return finalResults, nil
}

func (s *Searcher) searchVector(ctx context.Context, query string, namespace string, topK int) ([]cortexdb.MemorySearchHit, error) {
	req := cortexdb.MemorySearchRequest{
		Query:     query,
		Scope:     cortexdb.MemoryScopeGlobal,
		Namespace: namespace,
		TopK:      topK,
	}

	resp, err := s.vector.SearchMemory(ctx, req)
	if err != nil {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		req.RetrievalMode = cortexdb.RetrievalModeLexical
		lexResp, lexErr := s.vector.SearchMemory(ctx, req)
		if lexErr != nil {
			return nil, errors.Join(err, lexErr)
		}
		if lexResp != nil {
			return lexResp.Results, nil
		}
		return nil, err
	}

	if resp != nil {
		return resp.Results, nil
	}
	return nil, nil
}

func (s *Searcher) cleanupStaleIndices(ctx context.Context, versionIDs []string) {
	if len(versionIDs) == 0 || s.vector == nil {
		return
	}

	// Shared timeout budget bounded to 2 seconds respecting parent context
	cleanupCtx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()

	for _, id := range versionIDs {
		if cleanupCtx.Err() != nil {
			break
		}
		_, _ = s.vector.DeleteMemory(cleanupCtx, cortexdb.MemoryDeleteRequest{
			MemoryID: id,
		})
	}
}

func (s *Searcher) normalizeLimit(limit int) int {
	defaultLimit := 5
	if s.maxLimit < defaultLimit {
		defaultLimit = s.maxLimit
	}
	if limit <= 0 {
		return defaultLimit
	}
	if limit > s.maxLimit {
		return s.maxLimit
	}
	return limit
}

func (s *Searcher) initialTopK(limit int) int {
	topK := limit * 3
	if topK < 10 {
		topK = 10
	}
	return topK
}

func hasMatchingTag(skillTags []string, requestedTags []string) bool {
	for _, st := range skillTags {
		normST := strings.ToLower(strings.TrimSpace(st))
		for _, rt := range requestedTags {
			if normST == rt {
				return true
			}
		}
	}
	return false
}

func boundStringSlice(slice []string, maxItems, maxRunesPerItem int) []string {
	if len(slice) == 0 {
		return []string{}
	}
	count := len(slice)
	if count > maxItems {
		count = maxItems
	}
	res := make([]string, count)
	for i := 0; i < count; i++ {
		res[i] = TruncateRunes(slice[i], maxRunesPerItem)
	}
	return res
}

// TruncateRunes truncates a string to at most maxRunes Unicode characters, appending "..." if truncated.
func TruncateRunes(s string, maxRunes int) string {
	if maxRunes <= 0 {
		return ""
	}
	runeCount := utf8.RuneCountInString(s)
	if runeCount <= maxRunes {
		return s
	}
	if maxRunes <= 3 {
		runes := []rune(s)
		return string(runes[:maxRunes])
	}
	runes := []rune(s)
	return string(runes[:maxRunes-3]) + "..."
}
