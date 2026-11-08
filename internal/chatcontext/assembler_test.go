package chatcontext

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"
	"unicode/utf8"

	"bob/internal/knowledge"
	"bob/internal/memory"

	openai "github.com/sashabaranov/go-openai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestAssembler_LargeImagesRetainEarlierConversation(t *testing.T) {
	entries := make([]Entry, 0, 11)
	for i := 1; i <= 10; i++ {
		entries = append(entries, Entry{
			Role: openai.ChatMessageRoleUser, Content: fmt.Sprintf("Earlier message %d", i), Seq: int64(i),
		})
	}
	imageURL := "data:image/jpeg;base64," + strings.Repeat("A", 60*1024)
	entries = append(entries, Entry{
		Role: openai.ChatMessageRoleUser, SenderName: "Alice", Content: "Describe these images", Seq: 11,
		Images: []ImageAttachment{{URL: imageURL}, {URL: imageURL}, {URL: imageURL}},
	})

	res := NewAssembler().Assemble(AssemblyInput{RecentEntries: entries})

	require.Len(t, res.Messages, 11)
	for i := 0; i < 10; i++ {
		assert.Equal(t, entries[i].Content, res.Messages[i].Content)
	}
	require.Len(t, res.Messages[10].MultiContent, 4)
	assert.Equal(t, "Alice: Describe these images", res.Messages[10].MultiContent[0].Text)
	for _, part := range res.Messages[10].MultiContent[1:] {
		require.NotNil(t, part.ImageURL)
		assert.Equal(t, imageURL, part.ImageURL.URL)
	}
	assert.Equal(t, 0, res.Manifest.RingBuffer.DroppedCount)
	assert.False(t, res.Manifest.Usage.RecentTurns.Truncated)
	expectedBytes := 3 * EstimatedImageBytes
	for _, entry := range entries {
		expectedBytes += len(entry.Content)
	}
	expectedBytes += len("Alice: ")
	assert.Equal(t, expectedBytes, res.Manifest.Usage.RecentTurns.Bytes)
}

func TestAssembler_HistoricalImagesStrippedWithoutMutatingBuffer(t *testing.T) {
	mgr := NewManager(10)
	image := ImageAttachment{URL: "data:image/jpeg;base64," + strings.Repeat("A", 60*1024)}
	first := Entry{Role: openai.ChatMessageRoleUser, Content: "First image", Images: []ImageAttachment{image}, Seq: 1}
	mgr.Push("chat-images", first)
	res := mgr.Assemble("chat-images", AssemblyInput{})
	require.Len(t, res.Messages, 1)
	require.Len(t, res.Messages[0].MultiContent, 2)

	mgr.Push("chat-images", Entry{Role: openai.ChatMessageRoleAssistant, Content: "Description", Seq: 2})
	res = mgr.Assemble("chat-images", AssemblyInput{})
	require.Len(t, res.Messages, 3)
	assert.Empty(t, res.Messages[0].MultiContent)
	assert.Equal(t, first.Content, res.Messages[0].Content)

	latest := Entry{Role: openai.ChatMessageRoleUser, Content: "Next image", Images: []ImageAttachment{image}, Seq: 3}
	mgr.Push("chat-images", latest)
	res = mgr.Assemble("chat-images", AssemblyInput{}, SourceBudgets{
		RecentTurnsBytes: len(first.Content) + len("Description") + len(latest.Content) + EstimatedImageBytes,
	})
	require.Len(t, res.Messages, 3)
	assert.Equal(t, 0, res.Manifest.RingBuffer.DroppedCount)
	assert.Empty(t, res.Messages[0].MultiContent)
	assert.Equal(t, first.Content, res.Messages[0].Content)
	require.Len(t, res.Messages[2].MultiContent, 2)
	assert.Equal(t, image.URL, res.Messages[2].MultiContent[1].ImageURL.URL)
	buffered := mgr.GetOrCreate("chat-images").Entries()
	assert.Equal(t, first, buffered[0])
	assert.Equal(t, latest, buffered[2])
}

func TestEntryFormattedBytes_EstimatesImages(t *testing.T) {
	for _, encodedBytes := range []int{8, 60 * 1024} {
		t.Run(fmt.Sprintf("encoded_bytes_%d", encodedBytes), func(t *testing.T) {
			entry := Entry{
				SenderName: "Alice", Content: "Images",
				Images: []ImageAttachment{
					{URL: "data:image/jpeg;base64," + strings.Repeat("A", encodedBytes)},
					{URL: "data:image/jpeg;base64," + strings.Repeat("B", encodedBytes)},
				},
			}
			assert.Equal(t, len("Alice: Images")+2*EstimatedImageBytes, entryFormattedBytes(entry))
			entry.Content = ""
			entry.SenderName = ""
			assert.Equal(t, 2*EstimatedImageBytes, entryFormattedBytes(entry))
			entry.Role = openai.ChatMessageRoleAssistant
			assert.Zero(t, entryFormattedBytes(entry))
		})
	}
}

func TestTruncateEntryToFit_PreservesImagesAndMetadata(t *testing.T) {
	entry := Entry{
		Role: openai.ChatMessageRoleUser, SenderName: "Alice", SenderID: "user-1",
		Content: strings.Repeat("Ж", 50), Seq: 42, Timestamp: 1700000000,
		Images: []ImageAttachment{{URL: "data:image/jpeg;base64," + strings.Repeat("A", 60*1024)}},
	}
	for _, tc := range []struct {
		name       string
		budget     int
		content    string
		senderName string
		keepImages bool
	}{
		{name: "text truncated", budget: EstimatedImageBytes + len("Alice: ") + 5, content: "ЖЖ", senderName: "Alice", keepImages: true},
		{name: "prefix exceeds remaining budget", budget: EstimatedImageBytes + 3, content: "Ж", keepImages: true},
		{name: "image alone fits", budget: EstimatedImageBytes, keepImages: true},
		{name: "image exceeds entire budget", budget: len("Alice: ") + 5, content: "ЖЖ", senderName: "Alice"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := truncateEntryToFit(entry, tc.budget)
			expected := entry
			expected.Content = tc.content
			expected.SenderName = tc.senderName
			if !tc.keepImages {
				expected.Images = nil
			}
			assert.Equal(t, expected, got)
			assert.True(t, utf8.ValidString(got.Content))
			assert.LessOrEqual(t, entryFormattedBytes(got), tc.budget)
			res := NewAssembler(SourceBudgets{RecentTurnsBytes: tc.budget}).Assemble(AssemblyInput{RecentEntries: []Entry{entry}})
			require.Len(t, res.Messages, 1)
			assert.Equal(t, entryToLLMMessage(expected), res.Messages[0])
			assert.True(t, res.Manifest.Usage.RecentTurns.Truncated)
			assert.Equal(t, entryFormattedBytes(expected), res.Manifest.Usage.RecentTurns.Bytes)
		})
	}
}

func TestAssembler_MultiSourceAssemblyAndManifest(t *testing.T) {
	assembler := NewAssembler(DefaultSourceBudgets())

	res := assembler.Assemble(AssemblyInput{
		ChatID:       "chat-1",
		SystemPrompt: "You are Bob, a helpful assistant.",
		Memories: []knowledge.MemoryVersion{
			{
				ID:       "mem_1@2",
				ItemID:   "mem_1",
				Revision: 2,
				Type:     knowledge.MemoryTypeFact,
				Content:  "Production cluster is us-central1.",
				Status:   knowledge.VersionStatusApproved,
			},
		},
		Skills: []knowledge.SkillVersion{
			{
				ID:                   "skill_deploy@3",
				ItemID:               "skill_deploy",
				Revision:             3,
				Name:                 "k8s-deploy",
				Description:          "Deploy Kubernetes manifests safely",
				InstructionsMarkdown: "Run kubectl rollout status after apply.",
				Status:               knowledge.VersionStatusApproved,
			},
		},
		Episodes: []memory.Episode{
			{
				ID:             "ep_chat1_1_5",
				ChatID:         "chat-1",
				StartSeq:       1,
				EndSeq:         5,
				Participants:   []string{"Alice", "Bob"},
				TopicTags:      []string{"deploy"},
				TaskStatus:     memory.EpisodeStatusCompleted,
				Summary:        "Discussed initial staging rollout.",
				SourceChunkIDs: []string{"chunk_chat1_1_5"},
				ModelDerived:   true,
			},
		},
		HistoricalItems: []memory.MemoryItem{
			{
				ID:       "chunk_6_8",
				Kind:     "raw_chunk",
				ChatID:   "chat-1",
				StartSeq: 6,
				EndSeq:   8,
				Content:  "Alice: Let's check staging first.",
			},
		},
		RecentEntries: []Entry{
			{Role: openai.ChatMessageRoleUser, SenderName: "Alice", Content: "Ready to deploy?", Timestamp: 1700000100, Seq: 10},
			{Role: openai.ChatMessageRoleAssistant, Content: "Checking cluster status.", Timestamp: 1700000105, Seq: 11},
		},
		ContinuationPrompt: "Please continue.",
	})

	// System prompt contains base prompt + all 3 structured context blocks
	assert.Contains(t, res.SystemPrompt, "You are Bob, a helpful assistant.")
	assert.Contains(t, res.SystemPrompt, "[Approved Structured Memories]")
	assert.Contains(t, res.SystemPrompt, "[mem_1@2] (fact): Production cluster is us-central1.")
	assert.Contains(t, res.SystemPrompt, "[Approved Procedural Skills]")
	assert.Contains(t, res.SystemPrompt, "### Skill [skill_deploy@3] k8s-deploy")
	assert.Contains(t, res.SystemPrompt, "[Episodic & Historical Context]")
	assert.Contains(t, res.SystemPrompt, "[Model-Derived Episode Summary | seq 1-5]")
	assert.Contains(t, res.SystemPrompt, "chunk_6_8")

	// Messages: [0] user (seq 10), [1] assistant (seq 11), [2] user continuation
	require.Len(t, res.Messages, 3)
	assert.Equal(t, openai.ChatMessageRoleUser, res.Messages[0].Role)
	assert.Contains(t, res.Messages[0].Content, "Ready to deploy?")
	assert.Equal(t, openai.ChatMessageRoleAssistant, res.Messages[1].Role)
	assert.Equal(t, "Checking cluster status.", res.Messages[1].Content)
	assert.Equal(t, openai.ChatMessageRoleUser, res.Messages[2].Role)
	assert.Equal(t, "Please continue.", res.Messages[2].Content)

	// Verify Provenance Manifest and exact byte accounting (including headers, separators, and continuation)
	assert.Equal(t, []string{"mem_1@2"}, res.Manifest.MemoryVersions)
	assert.Equal(t, []string{"skill_deploy@3"}, res.Manifest.SkillVersions)
	assert.Equal(t, []string{"ep_chat1_1_5"}, res.Manifest.EpisodeIDs)
	assert.Equal(t, []string{"chunk_6_8"}, res.Manifest.ChunkIDs)
	assert.Empty(t, res.Manifest.DeduplicatedChunkIDs)

	assert.Equal(t, int64(10), res.Manifest.RingBuffer.MinSeq)
	assert.Equal(t, int64(11), res.Manifest.RingBuffer.MaxSeq)
	assert.Equal(t, 2, res.Manifest.RingBuffer.EntryCount)
	assert.Equal(t, 0, res.Manifest.RingBuffer.DroppedCount)

	expectedSystemBytes := res.Manifest.Usage.SystemPrompt.Bytes +
		res.Manifest.Usage.StructuredMemories.Bytes +
		res.Manifest.Usage.ProceduralSkills.Bytes +
		res.Manifest.Usage.HistoricalContext.Bytes
	assert.Equal(t, len(res.SystemPrompt), expectedSystemBytes)

	emittedMessageBytes := 0
	for _, m := range res.Messages {
		emittedMessageBytes += len(m.Content)
	}
	assert.Equal(t, emittedMessageBytes, res.Manifest.Usage.RecentTurns.Bytes)
	assert.Equal(t, len(res.SystemPrompt)+emittedMessageBytes, res.Manifest.Usage.TotalBytes)

	// Verify JSON serialization round-trip
	require.NotEmpty(t, res.ManifestJSON)
	var decoded AssemblyManifest
	require.NoError(t, json.Unmarshal(res.ManifestJSON, &decoded))
	assert.Equal(t, res.Manifest.MemoryVersions, decoded.MemoryVersions)
	assert.Equal(t, res.Manifest.SkillVersions, decoded.SkillVersions)
	assert.Equal(t, res.Manifest.EpisodeIDs, decoded.EpisodeIDs)
	assert.Equal(t, res.Manifest.ChunkIDs, decoded.ChunkIDs)
	assert.Equal(t, int64(10), decoded.RingBuffer.MinSeq)
	assert.Equal(t, int64(11), decoded.RingBuffer.MaxSeq)
}

func TestAssembler_PerSourceBudgetEnforcement(t *testing.T) {
	budgets := SourceBudgets{
		SystemPromptBytes:      25,  // Force truncation of system prompt (including multi-byte UTF-8)
		StructuredMemoryBytes:  90,  // Fits section header + first memory only
		ProceduralSkillBytes:   100, // Fits section header + first skill only
		HistoricalContextBytes: 300, // Fits section header + first episode only
		RecentTurnsBytes:       90,  // Fits only the newest entry + bounded continuation
	}
	assembler := NewAssembler(budgets)

	res := assembler.Assemble(AssemblyInput{
		ChatID:       "chat-budget",
		SystemPrompt: "System prompt with UTF-8 Привет мир that exceeds twenty-five bytes",
		Memories: []knowledge.MemoryVersion{
			{ID: "mem_1@1", ItemID: "mem_1", Revision: 1, Content: "Short memory 1", Status: knowledge.VersionStatusApproved},
			{ID: "mem_2@1", ItemID: "mem_2", Revision: 1, Content: "Second memory that exceeds the structured memory budget", Status: knowledge.VersionStatusApproved},
		},
		Skills: []knowledge.SkillVersion{
			{ID: "skill_1@1", ItemID: "skill_1", Revision: 1, Name: "S1", InstructionsMarkdown: "Short skill 1", Status: knowledge.VersionStatusApproved},
			{ID: "skill_2@1", ItemID: "skill_2", Revision: 1, Name: "S2", InstructionsMarkdown: "Second skill that exceeds the procedural skill budget", Status: knowledge.VersionStatusApproved},
		},
		HistoricalItems: []memory.MemoryItem{
			{ID: "ep_1_3", Kind: "episode", StartSeq: 1, EndSeq: 3, Content: "Episode 1 summary"},
			{ID: "ep_4_6", Kind: "episode", StartSeq: 4, EndSeq: 6, Content: "Episode 2 summary that exceeds the historical budget"},
		},
		RecentEntries: []Entry{
			{Role: openai.ChatMessageRoleUser, SenderName: "Alice", Content: "Old message that should be dropped due to RecentTurnsBytes cap", Timestamp: 1700000001, Seq: 10},
			{Role: openai.ChatMessageRoleAssistant, Content: "Newest assistant message", Timestamp: 1700000002, Seq: 11},
		},
		ContinuationPrompt: "Please continue with this very long continuation prompt that would otherwise exceed RecentTurnsBytes if not bounded!",
	})

	// 1. SystemPrompt truncated cleanly on UTF-8 boundary
	assert.True(t, res.Manifest.Usage.SystemPrompt.Truncated)
	assert.LessOrEqual(t, res.Manifest.Usage.SystemPrompt.Bytes, budgets.SystemPromptBytes)
	assert.True(t, utf8.ValidString(res.SystemPrompt))

	// 2. StructuredMemoryBytes enforced independently (including section heading)
	assert.Equal(t, []string{"mem_1@1"}, res.Manifest.MemoryVersions)
	assert.Equal(t, []string{"mem_2@1"}, res.Manifest.OmittedMemoryVersions)
	assert.True(t, res.Manifest.Usage.StructuredMemories.Truncated)
	assert.LessOrEqual(t, res.Manifest.Usage.StructuredMemories.Bytes, budgets.StructuredMemoryBytes)

	// 3. ProceduralSkillBytes enforced independently (including section heading)
	assert.Equal(t, []string{"skill_1@1"}, res.Manifest.SkillVersions)
	assert.Equal(t, []string{"skill_2@1"}, res.Manifest.OmittedSkillVersions)
	assert.True(t, res.Manifest.Usage.ProceduralSkills.Truncated)
	assert.LessOrEqual(t, res.Manifest.Usage.ProceduralSkills.Bytes, budgets.ProceduralSkillBytes)

	// 4. HistoricalContextBytes enforced independently (including section heading)
	assert.Equal(t, []string{"ep_1_3"}, res.Manifest.EpisodeIDs)
	assert.Equal(t, []string{"ep_4_6"}, res.Manifest.OmittedHistoricalIDs)
	assert.True(t, res.Manifest.Usage.HistoricalContext.Truncated)
	assert.LessOrEqual(t, res.Manifest.Usage.HistoricalContext.Bytes, budgets.HistoricalContextBytes)

	// Verify actual emitted SystemPrompt size is within the sum of the 4 system-prompt source budgets
	assert.Equal(t, len(res.SystemPrompt),
		res.Manifest.Usage.SystemPrompt.Bytes+
			res.Manifest.Usage.StructuredMemories.Bytes+
			res.Manifest.Usage.ProceduralSkills.Bytes+
			res.Manifest.Usage.HistoricalContext.Bytes)
	assert.LessOrEqual(t, len(res.SystemPrompt),
		budgets.SystemPromptBytes+budgets.StructuredMemoryBytes+budgets.ProceduralSkillBytes+budgets.HistoricalContextBytes)

	// 5. RecentTurnsBytes enforced independently (including continuation prompt)
	assert.Equal(t, 1, res.Manifest.RingBuffer.DroppedCount)
	assert.True(t, res.Manifest.Usage.RecentTurns.Truncated)
	assert.Equal(t, int64(11), res.Manifest.RingBuffer.MinSeq)
	assert.Equal(t, int64(11), res.Manifest.RingBuffer.MaxSeq)
	assert.Equal(t, 1, res.Manifest.RingBuffer.EntryCount)
	require.Len(t, res.Messages, 2) // [0] Newest assistant message, [1] bounded continuation message
	emittedRecentBytes := len(res.Messages[0].Content) + len(res.Messages[1].Content)
	assert.Equal(t, emittedRecentBytes, res.Manifest.Usage.RecentTurns.Bytes)
	assert.LessOrEqual(t, emittedRecentBytes, budgets.RecentTurnsBytes)
}

func TestAssembler_RejectsUnapprovedAndExpiredRecords(t *testing.T) {
	now := time.Date(2026, 11, 7, 12, 0, 0, 0, time.UTC)
	expiredAt := now.Add(-1 * time.Minute).Unix()
	exactNow := now.Unix()
	futureAt := now.Add(10 * time.Minute).Unix()

	assembler := NewAssembler(DefaultSourceBudgets())
	res := assembler.Assemble(AssemblyInput{
		Now:          now,
		SystemPrompt: "Base prompt",
		Memories: []knowledge.MemoryVersion{
			{ID: "mem_empty_status@1", ItemID: "mem_empty_status", Revision: 1, Content: "Empty status", Status: ""},
			{ID: "mem_proposed@1", ItemID: "mem_proposed", Revision: 1, Content: "Proposed memory", Status: knowledge.VersionStatusProposed},
			{ID: "mem_rejected@1", ItemID: "mem_rejected", Revision: 1, Content: "Rejected memory", Status: knowledge.VersionStatusRejected},
			{ID: "mem_expired@1", ItemID: "mem_expired", Revision: 1, Content: "Expired memory", Status: knowledge.VersionStatusApproved, ExpiresAt: &expiredAt},
			{ID: "mem_expired_now@1", ItemID: "mem_expired_now", Revision: 1, Content: "Expired at exact now", Status: knowledge.VersionStatusApproved, ExpiresAt: &exactNow},
			{ID: "mem_valid_ttl@1", ItemID: "mem_valid_ttl", Revision: 1, Content: "Valid TTL memory", Status: knowledge.VersionStatusApproved, ExpiresAt: &futureAt},
			{ID: "mem_valid_no_ttl@2", ItemID: "mem_valid_no_ttl", Revision: 2, Content: "Valid permanent memory", Status: knowledge.VersionStatusApproved},
		},
		Skills: []knowledge.SkillVersion{
			{ID: "skill_empty_status@1", ItemID: "skill_empty_status", Revision: 1, Name: "Empty", InstructionsMarkdown: "Do not include", Status: ""},
			{ID: "skill_proposed@1", ItemID: "skill_proposed", Revision: 1, Name: "Proposed", InstructionsMarkdown: "Do not include", Status: knowledge.VersionStatusProposed},
			{ID: "skill_rejected@1", ItemID: "skill_rejected", Revision: 1, Name: "Rejected", InstructionsMarkdown: "Do not include", Status: knowledge.VersionStatusRejected},
			{ID: "skill_approved@1", ItemID: "skill_approved", Revision: 1, Name: "ApprovedSkill", InstructionsMarkdown: "Approved instructions", Status: knowledge.VersionStatusApproved},
		},
	})

	assert.Equal(t, []string{"mem_valid_ttl@1", "mem_valid_no_ttl@2"}, res.Manifest.MemoryVersions)
	assert.Equal(t, []string{"skill_approved@1"}, res.Manifest.SkillVersions)
	assert.NotContains(t, res.SystemPrompt, "Empty status")
	assert.NotContains(t, res.SystemPrompt, "Proposed memory")
	assert.NotContains(t, res.SystemPrompt, "Rejected memory")
	assert.NotContains(t, res.SystemPrompt, "Expired memory")
	assert.NotContains(t, res.SystemPrompt, "Do not include")
	assert.Contains(t, res.SystemPrompt, "Valid TTL memory")
	assert.Contains(t, res.SystemPrompt, "Valid permanent memory")
	assert.Contains(t, res.SystemPrompt, "Approved instructions")
}

func TestAssembler_SingleOversizedRecentTurnTruncatedUTF8Safe(t *testing.T) {
	budgets := DefaultSourceBudgets()
	budgets.RecentTurnsBytes = 40
	assembler := NewAssembler(budgets)

	res := assembler.Assemble(AssemblyInput{
		RecentEntries: []Entry{
			{Role: openai.ChatMessageRoleUser, Content: "Older turn", Seq: 1},
			{Role: openai.ChatMessageRoleUser, Content: strings.Repeat("Ж", 50), Seq: 2},
		},
	})

	assert.Equal(t, 1, res.Manifest.RingBuffer.DroppedCount)
	assert.True(t, res.Manifest.Usage.RecentTurns.Truncated)
	require.Len(t, res.Messages, 1)
	assert.LessOrEqual(t, len(res.Messages[0].Content), 40)
	assert.True(t, utf8.ValidString(res.Messages[0].Content))
}

func TestAssembler_SequenceOverlapDeduplication_GapsAndCrossChat(t *testing.T) {
	assembler := NewAssembler(DefaultSourceBudgets())

	res := assembler.Assemble(AssemblyInput{
		ChatID:       "dm_alice",
		SystemPrompt: "Base prompt",
		HistoricalItems: []memory.MemoryItem{
			// Non-overlapping older raw chunk (seq 1..5) -> kept
			{ID: "chunk_dm_alice_1_5", Kind: "raw_chunk", ChatID: "dm_alice", StartSeq: 1, EndSeq: 5, Content: "Historical messages 1..5"},
			// Gap raw chunk (seq 11..19 lies strictly between retained ring buffer sequences {10, 20}) -> MUST BE KEPT!
			{ID: "chunk_dm_alice_11_19", Kind: "raw_chunk", ChatID: "dm_alice", StartSeq: 11, EndSeq: 19, Content: "Gap messages 11..19"},
			// Overlapping same-chat raw chunk (seq 8..12 overlaps retained seq 10) -> deduplicated!
			{ID: "chunk_dm_alice_8_12", Kind: "raw_chunk", ChatID: "dm_alice", StartSeq: 8, EndSeq: 12, Content: "Overlapping messages 8..12"},
			// Point overlap same-chat raw chunk (seq 20..20 overlaps retained seq 20) -> deduplicated!
			{ID: "chunk_dm_alice_20_20", Kind: "raw_chunk", ChatID: "dm_alice", StartSeq: 20, EndSeq: 20, Content: "Overlapping message 20"},
			// Cross-chat Townhall raw chunk with overlapping sequence numbers 10..20 -> MUST BE KEPT (different chat)!
			{ID: "chunk_townhall_10_20", Kind: "raw_chunk", ChatID: "townhall", StartSeq: 10, EndSeq: 20, Content: "Townhall messages 10..20"},
			// Episode summary covering 8..12 -> kept even if range overlaps (episodes are high-level summaries, not duplicate raw turns)
			{ID: "ep_dm_alice_8_12", Kind: "episode", ChatID: "dm_alice", StartSeq: 8, EndSeq: 12, Content: "Episode summary for 8..12"},
		},
		RecentEntries: []Entry{
			{Role: openai.ChatMessageRoleUser, SenderName: "Alice", Content: "Turn 10", Timestamp: 1700000010, Seq: 10},
			{Role: openai.ChatMessageRoleAssistant, Content: "Turn 20", Timestamp: 1700000020, Seq: 20},
		},
	})

	assert.Equal(t, []string{"chunk_dm_alice_1_5", "chunk_dm_alice_11_19", "chunk_townhall_10_20"}, res.Manifest.ChunkIDs)
	assert.Equal(t, []string{"ep_dm_alice_8_12"}, res.Manifest.EpisodeIDs)
	assert.Equal(t, []string{"chunk_dm_alice_8_12", "chunk_dm_alice_20_20"}, res.Manifest.DeduplicatedChunkIDs)
	assert.Contains(t, res.SystemPrompt, "Historical messages 1..5")
	assert.Contains(t, res.SystemPrompt, "Gap messages 11..19")
	assert.Contains(t, res.SystemPrompt, "Townhall messages 10..20")
	assert.Contains(t, res.SystemPrompt, "Episode summary for 8..12")
	assert.NotContains(t, res.SystemPrompt, "Overlapping messages 8..12")
	assert.NotContains(t, res.SystemPrompt, "Overlapping message 20")
}

func TestManager_Assemble(t *testing.T) {
	mgr := NewManager(10)
	mgr.Push("chat-1", Entry{Role: openai.ChatMessageRoleUser, SenderName: "Alice", Content: "Hello", Timestamp: 100, Seq: 1})
	mgr.Push("chat-1", Entry{Role: openai.ChatMessageRoleAssistant, Content: "Hi there!", Timestamp: 101, Seq: 2})

	res := mgr.Assemble("chat-1", AssemblyInput{
		SystemPrompt:       "System instructions",
		ContinuationPrompt: "Please continue.",
	})

	assert.Equal(t, "System instructions", res.SystemPrompt)
	require.Len(t, res.Messages, 3) // user + assistant + continuation
	assert.Equal(t, openai.ChatMessageRoleUser, res.Messages[0].Role)
	assert.Equal(t, openai.ChatMessageRoleAssistant, res.Messages[1].Role)
	assert.Equal(t, openai.ChatMessageRoleUser, res.Messages[2].Role)
	assert.Equal(t, "Please continue.", res.Messages[2].Content)
	assert.Equal(t, int64(1), res.Manifest.RingBuffer.MinSeq)
	assert.Equal(t, int64(2), res.Manifest.RingBuffer.MaxSeq)
}
