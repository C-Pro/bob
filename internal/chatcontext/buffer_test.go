package chatcontext

import (
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	openai "github.com/sashabaranov/go-openai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRingBuffer_UpdateLast(t *testing.T) {
	rb := NewRingBuffer(10)
	assert.False(t, rb.UpdateLast(func(e *Entry) bool { return true }))
	rb.Push(Entry{Role: "user", Content: "Older", Seq: 1})
	rb.Push(Entry{Role: "user", Content: "Newest", Seq: 2, Images: []ImageAttachment{{URL: "image"}}})
	rb.Push(Entry{Role: "assistant", Content: "Reply", Seq: 3})
	var visited []int64
	require.True(t, rb.UpdateLast(func(e *Entry) bool {
		visited = append(visited, e.Seq)
		if e.Role != "user" {
			return false
		}
		e.Content = "Enriched"
		e.Images = nil
		return true
	}))
	assert.Equal(t, []int64{3, 2}, visited)
	entries := rb.Entries()
	assert.Equal(t, "Older", entries[0].Content)
	assert.Equal(t, "Enriched", entries[1].Content)
	assert.Nil(t, entries[1].Images)
	assert.Equal(t, "Reply", entries[2].Content)
	assert.False(t, rb.UpdateLast(func(e *Entry) bool { return e.Seq == 99 }))

	var wg sync.WaitGroup
	for range 10 {
		wg.Go(func() {
			for range 100 {
				rb.UpdateLast(func(e *Entry) bool {
					if e.Role != "user" {
						return false
					}
					e.Timestamp++
					return true
				})
				rb.Entries()
			}
		})
	}
	wg.Wait()
	assert.Equal(t, int64(1000), rb.Entries()[1].Timestamp)
}

func TestRingBuffer_PushAndEviction(t *testing.T) {
	rb := NewRingBuffer(3)
	assert.Equal(t, 3, rb.Capacity())
	assert.Equal(t, 0, rb.Len())

	// Push 1
	rb.Push(Entry{Role: "user", SenderName: "Alice", Content: "Msg 1", Timestamp: 100})
	assert.Equal(t, 1, rb.Len())

	// Push 2 & 3
	rb.Push(Entry{Role: "assistant", Content: "Reply 1", Timestamp: 101})
	rb.Push(Entry{Role: "user", SenderName: "Bob", Content: "Msg 2", Timestamp: 102})
	assert.Equal(t, 3, rb.Len())

	entries := rb.Entries()
	require.Len(t, entries, 3)
	assert.Equal(t, "Msg 1", entries[0].Content)
	assert.Equal(t, "Reply 1", entries[1].Content)
	assert.Equal(t, "Msg 2", entries[2].Content)

	// Push 4th (capacity 3, evictCount = 3/3 = 1 -> evicts Msg 1)
	rb.Push(Entry{Role: "assistant", Content: "Reply 2", Timestamp: 103})
	assert.Equal(t, 3, rb.Len())

	entries = rb.Entries()
	require.Len(t, entries, 3)
	assert.Equal(t, "Reply 1", entries[0].Content)
	assert.Equal(t, "Msg 2", entries[1].Content)
	assert.Equal(t, "Reply 2", entries[2].Content)
}

func TestRingBuffer_ChunkedEvictionAndPrefixStability(t *testing.T) {
	// Capacity 6: evictCount = 6 / 3 = 2
	rb := NewRingBuffer(6)

	for i := 1; i <= 6; i++ {
		rb.Push(Entry{Role: "user", Content: fmt.Sprintf("Msg %d", i)})
	}
	require.Equal(t, 6, rb.Len())
	for i := 1; i <= 6; i++ {
		assert.Equal(t, fmt.Sprintf("Msg %d", i), rb.Entries()[i-1].Content)
	}

	// 7th push: exceeds capacity 6 -> evicts oldest 2 (Msg 1, Msg 2). Retains 4 + 1 = 5 items (Msg 3..7)
	rb.Push(Entry{Role: "user", Content: "Msg 7"})
	require.Equal(t, 5, rb.Len())
	entries := rb.Entries()
	assert.Equal(t, "Msg 3", entries[0].Content)
	assert.Equal(t, "Msg 4", entries[1].Content)
	assert.Equal(t, "Msg 5", entries[2].Content)
	assert.Equal(t, "Msg 6", entries[3].Content)
	assert.Equal(t, "Msg 7", entries[4].Content)

	// 8th push: reaches capacity 6 (Msg 3..8), no eviction occurred on this push so prefix [Msg 3..7] remained static
	rb.Push(Entry{Role: "user", Content: "Msg 8"})
	require.Equal(t, 6, rb.Len())
	entries = rb.Entries()
	assert.Equal(t, "Msg 3", entries[0].Content)
	assert.Equal(t, "Msg 4", entries[1].Content)
	assert.Equal(t, "Msg 5", entries[2].Content)
	assert.Equal(t, "Msg 6", entries[3].Content)
	assert.Equal(t, "Msg 7", entries[4].Content)
	assert.Equal(t, "Msg 8", entries[5].Content)

	// 9th push: exceeds capacity 6 -> evicts oldest 2 (Msg 3, Msg 4). Retains 4 + 1 = 5 items (Msg 5..9)
	rb.Push(Entry{Role: "user", Content: "Msg 9"})
	require.Equal(t, 5, rb.Len())
	entries = rb.Entries()
	assert.Equal(t, "Msg 5", entries[0].Content)
	assert.Equal(t, "Msg 6", entries[1].Content)
	assert.Equal(t, "Msg 7", entries[2].Content)
	assert.Equal(t, "Msg 8", entries[3].Content)
	assert.Equal(t, "Msg 9", entries[4].Content)
}

func TestRingBuffer_ChunkedEvictionEdgeCases(t *testing.T) {
	// Capacity 1: evictCount = max(1, 1/3) = 1
	rb1 := NewRingBuffer(1)
	rb1.Push(Entry{Content: "A"})
	assert.Equal(t, 1, rb1.Len())
	assert.Equal(t, "A", rb1.Entries()[0].Content)
	rb1.Push(Entry{Content: "B"})
	assert.Equal(t, 1, rb1.Len())
	assert.Equal(t, "B", rb1.Entries()[0].Content)

	// Capacity 2: evictCount = max(1, 2/3) = 1
	rb2 := NewRingBuffer(2)
	rb2.Push(Entry{Content: "A"})
	rb2.Push(Entry{Content: "B"})
	assert.Equal(t, 2, rb2.Len())
	rb2.Push(Entry{Content: "C"})
	assert.Equal(t, 2, rb2.Len())
	assert.Equal(t, "B", rb2.Entries()[0].Content)
	assert.Equal(t, "C", rb2.Entries()[1].Content)

	// Capacity 100: evictCount = 100 / 3 = 33
	rb100 := NewRingBuffer(100)
	for i := 1; i <= 100; i++ {
		rb100.Push(Entry{Content: fmt.Sprintf("M%d", i)})
	}
	assert.Equal(t, 100, rb100.Len())

	// Push 101: evicts 33 -> 67 items + 1 = 68 items (M34..M101)
	rb100.Push(Entry{Content: "M101"})
	assert.Equal(t, 68, rb100.Len())
	assert.Equal(t, "M34", rb100.Entries()[0].Content)
	assert.Equal(t, "M101", rb100.Entries()[67].Content)

	// Next 32 pushes (up to M133) should NOT evict anything, preserving prefix M34..M101
	for i := 102; i <= 133; i++ {
		rb100.Push(Entry{Content: fmt.Sprintf("M%d", i)})
	}
	assert.Equal(t, 100, rb100.Len())
	assert.Equal(t, "M34", rb100.Entries()[0].Content)
	assert.Equal(t, "M133", rb100.Entries()[99].Content)
}

func TestRingBuffer_ToLLMMessages(t *testing.T) {
	rb := NewRingBuffer(5)
	rb.Push(Entry{Role: "user", SenderName: "Alice", Content: "Hello world"})
	rb.Push(Entry{Role: "assistant", Content: "Hi Alice!"})
	rb.Push(Entry{Role: "user", SenderName: "", Content: "No sender name"})

	msgs := rb.ToLLMMessages()
	require.Len(t, msgs, 3)

	assert.Equal(t, openai.ChatMessageRoleUser, msgs[0].Role)
	assert.Equal(t, "Alice: Hello world", msgs[0].Content)

	assert.Equal(t, openai.ChatMessageRoleAssistant, msgs[1].Role)
	assert.Equal(t, "Hi Alice!", msgs[1].Content)

	assert.Equal(t, openai.ChatMessageRoleUser, msgs[2].Role)
	assert.Equal(t, "No sender name", msgs[2].Content)
	assert.Nil(t, msgs[2].MultiContent)
}

func TestRingBuffer_ToLLMMessages_Multimodal(t *testing.T) {
	rb := NewRingBuffer(5)

	// User message with single image
	rb.Push(Entry{
		Role:       "user",
		SenderName: "Alice",
		Content:    "Look at this screenshot",
		Images: []ImageAttachment{
			{
				URL: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
			},
		},
	})

	// Assistant response (text only)
	rb.Push(Entry{
		Role:    "assistant",
		Content: "I see a 1x1 pixel image!",
	})

	// User message with multiple images
	rb.Push(Entry{
		Role:       "user",
		SenderName: "Bob",
		Content:    "Here are two charts",
		Images: []ImageAttachment{
			{
				URL: "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD/",
			},
			{
				URL: "data:image/webp;base64,UklGRkAAAABXRUJQVlA4IDQAAADwAQCdASoBAAEAAQAcJaACdLoB+AA/v2AAA",
			},
		},
	})

	msgs := rb.ToLLMMessages()
	require.Len(t, msgs, 3)

	// Check 1st message (Alice)
	assert.Equal(t, openai.ChatMessageRoleUser, msgs[0].Role)
	assert.Empty(t, msgs[0].Content)
	require.Len(t, msgs[0].MultiContent, 2)
	assert.Equal(t, openai.ChatMessagePartTypeText, msgs[0].MultiContent[0].Type)
	assert.Equal(t, "Alice: Look at this screenshot", msgs[0].MultiContent[0].Text)
	assert.Equal(t, openai.ChatMessagePartTypeImageURL, msgs[0].MultiContent[1].Type)
	assert.Equal(t, "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==", msgs[0].MultiContent[1].ImageURL.URL)
	assert.Equal(t, openai.ImageURLDetailAuto, msgs[0].MultiContent[1].ImageURL.Detail)

	// Check 2nd message (Assistant)
	assert.Equal(t, openai.ChatMessageRoleAssistant, msgs[1].Role)
	assert.Equal(t, "I see a 1x1 pixel image!", msgs[1].Content)
	assert.Nil(t, msgs[1].MultiContent)

	// Check 3rd message (Bob with 2 images)
	assert.Equal(t, openai.ChatMessageRoleUser, msgs[2].Role)
	assert.Empty(t, msgs[2].Content)
	require.Len(t, msgs[2].MultiContent, 3)
	assert.Equal(t, openai.ChatMessagePartTypeText, msgs[2].MultiContent[0].Type)
	assert.Equal(t, "Bob: Here are two charts", msgs[2].MultiContent[0].Text)
	assert.Equal(t, "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD/", msgs[2].MultiContent[1].ImageURL.URL)
	assert.Equal(t, "data:image/webp;base64,UklGRkAAAABXRUJQVlA4IDQAAADwAQCdASoBAAEAAQAcJaACdLoB+AA/v2AAA", msgs[2].MultiContent[2].ImageURL.URL)
}

func TestRingBuffer_ToLLMMessages_AssistantImagesOmitted(t *testing.T) {
	rb := NewRingBuffer(5)

	// Assistant message with images (e.g. from sandbox_upload_attachment)
	// should NOT produce image_url parts — only text.
	rb.Push(Entry{
		Role:    "assistant",
		Content: "Here is the chart you requested",
		Images: []ImageAttachment{
			{URL: "data:image/png;base64,fakedata"},
		},
	})

	// Assistant message with images but no text content.
	rb.Push(Entry{
		Role: "assistant",
		Images: []ImageAttachment{
			{URL: "data:image/png;base64,moredata"},
		},
	})

	msgs := rb.ToLLMMessages()
	require.Len(t, msgs, 2)

	assert.Equal(t, openai.ChatMessageRoleAssistant, msgs[0].Role)
	assert.Equal(t, "Here is the chart you requested", msgs[0].Content)
	assert.Nil(t, msgs[0].MultiContent)

	assert.Equal(t, openai.ChatMessageRoleAssistant, msgs[1].Role)
	assert.Empty(t, msgs[1].Content)
	assert.Nil(t, msgs[1].MultiContent)
}

func TestRingBuffer_Clear(t *testing.T) {
	rb := NewRingBuffer(5)
	rb.Push(Entry{Role: "user", Content: "test"})
	assert.Equal(t, 1, rb.Len())

	rb.Clear()
	assert.Equal(t, 0, rb.Len())
	assert.Empty(t, rb.Entries())
	assert.Empty(t, rb.ToLLMMessages())
}

func TestRingBuffer_DefaultCapacity(t *testing.T) {
	rb := NewRingBuffer(0)
	assert.Equal(t, 100, rb.Capacity())

	rbNeg := NewRingBuffer(-5)
	assert.Equal(t, 100, rbNeg.Capacity())
}

func TestRingBuffer_Concurrency(t *testing.T) {
	rb := NewRingBuffer(50)
	var wg sync.WaitGroup

	numWriters := 10
	iterations := 100

	for i := 0; i < numWriters; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for j := 0; j < iterations; j++ {
				rb.Push(Entry{
					Role:       "user",
					SenderName: fmt.Sprintf("User%d", workerID),
					Content:    fmt.Sprintf("Msg %d-%d", workerID, j),
					Timestamp:  time.Now().UnixNano(),
				})
				_ = rb.Len()
				_ = rb.Entries()
				_ = rb.ToLLMMessages()
			}
		}(i)
	}

	wg.Wait()
	assert.GreaterOrEqual(t, rb.Len(), 50-50/3)
	assert.LessOrEqual(t, rb.Len(), 50)
}

func TestManager_PerChatSeparation(t *testing.T) {
	mgr := NewManager(2)

	mgr.Push("townhall", Entry{Role: "user", SenderName: "Alice", Content: "Townhall 1"})
	mgr.Push("dm_1", Entry{Role: "user", SenderName: "Bob", Content: "DM 1"})
	mgr.Push("townhall", Entry{Role: "user", SenderName: "Charlie", Content: "Townhall 2"})

	thMsgs := mgr.GetLLMMessages("townhall")
	require.Len(t, thMsgs, 2)
	assert.Equal(t, "Alice: Townhall 1", thMsgs[0].Content)
	assert.Equal(t, "Charlie: Townhall 2", thMsgs[1].Content)

	dmMsgs := mgr.GetLLMMessages("dm_1")
	require.Len(t, dmMsgs, 1)
	assert.Equal(t, "Bob: DM 1", dmMsgs[0].Content)

	// Non-existent chat should return empty
	emptyMsgs := mgr.GetLLMMessages("dm_nonexistent")
	assert.Empty(t, emptyMsgs)
}

func TestRingBuffer_EvictionHook(t *testing.T) {
	rb := NewRingBuffer(3)
	var evictedEntries []Entry
	var mu sync.Mutex

	rb.SetOnEvict(func(evicted []Entry) {
		mu.Lock()
		defer mu.Unlock()
		evictedEntries = append(evictedEntries, evicted...)
	})

	rb.Push(Entry{Seq: 1, Content: "Msg 1"})
	rb.Push(Entry{Seq: 2, Content: "Msg 2"})
	rb.Push(Entry{Seq: 3, Content: "Msg 3"})

	mu.Lock()
	assert.Empty(t, evictedEntries)
	mu.Unlock()

	// 4th push triggers eviction of 1 item (Msg 1)
	rb.Push(Entry{Seq: 4, Content: "Msg 4"})

	mu.Lock()
	require.Len(t, evictedEntries, 1)
	assert.Equal(t, int64(1), evictedEntries[0].Seq)
	assert.Equal(t, "Msg 1", evictedEntries[0].Content)
	mu.Unlock()
}

func TestManager_EvictionHook(t *testing.T) {
	mgr := NewManager(3)
	type evictedCall struct {
		chatID  string
		entries []Entry
	}
	var calls []evictedCall
	var mu sync.Mutex

	mgr.SetOnEvict(func(chatID string, evicted []Entry) {
		mu.Lock()
		defer mu.Unlock()
		calls = append(calls, evictedCall{chatID: chatID, entries: evicted})
	})

	for i := 1; i <= 4; i++ {
		mgr.Push("dm_user1", Entry{Seq: int64(i), Content: fmt.Sprintf("User1 Msg %d", i)})
	}

	mu.Lock()
	require.Len(t, calls, 1)
	assert.Equal(t, "dm_user1", calls[0].chatID)
	require.Len(t, calls[0].entries, 1)
	assert.Equal(t, int64(1), calls[0].entries[0].Seq)
	assert.Equal(t, "User1 Msg 1", calls[0].entries[0].Content)
	mu.Unlock()
}

func TestRingBuffer_ReconcileAssistantEcho(t *testing.T) {
	t.Run("reconciles pending outgoing entry even across interleaved user messages", func(t *testing.T) {
		rb := NewRingBuffer(10)
		rb.Push(Entry{Role: "assistant", SenderID: "bot-1", Content: "Proceed?"})
		rb.Push(Entry{Role: "user", SenderID: "alice", Seq: 11, Content: "Wait a sec"})

		appended := rb.ReconcileAssistantEcho(Entry{
			Seq:       10,
			Role:      "assistant",
			SenderID:  "bot-1",
			Content:   "  Proceed?  ",
			Timestamp: 1700000010,
		})
		assert.False(t, appended)
		entries := rb.Entries()
		require.Len(t, entries, 2)
		assert.Equal(t, int64(10), entries[0].Seq)
		assert.Equal(t, int64(11), entries[1].Seq)

		// Exact same Seq echo again is deduplicated
		assert.False(t, rb.ReconcileAssistantEcho(Entry{
			Seq:      10,
			Role:     "assistant",
			SenderID: "bot-1",
			Content:  "Proceed?",
		}))
		assert.Equal(t, 2, rb.Len())
	})

	t.Run("preserves repeated identical prompts with distinct sequences", func(t *testing.T) {
		rb := NewRingBuffer(10)
		assert.True(t, rb.ReconcileAssistantEcho(Entry{
			Seq:      21,
			Role:     "assistant",
			SenderID: "bot-1",
			Content:  "Deploy now?",
		}))
		assert.True(t, rb.ReconcileAssistantEcho(Entry{
			Seq:      22,
			Role:     "assistant",
			SenderID: "bot-1",
			Content:  "Deploy now?",
		}))
		entries := rb.Entries()
		require.Len(t, entries, 2)
		assert.Equal(t, int64(21), entries[0].Seq)
		assert.Equal(t, int64(22), entries[1].Seq)
	})

	t.Run("appends echo when pending outgoing entry was already evicted", func(t *testing.T) {
		rb := NewRingBuffer(3)
		rb.Push(Entry{Role: "assistant", SenderID: "bot-1", Content: "Proceed?"})
		rb.Push(Entry{Seq: 2, Role: "user", Content: "u2"})
		rb.Push(Entry{Seq: 3, Role: "user", Content: "u3"})
		rb.Push(Entry{Seq: 4, Role: "user", Content: "u4"}) // evicts pending Seq=0 entry

		appended := rb.ReconcileAssistantEcho(Entry{
			Seq:      1,
			Role:     "assistant",
			SenderID: "bot-1",
			Content:  "Proceed?",
		})
		assert.True(t, appended)
		entries := rb.Entries()
		assert.Equal(t, int64(1), entries[len(entries)-1].Seq)
	})
}

func TestRingBuffer_ToLLMMessages_TimestampGapInjection(t *testing.T) {
	baseTS := int64(1700000000) // 2023-11-14T22:13:20Z

	rb := NewRingBuffer(10)
	// First message: no previous message -> no timestamp injected
	rb.Push(Entry{Role: "user", SenderID: "u1", SenderName: "Alice", Content: "First", Timestamp: baseTS})
	// Within 1 hour (exactly 3600s, not > 1h) -> no timestamp injected
	rb.Push(Entry{Role: "user", SenderID: "u1", SenderName: "Alice", Content: "Within hour", Timestamp: baseTS + 3600})
	// Assistant reply > 1h later -> assistant messages never get timestamp injected
	rb.Push(Entry{Role: "assistant", SenderID: "bot-1", Content: "Bot reply", Timestamp: baseTS + 10000})
	// User reply > 1h after assistant reply (3601s) -> RFC3339 UTC timestamp injected
	userAfterGapTS := baseTS + 10000 + 3601
	rb.Push(Entry{Role: "user", SenderID: "u1", SenderName: "Alice", Content: "Back again", Timestamp: userAfterGapTS})
	// User reply with missing SenderName and > 1h gap
	userNoSenderTS := userAfterGapTS + 7200
	rb.Push(Entry{Role: "user", SenderID: "u1", Content: "Ping", Timestamp: userNoSenderTS})
	// Multimodal user message with empty text and > 1h gap
	imgGapTS := userNoSenderTS + 4000
	rb.Push(Entry{
		Role:      "user",
		SenderID:  "u1",
		Content:   "",
		Images:    []ImageAttachment{{URL: "data:image/png;base64,AAA"}},
		Timestamp: imgGapTS,
	})

	msgs := rb.ToLLMMessages()
	require.Len(t, msgs, 6)

	assert.Equal(t, "Alice: First", msgs[0].Content)
	assert.Equal(t, "Alice: Within hour", msgs[1].Content)
	assert.Equal(t, "Bot reply", msgs[2].Content)
	assert.Equal(t, "["+time.Unix(userAfterGapTS, 0).UTC().Format(time.RFC3339)+"] Alice: Back again", msgs[3].Content)
	assert.Equal(t, "["+time.Unix(userNoSenderTS, 0).UTC().Format(time.RFC3339)+"] Ping", msgs[4].Content)
	require.Len(t, msgs[5].MultiContent, 2)
	assert.Equal(t, "["+time.Unix(imgGapTS, 0).UTC().Format(time.RFC3339)+"]", msgs[5].MultiContent[0].Text)
	assert.Equal(t, openai.ChatMessagePartTypeImageURL, msgs[5].MultiContent[1].Type)

	// Assembler.Assemble applies the same timestamp injection and accounts for its byte cost.
	res := NewAssembler().Assemble(AssemblyInput{
		RecentEntries: rb.Entries(),
	})
	require.Len(t, res.Messages, 6)
	assert.Equal(t, msgs[3].Content, res.Messages[3].Content)
	assert.Equal(t, msgs[4].Content, res.Messages[4].Content)
	assert.Equal(t, emittedMessagesBytes(res.Messages), res.Manifest.Usage.RecentTurns.Bytes)

	t.Run("zero and backward timestamps and empty role", func(t *testing.T) {
		rbEdge := NewRingBuffer(10)
		rbEdge.Push(Entry{Role: "", SenderID: "u1", Content: "No ts 1", Timestamp: 0})
		rbEdge.Push(Entry{Role: "", SenderID: "u1", Content: "Has ts", Timestamp: baseTS})
		rbEdge.Push(Entry{Role: "", SenderID: "u1", Content: "Zero ts", Timestamp: 0})
		rbEdge.Push(Entry{Role: "", SenderID: "u1", Content: "Backward ts", Timestamp: baseTS - 7200})

		edgeMsgs := rbEdge.ToLLMMessages()
		require.Len(t, edgeMsgs, 4)
		assert.Equal(t, openai.ChatMessageRoleUser, edgeMsgs[0].Role)
		assert.Equal(t, "No ts 1", edgeMsgs[0].Content)
		assert.Equal(t, "Has ts", edgeMsgs[1].Content)
		assert.Equal(t, "Zero ts", edgeMsgs[2].Content)
		assert.Equal(t, "Backward ts", edgeMsgs[3].Content)
	})

	t.Run("dropped predecessor preserves timestamp and exact manifest bytes", func(t *testing.T) {
		entries := []Entry{
			{Role: "assistant", SenderID: "bot-1", Content: strings.Repeat("x", 100), Timestamp: baseTS},
			{Role: "user", SenderID: "u1", Content: "Ping", Timestamp: baseTS + 7200},
		}
		resDropped := NewAssembler(SourceBudgets{RecentTurnsBytes: 40}).Assemble(AssemblyInput{
			RecentEntries: entries,
		})
		require.Len(t, resDropped.Messages, 1)
		expected := "[" + time.Unix(baseTS+7200, 0).UTC().Format(time.RFC3339) + "] Ping"
		assert.Equal(t, expected, resDropped.Messages[0].Content)
		assert.Equal(t, len(expected), resDropped.Manifest.Usage.RecentTurns.Bytes)
		assert.Equal(t, 1, resDropped.Manifest.Usage.RecentTurns.OmittedCount)
	})

	t.Run("latest-turn truncation reserves timestamp overhead", func(t *testing.T) {
		prev := Entry{Role: "assistant", SenderID: "bot-1", Content: "Old", Timestamp: baseTS}
		longUser := Entry{
			Role:       "user",
			SenderID:   "u1",
			SenderName: "Alice",
			Content:    strings.Repeat("a", 100),
			Timestamp:  baseTS + 7200,
		}
		res40 := NewAssembler(SourceBudgets{RecentTurnsBytes: 40}).Assemble(AssemblyInput{
			RecentEntries: []Entry{prev, longUser},
		})
		require.Len(t, res40.Messages, 1)
		assert.LessOrEqual(t, len(res40.Messages[0].Content), 40)
		assert.Equal(t, len(res40.Messages[0].Content), res40.Manifest.Usage.RecentTurns.Bytes)
		assert.True(t, strings.HasPrefix(res40.Messages[0].Content, "["+time.Unix(baseTS+7200, 0).UTC().Format(time.RFC3339)+"]"))

		// UTF-8 multi-byte truncation with timestamp overhead
		utf8User := Entry{
			Role:      "user",
			SenderID:  "u1",
			Content:   strings.Repeat("é", 30),
			Timestamp: baseTS + 7200,
		}
		resUTF8 := NewAssembler(SourceBudgets{RecentTurnsBytes: 30}).Assemble(AssemblyInput{
			RecentEntries: []Entry{prev, utf8User},
		})
		require.Len(t, resUTF8.Messages, 1)
		assert.LessOrEqual(t, len(resUTF8.Messages[0].Content), 30)
		assert.Equal(t, len(resUTF8.Messages[0].Content), resUTF8.Manifest.Usage.RecentTurns.Bytes)

		// Budget smaller than timestamp length (22 bytes) drops timestamp rather than exceeding budget
		resTiny := NewAssembler(SourceBudgets{RecentTurnsBytes: 10}).Assemble(AssemblyInput{
			RecentEntries: []Entry{prev, longUser},
		})
		require.Len(t, resTiny.Messages, 1)
		assert.LessOrEqual(t, len(resTiny.Messages[0].Content), 10)
		assert.Equal(t, len(resTiny.Messages[0].Content), resTiny.Manifest.Usage.RecentTurns.Bytes)
	})
}

func emittedMessagesBytes(msgs []openai.ChatCompletionMessage) int {
	total := 0
	for _, m := range msgs {
		if len(m.MultiContent) > 0 {
			for _, p := range m.MultiContent {
				total += len(p.Text)
				if p.ImageURL != nil {
					total += EstimatedImageBytes
				}
			}
		} else {
			total += len(m.Content)
		}
	}
	return total
}
