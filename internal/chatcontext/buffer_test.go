package chatcontext

import (
	"fmt"
	"sync"
	"testing"
	"time"

	openai "github.com/sashabaranov/go-openai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

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

