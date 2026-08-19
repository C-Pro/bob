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

	// Push 4th (should evict Msg 1)
	rb.Push(Entry{Role: "assistant", Content: "Reply 2", Timestamp: 103})
	assert.Equal(t, 3, rb.Len())

	entries = rb.Entries()
	require.Len(t, entries, 3)
	assert.Equal(t, "Reply 1", entries[0].Content)
	assert.Equal(t, "Msg 2", entries[1].Content)
	assert.Equal(t, "Reply 2", entries[2].Content)
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
	assert.Equal(t, 50, rb.Len())
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
