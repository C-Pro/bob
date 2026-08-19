package chatcontext

import (
	"fmt"
	"strings"
	"sync"

	openai "github.com/sashabaranov/go-openai"
)

// Entry represents a single turn in a chat conversation.
type Entry struct {
	Role       string `json:"role"`       // "user" or "assistant"
	SenderID   string `json:"senderId"`   // user/bot ID
	SenderName string `json:"senderName"` // display name or username
	Content    string `json:"content"`    // message text content
	Timestamp  int64  `json:"timestamp"`  // unix timestamp
}

// RingBuffer is a thread-safe circular buffer holding up to `capacity` entries.
type RingBuffer struct {
	mu       sync.RWMutex
	capacity int
	entries  []Entry
}

// NewRingBuffer creates a new RingBuffer with the given maximum capacity.
func NewRingBuffer(capacity int) *RingBuffer {
	if capacity <= 0 {
		capacity = 100
	}
	return &RingBuffer{
		capacity: capacity,
		entries:  make([]Entry, 0, capacity),
	}
}

// Push adds an entry to the ring buffer, evicting the oldest entry if capacity is exceeded.
func (rb *RingBuffer) Push(entry Entry) {
	rb.mu.Lock()
	defer rb.mu.Unlock()

	rb.entries = append(rb.entries, entry)
	if len(rb.entries) > rb.capacity {
		// Retain only the latest capacity items
		rb.entries = rb.entries[len(rb.entries)-rb.capacity:]
	}
}

// Entries returns a copy of all entries in chronological order (oldest first).
func (rb *RingBuffer) Entries() []Entry {
	rb.mu.RLock()
	defer rb.mu.RUnlock()

	res := make([]Entry, len(rb.entries))
	copy(res, rb.entries)
	return res
}

// Len returns the current number of entries in the buffer.
func (rb *RingBuffer) Len() int {
	rb.mu.RLock()
	defer rb.mu.RUnlock()
	return len(rb.entries)
}

// Capacity returns the maximum capacity of the buffer.
func (rb *RingBuffer) Capacity() int {
	return rb.capacity
}

// Clear removes all entries from the buffer.
func (rb *RingBuffer) Clear() {
	rb.mu.Lock()
	defer rb.mu.Unlock()
	rb.entries = make([]Entry, 0, rb.capacity)
}

// ToLLMMessages converts the buffered entries into openai.ChatCompletionMessage objects.
// User messages are formatted with "<SenderName>: <Content>" if SenderName is present.
func (rb *RingBuffer) ToLLMMessages() []openai.ChatCompletionMessage {
	rb.mu.RLock()
	defer rb.mu.RUnlock()

	msgs := make([]openai.ChatCompletionMessage, 0, len(rb.entries))
	for _, e := range rb.entries {
		role := e.Role
		switch role {
		case "", "user":
			role = openai.ChatMessageRoleUser
		case "assistant":
			role = openai.ChatMessageRoleAssistant
		case "system":
			role = openai.ChatMessageRoleSystem
		}

		var content string
		senderName := strings.TrimSpace(e.SenderName)
		if role == openai.ChatMessageRoleUser && senderName != "" && senderName != e.SenderID {
			content = fmt.Sprintf("%s: %s", senderName, e.Content)
		} else {
			content = e.Content
		}

		msgs = append(msgs, openai.ChatCompletionMessage{
			Role:    role,
			Content: content,
		})
	}
	return msgs
}

// Manager manages per-chat RingBuffers.
type Manager struct {
	mu              sync.RWMutex
	defaultCapacity int
	buffers         map[string]*RingBuffer
}

// NewManager creates a new chat context manager.
func NewManager(defaultCapacity int) *Manager {
	if defaultCapacity <= 0 {
		defaultCapacity = 100
	}
	return &Manager{
		defaultCapacity: defaultCapacity,
		buffers:         make(map[string]*RingBuffer),
	}
}

// GetOrCreate returns the RingBuffer for the given chatID, creating one if it doesn't exist.
func (m *Manager) GetOrCreate(chatID string) *RingBuffer {
	m.mu.RLock()
	rb, exists := m.buffers[chatID]
	m.mu.RUnlock()
	if exists {
		return rb
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	// Double check
	if rb, exists = m.buffers[chatID]; exists {
		return rb
	}
	rb = NewRingBuffer(m.defaultCapacity)
	m.buffers[chatID] = rb
	return rb
}

// Push adds an entry to the ring buffer for the specified chatID.
func (m *Manager) Push(chatID string, entry Entry) {
	rb := m.GetOrCreate(chatID)
	rb.Push(entry)
}

// GetLLMMessages returns the formatted LLM messages for the specified chatID.
func (m *Manager) GetLLMMessages(chatID string) []openai.ChatCompletionMessage {
	rb := m.GetOrCreate(chatID)
	return rb.ToLLMMessages()
}
