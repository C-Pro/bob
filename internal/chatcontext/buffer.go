package chatcontext

import (
	"fmt"
	"strings"
	"sync"
	"time"

	openai "github.com/sashabaranov/go-openai"
)

// UserTimestampGapSeconds is the minimum elapsed seconds since the previous message
// before an RFC3339 UTC timestamp is automatically prepended to a user message.
const UserTimestampGapSeconds int64 = 3600

// ImageAttachment represents an image attached to a chat entry as a data URL.
type ImageAttachment struct {
	URL string `json:"url"` // data URL e.g. "data:image/jpeg;base64,..."
}

// Entry represents a single turn in a chat conversation.
type Entry struct {
	Seq        int64             `json:"seq,omitempty"`    // message sequence number
	Role       string            `json:"role"`             // "user" or "assistant"
	SenderID   string            `json:"senderId"`         // user/bot ID
	SenderName string            `json:"senderName"`       // display name or username
	Content    string            `json:"content"`          // message text content
	Images     []ImageAttachment `json:"images,omitempty"` // image attachments
	Timestamp  int64             `json:"timestamp"`        // unix timestamp
}

// EvictionHook is a callback invoked with evicted entries when capacity is exceeded.
type EvictionHook func(evicted []Entry)

type pendingEntry struct {
	token uint64
	entry Entry
}

// RingBuffer is a thread-safe circular buffer holding up to `capacity` entries.
type RingBuffer struct {
	mu               sync.RWMutex
	capacity         int
	entries          []Entry
	pending          []pendingEntry
	nextPendingToken uint64
	onEvict          EvictionHook
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

// SetOnEvict sets a callback function to be called when entries are evicted.
func (rb *RingBuffer) SetOnEvict(hook EvictionHook) {
	rb.mu.Lock()
	defer rb.mu.Unlock()
	rb.onEvict = hook
}

func (rb *RingBuffer) pushLocked(entry Entry) ([]Entry, EvictionHook) {
	rb.entries = append(rb.entries, entry)
	if len(rb.entries) <= rb.capacity {
		return nil, nil
	}
	evictCount := rb.capacity / 3
	if evictCount < 1 {
		evictCount = 1
	}
	evicted := make([]Entry, evictCount)
	copy(evicted, rb.entries[:evictCount])
	rb.entries = rb.entries[evictCount:]
	return evicted, rb.onEvict
}

func normalizeSpaces(s string) string {
	return strings.Join(strings.Fields(s), " ")
}

// Push adds an entry to the ring buffer. When capacity is exceeded, it evicts
// a chunk of the oldest messages (1/3 of capacity) at once to keep the conversation
// prefix stable and maximize LLM prefix caching.
func (rb *RingBuffer) Push(entry Entry) {
	rb.mu.Lock()
	evicted, hook := rb.pushLocked(entry)
	rb.mu.Unlock()

	if hook != nil && len(evicted) > 0 {
		hook(evicted)
	}
}

// RegisterPending records an outgoing assistant entry in the pre-send pending list
// without modifying or evicting retained conversation entries, returning a unique token.
func (rb *RingBuffer) RegisterPending(entry Entry) uint64 {
	rb.mu.Lock()
	defer rb.mu.Unlock()
	rb.nextPendingToken++
	token := rb.nextPendingToken
	rb.pending = append(rb.pending, pendingEntry{token: token, entry: entry})
	return token
}

// CommitPending promotes the pending entry identified by token into the ring buffer
// after transport succeeds. If a concurrent echo already promoted it, CommitPending is a no-op.
func (rb *RingBuffer) CommitPending(token uint64) {
	rb.mu.Lock()
	idx := -1
	var entry Entry
	for i := len(rb.pending) - 1; i >= 0; i-- {
		if rb.pending[i].token == token {
			idx = i
			entry = rb.pending[i].entry
			break
		}
	}
	if idx < 0 {
		rb.mu.Unlock()
		return
	}
	rb.pending = append(rb.pending[:idx], rb.pending[idx+1:]...)
	evicted, hook := rb.pushLocked(entry)
	rb.mu.Unlock()

	if hook != nil && len(evicted) > 0 {
		hook(evicted)
	}
}

// RemovePending removes the pre-send pending entry identified by token.
// Used to roll back pre-send entries when transport fails without evicting retained history.
func (rb *RingBuffer) RemovePending(token uint64) bool {
	rb.mu.Lock()
	defer rb.mu.Unlock()
	for i := len(rb.pending) - 1; i >= 0; i-- {
		if rb.pending[i].token == token {
			rb.pending = append(rb.pending[:i], rb.pending[i+1:]...)
			return true
		}
	}
	return false
}

// ReconcileAssistantEcho atomically reconciles an incoming assistant self-echo against existing
// ring buffer entries and pre-send pending entries:
//  1. If entry.Seq > 0 matches an existing assistant entry, it is ignored as an exact duplicate.
//  2. Otherwise, if a committed pending assistant entry (Seq == 0) in entries has matching
//     normalized content, its Seq is updated to entry.Seq (when > 0) without appending a duplicate.
//  3. Otherwise, if a pre-send pending entry matches, it is promoted into the ring buffer with
//     entry.Seq (when > 0) without appending a duplicate.
//  4. Otherwise, entry is appended to the ring buffer (evicting oldest entries if over capacity).
//
// Returns true if a new entry was appended, false if it matched or reconciled an existing entry.
func (rb *RingBuffer) ReconcileAssistantEcho(entry Entry) bool {
	rb.mu.Lock()

	if entry.Seq > 0 {
		for i := len(rb.entries) - 1; i >= 0; i-- {
			e := rb.entries[i]
			if e.Role == "assistant" && e.Seq == entry.Seq {
				rb.mu.Unlock()
				return false
			}
		}
	}

	normIncoming := normalizeSpaces(entry.Content)
	for i := 0; i < len(rb.entries); i++ {
		e := &rb.entries[i]
		if e.Role != "assistant" || e.Seq != 0 {
			continue
		}
		if e.SenderID != "" && entry.SenderID != "" && e.SenderID != entry.SenderID {
			continue
		}
		if normalizeSpaces(e.Content) == normIncoming {
			if entry.Seq > 0 {
				e.Seq = entry.Seq
				if e.Timestamp == 0 && entry.Timestamp > 0 {
					e.Timestamp = entry.Timestamp
				}
			}
			rb.mu.Unlock()
			return false
		}
	}

	for i := 0; i < len(rb.pending); i++ {
		p := rb.pending[i].entry
		if p.Role != "assistant" {
			continue
		}
		if p.SenderID != "" && entry.SenderID != "" && p.SenderID != entry.SenderID {
			continue
		}
		if normalizeSpaces(p.Content) == normIncoming {
			rb.pending = append(rb.pending[:i], rb.pending[i+1:]...)
			if entry.Seq > 0 {
				p.Seq = entry.Seq
			}
			if p.Timestamp == 0 && entry.Timestamp > 0 {
				p.Timestamp = entry.Timestamp
			}
			evicted, hook := rb.pushLocked(p)
			rb.mu.Unlock()
			if hook != nil && len(evicted) > 0 {
				hook(evicted)
			}
			return false
		}
	}

	evicted, hook := rb.pushLocked(entry)
	rb.mu.Unlock()

	if hook != nil && len(evicted) > 0 {
		hook(evicted)
	}
	return true
}

// Entries returns a copy of all entries in chronological order (oldest first).
func (rb *RingBuffer) Entries() []Entry {
	rb.mu.RLock()
	defer rb.mu.RUnlock()

	res := make([]Entry, len(rb.entries))
	copy(res, rb.entries)
	return res
}

// UpdateLast visits entries newest first, stopping when fn updates an entry and returns true.
// The callback runs under the buffer lock and must not call other buffer methods.
func (rb *RingBuffer) UpdateLast(fn func(e *Entry) bool) bool {
	rb.mu.Lock()
	defer rb.mu.Unlock()
	for i := len(rb.entries) - 1; i >= 0; i-- {
		if fn(&rb.entries[i]) {
			return true
		}
	}
	return false
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

// entryToLLMMessage converts a single chatcontext Entry into an openai.ChatCompletionMessage.
func entryToLLMMessage(e Entry) openai.ChatCompletionMessage {
	return entryToLLMMessageWithPrev(e, nil)
}

func shouldInjectUserTimestamp(e Entry, prev *Entry) bool {
	if prev == nil || prev.Timestamp <= 0 || e.Timestamp <= 0 {
		return false
	}
	return e.Timestamp-prev.Timestamp > UserTimestampGapSeconds
}

func formatUserTimestampPrefix(ts int64) string {
	return fmt.Sprintf("[%s]", time.Unix(ts, 0).UTC().Format(time.RFC3339))
}

// entryToLLMMessageWithPrev converts a single chatcontext Entry into an openai.ChatCompletionMessage,
// prepending an RFC3339 UTC timestamp when a user message arrives more than 1 hour after prev.
func entryToLLMMessageWithPrev(e Entry, prev *Entry) openai.ChatCompletionMessage {
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

	if role == openai.ChatMessageRoleUser && shouldInjectUserTimestamp(e, prev) {
		tsPrefix := formatUserTimestampPrefix(e.Timestamp)
		if content != "" {
			content = tsPrefix + " " + content
		} else {
			content = tsPrefix
		}
	}

	if len(e.Images) > 0 && role == openai.ChatMessageRoleUser {
		parts := make([]openai.ChatMessagePart, 0, 1+len(e.Images))
		if content != "" {
			parts = append(parts, openai.ChatMessagePart{
				Type: openai.ChatMessagePartTypeText,
				Text: content,
			})
		}
		for _, img := range e.Images {
			parts = append(parts, openai.ChatMessagePart{
				Type: openai.ChatMessagePartTypeImageURL,
				ImageURL: &openai.ChatMessageImageURL{
					URL:    img.URL,
					Detail: openai.ImageURLDetailAuto,
				},
			})
		}
		return openai.ChatCompletionMessage{
			Role:         role,
			MultiContent: parts,
		}
	}

	return openai.ChatCompletionMessage{
		Role:    role,
		Content: content,
	}
}

func entriesToLLMMessages(entries []Entry) []openai.ChatCompletionMessage {
	return entriesToLLMMessagesWithFirstPrev(entries, nil)
}

func entriesToLLMMessagesWithFirstPrev(entries []Entry, firstPrev *Entry) []openai.ChatCompletionMessage {
	msgs := make([]openai.ChatCompletionMessage, 0, len(entries))
	for i, e := range entries {
		prev := firstPrev
		if i > 0 {
			prev = &entries[i-1]
		}
		msgs = append(msgs, entryToLLMMessageWithPrev(e, prev))
	}
	return msgs
}

// ToLLMMessages converts the buffered entries into openai.ChatCompletionMessage objects.
// User messages are formatted with "<SenderName>: <Content>" if SenderName is present,
// and include an RFC3339 UTC timestamp prefix when the previous message was > 1 hour earlier.
func (rb *RingBuffer) ToLLMMessages() []openai.ChatCompletionMessage {
	rb.mu.RLock()
	defer rb.mu.RUnlock()

	return entriesToLLMMessages(rb.entries)
}

// Manager manages per-chat RingBuffers.
type Manager struct {
	mu              sync.RWMutex
	defaultCapacity int
	buffers         map[string]*RingBuffer
	onEvict         func(chatID string, evicted []Entry)
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

// SetOnEvict sets a global eviction hook called whenever entries are evicted from any chat buffer.
func (m *Manager) SetOnEvict(hook func(chatID string, evicted []Entry)) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.onEvict = hook
	for chatID, rb := range m.buffers {
		cid := chatID
		rb.SetOnEvict(func(evicted []Entry) {
			if hook != nil {
				hook(cid, evicted)
			}
		})
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
	if m.onEvict != nil {
		cid := chatID
		hook := m.onEvict
		rb.SetOnEvict(func(evicted []Entry) {
			hook(cid, evicted)
		})
	}
	m.buffers[chatID] = rb
	return rb
}

// Push adds an entry to the ring buffer for the specified chatID.
func (m *Manager) Push(chatID string, entry Entry) {
	rb := m.GetOrCreate(chatID)
	rb.Push(entry)
}

// RegisterPending records a pre-send pending entry for chatID without evicting retained entries.
func (m *Manager) RegisterPending(chatID string, entry Entry) uint64 {
	return m.GetOrCreate(chatID).RegisterPending(entry)
}

// CommitPending promotes a pre-send pending entry for chatID into the ring buffer.
func (m *Manager) CommitPending(chatID string, token uint64) {
	m.GetOrCreate(chatID).CommitPending(token)
}

// RemovePending removes a pre-send pending entry for chatID by token.
func (m *Manager) RemovePending(chatID string, token uint64) bool {
	return m.GetOrCreate(chatID).RemovePending(token)
}

// GetLLMMessages returns the formatted LLM messages for the specified chatID.
func (m *Manager) GetLLMMessages(chatID string) []openai.ChatCompletionMessage {
	rb := m.GetOrCreate(chatID)
	return rb.ToLLMMessages()
}
