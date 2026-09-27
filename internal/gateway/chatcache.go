package gateway

import (
	"sync"

	"bob/internal/models"
)

// ChatCache provides a thread-safe cache for chat metadata.
type ChatCache struct {
	mu    sync.RWMutex
	chats map[string]models.Chat
}

// NewChatCache creates a new ChatCache instance.
func NewChatCache() *ChatCache {
	return &ChatCache{
		chats: make(map[string]models.Chat),
	}
}

// Set stores a chat in the cache.
func (c *ChatCache) Set(chat models.Chat) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.chats[chat.ID] = chat
}

// SetAll stores multiple chats in the cache.
func (c *ChatCache) SetAll(chats []models.Chat) {
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, ch := range chats {
		c.chats[ch.ID] = ch
	}
}

// Get retrieves a chat from the cache.
func (c *ChatCache) Get(chatID string) (models.Chat, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	ch, ok := c.chats[chatID]
	return ch, ok
}
