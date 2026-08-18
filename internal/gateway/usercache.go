package gateway

import (
	"sync"

	"bob/internal/models"
)

// UserCache provides a thread-safe cache for resolving user IDs to display names and metadata.
type UserCache struct {
	mu    sync.RWMutex
	users map[string]models.User
}

// NewUserCache creates a new UserCache instance.
func NewUserCache() *UserCache {
	return &UserCache{
		users: make(map[string]models.User),
	}
}

// Set stores a user in the cache.
func (c *UserCache) Set(user models.User) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.users[user.ID] = user
}

// SetAll stores multiple users in the cache.
func (c *UserCache) SetAll(users []models.User) {
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, u := range users {
		c.users[u.ID] = u
	}
}

// Get retrieves a user from the cache.
func (c *UserCache) Get(userID string) (models.User, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	u, ok := c.users[userID]
	return u, ok
}

// GetDisplayName returns the display name for a given user ID, or falls back to "User" (never raw UUID).
func (c *UserCache) GetDisplayName(userID string) string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	if u, ok := c.users[userID]; ok {
		return u.GetDisplayName()
	}
	return "User"
}

// GetUserName returns the username handle for a given user ID, or empty string if unknown.
func (c *UserCache) GetUserName(userID string) string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	if u, ok := c.users[userID]; ok {
		return u.GetUserName()
	}
	return ""
}

// All returns a slice of all cached users.
func (c *UserCache) All() []models.User {
	c.mu.RLock()
	defer c.mu.RUnlock()
	res := make([]models.User, 0, len(c.users))
	for _, u := range c.users {
		res = append(res, u)
	}
	return res
}
