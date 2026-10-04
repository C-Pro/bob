package gateway

import (
	"testing"

	"bob/internal/models"

	"github.com/stretchr/testify/assert"
)

func TestUserCache_TimeZoneAndPreferredLanguage(t *testing.T) {
	cache := NewUserCache()
	u := models.User{
		ID:                "u1",
		DisplayName:       "Alice",
		TimeZone:          "Europe/London",
		PreferredLanguage: "en-GB",
	}

	cache.Set(u)
	retrieved, ok := cache.Get("u1")
	assert.True(t, ok)
	assert.Equal(t, "Europe/London", retrieved.TimeZone)
	assert.Equal(t, "en-GB", retrieved.PreferredLanguage)

	// Test SetAll
	u2 := models.User{
		ID:                "u2",
		DisplayName:       "Bob",
		TimeZone:          "Asia/Tokyo",
		PreferredLanguage: "ja-JP",
	}
	cache.SetAll([]models.User{u, u2})

	retrieved2, ok2 := cache.Get("u2")
	assert.True(t, ok2)
	assert.Equal(t, "Asia/Tokyo", retrieved2.TimeZone)
	assert.Equal(t, "ja-JP", retrieved2.PreferredLanguage)
}
