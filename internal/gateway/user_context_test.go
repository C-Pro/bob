package gateway

import (
	"sync"
	"testing"

	"bob/internal/chatcontext"
	"bob/internal/models"

	"github.com/c-pro/geche"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestFormatUserContext(t *testing.T) {
	tests := []struct {
		name     string
		timeZone string
		lang     string
		expected string
	}{
		{
			name:     "both timezone and language",
			timeZone: "America/New_York",
			lang:     "en-US",
			expected: "[User context: timezone=America/New_York, language=en-US]",
		},
		{
			name:     "timezone only",
			timeZone: "Europe/London",
			lang:     "",
			expected: "[User context: timezone=Europe/London]",
		},
		{
			name:     "language only",
			timeZone: "",
			lang:     "fr-FR",
			expected: "[User context: language=fr-FR]",
		},
		{
			name:     "both with leading and trailing whitespace",
			timeZone: "  Asia/Tokyo  ",
			lang:     "  ja-JP  ",
			expected: "[User context: timezone=Asia/Tokyo, language=ja-JP]",
		},
		{
			name:     "neither present",
			timeZone: "",
			lang:     "",
			expected: "",
		},
		{
			name:     "whitespace only",
			timeZone: "   ",
			lang:     "   ",
			expected: "",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			actual := FormatUserContext(tc.timeZone, tc.lang)
			assert.Equal(t, tc.expected, actual)
		})
	}
}

func TestFormatUserContextFor(t *testing.T) {
	tests := []struct {
		name     string
		userTag  string
		timeZone string
		lang     string
		expected string
	}{
		{
			name:     "with handle prefix @",
			userTag:  "@alice",
			timeZone: "America/New_York",
			lang:     "en-US",
			expected: "[User context for @alice: timezone=America/New_York, language=en-US]",
		},
		{
			name:     "with raw username without @",
			userTag:  "bob",
			timeZone: "Europe/London",
			lang:     "",
			expected: "[User context for @bob: timezone=Europe/London]",
		},
		{
			name:     "empty userTag delegates to standard format",
			userTag:  "",
			timeZone: "Europe/London",
			lang:     "en",
			expected: "[User context: timezone=Europe/London, language=en]",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			actual := FormatUserContextFor(tc.userTag, tc.timeZone, tc.lang)
			assert.Equal(t, tc.expected, actual)
		})
	}
}

func TestInjectUserContextOnce(t *testing.T) {
	gw := &Gateway{
		contextManager:      chatcontext.NewManager(10),
		userCache:           NewUserCache(),
		userContextInjected: geche.NewMapCache[string, bool](),
	}

	gw.userCache.Set(models.User{
		ID:                "u1",
		DisplayName:       "Alice",
		TimeZone:          "America/New_York",
		PreferredLanguage: "en-US",
	})
	gw.userCache.Set(models.User{
		ID:          "u2",
		DisplayName: "Bob",
		// No timezone or language
	})

	ctx := t.Context()
	rb1 := gw.contextManager.GetOrCreate("chat_1")

	// First message from u1 in chat_1 (DM): should inject
	gw.injectUserContextOnce(ctx, "chat_1", "u1", rb1, true)
	entries := rb1.Entries()
	require.Len(t, entries, 1)
	assert.Equal(t, "system", entries[0].Role)
	assert.Equal(t, "[User context: timezone=America/New_York, language=en-US]", entries[0].Content)

	// Second call for same chat_1: should NOT inject again
	gw.injectUserContextOnce(ctx, "chat_1", "u1", rb1, true)
	assert.Len(t, rb1.Entries(), 1)

	// Message from u2 (no timezone or language): should not inject
	rb2 := gw.contextManager.GetOrCreate("chat_2")
	gw.injectUserContextOnce(ctx, "chat_2", "u2", rb2, true)
	assert.Empty(t, rb2.Entries())

	// Reset/Clear chat_1 session
	gw.ResetUserContextSession("chat_1")
	rb1.Clear()
	assert.Empty(t, rb1.Entries())

	// After reset, should inject once more
	gw.injectUserContextOnce(ctx, "chat_1", "u1", rb1, true)
	require.Len(t, rb1.Entries(), 1)
	assert.Equal(t, "system", rb1.Entries()[0].Role)
}

func TestInjectUserContextOnce_TownhallMultipleUsers(t *testing.T) {
	gw := &Gateway{
		contextManager:      chatcontext.NewManager(10),
		userCache:           NewUserCache(),
		userContextInjected: geche.NewMapCache[string, bool](),
	}

	gw.userCache.Set(models.User{
		ID:                "u1",
		UserName:          "alice",
		DisplayName:       "Alice W",
		TimeZone:          "America/New_York",
		PreferredLanguage: "en-US",
	})
	gw.userCache.Set(models.User{
		ID:                "u2",
		UserName:          "charlie",
		DisplayName:       "Charlie B",
		TimeZone:          "Europe/Paris",
		PreferredLanguage: "fr-FR",
	})

	ctx := t.Context()
	rbTownhall := gw.contextManager.GetOrCreate("townhall")

	// Alice messages the bot in Townhall
	gw.injectUserContextOnce(ctx, "townhall", "u1", rbTownhall, false)
	entries := rbTownhall.Entries()
	require.Len(t, entries, 1)
	assert.Equal(t, "system", entries[0].Role)
	assert.Equal(t, "[User context for @alice: timezone=America/New_York, language=en-US]", entries[0].Content)

	// Alice messages again: should NOT inject duplicate
	gw.injectUserContextOnce(ctx, "townhall", "u1", rbTownhall, false)
	assert.Len(t, rbTownhall.Entries(), 1)

	// Charlie messages the bot in Townhall: MUST inject Charlie's context
	gw.injectUserContextOnce(ctx, "townhall", "u2", rbTownhall, false)
	entries = rbTownhall.Entries()
	require.Len(t, entries, 2)
	assert.Equal(t, "system", entries[1].Role)
	assert.Equal(t, "[User context for @charlie: timezone=Europe/Paris, language=fr-FR]", entries[1].Content)

	// ResetTownhall session clears all user session markers in townhall
	gw.ResetUserContextSession("townhall")
	rbTownhall.Clear()

	// After reset, Alice can inject again
	gw.injectUserContextOnce(ctx, "townhall", "u1", rbTownhall, false)
	require.Len(t, rbTownhall.Entries(), 1)
	assert.Equal(t, "[User context for @alice: timezone=America/New_York, language=en-US]", rbTownhall.Entries()[0].Content)
}

func TestInjectUserContextOnce_Concurrent(t *testing.T) {
	gw := &Gateway{
		contextManager:      chatcontext.NewManager(50),
		userCache:           NewUserCache(),
		userContextInjected: geche.NewMapCache[string, bool](),
	}

	gw.userCache.Set(models.User{
		ID:                "u1",
		DisplayName:       "Alice",
		TimeZone:          "UTC",
		PreferredLanguage: "en",
	})

	ctx := t.Context()
	rb := gw.contextManager.GetOrCreate("chat_concurrent")

	var wg sync.WaitGroup
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			gw.injectUserContextOnce(ctx, "chat_concurrent", "u1", rb, true)
		}()
	}
	wg.Wait()

	// Must be injected exactly once
	assert.Equal(t, 1, rb.Len())
	assert.Equal(t, "system", rb.Entries()[0].Role)
	assert.Equal(t, "[User context: timezone=UTC, language=en]", rb.Entries()[0].Content)
}
