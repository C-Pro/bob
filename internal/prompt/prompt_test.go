package prompt

import (
	"testing"

	"bob/internal/models"

	"github.com/stretchr/testify/assert"
)

func TestRenderTownhallPrompt(t *testing.T) {
	bot := models.User{
		ID:          "bot-1",
		UserName:    "bob_bot",
		DisplayName: "Bob AI",
	}

	prompt := RenderTownhallPrompt(bot, "@bob_bot", 3)
	assert.Contains(t, prompt, "Bob AI")
	assert.Contains(t, prompt, "@bob_bot")
	assert.Contains(t, prompt, "maximum 3 paragraphs")
	assert.Contains(t, prompt, "Townhall")

	// Fallback handling: fallback bot should use AI Assistant (@bot), never expose raw UUID
	botEmpty := models.User{ID: "bot-2"}
	prompt2 := RenderTownhallPrompt(botEmpty, "", 0)
	assert.Contains(t, prompt2, "AI Assistant")
	assert.Contains(t, prompt2, "@bot")
	assert.NotContains(t, prompt2, "bot-2")
	assert.Contains(t, prompt2, "maximum 2 paragraphs")
}

func TestRenderDMPrompt(t *testing.T) {
	bot := models.User{
		ID:          "bot-1",
		UserName:    "bob_bot",
		DisplayName: "Bob AI",
	}
	user := models.User{
		ID:          "user-1",
		UserName:    "alice",
		DisplayName: "Alice Wonder",
	}

	prompt := RenderDMPrompt(bot, "@bob_bot", user, 5)
	assert.Contains(t, prompt, "Bob AI")
	assert.Contains(t, prompt, "@bob_bot")
	assert.Contains(t, prompt, "Alice Wonder")
	assert.Contains(t, prompt, "maximum 5 paragraphs")
	assert.Contains(t, prompt, "direct message conversation")

	// Fallback user display name should fall back to username or "the user", never expose raw UUID user-99
	userWithUsername := models.User{ID: "user-99", UserName: "alice99"}
	prompt2 := RenderDMPrompt(bot, "@bob_bot", userWithUsername, 0)
	assert.Contains(t, prompt2, "alice99")
	assert.NotContains(t, prompt2, "user-99")
	assert.Contains(t, prompt2, "maximum 10 paragraphs")

	userEmpty := models.User{ID: "user-99"}
	prompt3 := RenderDMPrompt(bot, "@bob_bot", userEmpty, 0)
	assert.Contains(t, prompt3, "the user")
	assert.NotContains(t, prompt3, "user-99")
}
