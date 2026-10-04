package prompt

import (
	"testing"

	"bob/internal/models"

	"github.com/stretchr/testify/assert"
)

// TestRenderDMPromptWithSandbox_EdgeCases verifies edge case handling in DM prompt rendering
// with sandbox state, including inactive states, boundary TTL values, special characters,
// and user/bot fallback resolutions.
func TestRenderDMPromptWithSandbox_EdgeCases(t *testing.T) {
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

	t.Run("sandbox instructions are static across any sandbox parameter to preserve KV cache", func(t *testing.T) {
		basePrompt := RenderDMPrompt(bot, "@bob_bot", user, 5)
		assert.Contains(t, basePrompt, "Bob AI")
		assert.Contains(t, basePrompt, "Alice Wonder")
		assert.Contains(t, basePrompt, "sandbox_request")
		assert.Contains(t, basePrompt, "sandbox_exec")
		assert.Contains(t, basePrompt, "sandbox_destroy")
		assert.Contains(t, basePrompt, "/workspace")
		assert.NotContains(t, basePrompt, "active sandbox")
		assert.NotContains(t, basePrompt, "unconditional destruction")

		ttlCases := []struct {
			name   string
			active bool
			ttl    string
		}{
			{"inactive with minutes", false, "30 minutes"},
			{"active standard minutes", true, "25 minutes"},
			{"empty TTL", true, ""},
			{"expired TTL", true, "expired"},
			{"zero seconds", true, "0s"},
			{"negative TTL", true, "-5m"},
			{"complex duration", true, "1h 45m 30s"},
		}

		for _, tc := range ttlCases {
			t.Run(tc.name, func(t *testing.T) {
				p := RenderDMPromptWithSandbox(bot, "@bob_bot", user, 5, tc.active, tc.ttl)
				assert.Equal(t, basePrompt, p, "prompt must be static regardless of sandbox state to preserve LLM KV cache")
			})
		}
	})

	t.Run("special characters and template injection attempts in names", func(t *testing.T) {
		maliciousUser := models.User{
			ID:          "user-evil",
			UserName:    "attacker",
			DisplayName: "{{.EvilVariable}} <script>alert('xss')</script> `code`",
		}
		maliciousBot := models.User{
			ID:          "bot-evil",
			UserName:    "bot_injected",
			DisplayName: "{{if .SandboxActive}}INJECTED{{end}} 🤖",
		}

		// Rendering must complete safely without executing template directives in data
		p := RenderDMPromptWithSandbox(maliciousBot, "bot_injected", maliciousUser, 5, true, "10m")
		// The template text should contain the raw string, not evaluate it
		assert.Contains(t, p, "{{.EvilVariable}}")
		assert.Contains(t, p, "<script>alert('xss')</script>")
		assert.Contains(t, p, "{{if .SandboxActive}}INJECTED{{end}}")
		// Handle normalization prepends @
		assert.Contains(t, p, "@bot_injected")
	})

	t.Run("unicode, emojis, and non-Latin scripts", func(t *testing.T) {
		intlBot := models.User{
			ID:          "bot-intl",
			DisplayName: "ボット AI 🤖",
			UserName:    "bot_cjk",
		}
		intlUser := models.User{
			ID:          "user-intl",
			DisplayName: "Алиса 👩‍💻 (مرحبا)",
			UserName:    "alisa",
		}

		p := RenderDMPromptWithSandbox(intlBot, "@bot_cjk", intlUser, 5, true, "15m")
		assert.Contains(t, p, "ボット AI 🤖")
		assert.Contains(t, p, "Алиса 👩‍💻 (مرحبا)")
		assert.Contains(t, p, "sandbox_request")
	})

	t.Run("newlines and tabs in names", func(t *testing.T) {
		spacedUser := models.User{
			ID:          "user-space",
			DisplayName: "Line1\nLine2\tTabbed",
		}
		p := RenderDMPromptWithSandbox(bot, "@bot", spacedUser, 5, true, "10m\nremaining")
		assert.Contains(t, p, "Line1\nLine2\tTabbed")
		assert.Contains(t, p, "sandbox_request")
	})

	t.Run("fallback resolution hierarchy", func(t *testing.T) {
		// 1. User with empty DisplayName and empty UserName falls back to "the user"
		u1 := models.User{ID: "uuid-1234"}
		p1 := RenderDMPromptWithSandbox(bot, "@bot", u1, 5, false, "")
		assert.Contains(t, p1, "the user")
		assert.NotContains(t, p1, "uuid-1234")

		// 2. User with DisplayName matching ID falls back to UserName
		u2 := models.User{ID: "uuid-5678", DisplayName: "uuid-5678", UserName: "bob_user"}
		p2 := RenderDMPromptWithSandbox(bot, "@bot", u2, 5, false, "")
		assert.Contains(t, p2, "bob_user")
		assert.NotContains(t, p2, "uuid-5678")

		// 3. Bot with empty DisplayName and empty UserName falls back to "AI Assistant"
		b1 := models.User{ID: "bot-uuid"}
		p3 := RenderDMPromptWithSandbox(b1, "", user, 5, false, "")
		assert.Contains(t, p3, "AI Assistant")
		assert.Contains(t, p3, "@bot")

		// 4. Bot handle without leading @ is normalized with @
		p4 := RenderDMPromptWithSandbox(bot, "custom_handle", user, 5, false, "")
		assert.Contains(t, p4, "@custom_handle")
	})

	t.Run("paragraph limit boundary clamping", func(t *testing.T) {
		// Zero paragraphs defaults to 10 for DM
		pZero := RenderDMPromptWithSandbox(bot, "@bot", user, 0, false, "")
		assert.Contains(t, pZero, "maximum 10 paragraphs")

		// Negative paragraphs defaults to 10
		pNeg := RenderDMPromptWithSandbox(bot, "@bot", user, -5, false, "")
		assert.Contains(t, pNeg, "maximum 10 paragraphs")

		// Positive boundaries
		pOne := RenderDMPromptWithSandbox(bot, "@bot", user, 1, false, "")
		assert.Contains(t, pOne, "maximum 1 paragraphs")

		pHundred := RenderDMPromptWithSandbox(bot, "@bot", user, 100, false, "")
		assert.Contains(t, pHundred, "maximum 100 paragraphs")
	})
}
