package prompt

import (
	"bytes"
	"fmt"
	"strings"
	"text/template"

	"bob/internal/models"
)

const (
	DefaultTownhallTemplate = "You are {{.BotDisplayName}} ({{.BotHandle}}), an AI assistant participating in the Besedka Townhall chat. Answer directly, accurately, and professionally. Keep your answer concise and brief (maximum {{.MaxParagraphs}} paragraphs) without unnecessary conversational filler."
	DefaultDMTemplate       = "You are {{.BotDisplayName}} ({{.BotHandle}}), an AI assistant in a direct message conversation with {{.UserDisplayName}} in the Besedka chat application. Answer clearly, accurately, and helpfully using markdown formatting. Keep your answer brief (maximum {{.MaxParagraphs}} paragraphs)."
)

var (
	townhallTmpl = template.Must(template.New("townhall").Parse(DefaultTownhallTemplate))
	dmTmpl       = template.Must(template.New("dm").Parse(DefaultDMTemplate))
)

// TownhallPromptData contains parameters for rendering the Townhall system prompt.
type TownhallPromptData struct {
	BotDisplayName string
	BotHandle      string
	MaxParagraphs  int
}

// DMPromptData contains parameters for rendering the DM system prompt.
type DMPromptData struct {
	BotDisplayName  string
	BotHandle       string
	UserDisplayName string
	MaxParagraphs   int
}

// RenderTownhallPrompt builds the system prompt for Townhall conversations.
func RenderTownhallPrompt(bot models.User, botHandle string, maxParagraphs int) string {
	botDisplayName := bot.GetDisplayName()
	if botDisplayName == "" || botDisplayName == bot.ID || botDisplayName == "User" {
		if bot.GetUserName() != "" {
			botDisplayName = bot.GetUserName()
		} else {
			botDisplayName = "AI Assistant"
		}
	}

	handle := botHandle
	if handle == "" {
		if bot.GetUserName() != "" {
			handle = "@" + bot.GetUserName()
		} else {
			handle = "@bot"
		}
	}
	if !strings.HasPrefix(handle, "@") {
		handle = "@" + handle
	}

	if maxParagraphs <= 0 {
		maxParagraphs = 2
	}

	data := TownhallPromptData{
		BotDisplayName: botDisplayName,
		BotHandle:      handle,
		MaxParagraphs:  maxParagraphs,
	}

	var buf bytes.Buffer
	if err := townhallTmpl.Execute(&buf, data); err != nil {
		return fmt.Sprintf("You are %s (%s) for Besedka Townhall. Keep responses under %d paragraphs.", data.BotDisplayName, data.BotHandle, data.MaxParagraphs)
	}
	return buf.String()
}

// RenderDMPrompt builds the system prompt for Direct Message conversations with a specific user.
func RenderDMPrompt(bot models.User, botHandle string, targetUser models.User, maxParagraphs int) string {
	botDisplayName := bot.GetDisplayName()
	if botDisplayName == "" || botDisplayName == bot.ID || botDisplayName == "User" {
		if bot.GetUserName() != "" {
			botDisplayName = bot.GetUserName()
		} else {
			botDisplayName = "AI Assistant"
		}
	}

	handle := botHandle
	if handle == "" {
		if bot.GetUserName() != "" {
			handle = "@" + bot.GetUserName()
		} else {
			handle = "@bot"
		}
	}
	if !strings.HasPrefix(handle, "@") {
		handle = "@" + handle
	}

	userDisplayName := targetUser.GetDisplayName()
	if userDisplayName == "" || userDisplayName == targetUser.ID {
		if targetUser.GetUserName() != "" {
			userDisplayName = targetUser.GetUserName()
		} else {
			userDisplayName = "User"
		}
	}

	if maxParagraphs <= 0 {
		maxParagraphs = 10
	}

	data := DMPromptData{
		BotDisplayName:  botDisplayName,
		BotHandle:       handle,
		UserDisplayName: userDisplayName,
		MaxParagraphs:   maxParagraphs,
	}

	var buf bytes.Buffer
	if err := dmTmpl.Execute(&buf, data); err != nil {
		return fmt.Sprintf("You are %s (%s) in a DM with %s. Keep responses under %d paragraphs.", data.BotDisplayName, data.BotHandle, data.UserDisplayName, data.MaxParagraphs)
	}
	return buf.String()
}
