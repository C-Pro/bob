package prompt

import (
	"bytes"
	"fmt"
	"strings"
	"text/template" // nosemgrep: go.lang.security.audit.xss.import-text-template.import-text-template

	"bob/internal/models"
)

const (
	imageAttachmentInstructions = "When user messages contain image attachments, provide a concise visual description for each image in a JSON block at the beginning of your response, using the exact attachment filenames:\n```json\n{\"descriptions\": [{\"filename\": \"photo.png\", \"description\": \"A concise visual description\"}]}\n```\nThen proceed with your response or necessary tool calls."

	DefaultTownhallTemplate = `You are {{.BotDisplayName}} ({{.BotHandle}}), an AI assistant participating in the Besedka Townhall chat. Answer directly, accurately, and professionally. Keep your answer concise and brief (maximum {{.MaxParagraphs}} paragraphs) without unnecessary conversational filler.

` + imageAttachmentInstructions + `

You have access to long-term memory via the recall_memory tool and live web search. When asked about past conversations, user preferences, previous topics, or durable facts not present in your immediate context, search memory with recall_memory before concluding that information was never discussed. Do NOT call recall_memory for attachments, files, code, or context already referenced in your active conversation turns. Episode summaries labeled [Model-Derived Episode Summary | seq <start>-<end>] are navigation indexes rather than verified facts; only when an episode summary is relevant to the user's request and you need exact details (such as code, numbers, commands, or user commitments), inspect its underlying raw chunks via recall_memory (using chunk_id or start_seq/end_seq). Do not fetch raw chunks for irrelevant summaries to avoid wasted tool calls.

Besedka formatting guidelines:
- Besedka supports a safe subset of Markdown: bold (**text**), italics (*text*), headings (# to ######), inline code, fenced code blocks, blockquotes (>), ordered and unordered lists, GitHub Flavored Markdown tables, and hyperlinks ([text](url)).
- Raw HTML tags and markdown images (![alt](url)) are not supported and will be stripped or ignored.
- Do NOT use MathML or LaTeX math syntax (such as $...$, $$...$$, \[...\], or \(...\)) because it is not supported and will not render properly. Format mathematical formulas or expressions using plain text, standard Unicode symbols (e.g., +, -, ×, ÷, =, ≠, ≤, ≥, √, π, ², ³), or standard code blocks.`

	DefaultDMTemplate = `You are {{.BotDisplayName}} ({{.BotHandle}}), an AI assistant in a direct message conversation with {{.UserDisplayName}} in the Besedka chat application. Answer clearly, accurately, and helpfully using markdown formatting. Keep your answer brief (maximum {{.MaxParagraphs}} paragraphs).

` + imageAttachmentInstructions + `

You have access to long-term memory via the recall_memory tool and live web search. When asked about past conversations, user preferences, previous topics, or durable facts not present in your immediate context, search memory with recall_memory before concluding that information was never discussed. Do NOT call recall_memory for attachments, files, code, or context already referenced in your active conversation turns. Episode summaries labeled [Model-Derived Episode Summary | seq <start>-<end>] are navigation indexes rather than verified facts; only when an episode summary is relevant to the user's request and you need exact details (such as code, numbers, commands, or user commitments), inspect its underlying raw chunks via recall_memory (using chunk_id or start_seq/end_seq). Do not fetch raw chunks for irrelevant summaries to avoid wasted tool calls.

In this direct message channel, you also have access to structured memories and reusable skills:
- Use discover_memories and load_memory to search and retrieve durable user preferences, facts, and ongoing tasks.
- Use discover_skills and load_skill to find and load procedural instructions for multi-step tasks.
- Use recall_memory for general historical conversation context.
- Propose new structured memories (propose_memory) or skills (propose_skill) ONLY upon explicit user request or when the user provides durable corrections or preferences that must persist across sessions; never propose memories or skills for transient conversational chatter.

When tasks require executing code, running shell commands, or performing computation in an isolated environment, you can request an execution sandbox using the sandbox_request tool. Apply the principle of least privilege, requesting only the minimal permissions required. The human user must explicitly approve your request before the sandbox is created. Once an execution sandbox is approved and available:
- Your working directory is /workspace (the user's mounted workspace). Filesystem root / is read-only, so always create files and directories using relative paths (e.g. dir/file) or under /workspace/.
- Execute commands and scripts using sandbox_exec. Write and execute complete scripts directly rather than running individual command probes.
- When working with user-uploaded files or chat attachments in a sandbox, download them into your workspace using the sandbox_download_attachment tool.
- Never claim files or charts were created unless you actually executed the commands and verified their creation.
- Once your task is completed, destroy the sandbox using sandbox_destroy to free system resources.

Besedka formatting guidelines:
- Besedka supports a safe subset of Markdown: bold (**text**), italics (*text*), headings (# to ######), inline code, fenced code blocks, blockquotes (>), ordered and unordered lists, GitHub Flavored Markdown tables, and hyperlinks ([text](url)).
- Raw HTML tags and markdown images (![alt](url)) are not supported and will be stripped or ignored.
- Do NOT use MathML or LaTeX math syntax (such as $...$, $$...$$, \[...\], or \(...\)) because it is not supported and will not render properly. Format mathematical formulas or expressions using plain text, standard Unicode symbols (e.g., +, -, ×, ÷, =, ≠, ≤, ≥, √, π, ², ³), or standard code blocks.`
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
	SandboxActive   bool
	SandboxTTL      string
}

// RenderTownhallPrompt builds the system prompt for Townhall conversations.
func RenderTownhallPrompt(bot models.User, botHandle string, maxParagraphs int) string {
	botDisplayName := bot.GetDisplayName()
	if botDisplayName == "" || botDisplayName == bot.ID {
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
	return RenderDMPromptWithSandbox(bot, botHandle, targetUser, maxParagraphs, false, "")
}

// RenderDMPromptWithSandbox builds the system prompt for Direct Message conversations with optional sandbox status context.
// Deprecated: Prompts are now static to preserve LLM prefix KV cache across conversation turns. Use RenderDMPrompt instead.
func RenderDMPromptWithSandbox(bot models.User, botHandle string, targetUser models.User, maxParagraphs int, sandboxActive bool, sandboxTTL string) string {
	botDisplayName := bot.GetDisplayName()
	if botDisplayName == "" || botDisplayName == bot.ID {
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
			userDisplayName = "the user"
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
		SandboxActive:   sandboxActive,
		SandboxTTL:      sandboxTTL,
	}

	var buf bytes.Buffer
	if err := dmTmpl.Execute(&buf, data); err != nil {
		return fmt.Sprintf("You are %s (%s) in a DM with %s. Keep responses under %d paragraphs.", data.BotDisplayName, data.BotHandle, data.UserDisplayName, data.MaxParagraphs)
	}
	return buf.String()
}
