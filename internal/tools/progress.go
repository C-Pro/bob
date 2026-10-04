package tools

import (
	"encoding/json"
	"fmt"
	"path/filepath"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"

	openai "github.com/sashabaranov/go-openai"
)

const (
	// MaxStepTitleBytes limits the byte length of step titles in progress cards.
	MaxStepTitleBytes = 80
	// MaxStepDescBytes limits the byte length of step descriptions in progress cards.
	MaxStepDescBytes = 512
)

// FormatProgressStep extracts a meaningful title and short description for a tool execution step.
func FormatProgressStep(toolCall openai.ToolCall) (title, desc string) {
	name := strings.TrimSpace(toolCall.Function.Name)
	rawArgs := toolCall.Function.Arguments

	switch name {
	case "web_search", "tavily_search":
		title = "Web Search"
		if q := parseStringArg(rawArgs, "query"); q != "" {
			desc = "Searching for: " + q
		} else {
			desc = "Searching the web"
		}

	case "web_fetch":
		title = "Fetch Web Page"
		if u := parseStringArg(rawArgs, "url"); u != "" {
			desc = "Retrieving " + u
		} else {
			desc = "Fetching web page"
		}

	case "recall_memory":
		title = "Recall Memory"
		if q := parseStringArg(rawArgs, "query"); q != "" {
			desc = "Querying memory for: " + q
		} else {
			desc = "Querying memory"
		}

	case "sandbox_exec":
		cmd := parseStringArg(rawArgs, "command")
		explicitDesc := parseStringArg(rawArgs, "description")
		if explicitDesc != "" {
			title = capitalizeFirst(explicitDesc)
			if cmd != "" && cmd != explicitDesc {
				desc = cmd
			}
		} else {
			title = extractCommandTitle(cmd)
			if cmd != "" && cmd != title {
				desc = cmd
			} else {
				desc = "Executing command in sandbox"
			}
		}

	case "sandbox_request":
		title = "Request Sandbox"
		desc = "Requesting sandbox approval"

	case "sandbox_destroy":
		title = "Destroy Sandbox"
		desc = "Tearing down sandbox environment"

	case "sandbox_download_attachment":
		title = "Download Attachment"
		if fid := parseStringArg(rawArgs, "file_id"); fid != "" {
			desc = "Downloading attachment: " + fid
		} else {
			desc = "Downloading attachment"
		}

	case "sandbox_upload_attachment":
		title = "Upload Attachment"
		nameArg := parseStringArg(rawArgs, "name")
		if nameArg == "" {
			if src := parseStringArg(rawArgs, "source_path"); src != "" {
				nameArg = filepath.Base(src)
			}
		}
		if nameArg != "" {
			desc = "Uploading attachment: " + nameArg
		} else {
			desc = "Uploading attachment"
		}

	case "discover_memories":
		title = "Discover Memories"
		if q := parseStringArg(rawArgs, "query"); q != "" {
			desc = "Searching memories for: " + q
		} else {
			desc = "Searching memories"
		}

	case "load_memory":
		title = "Load Memory"
		if mid := parseStringArg(rawArgs, "memory_id"); mid != "" {
			desc = "Loading memory: " + mid
		} else {
			desc = "Loading memory"
		}

	case "propose_memory":
		title = "Propose Memory"
		typ := parseStringArg(rawArgs, "type")
		d := parseStringArg(rawArgs, "description")
		if typ != "" {
			desc = "Proposing memory: " + typ
		} else if d != "" {
			desc = "Proposing memory: " + d
		} else {
			desc = "Proposing memory"
		}

	case "discover_skills":
		title = "Discover Skills"
		if q := parseStringArg(rawArgs, "query"); q != "" {
			desc = "Searching skills for: " + q
		} else {
			desc = "Searching skills"
		}

	case "load_skill":
		title = "Load Skill"
		if sid := parseStringArg(rawArgs, "skill_id"); sid != "" {
			desc = "Loading skill: " + sid
		} else {
			desc = "Loading skill"
		}

	case "propose_skill":
		title = "Propose Skill"
		if n := parseStringArg(rawArgs, "name"); n != "" {
			desc = "Proposing skill: " + n
		} else {
			desc = "Proposing skill"
		}

	case "schedule_task":
		title = "Schedule Task"
		if sn := parseStringArg(rawArgs, "schedule_name"); sn != "" {
			desc = "Scheduling task: " + sn
		} else {
			desc = "Scheduling task"
		}

	case "cancel_schedule":
		title = "Cancel Task"
		if sn := parseStringArg(rawArgs, "schedule_name"); sn != "" {
			desc = "Canceling task: " + sn
		} else {
			desc = "Canceling task"
		}

	case "list_schedules":
		title = "List Schedules"
		desc = "Listing scheduled tasks"

	default:
		if name == "" {
			title = "Tool"
		} else {
			title = snakeToTitle(name)
		}
		desc = summarizeFallbackArgs(rawArgs)
	}

	title = truncateUTF8(title, MaxStepTitleBytes)
	desc = truncateUTF8(desc, MaxStepDescBytes)
	return title, desc
}

func parseStringArg(rawArgs, key string) string {
	if strings.TrimSpace(rawArgs) == "" {
		return ""
	}
	var m map[string]json.RawMessage
	if err := json.Unmarshal([]byte(rawArgs), &m); err != nil {
		return ""
	}
	raw, ok := m[key]
	if !ok {
		return ""
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return strings.TrimSpace(s)
	}
	return ""
}

func capitalizeFirst(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return ""
	}
	r, size := utf8.DecodeRuneInString(s)
	return string(unicode.ToUpper(r)) + s[size:]
}

func extractCommandTitle(cmd string) string {
	cmd = strings.TrimSpace(cmd)
	if cmd == "" {
		return "Execute Command"
	}

	segments := splitCommandChain(cmd)
	if len(segments) == 0 {
		return "Execute Command"
	}
	mainCmd := segments[len(segments)-1]
	for _, seg := range segments {
		segTrim := strings.TrimSpace(seg)
		if !isSetupCommand(segTrim) {
			mainCmd = segTrim
			break
		}
	}

	fields := strings.Fields(mainCmd)
	for len(fields) > 0 && strings.Contains(fields[0], "=") {
		fields = fields[1:]
	}
	if len(fields) == 0 {
		return "Execute Command"
	}
	first := fields[0]
	first = strings.Trim(first, "\"'")
	base := filepath.Base(first)
	base = strings.TrimLeft(base, "./")
	if base == "" {
		return "Execute Command"
	}

	if len(fields) > 1 {
		sub := strings.Trim(fields[1], "\"'")
		if !strings.HasPrefix(sub, "-") && isKnownToolWithSubcommand(base) {
			return capitalizeFirst(base) + " " + sub
		}
	}

	return capitalizeFirst(base)
}

func splitCommandChain(cmd string) []string {
	var segments []string
	curr := strings.Builder{}
	runes := []rune(cmd)
	inSingleQuote := false
	inDoubleQuote := false
	for i := 0; i < len(runes); i++ {
		r := runes[i]
		if r == '\'' && !inDoubleQuote {
			inSingleQuote = !inSingleQuote
		} else if r == '"' && !inSingleQuote {
			inDoubleQuote = !inDoubleQuote
		}

		if !inSingleQuote && !inDoubleQuote {
			if (i+1 < len(runes) && r == '&' && runes[i+1] == '&') ||
				(i+1 < len(runes) && r == '|' && runes[i+1] == '|') {
				if s := strings.TrimSpace(curr.String()); s != "" {
					segments = append(segments, s)
				}
				curr.Reset()
				i++
				continue
			}
			if r == ';' {
				if s := strings.TrimSpace(curr.String()); s != "" {
					segments = append(segments, s)
				}
				curr.Reset()
				continue
			}
		}
		curr.WriteRune(r)
	}
	if s := strings.TrimSpace(curr.String()); s != "" {
		segments = append(segments, s)
	}
	return segments
}

func isSetupCommand(seg string) bool {
	fields := strings.Fields(strings.TrimSpace(seg))
	if len(fields) == 0 {
		return false
	}
	cmd := strings.ToLower(filepath.Base(fields[0]))
	switch cmd {
	case "cd", "export", "set", "env", "source", ".", "mkdir", "echo":
		return true
	default:
		return strings.Contains(cmd, "=")
	}
}

func isKnownToolWithSubcommand(base string) bool {
	switch strings.ToLower(base) {
	case "git", "pip", "npm", "yarn", "pnpm", "go", "cargo", "asv", "docker", "kubectl", "apt", "apt-get", "brew", "poetry":
		return true
	default:
		return false
	}
}

func snakeToTitle(s string) string {
	parts := strings.Split(s, "_")
	var formatted []string
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		r := []rune(p)
		r[0] = []rune(strings.ToUpper(string(r[0])))[0]
		formatted = append(formatted, string(r))
	}
	if len(formatted) == 0 {
		return "Execute Tool"
	}
	return strings.Join(formatted, " ")
}

func summarizeFallbackArgs(rawArgs string) string {
	if strings.TrimSpace(rawArgs) == "" {
		return ""
	}
	var m map[string]any
	if err := json.Unmarshal([]byte(rawArgs), &m); err != nil || len(m) == 0 {
		return ""
	}
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	var parts []string
	for _, k := range keys {
		v := m[k]
		if s, ok := v.(string); ok && s != "" {
			parts = append(parts, fmt.Sprintf("%s: %s", k, s))
		}
	}
	return strings.Join(parts, ", ")
}

func truncateUTF8(s string, maxBytes int) string {
	if maxBytes <= 0 {
		return ""
	}
	if len(s) <= maxBytes {
		return s
	}

	const ellipsis = "..."
	if maxBytes <= len(ellipsis) {
		return s[:maxBytes]
	}

	target := maxBytes - len(ellipsis)
	valid := 0
	for i := 0; i < len(s); {
		r, size := utf8.DecodeRuneInString(s[i:])
		if r == utf8.RuneError && size == 1 {
			break
		}
		if i+size > target {
			break
		}
		i += size
		valid = i
	}
	return strings.TrimRightFunc(s[:valid], unicode.IsSpace) + ellipsis
}
