package tools

import (
	"strings"
	"testing"

	openai "github.com/sashabaranov/go-openai"
	"github.com/stretchr/testify/assert"
)

func TestFormatProgressStep_RegisteredTools(t *testing.T) {
	tests := []struct {
		name         string
		toolName     string
		args         string
		expectedTitle string
		expectedDesc string
	}{
		{
			name:         "web_search with query",
			toolName:     "web_search",
			args:         `{"query":"golang concurrency patterns"}`,
			expectedTitle: "Web Search",
			expectedDesc: "Searching for: golang concurrency patterns",
		},
		{
			name:         "web_search empty query",
			toolName:     "web_search",
			args:         `{"query":""}`,
			expectedTitle: "Web Search",
			expectedDesc: "Searching the web",
		},
		{
			name:         "web_fetch with url",
			toolName:     "web_fetch",
			args:         `{"url":"https://example.com/docs"}`,
			expectedTitle: "Fetch Web Page",
			expectedDesc: "Retrieving https://example.com/docs",
		},
		{
			name:         "web_fetch empty url",
			toolName:     "web_fetch",
			args:         `{}`,
			expectedTitle: "Fetch Web Page",
			expectedDesc: "Fetching web page",
		},
		{
			name:         "recall_memory with query",
			toolName:     "recall_memory",
			args:         `{"query":"favorite editor"}`,
			expectedTitle: "Recall Memory",
			expectedDesc: "Querying memory for: favorite editor",
		},
		{
			name:          "sandbox_exec with explicit description",
			toolName:      "sandbox_exec",
			args:          `{"command":"python3 test.py","description":"Running test suite"}`,
			expectedTitle: "Running test suite",
			expectedDesc:  "python3 test.py",
		},
		{
			name:          "sandbox_exec with explicit description and compound command",
			toolName:      "sandbox_exec",
			args:          `{"command":"export HOME=/tmp && asv run bench_linalg","description":"running bench_linalg benchmarks"}`,
			expectedTitle: "Running bench_linalg benchmarks",
			expectedDesc:  "export HOME=/tmp && asv run bench_linalg",
		},
		{
			name:          "sandbox_exec compound command with cd without description",
			toolName:      "sandbox_exec",
			args:          `{"command":"cd /tmp && git clone https://github.com/numpy/numpy.git"}`,
			expectedTitle: "Git clone",
			expectedDesc:  "cd /tmp && git clone https://github.com/numpy/numpy.git",
		},
		{
			name:          "sandbox_exec compound command with export without description",
			toolName:      "sandbox_exec",
			args:          `{"command":"export HOME=/tmp && asv run bench_linalg"}`,
			expectedTitle: "Asv run",
			expectedDesc:  "export HOME=/tmp && asv run bench_linalg",
		},
		{
			name:          "sandbox_exec without description",
			toolName:      "sandbox_exec",
			args:          `{"command":"/usr/bin/grep -rn 'foo' ."}`,
			expectedTitle: "Grep",
			expectedDesc:  "/usr/bin/grep -rn 'foo' .",
		},
		{
			name:          "sandbox_exec with var assignment",
			toolName:      "sandbox_exec",
			args:          `{"command":"ENV=prod ./run.sh"}`,
			expectedTitle: "Run.sh",
			expectedDesc:  "ENV=prod ./run.sh",
		},
		{
			name:          "sandbox_exec empty command",
			toolName:      "sandbox_exec",
			args:          `{"command":""}`,
			expectedTitle: "Execute Command",
			expectedDesc:  "Executing command in sandbox",
		},
		{
			name:         "sandbox_request",
			toolName:     "sandbox_request",
			args:         `{}`,
			expectedTitle: "Request Sandbox",
			expectedDesc: "Requesting sandbox approval",
		},
		{
			name:         "sandbox_destroy",
			toolName:     "sandbox_destroy",
			args:         `{}`,
			expectedTitle: "Destroy Sandbox",
			expectedDesc: "Tearing down sandbox environment",
		},
		{
			name:         "sandbox_download_attachment",
			toolName:     "sandbox_download_attachment",
			args:         `{"file_id":"att-123"}`,
			expectedTitle: "Download Attachment",
			expectedDesc: "Downloading attachment: att-123",
		},
		{
			name:         "sandbox_upload_attachment with name",
			toolName:     "sandbox_upload_attachment",
			args:         `{"name":"output.png","source_path":"/tmp/out.png"}`,
			expectedTitle: "Upload Attachment",
			expectedDesc: "Uploading attachment: output.png",
		},
		{
			name:         "sandbox_upload_attachment fallback to source_path basename",
			toolName:     "sandbox_upload_attachment",
			args:         `{"source_path":"/workspace/data.csv"}`,
			expectedTitle: "Upload Attachment",
			expectedDesc: "Uploading attachment: data.csv",
		},
		{
			name:         "discover_memories",
			toolName:     "discover_memories",
			args:         `{"query":"travel preferences"}`,
			expectedTitle: "Discover Memories",
			expectedDesc: "Searching memories for: travel preferences",
		},
		{
			name:         "load_memory",
			toolName:     "load_memory",
			args:         `{"memory_id":"mem-abc-123"}`,
			expectedTitle: "Load Memory",
			expectedDesc: "Loading memory: mem-abc-123",
		},
		{
			name:         "propose_memory with type",
			toolName:     "propose_memory",
			args:         `{"type":"preference","description":"Prefers dark theme"}`,
			expectedTitle: "Propose Memory",
			expectedDesc: "Proposing memory: preference",
		},
		{
			name:         "discover_skills",
			toolName:     "discover_skills",
			args:         `{"query":"code formatting"}`,
			expectedTitle: "Discover Skills",
			expectedDesc: "Searching skills for: code formatting",
		},
		{
			name:         "load_skill",
			toolName:     "load_skill",
			args:         `{"skill_id":"skill-xyz"}`,
			expectedTitle: "Load Skill",
			expectedDesc: "Loading skill: skill-xyz",
		},
		{
			name:         "propose_skill",
			toolName:     "propose_skill",
			args:         `{"name":"deploy_helper"}`,
			expectedTitle: "Propose Skill",
			expectedDesc: "Proposing skill: deploy_helper",
		},
		{
			name:         "schedule_task",
			toolName:     "schedule_task",
			args:         `{"schedule_name":"daily_backup"}`,
			expectedTitle: "Schedule Task",
			expectedDesc: "Scheduling task: daily_backup",
		},
		{
			name:         "cancel_schedule",
			toolName:     "cancel_schedule",
			args:         `{"schedule_name":"daily_backup"}`,
			expectedTitle: "Cancel Task",
			expectedDesc: "Canceling task: daily_backup",
		},
		{
			name:         "list_schedules",
			toolName:     "list_schedules",
			args:         `{}`,
			expectedTitle: "List Schedules",
			expectedDesc: "Listing scheduled tasks",
		},
		{
			name:         "fallback unknown tool",
			toolName:     "custom_analysis_tool",
			args:         `{"target":"dataset.csv"}`,
			expectedTitle: "Custom Analysis Tool",
			expectedDesc: "target: dataset.csv",
		},
		{
			name:         "empty tool name fallback",
			toolName:     "",
			args:         `{}`,
			expectedTitle: "Tool",
			expectedDesc: "",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			tcCall := openai.ToolCall{
				Function: openai.FunctionCall{
					Name:      tc.toolName,
					Arguments: tc.args,
				},
			}
			title, desc := FormatProgressStep(tcCall)
			assert.Equal(t, tc.expectedTitle, title)
			assert.Equal(t, tc.expectedDesc, desc)
		})
	}
}

func TestFormatProgressStep_MalformedJSON(t *testing.T) {
	tcCall := openai.ToolCall{
		Function: openai.FunctionCall{
			Name:      "web_search",
			Arguments: "invalid-json{{{",
		},
	}
	title, desc := FormatProgressStep(tcCall)
	assert.Equal(t, "Web Search", title)
	assert.Equal(t, "Searching the web", desc)

	// Fallback tool with malformed JSON
	tcFallback := openai.ToolCall{
		Function: openai.FunctionCall{
			Name:      "unknown_tool",
			Arguments: "12345",
		},
	}
	title2, desc2 := FormatProgressStep(tcFallback)
	assert.Equal(t, "Unknown Tool", title2)
	assert.Empty(t, desc2)
}

func TestFormatProgressStep_Truncation(t *testing.T) {
	// Long description
	longQuery := strings.Repeat("абвг", 200) // multibyte Cyrillic characters
	tcCall := openai.ToolCall{
		Function: openai.FunctionCall{
			Name:      "web_search",
			Arguments: `{"query":"` + longQuery + `"}`,
		},
	}
	title, desc := FormatProgressStep(tcCall)
	assert.Equal(t, "Web Search", title)
	assert.True(t, len(desc) <= 512, "desc byte length %d must be <= 512", len(desc))
	assert.True(t, strings.HasSuffix(desc, "..."))

	// Long title
	tcLongTitle := openai.ToolCall{
		Function: openai.FunctionCall{
			Name: strings.Repeat("long_tool_name_", 10),
		},
	}
	title2, _ := FormatProgressStep(tcLongTitle)
	assert.True(t, len(title2) <= 80, "title byte length %d must be <= 80", len(title2))
	assert.True(t, strings.HasSuffix(title2, "..."))
}

func TestFormatProgressStep_MalformedCommandsAndQuotes(t *testing.T) {
	testCases := []struct {
		name          string
		cmd           string
		expectedTitle string
	}{
		{
			name:          "only semicolons",
			cmd:           ";;;",
			expectedTitle: "Execute Command",
		},
		{
			name:          "only boolean operators",
			cmd:           "&& || &&",
			expectedTitle: "Execute Command",
		},
		{
			name:          "multiple environment variables",
			cmd:           "VAR1=1 VAR2=2 asv run bench_linalg",
			expectedTitle: "Asv run",
		},
		{
			name:          "semicolon inside double quotes",
			cmd:           `echo "foo; bar"; python main.py`,
			expectedTitle: "Python",
		},
		{
			name:          "double ampersand inside single quotes",
			cmd:           `echo 'first && second'`,
			expectedTitle: "Echo",
		},
		{
			name:          "chain of setup commands before real tool",
			cmd:           `cd /tmp && export FOO=bar && VAR=1 pytest`,
			expectedTitle: "Pytest",
		},
		{
			name:          "only setup commands falls back cleanly",
			cmd:           `cd /tmp && export FOO=bar`,
			expectedTitle: "Export",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			toolCall := openai.ToolCall{
				Function: openai.FunctionCall{
					Name:      "sandbox_exec",
					Arguments: `{"command":` + `"` + strings.ReplaceAll(tc.cmd, `"`, `\"`) + `"}`,
				},
			}
			title, _ := FormatProgressStep(toolCall)
			assert.Equal(t, tc.expectedTitle, title)
		})
	}
}

