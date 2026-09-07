package tools

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"time"

	"bob/internal/memory"
	"bob/internal/sandbox"
	"bob/internal/tools/tavily"
	"bob/internal/tools/webfetch"

	openai "github.com/sashabaranov/go-openai"
)

type chatContextKey struct{}

// ChatSessionContext holds request-scoped chat context for tool execution.
type ChatSessionContext struct {
	ChatID                string
	UserID                string
	IsDM                  bool
	Notifier              func(chatID, text string) error
	Progress              *ProgressReporter
	SandboxRequestCreated *bool
}

// WithChatSession returns a new context with the given ChatSessionContext attached.
func WithChatSession(ctx context.Context, session ChatSessionContext) context.Context {
	return context.WithValue(ctx, chatContextKey{}, session)
}

// ChatSessionFromContext retrieves the ChatSessionContext from context if present.
func ChatSessionFromContext(ctx context.Context) (ChatSessionContext, bool) {
	s, ok := ctx.Value(chatContextKey{}).(ChatSessionContext)
	return s, ok
}

// Registry manages available LLM tool definitions and executes tool calls.
type Registry struct {
	tavilyClient      *tavily.Client
	memoryManager     *memory.Manager
	sandboxManager    *sandbox.Manager
	toolDefinitions   []openai.Tool
	dmToolDefinitions []openai.Tool
}

// NewRegistry creates a new tool registry and initializes static tool definitions once.
func NewRegistry(tavilyClient *tavily.Client, memoryManager *memory.Manager, sandboxManager ...*sandbox.Manager) *Registry {
	var sm *sandbox.Manager
	if len(sandboxManager) > 0 {
		sm = sandboxManager[0]
	}
	r := &Registry{
		tavilyClient:   tavilyClient,
		memoryManager:  memoryManager,
		sandboxManager: sm,
	}
	r.initToolDefinitions()
	return r
}

// SetMemoryManager updates the memory manager.
func (r *Registry) SetMemoryManager(m *memory.Manager) {
	r.memoryManager = m
}

// SetSandboxManager updates the sandbox manager.
func (r *Registry) SetSandboxManager(sm *sandbox.Manager) {
	r.sandboxManager = sm
	r.initToolDefinitions()
}

func (r *Registry) initToolDefinitions() {
	webSearchSchema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"query": map[string]interface{}{
				"type":        "string",
				"description": "The search query to look up on the live web.",
			},
			"search_depth": map[string]interface{}{
				"type":        "string",
				"enum":        []string{"basic", "advanced"},
				"description": "Search depth. 'basic' is faster; 'advanced' performs deeper search.",
			},
			"max_results": map[string]interface{}{
				"type":        "integer",
				"description": "Maximum number of search results to return (1-5, default 5).",
			},
		},
		"required": []string{"query"},
	}

	webFetchSchema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"url": map[string]interface{}{
				"type":        "string",
				"description": "The target HTTP or HTTPS URL to fetch content from.",
			},
			"mode": map[string]interface{}{
				"type":        "string",
				"enum":        []string{"auto", "raw", "extract"},
				"description": "Operational fetching mode. 'auto' (default) parses HTML into readable text with dynamic fallback; 'raw' returns the direct body without HTML parsing (recommended for code, scripts, configs, JSON); 'extract' queries Tavily Extract directly for heavy dynamic SPAs or if 'auto' failed to capture needed content). All modes truncate output to 16KB.",
			},
		},
		"required": []string{"url"},
	}

	recallMemorySchema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"query": map[string]interface{}{
				"type":        "string",
				"description": "The search query to recall past conversations, topics, facts, or discussions from long-term memory.",
			},
			"limit": map[string]interface{}{
				"type":        "integer",
				"description": "Maximum number of memory passages to retrieve (1-10, default 5).",
			},
		},
		"required": []string{"query"},
	}

	allowedNetworkModes := []string{"none", "restricted"}
	if r.sandboxManager != nil {
		allowedNetworkModes = r.sandboxManager.AllowedNetworkModes()
	}

	var networkDescriptions []string
	for _, m := range allowedNetworkModes {
		switch strings.ToLower(m) {
		case "none":
			networkDescriptions = append(networkDescriptions, "'none' (default, completely offline)")
		case "restricted":
			networkDescriptions = append(networkDescriptions, "'restricted' (access limited to whitelisted domains)")
		case "full":
			networkDescriptions = append(networkDescriptions, "'full' (unrestricted internet access)")
		}
	}
	networkDesc := "Network access level. Permitted modes on this server: " + strings.Join(networkDescriptions, ", ") + ". Apply least privilege."

	sandboxRequestSchema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"driver": map[string]interface{}{
				"type":        "string",
				"enum":        []string{"bwrap", "docker"},
				"description": "Sandbox driver: 'bwrap' (fast local isolation via Bubblewrap) or 'docker' (container isolation).",
			},
			"image": map[string]interface{}{
				"type":        "string",
				"description": "Container image name when using 'docker' (e.g. 'alpine:latest', 'golang:alpine', 'python:3.11-slim', 'node:20-slim').",
			},
			"network": map[string]interface{}{
				"type":        "string",
				"enum":        allowedNetworkModes,
				"description": networkDesc,
			},
			"domains": map[string]interface{}{
				"type": "array",
				"items": map[string]interface{}{
					"type": "string",
				},
				"description": "List of domains to allow when network is 'restricted' (e.g. ['pypi.org', 'files.pythonhosted.org']).",
			},
			"mounts": map[string]interface{}{
				"type": "array",
				"items": map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"path": map[string]interface{}{
							"type":        "string",
							"description": "Relative directory path inside your private workspace (e.g. '.' for whole workspace, or 'project1' for a specific subdirectory).",
						},
						"read_only": map[string]interface{}{
							"type":        "boolean",
							"description": "Whether the mount is read-only (default false).",
						},
					},
					"required": []string{"path"},
				},
				"description": "Subdirectories or whole workspace ('.') in your user workspace to mount into the sandbox.",
			},
			"lifetime_minutes": map[string]interface{}{
				"type":        "integer",
				"description": "Requested sandbox lifetime in minutes (1-30, default 30).",
			},
			"reason": map[string]interface{}{
				"type":        "string",
				"description": "Plain explanation to the user describing why the sandbox and each requested permission are needed.",
			},
		},
		"required": []string{"driver", "reason"},
	}

	sandboxExecSchema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"command": map[string]interface{}{
				"type":        "string",
				"description": "Shell command line to execute inside your active sandbox. IMPORTANT: Use one-shot, non-interactive commands only (do NOT use TUI tools like nano, vim, top, or interactive prompts).",
			},
			"timeout_seconds": map[string]interface{}{
				"type":        "integer",
				"description": "Command execution timeout in seconds (default 60, min 5, max 600).",
			},
		},
		"required": []string{"command"},
	}

	sandboxDestroySchema := map[string]interface{}{
		"type":       "object",
		"properties": map[string]interface{}{},
	}

	r.toolDefinitions = []openai.Tool{
		{
			Type: openai.ToolTypeFunction,
			Function: &openai.FunctionDefinition{
				Name:        "web_search",
				Description: "Search the live web for current information, news, documentation, and facts using Tavily search. When using results from this tool in your response, always cite sources and include the original markdown links [Title](URL) provided in the search results.",
				Parameters:  webSearchSchema,
			},
		},
		{
			Type: openai.ToolTypeFunction,
			Function: &openai.FunctionDefinition{
				Name:        "web_fetch",
				Description: "Fetch and extract text content from a web page URL or raw text/code file. Supports 3 modes: 'auto' (default: parses HTML into clean readable text with dynamic fallback), 'raw' (recommended for raw code, scripts, JSON, or configs without HTML parsing), and 'extract' (uses Tavily Extract directly for heavy dynamic SPAs or if 'auto' failed to capture needed content). All outputs are capped at 16KB.",
				Parameters:  webFetchSchema,
			},
		},
		{
			Type: openai.ToolTypeFunction,
			Function: &openai.FunctionDefinition{
				Name:        "recall_memory",
				Description: "Recall relevant past conversation history, discussions, topics, and facts from long-term memory. Use this tool when users ask about previous topics, past conversations, or shared history.",
				Parameters:  recallMemorySchema,
			},
		},
	}

	sandboxTools := []openai.Tool{
		{
			Type: openai.ToolTypeFunction,
			Function: &openai.FunctionDefinition{
				Name:        "sandbox_request",
				Description: "Request creation of a secure isolated execution sandbox. You MUST apply the principle of least privilege, requesting only the minimal permissions required for the task. The human user must explicitly approve your request before the sandbox is created. Available only in 1-on-1 direct messages.",
				Parameters:  sandboxRequestSchema,
			},
		},
		{
			Type: openai.ToolTypeFunction,
			Function: &openai.FunctionDefinition{
				Name:        "sandbox_exec",
				Description: "Execute a shell command inside your active sandbox. Output (stdout, stderr, exit code) is returned upon completion. Use non-interactive, one-shot commands only.",
				Parameters:  sandboxExecSchema,
			},
		},
		{
			Type: openai.ToolTypeFunction,
			Function: &openai.FunctionDefinition{
				Name:        "sandbox_destroy",
				Description: "Explicitly terminate and tear down your active sandbox when you have finished your tasks, freeing system resources.",
				Parameters:  sandboxDestroySchema,
			},
		},
	}

	r.dmToolDefinitions = append(append([]openai.Tool{}, r.toolDefinitions...), sandboxTools...)
}

// ToolDefinitions returns the cached slice of OpenAI tool definitions (for Townhall / public chats).
func (r *Registry) ToolDefinitions() []openai.Tool {
	return r.toolDefinitions
}

// ToolDefinitionsForSession returns tool definitions tailored to the session.
// In DM chats with a configured sandboxManager, sandbox tools are included.
func (r *Registry) ToolDefinitionsForSession(session ChatSessionContext) []openai.Tool {
	if session.IsDM && r.sandboxManager != nil {
		return r.dmToolDefinitions
	}
	return r.toolDefinitions
}

// WebSearchArgs defines arguments for the web_search tool.
type WebSearchArgs struct {
	Query       string `json:"query"`
	SearchDepth string `json:"search_depth,omitempty"`
	MaxResults  int    `json:"max_results,omitempty"`
}

// WebFetchArgs defines arguments for the web_fetch tool.
type WebFetchArgs struct {
	URL  string `json:"url"`
	Mode string `json:"mode,omitempty"`
}

// RecallMemoryArgs defines arguments for the recall_memory tool.
type RecallMemoryArgs struct {
	Query string `json:"query"`
	Limit int    `json:"limit,omitempty"`
}

// Execute dispatches a tool execution by function name and arguments JSON.
func (r *Registry) Execute(ctx context.Context, name string, argsJSON string) (string, error) {
	switch name {
	case "web_search":
		return r.executeWebSearch(ctx, argsJSON)
	case "web_fetch":
		return r.executeWebFetch(ctx, argsJSON)
	case "recall_memory":
		return r.executeRecallMemory(ctx, argsJSON)
	case "sandbox_request":
		return r.executeSandboxRequest(ctx, argsJSON)
	case "sandbox_exec":
		return r.executeSandboxExec(ctx, argsJSON)
	case "sandbox_destroy":
		return r.executeSandboxDestroy(ctx, argsJSON)
	default:
		return "", fmt.Errorf("unknown tool: %s", name)
	}
}

func (r *Registry) executeRecallMemory(ctx context.Context, argsJSON string) (string, error) {
	if r.memoryManager == nil {
		return "", errors.New("memory manager is not configured")
	}

	var args RecallMemoryArgs
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
		return "", fmt.Errorf("failed to parse arguments: %w", err)
	}

	args.Query = strings.TrimSpace(args.Query)
	if args.Query == "" {
		return "", errors.New("query cannot be empty")
	}

	limit := args.Limit
	if limit <= 0 {
		limit = 5
	}

	session, _ := ChatSessionFromContext(ctx)
	chatID := session.ChatID
	if chatID == "" {
		chatID = "townhall"
	}
	isDM := session.IsDM
	if chatID == "townhall" {
		isDM = false
	}

	items, err := r.memoryManager.Search(ctx, args.Query, chatID, isDM, limit)
	if err != nil {
		return "", fmt.Errorf("memory search error: %w", err)
	}

	slog.Info("executed recall_memory", "query", args.Query, "hits", len(items))

	if len(items) == 0 {
		payload := map[string]interface{}{
			"query":    args.Query,
			"count":    0,
			"memories": []interface{}{},
			"message":  "No relevant memories found in long-term memory for the given query.",
		}
		respBytes, _ := json.Marshal(payload)
		return string(respBytes), nil
	}

	var sb strings.Builder
	sb.WriteString("### Recalled Conversation Chunks:\n")
	for _, item := range items {
		var timeRange string
		if item.StartTime > 0 && item.EndTime > 0 {
			startT := time.Unix(item.StartTime, 0).UTC().Format("2006-01-02 15:04:05")
			endT := time.Unix(item.EndTime, 0).UTC().Format("2006-01-02 15:04:05")
			timeRange = fmt.Sprintf(" (%s to %s UTC)", startT, endT)
		} else if item.StartTime > 0 {
			timeRange = fmt.Sprintf(" (%s UTC)", time.Unix(item.StartTime, 0).UTC().Format("2006-01-02 15:04:05"))
		}

		seqRange := fmt.Sprintf("seq %d-%d", item.StartSeq, item.EndSeq)
		if item.StartSeq == item.EndSeq {
			seqRange = fmt.Sprintf("seq %d", item.StartSeq)
		}

		fmt.Fprintf(&sb, "\n--- %s [%s]%s ---\n%s\n", item.Source, seqRange, timeRange, strings.TrimSpace(item.Content))
	}

	payload := map[string]interface{}{
		"query":             args.Query,
		"count":             len(items),
		"memories":          items,
		"formatted_results": strings.TrimSpace(sb.String()),
	}

	respBytes, err := json.Marshal(payload)
	if err != nil {
		return "", fmt.Errorf("failed to format memory response: %w", err)
	}
	return string(respBytes), nil
}

func (r *Registry) executeWebSearch(ctx context.Context, argsJSON string) (string, error) {
	if r.tavilyClient == nil {
		return "", errors.New("tavily client is not configured")
	}

	var args WebSearchArgs
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
		return "", fmt.Errorf("failed to parse arguments: %w", err)
	}

	args.Query = strings.TrimSpace(args.Query)
	if args.Query == "" {
		return "", errors.New("query cannot be empty")
	}

	req := tavily.SearchRequest{
		Query:         args.Query,
		SearchDepth:   args.SearchDepth,
		MaxResults:    args.MaxResults,
		IncludeAnswer: true,
	}

	resp, err := r.tavilyClient.Search(ctx, req)
	if err != nil {
		return "", fmt.Errorf("search error: %w", err)
	}

	payload := map[string]interface{}{
		"query":       resp.Query,
		"answer":      resp.Answer,
		"results":     resp.Results,
		"instruction": "When presenting these search findings to the user, cite your sources with original markdown links [Title](URL) from the results.",
	}

	respBytes, err := json.Marshal(payload)
	if err != nil {
		return "", fmt.Errorf("failed to format search response: %w", err)
	}

	return string(respBytes), nil
}

func (r *Registry) executeWebFetch(ctx context.Context, argsJSON string) (string, error) {
	var args WebFetchArgs
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
		return "", fmt.Errorf("failed to parse arguments: %w", err)
	}

	args.URL = strings.TrimSpace(args.URL)
	if args.URL == "" {
		return "", errors.New("url cannot be empty")
	}

	mode := strings.ToLower(strings.TrimSpace(args.Mode))
	if mode == "" {
		mode = "auto"
	}

	switch mode {
	case "extract":
		return r.executeFetchExtractMode(ctx, args.URL)
	case "raw":
		return r.executeFetchRawMode(ctx, args.URL)
	case "auto":
		return r.executeFetchAutoMode(ctx, args.URL)
	default:
		return "", fmt.Errorf("invalid mode %q: supported modes are 'auto', 'raw', and 'extract'", mode)
	}
}

func (r *Registry) executeFetchExtractMode(ctx context.Context, targetURL string) (string, error) {
	if r.tavilyClient == nil {
		return "", errors.New("tavily client is not configured for extract mode")
	}

	resp, err := r.tavilyClient.Extract(ctx, targetURL)
	if err != nil {
		return "", fmt.Errorf("extract error: %w", err)
	}

	if len(resp.Results) == 0 {
		if len(resp.FailedResults) > 0 {
			return "", fmt.Errorf("extract failed for url %s: %s", resp.FailedResults[0].URL, resp.FailedResults[0].Error)
		}
		return "", fmt.Errorf("no content extracted from %s", targetURL)
	}

	res := resp.Results[0]
	truncatedContent := webfetch.TruncateText(res.RawContent, len(res.RawContent) > webfetch.MaxContentSize)

	payload := map[string]interface{}{
		"url":     res.URL,
		"mode":    "extract",
		"content": truncatedContent,
	}

	respBytes, err := json.Marshal(payload)
	if err != nil {
		return "", fmt.Errorf("failed to format extract response: %w", err)
	}
	return string(respBytes), nil
}

func (r *Registry) executeFetchRawMode(ctx context.Context, targetURL string) (string, error) {
	fetchRes, err := webfetch.Fetch(ctx, targetURL, nil)
	if err != nil {
		return "", fmt.Errorf("fetch error: %w", err)
	}

	truncatedContent := webfetch.TruncateText(string(fetchRes.RawBody), fetchRes.Truncated)

	payload := map[string]interface{}{
		"url":          fetchRes.URL,
		"mode":         "raw",
		"content_type": fetchRes.ContentType,
		"status_code":  fetchRes.StatusCode,
		"content":      truncatedContent,
	}

	respBytes, err := json.Marshal(payload)
	if err != nil {
		return "", fmt.Errorf("failed to format raw fetch response: %w", err)
	}
	return string(respBytes), nil
}

func (r *Registry) executeFetchAutoMode(ctx context.Context, targetURL string) (string, error) {
	fetchRes, fetchErr := webfetch.Fetch(ctx, targetURL, nil)
	if fetchErr != nil {
		// If direct fetch fails and Tavily is available, try dynamic extraction
		if r.tavilyClient != nil {
			extractPayload, extractErr := r.executeFetchExtractMode(ctx, targetURL)
			if extractErr == nil {
				return extractPayload, nil
			}
		}
		return "", fmt.Errorf("fetch error: %w", fetchErr)
	}

	if webfetch.IsBinary(fetchRes.ContentType) {
		payload := map[string]interface{}{
			"url":          fetchRes.URL,
			"mode":         "auto",
			"content_type": fetchRes.ContentType,
			"status_code":  fetchRes.StatusCode,
			"content":      fmt.Sprintf("[Binary file of type %s (%d bytes). Direct display not supported; file download support will be added in a future update.]", fetchRes.ContentType, len(fetchRes.RawBody)),
		}
		respBytes, err := json.Marshal(payload)
		if err != nil {
			return "", fmt.Errorf("failed to format response: %w", err)
		}
		return string(respBytes), nil
	}

	if !webfetch.IsHTML(fetchRes.ContentType) {
		truncatedContent := webfetch.TruncateText(string(fetchRes.RawBody), fetchRes.Truncated)
		payload := map[string]interface{}{
			"url":          fetchRes.URL,
			"mode":         "auto",
			"content_type": fetchRes.ContentType,
			"status_code":  fetchRes.StatusCode,
			"content":      truncatedContent,
		}
		respBytes, err := json.Marshal(payload)
		if err != nil {
			return "", fmt.Errorf("failed to format response: %w", err)
		}
		return string(respBytes), nil
	}

	// HTML handling with readability and heuristics
	rResult, rErr := webfetch.ParseReadability(fetchRes.RawBody, fetchRes.URL)
	needsFallback, fallbackReason := webfetch.NeedsDynamicFallback(rResult, rErr, string(fetchRes.RawBody))

	if needsFallback {
		if r.tavilyClient != nil {
			extractResp, extractErr := r.tavilyClient.Extract(ctx, targetURL)
			if extractErr == nil && len(extractResp.Results) > 0 && strings.TrimSpace(extractResp.Results[0].RawContent) != "" {
				res := extractResp.Results[0]
				payload := map[string]interface{}{
					"url":             res.URL,
					"mode":            "auto",
					"source":          "tavily_extract",
					"fallback_reason": fallbackReason,
					"content":         webfetch.TruncateText(res.RawContent, len(res.RawContent) > webfetch.MaxContentSize),
				}
				respBytes, err := json.Marshal(payload)
				if err != nil {
					return "", fmt.Errorf("failed to format extract response: %w", err)
				}
				return string(respBytes), nil
			}
		}

		// If Tavily is unconfigured or failed, return best-effort direct content with note
		var content string
		if rResult != nil && strings.TrimSpace(rResult.TextContent) != "" {
			content = rResult.TextContent + "\n\n[Note: Dynamic rendering fallback could not be executed: " + fallbackReason + "]"
		} else {
			content = string(fetchRes.RawBody) + "\n\n[Note: Dynamic rendering fallback could not be executed: " + fallbackReason + "]"
		}

		payload := map[string]interface{}{
			"url":             fetchRes.URL,
			"mode":            "auto",
			"source":          "direct_html_fallback",
			"fallback_reason": fallbackReason,
			"content":         webfetch.TruncateText(content, fetchRes.Truncated),
		}
		respBytes, err := json.Marshal(payload)
		if err != nil {
			return "", fmt.Errorf("failed to format response: %w", err)
		}
		return string(respBytes), nil
	}

	// Direct Readability Succeeded
	payload := map[string]interface{}{
		"url":     fetchRes.URL,
		"mode":    "auto",
		"title":   rResult.Title,
		"byline":  rResult.Byline,
		"excerpt": rResult.Excerpt,
		"content": webfetch.TruncateText(rResult.TextContent, fetchRes.Truncated),
	}

	respBytes, err := json.Marshal(payload)
	if err != nil {
		return "", fmt.Errorf("failed to format readability response: %w", err)
	}
	return string(respBytes), nil
}

// SandboxRequestArgs defines arguments for the sandbox_request tool.
type SandboxRequestArgs struct {
	Driver  string   `json:"driver"`
	Image   string   `json:"image,omitempty"`
	Network string   `json:"network,omitempty"`
	Domains []string `json:"domains,omitempty"`
	Mounts  []struct {
		Path     string `json:"path"`
		ReadOnly bool   `json:"read_only,omitempty"`
	} `json:"mounts,omitempty"`
	LifetimeMinutes int    `json:"lifetime_minutes,omitempty"`
	Reason          string `json:"reason"`
}

// SandboxExecArgs defines arguments for the sandbox_exec tool.
type SandboxExecArgs struct {
	Command        string `json:"command"`
	TimeoutSeconds int    `json:"timeout_seconds,omitempty"`
}

func (r *Registry) executeSandboxRequest(ctx context.Context, argsJSON string) (string, error) {
	if r.sandboxManager == nil {
		return "", errors.New("sandbox execution is not configured on this server")
	}
	session, ok := ChatSessionFromContext(ctx)
	if !ok || !session.IsDM {
		return "", errors.New("sandbox tools are strictly available in 1-on-1 direct messages")
	}
	if session.UserID == "" {
		return "", errors.New("cannot determine user ID from chat session")
	}

	var args SandboxRequestArgs
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
		return "", fmt.Errorf("failed to parse sandbox_request arguments: %w", err)
	}

	var mounts []sandbox.UserMount
	for _, m := range args.Mounts {
		mounts = append(mounts, sandbox.UserMount{
			RelativePath: m.Path,
			ReadOnly:     m.ReadOnly,
		})
	}

	netMode := sandbox.NetworkMode(strings.ToLower(strings.TrimSpace(args.Network)))
	if netMode == "" {
		netMode = sandbox.NetworkNone
	}

	params := sandbox.RequestParams{
		Driver:          sandbox.DriverType(strings.ToLower(strings.TrimSpace(args.Driver))),
		DockerImage:     strings.TrimSpace(args.Image),
		NetworkMode:     netMode,
		AllowedDomains:  args.Domains,
		Mounts:          mounts,
		LifetimeMinutes: args.LifetimeMinutes,
		Reason:          strings.TrimSpace(args.Reason),
	}

	sbx, err := r.sandboxManager.RequestSandbox(ctx, session.UserID, session.ChatID, params)
	if err != nil {
		return "", fmt.Errorf("sandbox request rejected: %w", err)
	}

	var card strings.Builder
	card.WriteString("🔒 **Sandbox Approval Requested**\n")
	card.WriteString(fmt.Sprintf("• **Driver:** `%s`\n", sbx.Driver))
	if sbx.Driver == sandbox.DriverDocker && sbx.DockerImage != "" {
		card.WriteString(fmt.Sprintf("• **Docker Image:** `%s`\n", sbx.DockerImage))
	}
	card.WriteString(fmt.Sprintf("• **Network:** `%s`\n", sbx.Network.Mode))
	if len(sbx.Network.AllowedHosts) > 0 {
		card.WriteString(fmt.Sprintf("• **Allowed Domains:** `%s`\n", strings.Join(sbx.Network.AllowedHosts, ", ")))
	}
	if len(sbx.Mounts) > 0 {
		card.WriteString("• **Mounts:**\n")
		for _, m := range sbx.Mounts {
			ro := "read-write"
			if m.ReadOnly {
				ro = "read-only"
			}
			card.WriteString(fmt.Sprintf("  - `%s` (%s)\n", m.RelativePath, ro))
		}
	} else {
		card.WriteString("• **Workspace Mount:** none (isolated scratch space)\n")
	}
	cleanReason := strings.ReplaceAll(strings.ReplaceAll(sbx.Reason, "\r", ""), "\n", " ")
	cleanReason = strings.TrimSpace(cleanReason)
	if cleanReason == "" {
		cleanReason = "No reason provided"
	}
	card.WriteString(fmt.Sprintf("• **Reason:** %s\n\n", cleanReason))
	card.WriteString("Reply `/sandbox approve` to approve or `/sandbox deny` to reject.")

	if session.Notifier != nil {
		if err := session.Notifier(session.ChatID, card.String()); err != nil {
			return "", fmt.Errorf("failed to send approval card: %w", err)
		}
	}
	if session.SandboxRequestCreated != nil {
		*session.SandboxRequestCreated = true
	}

	var b strings.Builder
	b.WriteString("Sandbox request created and approval card has already been sent to the user in chat.\n")
	b.WriteString("DO NOT generate any conversational message or ask for approval again, as it duplicates the card already displayed.\n")
	b.WriteString("Conclude your turn now.")
	return b.String(), nil
}

func (r *Registry) executeSandboxExec(ctx context.Context, argsJSON string) (string, error) {
	if r.sandboxManager == nil {
		return "", errors.New("sandbox execution is not configured on this server")
	}
	session, ok := ChatSessionFromContext(ctx)
	if !ok || !session.IsDM {
		return "", errors.New("sandbox tools are strictly available in 1-on-1 direct messages")
	}
	if session.UserID == "" {
		return "", errors.New("cannot determine user ID from chat session")
	}

	var args SandboxExecArgs
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
		return "", fmt.Errorf("failed to parse sandbox_exec arguments: %w", err)
	}

	cmdStr := strings.TrimSpace(args.Command)
	if cmdStr == "" {
		return "", errors.New("command cannot be empty")
	}

	if session.Progress != nil {
		session.Progress.SetCommand(cmdStr)
	}

	timeout := time.Duration(args.TimeoutSeconds) * time.Second

	res, err := r.sandboxManager.Exec(ctx, session.UserID, []string{"sh", "-c", cmdStr}, timeout)
	if session.Progress != nil {
		session.Progress.SetCurrent("Analyzing output")
	}
	if err != nil {
		return "", fmt.Errorf("execution failed: %w", err)
	}

	var b strings.Builder
	b.WriteString(fmt.Sprintf("Exit Code: %d\nDuration: %s\n", res.ExitCode, res.Duration.Round(time.Millisecond)))
	if res.Stdout != "" {
		b.WriteString("\n[stdout]\n" + res.Stdout)
	}
	if res.Stderr != "" {
		b.WriteString("\n[stderr]\n" + res.Stderr)
	}
	if res.Stdout == "" && res.Stderr == "" {
		b.WriteString("\n(No output)")
	}

	return b.String(), nil
}

func (r *Registry) executeSandboxDestroy(ctx context.Context, argsJSON string) (string, error) {
	if r.sandboxManager == nil {
		return "", errors.New("sandbox execution is not configured on this server")
	}
	session, ok := ChatSessionFromContext(ctx)
	if !ok || !session.IsDM {
		return "", errors.New("sandbox tools are strictly available in 1-on-1 direct messages")
	}
	if session.UserID == "" {
		return "", errors.New("cannot determine user ID from chat session")
	}

	if err := r.sandboxManager.Destroy(ctx, session.UserID); err != nil {
		return "", fmt.Errorf("failed to destroy sandbox: %w", err)
	}

	return "Sandbox successfully destroyed and resources released.", nil
}
