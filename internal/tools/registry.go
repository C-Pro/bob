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
	"bob/internal/tools/tavily"
	"bob/internal/tools/webfetch"

	openai "github.com/sashabaranov/go-openai"
)

type chatContextKey struct{}

// ChatSessionContext holds request-scoped chat context for tool execution.
type ChatSessionContext struct {
	ChatID string
	IsDM   bool
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
	tavilyClient    *tavily.Client
	memoryManager   *memory.Manager
	toolDefinitions []openai.Tool
}

// NewRegistry creates a new tool registry and initializes static tool definitions once.
func NewRegistry(tavilyClient *tavily.Client, memoryManager *memory.Manager) *Registry {
	r := &Registry{
		tavilyClient:  tavilyClient,
		memoryManager: memoryManager,
	}
	r.initToolDefinitions()
	return r
}

// SetMemoryManager updates the memory manager.
func (r *Registry) SetMemoryManager(m *memory.Manager) {
	r.memoryManager = m
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
				"description": "Operational fetching mode. 'auto' (default) parses HTML into readable text with dynamic fallback; 'raw' returns the direct body without HTML parsing (recommended for code, scripts, configs, JSON); 'extract' queries Tavily Extract directly for heavy JavaScript SPAs or if 'auto' missed content. All modes truncate output to 16KB.",
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
}

// ToolDefinitions returns the cached slice of OpenAI tool definitions.
func (r *Registry) ToolDefinitions() []openai.Tool {
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

