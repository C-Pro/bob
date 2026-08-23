package tools

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"bob/internal/tools/tavily"

	openai "github.com/sashabaranov/go-openai"
)

// Registry manages available LLM tool definitions and executes tool calls.
type Registry struct {
	tavilyClient    *tavily.Client
	toolDefinitions []openai.Tool
}

// NewRegistry creates a new tool registry and initializes static tool definitions once.
func NewRegistry(tavilyClient *tavily.Client) *Registry {
	r := &Registry{
		tavilyClient: tavilyClient,
	}
	r.initToolDefinitions()
	return r
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

	r.toolDefinitions = []openai.Tool{
		{
			Type: openai.ToolTypeFunction,
			Function: &openai.FunctionDefinition{
				Name:        "web_search",
				Description: "Search the live web for current information, news, documentation, and facts using Tavily search.",
				Parameters:  webSearchSchema,
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

// Execute dispatches a tool execution by function name and arguments JSON.
func (r *Registry) Execute(ctx context.Context, name string, argsJSON string) (string, error) {
	switch name {
	case "web_search":
		return r.executeWebSearch(ctx, argsJSON)
	default:
		return "", fmt.Errorf("unknown tool: %s", name)
	}
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

	respBytes, err := json.Marshal(resp)
	if err != nil {
		return "", fmt.Errorf("failed to format search response: %w", err)
	}

	return string(respBytes), nil
}
