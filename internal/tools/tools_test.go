package tools

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"bob/internal/tools/tavily"

	openai "github.com/sashabaranov/go-openai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestWebSearchToolDefinition(t *testing.T) {
	registry := NewRegistry(nil)
	tools := registry.ToolDefinitions()

	require.Len(t, tools, 1)
	tool := tools[0]
	assert.Equal(t, openai.ToolTypeFunction, tool.Type)
	require.NotNil(t, tool.Function)
	assert.Equal(t, "web_search", tool.Function.Name)
	assert.NotEmpty(t, tool.Function.Description)

	// Verify parameter schema is valid JSON Schema
	paramBytes, err := json.Marshal(tool.Function.Parameters)
	require.NoError(t, err)

	var schema map[string]interface{}
	err = json.Unmarshal(paramBytes, &schema)
	require.NoError(t, err)
	assert.Equal(t, "object", schema["type"])
	props, ok := schema["properties"].(map[string]interface{})
	require.True(t, ok)
	assert.Contains(t, props, "query")
	assert.Contains(t, props, "search_depth")
	assert.Contains(t, props, "max_results")
	required, ok := schema["required"].([]interface{})
	require.True(t, ok)
	assert.Contains(t, required, "query")
}

func TestExecuteWebSearchSuccess(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := tavily.SearchResponse{
			Query:  "golang news",
			Answer: "Go 1.26 is the latest version.",
			Results: []tavily.SearchResult{
				{
					Title:   "Go News",
					URL:     "https://go.dev/blog",
					Content: "Go 1.26 release notes...",
				},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer ts.Close()

	tavilyClient := tavily.NewClient("test-key", ts.URL, ts.Client())
	registry := NewRegistry(tavilyClient)

	args := `{"query": "golang news", "search_depth": "basic", "max_results": 3}`
	resultStr, err := registry.Execute(context.Background(), "web_search", args)
	require.NoError(t, err)
	assert.NotEmpty(t, resultStr)

	var resultObj map[string]interface{}
	err = json.Unmarshal([]byte(resultStr), &resultObj)
	require.NoError(t, err)
	assert.Equal(t, "golang news", resultObj["query"])
	assert.Equal(t, "Go 1.26 is the latest version.", resultObj["answer"])
	resultsArr, ok := resultObj["results"].([]interface{})
	require.True(t, ok)
	require.Len(t, resultsArr, 1)
}

func TestExecuteUnknownTool(t *testing.T) {
	registry := NewRegistry(nil)
	_, err := registry.Execute(context.Background(), "unknown_tool", `{}`)
	assert.ErrorContains(t, err, "unknown tool: unknown_tool")
}

func TestExecuteInvalidJSONArgs(t *testing.T) {
	tavilyClient := tavily.NewClient("test-key", "https://api.tavily.com", nil)
	registry := NewRegistry(tavilyClient)

	_, err := registry.Execute(context.Background(), "web_search", `invalid-json`)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to parse arguments")
}

func TestExecuteEmptyQuery(t *testing.T) {
	tavilyClient := tavily.NewClient("test-key", "https://api.tavily.com", nil)
	registry := NewRegistry(tavilyClient)

	_, err := registry.Execute(context.Background(), "web_search", `{"query": ""}`)
	assert.ErrorContains(t, err, "query cannot be empty")
}
