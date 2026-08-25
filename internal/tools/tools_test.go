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

	require.Len(t, tools, 2)
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

func TestWebFetchToolDefinition(t *testing.T) {
	registry := NewRegistry(nil)
	tools := registry.ToolDefinitions()

	require.Len(t, tools, 2)
	tool := tools[1]
	assert.Equal(t, openai.ToolTypeFunction, tool.Type)
	require.NotNil(t, tool.Function)
	assert.Equal(t, "web_fetch", tool.Function.Name)
	assert.Contains(t, tool.Function.Description, "auto")
	assert.Contains(t, tool.Function.Description, "raw")
	assert.Contains(t, tool.Function.Description, "extract")

	paramBytes, err := json.Marshal(tool.Function.Parameters)
	require.NoError(t, err)

	var schema map[string]interface{}
	err = json.Unmarshal(paramBytes, &schema)
	require.NoError(t, err)
	assert.Equal(t, "object", schema["type"])
	props, ok := schema["properties"].(map[string]interface{})
	require.True(t, ok)
	assert.Contains(t, props, "url")
	assert.Contains(t, props, "mode")
	required, ok := schema["required"].([]interface{})
	require.True(t, ok)
	assert.Contains(t, required, "url")
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
	assert.Contains(t, resultObj["instruction"], "markdown links [Title](URL)")
	resultsArr, ok := resultObj["results"].([]interface{})
	require.True(t, ok)
	require.Len(t, resultsArr, 1)
}

func TestExecuteWebFetchAutoStaticSuccess(t *testing.T) {
	articleHTML := `<!DOCTYPE html><html><head><title>Go Concurrency</title></head><body><article><h1>Go Concurrency</h1><p>Go is an open source programming language that makes it easy to build simple, reliable, and efficient software. Goroutines are lightweight threads managed by the Go runtime. They allow functions to run asynchronously and communicate with other routines via typed channels. Concurrency in Go differs substantially from traditional thread-based paradigms.</p></article></body></html>`

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		_, _ = w.Write([]byte(articleHTML))
	}))
	defer ts.Close()

	registry := NewRegistry(nil)
	args, err := json.Marshal(WebFetchArgs{URL: ts.URL, Mode: "auto"})
	require.NoError(t, err)

	resultStr, err := registry.Execute(context.Background(), "web_fetch", string(args))
	require.NoError(t, err)

	var resultObj map[string]interface{}
	err = json.Unmarshal([]byte(resultStr), &resultObj)
	require.NoError(t, err)
	assert.Equal(t, "auto", resultObj["mode"])
	assert.Equal(t, "Go Concurrency", resultObj["title"])
	assert.Contains(t, resultObj["content"], "Goroutines are lightweight threads")
}

func TestExecuteWebFetchAutoDynamicFallbackToTavily(t *testing.T) {
	spaHTML := `<!DOCTYPE html><html><head><title>SPA</title></head><body><div id="root"></div><script src="/bundle.js"></script></body></html>`

	webServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html")
		_, _ = w.Write([]byte(spaHTML))
	}))
	defer webServer.Close()

	tavilyServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/extract", r.URL.Path)
		resp := tavily.ExtractResponse{
			Results: []tavily.ExtractResult{
				{
					URL:        webServer.URL,
					RawContent: "# Dynamic SPA Content\nExtracted by Tavily extract API successfully.",
				},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer tavilyServer.Close()

	tavilyClient := tavily.NewClient("test-key", tavilyServer.URL, tavilyServer.Client())
	registry := NewRegistry(tavilyClient)

	args, err := json.Marshal(WebFetchArgs{URL: webServer.URL, Mode: "auto"})
	require.NoError(t, err)

	resultStr, err := registry.Execute(context.Background(), "web_fetch", string(args))
	require.NoError(t, err)

	var resultObj map[string]interface{}
	err = json.Unmarshal([]byte(resultStr), &resultObj)
	require.NoError(t, err)
	assert.Equal(t, "auto", resultObj["mode"])
	assert.Equal(t, "tavily_extract", resultObj["source"])
	assert.Contains(t, resultObj["content"], "Dynamic SPA Content")
}

func TestExecuteWebFetchAutoDynamicFallbackDirect(t *testing.T) {
	spaHTML := `<!DOCTYPE html><html><head><title>SPA</title></head><body><div id="root"></div><script src="/bundle.js"></script></body></html>`

	webServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html")
		_, _ = w.Write([]byte(spaHTML))
	}))
	defer webServer.Close()

	// Registry without Tavily client
	registry := NewRegistry(nil)

	args, err := json.Marshal(WebFetchArgs{URL: webServer.URL, Mode: "auto"})
	require.NoError(t, err)

	resultStr, err := registry.Execute(context.Background(), "web_fetch", string(args))
	require.NoError(t, err)

	var resultObj map[string]interface{}
	err = json.Unmarshal([]byte(resultStr), &resultObj)
	require.NoError(t, err)
	assert.Equal(t, "auto", resultObj["mode"])
	assert.Equal(t, "direct_html_fallback", resultObj["source"])
	assert.Contains(t, resultObj["content"], "[Note: Dynamic rendering fallback could not be executed")
}

func TestExecuteWebFetchRawMode(t *testing.T) {
	codeContent := `package main\n\nfunc main() {\n\tprintln("Hello")\n}`

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/x-go")
		_, _ = w.Write([]byte(codeContent))
	}))
	defer ts.Close()

	registry := NewRegistry(nil)
	args, err := json.Marshal(WebFetchArgs{URL: ts.URL, Mode: "raw"})
	require.NoError(t, err)

	resultStr, err := registry.Execute(context.Background(), "web_fetch", string(args))
	require.NoError(t, err)

	var resultObj map[string]interface{}
	err = json.Unmarshal([]byte(resultStr), &resultObj)
	require.NoError(t, err)
	assert.Equal(t, "raw", resultObj["mode"])
	assert.Equal(t, "text/x-go", resultObj["content_type"])
	assert.Equal(t, codeContent, resultObj["content"])
}

func TestExecuteWebFetchExtractMode(t *testing.T) {
	tavilyServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/extract", r.URL.Path)
		resp := tavily.ExtractResponse{
			Results: []tavily.ExtractResult{
				{
					URL:        "https://example.com/app",
					RawContent: "# Direct Extract\nRaw markdown extracted content.",
				},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer tavilyServer.Close()

	tavilyClient := tavily.NewClient("test-key", tavilyServer.URL, tavilyServer.Client())
	registry := NewRegistry(tavilyClient)

	args, err := json.Marshal(WebFetchArgs{URL: "https://example.com/app", Mode: "extract"})
	require.NoError(t, err)

	resultStr, err := registry.Execute(context.Background(), "web_fetch", string(args))
	require.NoError(t, err)

	var resultObj map[string]interface{}
	err = json.Unmarshal([]byte(resultStr), &resultObj)
	require.NoError(t, err)
	assert.Equal(t, "extract", resultObj["mode"])
	assert.Equal(t, "# Direct Extract\nRaw markdown extracted content.", resultObj["content"])
}

func TestExecuteWebFetchBinary(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/pdf")
		_, _ = w.Write([]byte("%PDF-1.4 binary data"))
	}))
	defer ts.Close()

	registry := NewRegistry(nil)
	args, err := json.Marshal(WebFetchArgs{URL: ts.URL, Mode: "auto"})
	require.NoError(t, err)

	resultStr, err := registry.Execute(context.Background(), "web_fetch", string(args))
	require.NoError(t, err)

	var resultObj map[string]interface{}
	err = json.Unmarshal([]byte(resultStr), &resultObj)
	require.NoError(t, err)
	assert.Equal(t, "auto", resultObj["mode"])
	assert.Equal(t, "application/pdf", resultObj["content_type"])
	assert.Contains(t, resultObj["content"], "Binary file of type application/pdf")
}

func TestExecuteWebFetchValidation(t *testing.T) {
	registry := NewRegistry(nil)

	// Empty URL
	_, err := registry.Execute(context.Background(), "web_fetch", `{"url": ""}`)
	assert.ErrorContains(t, err, "url cannot be empty")

	// Invalid Mode
	_, err = registry.Execute(context.Background(), "web_fetch", `{"url": "https://example.com", "mode": "invalid_mode"}`)
	assert.ErrorContains(t, err, "invalid mode \"invalid_mode\"")

	// Extract mode without Tavily client
	_, err = registry.Execute(context.Background(), "web_fetch", `{"url": "https://example.com", "mode": "extract"}`)
	assert.ErrorContains(t, err, "tavily client is not configured for extract mode")
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

