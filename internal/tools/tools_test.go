package tools

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"bob/internal/config"
	"bob/internal/memory"
	"bob/internal/sandbox"
	"bob/internal/tools/tavily"

	openai "github.com/sashabaranov/go-openai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestWebSearchToolDefinition(t *testing.T) {
	registry := NewRegistry(nil, nil)
	tools := registry.ToolDefinitions()

	require.Len(t, tools, 3)
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
	registry := NewRegistry(nil, nil)
	tools := registry.ToolDefinitions()

	require.Len(t, tools, 3)
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

func TestRecallMemoryToolDefinition(t *testing.T) {
	registry := NewRegistry(nil, nil)
	tools := registry.ToolDefinitions()

	require.Len(t, tools, 3)
	tool := tools[2]
	assert.Equal(t, openai.ToolTypeFunction, tool.Type)
	require.NotNil(t, tool.Function)
	assert.Equal(t, "recall_memory", tool.Function.Name)
	assert.Contains(t, tool.Function.Description, "memory")
	assert.Contains(t, tool.Function.Description, "[Model-Derived Episode Summary | seq <start>-<end>]")
	assert.Contains(t, tool.Function.Description, "chunk_id")

	paramBytes, err := json.Marshal(tool.Function.Parameters)
	require.NoError(t, err)

	var schema map[string]interface{}
	err = json.Unmarshal(paramBytes, &schema)
	require.NoError(t, err)
	assert.Equal(t, "object", schema["type"])
	props, ok := schema["properties"].(map[string]interface{})
	require.True(t, ok)
	assert.Contains(t, props, "query")
	assert.Contains(t, props, "chunk_id")
	assert.Contains(t, props, "start_seq")
	assert.Contains(t, props, "end_seq")
	assert.Contains(t, props, "limit")
	_, hasRequired := schema["required"]
	assert.False(t, hasRequired, "query should not be strictly required when chunk_id or sequence bounds are provided")
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
	registry := NewRegistry(tavilyClient, nil)

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

	registry := NewRegistry(nil, nil)
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
	registry := NewRegistry(tavilyClient, nil)

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
	registry := NewRegistry(nil, nil)

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

	registry := NewRegistry(nil, nil)
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
	registry := NewRegistry(tavilyClient, nil)

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

	registry := NewRegistry(nil, nil)
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
	registry := NewRegistry(nil, nil)

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
	registry := NewRegistry(nil, nil)
	_, err := registry.Execute(context.Background(), "unknown_tool", `{}`)
	assert.ErrorContains(t, err, "unknown tool: unknown_tool")
}

func TestExecuteInvalidJSONArgs(t *testing.T) {
	tavilyClient := tavily.NewClient("test-key", "https://api.tavily.com", nil)
	registry := NewRegistry(tavilyClient, nil)

	_, err := registry.Execute(context.Background(), "web_search", `invalid-json`)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to parse arguments")
}

func TestExecuteEmptyQuery(t *testing.T) {
	tavilyClient := tavily.NewClient("test-key", "https://api.tavily.com", nil)
	registry := NewRegistry(tavilyClient, nil)

	_, err := registry.Execute(context.Background(), "web_search", `{"query": ""}`)
	assert.ErrorContains(t, err, "query cannot be empty")
}

func TestExecuteRecallMemory_Success(t *testing.T) {
	tempDir := t.TempDir()

	cfg := &config.Config{
		DataDir:        tempDir,
		EmbeddingModel: "none",
	}
	memMgr := memory.NewManager(cfg, nil)
	defer func() { _ = memMgr.Close() }()

	ctx := context.Background()
	now := time.Now().Unix()

	// Index townhall message
	err := memMgr.IndexMessages(ctx, "townhall", false, []memory.MessageToStore{
		{
			Seq:        1,
			Timestamp:  now,
			ChatID:     "townhall",
			UserID:     "alice",
			SenderName: "Alice",
			Role:       "user",
			Content:    "We agreed to use Postgres 16 database for all production clusters.",
		},
	})
	require.NoError(t, err)

	// Index private DM message
	err = memMgr.IndexMessages(ctx, "dm_user1", true, []memory.MessageToStore{
		{
			Seq:        10,
			Timestamp:  now + 1,
			ChatID:     "dm_user1",
			UserID:     "user1",
			SenderName: "User One",
			Role:       "user",
			Content:    "My personal favorite database is SQLite with WAL mode.",
		},
	})
	require.NoError(t, err)

	registry := NewRegistry(nil, memMgr)

	// 1. Query in Townhall context
	thCtx := WithChatSession(ctx, ChatSessionContext{ChatID: "townhall", IsDM: false})
	thRes, err := registry.Execute(thCtx, "recall_memory", `{"query": "Postgres", "limit": 3}`)
	require.NoError(t, err)

	var thObj map[string]interface{}
	err = json.Unmarshal([]byte(thRes), &thObj)
	require.NoError(t, err)
	assert.Equal(t, float64(1), thObj["count"])
	assert.Contains(t, thObj["formatted_results"], "[Townhall]")
	assert.Contains(t, thObj["formatted_results"], "Postgres 16")

	// 2. Query in DM context
	dmCtx := WithChatSession(ctx, ChatSessionContext{ChatID: "dm_user1", IsDM: true})
	dmRes, err := registry.Execute(dmCtx, "recall_memory", `{"query": "database", "limit": 5}`)
	require.NoError(t, err)

	var dmObj map[string]interface{}
	err = json.Unmarshal([]byte(dmRes), &dmObj)
	require.NoError(t, err)
	assert.Equal(t, float64(2), dmObj["count"])
	assert.Contains(t, dmObj["formatted_results"], "[Direct Message]")
	assert.Contains(t, dmObj["formatted_results"], "SQLite with WAL")
	assert.Contains(t, dmObj["formatted_results"], "[Townhall]")
}

func TestExecuteRecallMemory_EmptyResults(t *testing.T) {
	tempDir := t.TempDir()

	cfg := &config.Config{
		DataDir:        tempDir,
		EmbeddingModel: "none",
	}
	memMgr := memory.NewManager(cfg, nil)
	defer func() { _ = memMgr.Close() }()

	registry := NewRegistry(nil, memMgr)
	res, err := registry.Execute(context.Background(), "recall_memory", `{"query": "nonexistent"}`)
	require.NoError(t, err)

	var obj map[string]interface{}
	err = json.Unmarshal([]byte(res), &obj)
	require.NoError(t, err)
	assert.Equal(t, float64(0), obj["count"])
	assert.Contains(t, obj["message"], "No relevant memories found")
}

func TestExecuteRecallMemory_NoMemoryManager(t *testing.T) {
	registry := NewRegistry(nil, nil)
	_, err := registry.Execute(context.Background(), "recall_memory", `{"query": "hello"}`)
	assert.ErrorContains(t, err, "memory manager is not configured")
}

func TestExecuteRecallMemory_InvalidArgs(t *testing.T) {
	tempDir := t.TempDir()

	memMgr := memory.NewManager(&config.Config{DataDir: tempDir, EmbeddingModel: "none"}, nil)
	defer func() { _ = memMgr.Close() }()

	registry := NewRegistry(nil, memMgr)

	_, err := registry.Execute(context.Background(), "recall_memory", `invalid-json`)
	assert.ErrorContains(t, err, "failed to parse arguments")

	_, err = registry.Execute(context.Background(), "recall_memory", `{"query": ""}`)
	assert.ErrorContains(t, err, "query cannot be empty")
}

func TestExecuteRecallMemory_EpisodesAndRawChunkDrillDown(t *testing.T) {
	tempDir := t.TempDir()
	memMgr := memory.NewManager(&config.Config{DataDir: tempDir, EmbeddingModel: "none"}, nil)
	defer func() { _ = memMgr.Close() }()

	ctx := context.Background()
	now := time.Now().Unix()

	require.NoError(t, memMgr.IndexMessages(ctx, "townhall", false, []memory.MessageToStore{
		{
			Seq:        42,
			Timestamp:  now,
			ChatID:     "townhall",
			UserID:     "alice",
			SenderName: "Alice",
			Role:       "user",
			Content:    "Deployed envoy canary proxy to region eu-central-1.",
		},
		{
			Seq:        43,
			Timestamp:  now + 2,
			ChatID:     "townhall",
			UserID:     "bob",
			SenderName: "Bob",
			Role:       "user",
			Content:    "Envoy canary rollout completed with zero 5xx errors.",
		},
	}))

	registry := NewRegistry(nil, memMgr)
	thCtx := WithChatSession(ctx, ChatSessionContext{ChatID: "townhall", IsDM: false})

	// 1. Search returns raw chunk labeled as Raw Conversation Chunk with episode_id reference (never mislabeled as Model-Derived Episode Summary)
	res, err := registry.Execute(thCtx, "recall_memory", `{"query": "envoy"}`)
	require.NoError(t, err)
	assert.Contains(t, res, "[Raw Conversation Chunk | chunk_id: chunk_townhall_42_43 | episode_id: ep_townhall_42_43]")
	assert.NotContains(t, res, "[Model-Derived Episode Summary | seq 42-43]")

	// 1b. Querying episode ID directly returns standalone episode summary with complete mandatory header and pointers
	epRes, err := registry.Execute(thCtx, "recall_memory", `{"query": "ep_townhall_42_43"}`)
	require.NoError(t, err)
	assert.Contains(t, epRes, "[Model-Derived Episode Summary | seq 42-43]")
	assert.Contains(t, epRes, "Source Chunks: chunk_townhall_42_43")
	assert.Contains(t, epRes, "Navigation summary only")

	// 2. Drill down into raw chunk by chunk_id without query
	chunkRes, err := registry.Execute(thCtx, "recall_memory", `{"chunk_id": "chunk_townhall_42_43"}`)
	require.NoError(t, err)
	assert.Contains(t, chunkRes, "Deployed envoy canary proxy")
	assert.Contains(t, chunkRes, "zero 5xx errors")

	// 3. Drill down into raw chunk by start_seq and end_seq
	seqRes, err := registry.Execute(thCtx, "recall_memory", `{"start_seq": 40, "end_seq": 45}`)
	require.NoError(t, err)
	assert.Contains(t, seqRes, "Deployed envoy canary proxy")

	// 4. Verify bounded <= 16 KiB response when multiple large chunks and oversized participant metadata match
	largePayload := strings.Repeat("envoy proxy configuration line with utf8 🚀\n", 140)
	oversizedSender := "Alice_" + strings.Repeat("P", 24*1024)
	for batchIdx := 0; batchIdx < 5; batchIdx++ {
		seq := int64(200 + batchIdx*10)
		require.NoError(t, memMgr.IndexMessages(ctx, "townhall", false, []memory.MessageToStore{
			{
				Seq:        seq,
				Timestamp:  now + int64(batchIdx+10),
				ChatID:     "townhall",
				UserID:     "alice",
				SenderName: oversizedSender,
				Role:       "user",
				Content:    largePayload,
			},
		}))
	}

	boundedRes, err := registry.Execute(thCtx, "recall_memory", `{"query": "envoy", "limit": 10}`)
	require.NoError(t, err)
	assert.LessOrEqual(t, len(boundedRes), MaxRecallMemoryResponseBytes)

	var boundedObj map[string]interface{}
	require.NoError(t, json.Unmarshal([]byte(boundedRes), &boundedObj))
	assert.NotEmpty(t, boundedObj["memories"])
	assert.Contains(t, boundedObj["formatted_results"], "envoy")

	// 5. Verify crowded response with an episode item preserves complete trust label or omits episode
	crowdedEpRes, err := registry.Execute(thCtx, "recall_memory", `{"query": "ep_townhall_200_200", "limit": 10}`)
	require.NoError(t, err)
	assert.LessOrEqual(t, len(crowdedEpRes), MaxRecallMemoryResponseBytes)
	if strings.Contains(crowdedEpRes, "Model-Derived Episode Summary") {
		assert.Contains(t, crowdedEpRes, "[Model-Derived Episode Summary | seq 200-200]")
		assert.Contains(t, crowdedEpRes, "Navigation summary only")
	}

	// 6. Long query with no matches is also bounded <= 16 KiB
	hugeQueryArgs, err := json.Marshal(RecallMemoryArgs{Query: "nonexistent_" + strings.Repeat("Q", 24*1024)})
	require.NoError(t, err)
	emptyBoundedRes, err := registry.Execute(thCtx, "recall_memory", string(hugeQueryArgs))
	require.NoError(t, err)
	assert.LessOrEqual(t, len(emptyBoundedRes), MaxRecallMemoryResponseBytes)

	// 7. Regression: long chunk_id / episode_id (> 64 bytes) under response pressure must preserve exact lookup IDs and drill down cleanly
	longChatID := "dm_" + strings.Repeat("organization_tenant_workspace_", 3) + "user42"
	require.Greater(t, len(longChatID), 64)
	longDMCtx := WithChatSession(ctx, ChatSessionContext{ChatID: longChatID, UserID: "user42", IsDM: true})
	for i := 0; i < 4; i++ {
		seq := int64(500 + i*10)
		require.NoError(t, memMgr.IndexMessages(ctx, longChatID, true, []memory.MessageToStore{
			{
				Seq:        seq,
				Timestamp:  now + int64(100+i),
				ChatID:     longChatID,
				UserID:     "user42",
				SenderName: "User42",
				Role:       "user",
				Content:    strings.Repeat("long_pointer_verification_payload ", 160),
			},
		}))
	}

	longRes, err := registry.Execute(longDMCtx, "recall_memory", `{"query": "long_pointer_verification_payload", "limit": 10}`)
	require.NoError(t, err)
	assert.LessOrEqual(t, len(longRes), MaxRecallMemoryResponseBytes)

	var longPayload struct {
		Memories []memory.MemoryItem `json:"memories"`
	}
	require.NoError(t, json.Unmarshal([]byte(longRes), &longPayload))
	require.NotEmpty(t, longPayload.Memories)
	for _, memItem := range longPayload.Memories {
		assert.False(t, strings.HasSuffix(memItem.ID, "..."), "item ID must never be truncated")
		assert.False(t, strings.HasSuffix(memItem.EpisodeID, "..."), "episode ID must never be truncated")
		for _, srcChunkID := range memItem.SourceChunkIDs {
			assert.Greater(t, len(srcChunkID), 64)
			assert.False(t, strings.HasSuffix(srcChunkID, "..."), "source_chunk_id pointer must never be truncated")
			drillArgs, err := json.Marshal(RecallMemoryArgs{ChunkID: srcChunkID})
			require.NoError(t, err)
			drillRes, err := registry.Execute(longDMCtx, "recall_memory", string(drillArgs))
			require.NoError(t, err)
			assert.Contains(t, drillRes, "long_pointer_verification_payload")
		}
	}
}

type mockSandboxDriver struct {
	available bool
}

func (m *mockSandboxDriver) Type() sandbox.DriverType {
	return sandbox.DriverBwrap
}

func (m *mockSandboxDriver) Available(ctx context.Context) bool {
	return m.available
}

func (m *mockSandboxDriver) Create(ctx context.Context, sbx *sandbox.UserSandbox, userWorkspaceDir string) error {
	return nil
}

func (m *mockSandboxDriver) Exec(ctx context.Context, sbx *sandbox.UserSandbox, cmd []string, timeout time.Duration) (*sandbox.ExecResult, error) {
	return &sandbox.ExecResult{
		ExitCode: 0,
		Stdout:   "sandbox exec result",
		Duration: 50 * time.Millisecond,
	}, nil
}

func (m *mockSandboxDriver) Destroy(ctx context.Context, sbx *sandbox.UserSandbox) error {
	return nil
}

func TestSandboxToolDefinitionsScope(t *testing.T) {
	tempDir := t.TempDir()
	cfg := &config.Config{
		DataDir:                   tempDir,
		SandboxEnabled:            true,
		SandboxDrivers:            []string{"bwrap"},
		SandboxMaxLifetime:        30 * time.Minute,
		SandboxDefaultExecTimeout: 1 * time.Minute,
		SandboxMaxExecTimeout:     10 * time.Minute,
	}

	mockDriver := &mockSandboxDriver{available: true}
	sandboxMgr := sandbox.NewManager(cfg.SandboxConfig(), []sandbox.Driver{mockDriver})
	defer func() { _ = sandboxMgr.Close() }()

	reg := NewRegistry(nil, nil, sandboxMgr)

	// Base ToolDefinitions (or Townhall session) has 3 tools (search, fetch, recall)
	assert.Len(t, reg.ToolDefinitions(), 3)
	assert.Len(t, reg.ToolDefinitionsForSession(ChatSessionContext{IsDM: false}), 3)

	thNames := make([]string, len(reg.ToolDefinitions()))
	for i, tool := range reg.ToolDefinitions() {
		thNames[i] = tool.Function.Name
	}
	assert.NotContains(t, thNames, "schedule_task")
	assert.NotContains(t, thNames, "list_schedules")
	assert.NotContains(t, thNames, "cancel_schedule")
	assert.NotContains(t, thNames, "sandbox_request")
	assert.NotContains(t, thNames, "sandbox_download_attachment")
	assert.NotContains(t, thNames, "sandbox_upload_attachment")
	assert.NotContains(t, thNames, "propose_memory")
	assert.NotContains(t, thNames, "discover_memories")
	assert.NotContains(t, thNames, "load_memory")
	assert.NotContains(t, thNames, "propose_skill")
	assert.NotContains(t, thNames, "discover_skills")
	assert.NotContains(t, thNames, "load_skill")

	// DM session has 15 tools (3 base + 3 scheduler + 3 sandbox + 6 knowledge) when no sandbox running
	dmTools := reg.ToolDefinitionsForSession(ChatSessionContext{IsDM: true})
	assert.Len(t, dmTools, 15)

	names := make([]string, len(dmTools))
	for i, tool := range dmTools {
		names[i] = tool.Function.Name
	}
	assert.Contains(t, names, "schedule_task")
	assert.Contains(t, names, "list_schedules")
	assert.Contains(t, names, "cancel_schedule")
	assert.Contains(t, names, "sandbox_request")
	assert.Contains(t, names, "sandbox_exec")
	assert.Contains(t, names, "sandbox_destroy")
	assert.Contains(t, names, "propose_memory")
	assert.Contains(t, names, "discover_memories")
	assert.Contains(t, names, "load_memory")
	assert.Contains(t, names, "propose_skill")
	assert.Contains(t, names, "discover_skills")
	assert.Contains(t, names, "load_skill")
	assert.NotContains(t, names, "sandbox_download_attachment")
	assert.NotContains(t, names, "sandbox_upload_attachment")

	// When user has an active running sandbox, sandbox_request is filtered out and attachment tools are added
	_, err := sandboxMgr.RequestSandbox(context.Background(), "active_user", "chat_dm", sandbox.RequestParams{
		Driver:      sandbox.DriverBwrap,
		NetworkMode: sandbox.NetworkNone,
	})
	require.NoError(t, err)
	_, err = sandboxMgr.ApproveSandbox(context.Background(), "active_user")
	require.NoError(t, err)

	activeDMTools := reg.ToolDefinitionsForSession(ChatSessionContext{IsDM: true, UserID: "active_user"})
	assert.Equal(t, len(dmTools)+1, len(activeDMTools))

	activeNames := make([]string, len(activeDMTools))
	for i, tool := range activeDMTools {
		activeNames[i] = tool.Function.Name
	}
	assert.NotContains(t, activeNames, "sandbox_request")
	assert.Contains(t, activeNames, "sandbox_exec")
	assert.Contains(t, activeNames, "sandbox_destroy")
	assert.Contains(t, activeNames, "sandbox_download_attachment")
	assert.Contains(t, activeNames, "sandbox_upload_attachment")

	// ToolDefinitionsForChat preserves UserID when passed in context (e.g. from persistent FSM runCtx)
	runCtx := WithChatSession(context.Background(), ChatSessionContext{
		ChatID: "chat_dm",
		UserID: "active_user",
		IsDM:   true,
	})
	fsmTools := reg.ToolDefinitionsForChat(runCtx, "chat_dm", true)
	fsmNames := make([]string, len(fsmTools))
	for i, tool := range fsmTools {
		fsmNames[i] = tool.Function.Name
	}
	assert.NotContains(t, fsmNames, "sandbox_request")
	assert.Contains(t, fsmNames, "sandbox_exec")
	assert.Contains(t, fsmNames, "sandbox_download_attachment")
	assert.Contains(t, fsmNames, "sandbox_upload_attachment")

	// Another user without active sandbox should still have sandbox_request available and no attachment tools
	otherUserTools := reg.ToolDefinitionsForSession(ChatSessionContext{IsDM: true, UserID: "other_user"})
	otherNames := make([]string, len(otherUserTools))
	for i, tool := range otherUserTools {
		otherNames[i] = tool.Function.Name
	}
	assert.Contains(t, otherNames, "sandbox_request")
	assert.NotContains(t, otherNames, "sandbox_download_attachment")
	assert.NotContains(t, otherNames, "sandbox_upload_attachment")
}

func TestSandboxToolExecution(t *testing.T) {
	tempDir := t.TempDir()
	cfg := &config.Config{
		DataDir:                   tempDir,
		SandboxEnabled:            true,
		SandboxDrivers:            []string{"bwrap"},
		SandboxMaxLifetime:        30 * time.Minute,
		SandboxDefaultExecTimeout: 1 * time.Minute,
		SandboxMaxExecTimeout:     10 * time.Minute,
	}

	mockDriver := &mockSandboxDriver{available: true}
	sandboxMgr := sandbox.NewManager(cfg.SandboxConfig(), []sandbox.Driver{mockDriver})
	defer func() { _ = sandboxMgr.Close() }()

	reg := NewRegistry(nil, nil, sandboxMgr)
	ctx := context.Background()

	// 1. Calling from Townhall (non-DM) must fail
	thCtx := WithChatSession(ctx, ChatSessionContext{
		ChatID: "townhall",
		UserID: "user1",
		IsDM:   false,
	})
	_, err := reg.Execute(thCtx, "sandbox_request", `{"driver":"bwrap","reason":"test"}`)
	assert.ErrorContains(t, err, "strictly available in 1-on-1 direct messages")

	// 2. Calling from DM with UserID
	var sbxRequested bool
	dmCtx := WithChatSession(ctx, ChatSessionContext{
		ChatID:                "dm_user1",
		UserID:                "user1",
		IsDM:                  true,
		SandboxRequestCreated: &sbxRequested,
	})

	// Request sandbox
	out, err := reg.Execute(dmCtx, "sandbox_request", `{
		"driver": "bwrap",
		"network": "none",
		"reason": "Run tests and scripts"
	}`)
	require.NoError(t, err)
	assert.True(t, sbxRequested)
	assert.Contains(t, out, "Sandbox request created and approval card has already been sent")
	assert.Contains(t, out, "DO NOT generate any conversational message")

	// Exec before approval fails
	_, err = reg.Execute(dmCtx, "sandbox_exec", `{"command": "echo 1"}`)
	assert.ErrorContains(t, err, "no active sandbox found")

	// Approve sandbox out-of-band
	_, err = sandboxMgr.ApproveSandbox(ctx, "user1")
	require.NoError(t, err)

	// Exec after approval succeeds
	execOut, err := reg.Execute(dmCtx, "sandbox_exec", `{"command": "echo 1"}`)
	require.NoError(t, err)
	assert.Contains(t, execOut, "Exit Code: 0")
	assert.Contains(t, execOut, "sandbox exec result")

	// Destroy sandbox
	destroyOut, err := reg.Execute(dmCtx, "sandbox_destroy", `{}`)
	require.NoError(t, err)
	assert.Contains(t, destroyOut, "successfully destroyed")

	// Exec after destroy fails
	_, err = reg.Execute(dmCtx, "sandbox_exec", `{"command": "echo 1"}`)
	assert.ErrorContains(t, err, "no active sandbox found")
}

func TestExecuteSandboxRequest_NotifierError(t *testing.T) {
	tempDir := t.TempDir()
	cfg := &config.Config{
		DataDir:        tempDir,
		SandboxEnabled: true,
		SandboxDrivers: []string{"bwrap"},
	}
	mockDriver := &mockSandboxDriver{available: true}
	sandboxMgr := sandbox.NewManager(cfg.SandboxConfig(), []sandbox.Driver{mockDriver})
	defer func() { _ = sandboxMgr.Close() }()

	reg := NewRegistry(nil, nil, sandboxMgr)
	ctx := context.Background()

	var sbxRequested bool
	dmCtx := WithChatSession(ctx, ChatSessionContext{
		ChatID: "dm_user1",
		UserID: "user1",
		IsDM:   true,
		Notifier: func(chatID, text string) error {
			return errors.New("network disconnection")
		},
		SandboxRequestCreated: &sbxRequested,
	})

	_, err := reg.Execute(dmCtx, "sandbox_request", `{
		"driver": "bwrap",
		"network": "none",
		"reason": "Run tests"
	}`)
	assert.ErrorContains(t, err, "failed to send approval card")

	// Verify that the pending sandbox request was rolled back and is not stuck in PendingApproval
	status, exists := sandboxMgr.GetStatus("user1")
	assert.False(t, exists, "pending sandbox request must be rolled back on notifier error")
	assert.Nil(t, status)

	// User must be able to request a new sandbox immediately without being told "already have a pending request"
	dmCtxSuccess := WithChatSession(ctx, ChatSessionContext{
		ChatID: "dm_user1",
		UserID: "user1",
		IsDM:   true,
		Notifier: func(chatID, text string) error {
			return nil
		},
		SandboxRequestCreated: &sbxRequested,
	})
	_, err = reg.Execute(dmCtxSuccess, "sandbox_request", `{
		"driver": "bwrap",
		"network": "none",
		"reason": "Retry after failure"
	}`)
	assert.NoError(t, err, "subsequent request must succeed after rollback")
}

func TestSandboxToolDefinitions_Descriptions(t *testing.T) {
	tempDir := t.TempDir()
	cfg := &config.Config{
		DataDir:                    tempDir,
		SandboxEnabled:             true,
		SandboxDrivers:             []string{"bwrap"},
		SandboxAllowedNetworkModes: []string{"none", "restricted"},
		SandboxDefaultExecTimeout:  1 * time.Minute,
		SandboxMaxExecTimeout:      10 * time.Minute,
	}
	mockDriver := &mockSandboxDriver{available: true}
	sandboxMgr := sandbox.NewManager(cfg.SandboxConfig(), []sandbox.Driver{mockDriver})
	defer func() { _ = sandboxMgr.Close() }()

	reg := NewRegistry(nil, nil, sandboxMgr)
	tools := reg.ToolDefinitionsForSession(ChatSessionContext{IsDM: true, UserID: "user1"})

	var sbxRequest, sbxDestroy *openai.Tool
	for i := range tools {
		if tools[i].Function.Name == "sandbox_request" {
			sbxRequest = &tools[i]
		}
		if tools[i].Function.Name == "sandbox_destroy" {
			sbxDestroy = &tools[i]
		}
	}

	require.NotNil(t, sbxRequest)
	require.NotNil(t, sbxDestroy)

	// Verify sandbox_request description uses milder, task-based wording
	assert.Contains(t, sbxRequest.Function.Description, "appropriate permissions and allowed domains, keeping them minimal")
	assert.NotContains(t, sbxRequest.Function.Description, "least privilege")
	assert.NotContains(t, sbxRequest.Function.Description, "network: full")
	assert.NotContains(t, sbxRequest.Function.Description, "network:full")

	// Verify parameter schema descriptions
	paramsMap, ok := sbxRequest.Function.Parameters.(map[string]interface{})
	require.True(t, ok)
	props, ok := paramsMap["properties"].(map[string]interface{})
	require.True(t, ok)

	netProp, ok := props["network"].(map[string]interface{})
	require.True(t, ok)
	assert.Contains(t, netProp["description"], "Request minimal permissions needed to achieve the task.")
	assert.NotContains(t, netProp["description"], "least privilege")

	domProp, ok := props["domains"].(map[string]interface{})
	require.True(t, ok)
	assert.Contains(t, domProp["description"], "If internet access for dependencies or APIs is required, provide the specific domains needed.")

	// Verify sandbox_destroy description includes reconfiguration guidance
	assert.Contains(t, sbxDestroy.Function.Description, "permissions/domains need to be reconfigured with sandbox_request")
}
