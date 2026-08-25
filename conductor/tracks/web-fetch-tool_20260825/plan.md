# Implementation Plan: Web Page Fetch Tool Support (`web_fetch`)

## Phase 1: Dependencies & Tavily Extract Client
- [ ] Task: Add `go-readability` dependency & implement Tavily Extract API in `internal/tools/tavily`
    - [ ] Add `github.com/go-shiori/go-readability` to `go.mod` and run `go mod tidy` and `go mod vendor`
    - [ ] Define Extract data models (`ExtractRequest`, `ExtractResponse`, `ExtractResult`) in `internal/tools/tavily/models.go`
    - [ ] Write unit tests for `Extract(ctx, urls...)` in `internal/tools/tavily/client_test.go` covering success, empty results, 429/5xx retries, and errors
    - [ ] Implement `Extract` method on `Client` in `internal/tools/tavily/client.go`
    - [ ] Run `go test -race ./internal/tools/tavily/...` and verify tests pass

## Phase 2: Web Fetcher Core, Heuristic Quality Engine & Realistic Test Corpus
- [ ] Task: Implement HTTP fetcher, readability parser, and test corpus with dynamic fallback heuristics in `internal/tools/webfetch`
    - [ ] Curate 30 realistic HTML test fixtures in `internal/tools/webfetch/testdata/`:
        - 10 static HTML sites (Wikipedia, blogs, documentation) where readability succeeds directly
        - 10 dynamic client-rendered SPA sites (React, Vue, Angular shells with `<div id="root"></div>` / loading states)
        - 10 bot protection / captcha challenge pages (Cloudflare "Just a moment", DataDome, PerimeterX, Access Denied, etc.)
    - [ ] Write unit tests for Content-Type detection, raw body fetching, and 16KB truncation in `internal/tools/webfetch/fetcher_test.go`
    - [ ] Write unit tests for `go-readability` article extraction and heuristic evaluation against the 30-case fixture dataset in `internal/tools/webfetch/heuristic_test.go`
    - [ ] Implement direct HTTP fetching with timeouts, redirects, and 16KB truncation in `internal/tools/webfetch/fetcher.go`
    - [ ] Implement readability parsing and fallback heuristics (`needsDynamicFallback`) in `internal/tools/webfetch/readability.go`
    - [ ] Run `go test -race ./internal/tools/webfetch/...` and verify all fixture tests pass

## Phase 3: Tool Definition, Mode Handling, & Registry Dispatch
- [ ] Task: Implement `web_fetch` tool schema and dispatch in `internal/tools`
    - [ ] Write unit tests in `internal/tools/tools_test.go` verifying `web_fetch` schema, `mode` parameter parsing (`auto`, `raw`, `extract`), and execution paths
    - [ ] Update `internal/tools/registry.go` with `web_fetch` tool definition (including detailed description of `auto`, `raw`, and `extract` modes and 16KB truncation)
    - [ ] Implement `executeWebFetch` in `internal/tools/registry.go` connecting direct fetcher, heuristics, and Tavily Extract fallback
    - [ ] Run `go test -race ./internal/tools/...` and verify tests pass

## Phase 4: Gateway & LLM Tool Loop Integration
- [ ] Task: Integrate `web_fetch` into agent gateway and verify multi-tool execution
    - [ ] Update `internal/gateway/gateway.go` to ensure `web_fetch` tool is available in tool registry
    - [ ] Write integration tests in `internal/gateway/gateway_test.go` and `internal/llm/client_test.go` verifying end-to-end `web_fetch` execution in LLM completions
    - [ ] Run `go test -race ./internal/gateway/...` and `go test -race ./internal/llm/...`

## Phase 5: Verification & Quality Checks
- [ ] Task: Run full test suite and quality enforcement
    - [ ] Run `go test -race ./...` across all packages
    - [ ] Run `make check` (`golangci-lint`, `go test -race`, `semgrep`, `osv-scanner`)
    - [ ] Verify zero credential leakage and correct truncation annotations in tool outputs
