# Implementation Plan: Web Search Tool Support (Tavily API)

## Phase 1: Configuration & Models
- [x] Task: Add Tavily configuration options in `internal/config`
    - [x] Write unit tests for `TAVILY_API_KEY` and `TAVILY_BASE_URL` parsing and validation in `internal/config/config_test.go`
    - [x] Implement `TavilyAPIKey` and `TavilyBaseURL` (defaulting to `https://api.tavily.com`) in `internal/config/config.go`
    - [x] Run `go test -race ./internal/config/...` and verify tests pass

## Phase 2: Tavily Search Client
- [x] Task: Implement native Go Tavily Search HTTP client in `internal/tools/tavily`
    - [x] Define Tavily data models (`SearchRequest`, `SearchResponse`, `SearchResult`) in `internal/tools/tavily/models.go`
    - [x] Write comprehensive unit tests in `internal/tools/tavily/client_test.go` using `httptest.Server` (success response, answer extraction, retries on 429/5xx, timeouts, error cases)
    - [x] Implement `Client` with `Search(ctx, req)` method, timeout management, and retry backoff in `internal/tools/tavily/client.go`
    - [x] Run `go test -race ./internal/tools/tavily/...` and verify all tests pass

## Phase 3: Tool Definition & Dispatcher
- [x] Task: Implement `web_search` tool definition and dispatcher in `internal/tools`
    - [x] Write unit tests in `internal/tools/tools_test.go` verifying OpenAI tool schema generation and JSON argument parsing
    - [x] Implement tool registry and `web_search` tool definition/executor in `internal/tools/search.go`
    - [x] Ensure tool definitions and system context precede ring buffer messages and remain independent of chat history
    - [x] Run `go test -race ./internal/tools/...` and verify all tests pass

## Phase 4: Multi-Turn LLM Tool Loop & Gateway Integration
- [x] Task: Implement multi-turn tool execution loop in `internal/llm` and wire into `internal/gateway`
    - [x] Write unit tests in `internal/llm/client_test.go` mocking OpenAI tool calls, multi-turn tool message appending, and loop termination
    - [x] Implement iterative tool loop handling (up to 5 iterations max) with error recovery and text synthesis
    - [x] Update `internal/gateway/gateway.go` to supply tools when `TavilyAPIKey` is present and handle tool loop execution
    - [x] Write integration tests in `internal/gateway/gateway_test.go` simulating search tool invocation and response generation
    - [x] Run `go test -race ./internal/llm/...` and `go test -race ./internal/gateway/...`

## Phase 5: Verification & Quality Checks
- [x] Task: Run full test suite and quality enforcement
    - [x] Run `go test -race ./...` across all packages
    - [x] Run `make check` (`golangci-lint`, `go test -race`, `semgrep`, `osv-scanner`)
    - [x] Verify zero API key leakage in logs or error messages
