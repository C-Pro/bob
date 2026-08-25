# Specification: Web Page Fetch Tool Support (`web_fetch`)

## 1. Overview
Add a flexible URL fetching and content extraction tool (`web_fetch`) to Bob supporting three operational modes (`auto`, `raw`, and `extract`):
1. **`auto` Mode (Default):** Performs a direct HTTP `GET`. If HTML, parses via `go-readability` (`github.com/go-shiori/go-readability`). If heuristics trip (parse error, text < 200 chars, bot/JS challenge indicators), automatically falls back to Tavily Extract API (`POST /extract`). Non-HTML content is returned as-is.
2. **`raw` Mode:** Performs a direct HTTP `GET` and returns the raw response body as plain text without HTML readability parsing or dynamic extraction. Recommended when retrieving source code, raw scripts, JSON, configs, or plain text files.
3. **`extract` Mode:** Skips direct HTTP retrieval and calls Tavily Extract API directly. Recommended when targeting known heavy JavaScript SPAs or when `auto` heuristics misfired.

All modes enforce a uniform 16KB (16,384 bytes) truncation limit on returned content and append an explicit truncation indicator if exceeded.

## 2. Functional Requirements

### 2.1 Tool Definition & Registration (`internal/tools`)
- Add the `web_fetch` tool to `Registry.ToolDefinitions()`:
  - Function name: `web_fetch`
  - Description: "Fetch and extract text content from a web page URL or raw text/code file. Supports 3 modes: 'auto' (default: parses HTML into clean readable text with dynamic fallback), 'raw' (recommended for raw code, scripts, JSON, or configs without HTML parsing), and 'extract' (uses Tavily Extract directly for heavy dynamic SPAs or if 'auto' failed to capture needed content). All outputs are capped at 16KB."
  - Parameters:
    - `url` (string, required): The target HTTP/HTTPS URL to fetch.
    - `mode` (string, optional, enum: `["auto", "raw", "extract"]`, default: `"auto"`): Operational fetching mode.
- Dispatch `web_fetch` calls in `Registry.Execute(ctx, name, argsJSON)`.

### 2.2 Direct HTTP Fetcher & Readability (`internal/tools/webfetch`)
- Implement a dedicated fetcher package (`internal/tools/webfetch`):
  - HTTP client with 15s timeout, standard User-Agent header, and automatic redirect following (up to 10 redirects).
  - Response body streaming with 16KB limit (16,384 bytes) enforcement:
    - If content exceeds 16KB, truncate cleanly and append `\n\n[Content truncated to 16KB. Full download support will be added in a future update.]`.
  - Content parsing logic:
    - **`raw` mode**: Return fetched body as text regardless of Content-Type.
    - **`auto` mode**:
      - If Content-Type is HTML (`text/html`, `application/xhtml+xml`): parse using `github.com/go-shiori/go-readability`.
      - If Content-Type is non-HTML text/code (`text/*`, `application/json`, `application/xml`, etc.): return content as text.
      - If Content-Type is binary media (`image/*`, `application/pdf`, `application/octet-stream`): return MIME metadata and a notice that binary download is planned for a future update.

### 2.3 Heuristic Quality Check & Dynamic Fallback
- In `auto` mode, evaluate output with `needsDynamicFallback(readabilityText, err)`:
  - Triggered if `go-readability` returns an error.
  - Triggered if extracted text length is < 200 characters.
  - Triggered if text matches common dynamic rendering / challenge keywords (e.g., "JavaScript is required", "Enable JavaScript to view this page", "Access Denied", "Attention Required! | Cloudflare", "Just a moment...").
- Fallback execution:
  - If dynamic fallback is needed and `tavilyClient` is configured: call `tavilyClient.Extract(ctx, url)`.
  - If `tavilyClient` is unconfigured or fails: return the direct HTTP / readability result with an informational note: `[Note: Dynamic rendering fallback could not be executed: <reason>]`.

### 2.4 Tavily Client Extract API (`internal/tools/tavily`)
- Extend `internal/tools/tavily/client.go`:
  - Add `Extract(ctx context.Context, urls ...string) (*ExtractResponse, error)`.
  - Endpoint: `POST /extract`.
  - Request body: `{"urls": ["..."], "api_key": "..."}`.
  - Parse Tavily extract response (extract results with `url`, `raw_content`, etc.).
  - Reuse retry backoff (429/5xx) and timeout handling.

## 3. Non-Functional Requirements
- **Performance & Timeouts:** 15s context timeout for network calls.
- **Safety & Error Handling:** Explicit error checking; no raw unhandled panics or memory leaks during DOM parsing.
- **Dependencies:** Add `github.com/go-shiori/go-readability` (`go mod tidy` & `go mod vendor`).
- **Code Quality:** Pass full `make check` (golangci-lint, unit tests with `-race`, semgrep, osv-scanner) with 0 errors.

## 4. Acceptance Criteria
- Unit tests for direct HTTP fetching and Content-Type handling across `raw`, `auto`, and `extract` modes.
- Unit tests for HTML `go-readability` extraction with mocked HTML fixtures.
- Unit tests for the heuristic detector (`needsDynamicFallback`) evaluated against a 30-case realistic test corpus (10 static, 10 dynamic SPA, 10 bot/captcha protection).
- Unit tests for Tavily `Extract` API client with mocked HTTP server.
- Unit tests for `web_fetch` tool execution in `internal/tools/registry_test.go` covering all 3 modes, fallback conditions, and 16KB truncation.
- Integration tests in `internal/llm` and `internal/gateway` with `web_fetch` tool calls.
- `make check` passes cleanly with 0 errors.

## 5. Out of Scope
- Full file downloading / local saving feature (deferred to future track).
- Screenshot or PDF rendering capture.
- Multi-page spidering / crawling (deferred).
