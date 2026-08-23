# Specification: Web Search Tool Support (Tavily API)

## 1. Overview
Add live web search capabilities to Bob using the Tavily REST API (`https://api.tavily.com/search`). When configured with `TAVILY_API_KEY`, Bob provides tool definitions alongside the system prompt that precedes the ring buffer conversation history. If the LLM generates a tool call during completion, Bob executes the search via a dedicated native Go Tavily client and conducts an iterative multi-turn tool execution loop (up to 5 iterations) until a final response is generated.

## 2. Functional Requirements
### 2.1 Configuration & Environment
- Add `TavilyAPIKey` and `TavilyBaseURL` (default: `https://api.tavily.com`) to `internal/config/config.go`.
- Support loading `TAVILY_API_KEY` and `TAVILY_BASE_URL` from environment variables and `.env`.
- Automatically enable the `web_search` tool if `TAVILY_API_KEY` is present and non-empty.

### 2.2 Tavily API Client (`internal/tools/tavily`)
- Implement a lightweight, native Go HTTP client for Tavily Search API (`POST /search`).
- Request parameters:
  - `query` (string, required)
  - `search_depth` (string, optional: `"basic"` | `"advanced"`, default `"basic"`)
  - `max_results` (int, optional: 1–5, default 5)
  - `include_answer` (bool, optional: default `true`)
- Response parsing:
  - Title, URL, content snippet, and optional direct answer.
- Robust HTTP handling: configurable timeout (15s), exponential retry backoff for 429/5xx status codes, and sanitized error reporting (never logging API keys).

### 2.3 Tool Definition & Context Independence (`internal/prompt` & `internal/tools`)
- Provide standard OpenAI-compatible tool definitions (`openai.Tool`) for `web_search`.
- Tool definitions and any tool system guidance are configured as part of the initial system context that precedes the ring buffer messages and remains strictly independent from ring buffer history.
- Tool definitions are instantiated once per runtime/context and passed directly in the completion request.

### 2.4 Multi-Turn LLM Tool Execution Loop (`internal/llm` & `internal/gateway`)
- Enhance LLM interaction to handle tool calls:
  - Pass available tool definitions once in `ChatCompletionRequest.Tools`.
  - When LLM returns `ToolCalls`, execute each tool call through the dispatcher.
  - Append the assistant message with `ToolCalls` and corresponding `tool` role messages (`role: "tool"`, `ToolCallID: id`, `Content: result_json`).
  - Send updated conversation back to LLM until the LLM returns text content or max iterations (e.g. 5) is reached.
  - If tool execution fails, return error payload in tool message so LLM can adapt or inform the user.
- Ensure ring buffer stores only clean final conversational turns (user message & final assistant response) without polluting the ring buffer with raw intermediate tool frames.
- Apply standard paragraph formatting (`FormatResponse`) to the final assistant text.

## 3. Non-Functional Requirements
- **Performance & Concurrency:** Thread-safe, context-aware execution with timeouts to prevent hanging on external API calls.
- **Security:** Zero leakage of `TAVILY_API_KEY` in logs, traces, or user messages.
- **Dependencies:** Standard library `net/http` + existing `github.com/sashabaranov/go-openai`; zero new external heavy SDKs.
- **Quality:** `make check` must pass cleanly (100% pass on linting, race detection, unit tests, osv-scanner, and semgrep).

## 4. Acceptance Criteria
- Unit tests for Tavily client covering search requests, responses, HTTP retries, and error handling.
- Unit tests for tool schema definition and argument deserialization/validation.
- Unit tests for LLM multi-turn tool calling loop with mocked OpenAI tool responses and multi-step resolution.
- Gateway integration tests verifying end-to-end handling of messages triggering web searches.
- Full `make check` suite passes with 0 errors.

## 5. Out of Scope
- Standalone `/crawl`, `/map`, and `/extract` tools (deferred to future tracks).
- Tavily CLI (`tvly`) subprocess execution.
