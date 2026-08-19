# Track Spec: Refactor LLM Client to go-openai

## 1. Overview & Context
This track refactors the LLM communication layer from the hand-rolled HTTP client and JSON structures to the community-standard Go OpenAI client (`github.com/sashabaranov/go-openai`). This modernizes message representations to standard `openai.ChatCompletionMessage` structs, provides native support for OpenAI tool/function definitions out of the box, and supports standard OpenAI environment variables (`OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`) with full backward compatibility for legacy `GEMINI_*` environment variables during the transition period.

## 2. Functional Requirements

### 2.1 Dependencies & Module Management
- Add `github.com/sashabaranov/go-openai` to `go.mod`.
- Ensure clean dependency resolution with `go mod tidy`.

### 2.2 Standard OpenAI Configuration with Backward Compatibility (`internal/config`)
- Refactor `Config` to support standard OpenAI environment variables with fallback to legacy `GEMINI_*` variables:
  - **API Key:** `OPENAI_API_KEY` (fallback to `GEMINI_API_KEY`).
  - **Base URL:** `OPENAI_BASE_URL` (fallback to `GEMINI_BASE_URL`, defaulting to `https://generativelanguage.googleapis.com/v1beta/openai/`).
  - **Model:** `OPENAI_MODEL` (fallback to `GEMINI_MODEL`, defaulting to `gemini-3.7-flash`).
- Backward-compatible struct fields / getters so existing code and deployments operate seamlessly.
- Update `Config.Validate(requireAPIKey bool)` to verify that either `OPENAI_API_KEY` or `GEMINI_API_KEY` is present.

### 2.3 LLM Client Refactor (`internal/llm`)
- Reimplement `llm.Client` around `openai.Client`:
  - Initialize via `openai.NewClientWithConfig` using `openai.DefaultConfig(apiKey)` with custom `BaseURL` and custom `http.Client`.
  - Maintain retry handling with exponential backoff on transient errors (5xx, 429).
  - Update `GenerateChatResponse(ctx context.Context, messages []openai.ChatCompletionMessage) (string, error)`.
  - Update `GenerateResponse(ctx context.Context, systemPrompt, userMessage string) (string, error)`.
  - Expose client support/types for tool definitions (e.g., accepting `[]openai.Tool` in chat completion requests) to enable future tool calling tracks.
  - Delete obsolete custom JSON structs (`CompletionRequest`, `CompletionResponse`, `Choice`, `APIError`, `Message`).

### 2.4 Message Struct Migration (`internal/chatcontext` & `internal/gateway`)
- Update `chatcontext.RingBuffer.ToLLMMessages()` to return `[]openai.ChatCompletionMessage` with standard roles (`openai.ChatMessageRoleUser`, `openai.ChatMessageRoleAssistant`).
- Update `chatcontext.Manager.GetLLMMessages(chatID)` to return `[]openai.ChatCompletionMessage`.
- Update `internal/gateway/gateway.go` to construct system messages using `openai.ChatMessageRoleSystem` and pass `[]openai.ChatCompletionMessage` to `llmClient.GenerateChatResponse`.

### 2.5 Service Entrypoint & Testing Harness
- Update `cmd/agent/main.go` to log active OpenAI configuration.
- Update unit tests (`internal/config_test.go`, `internal/llm/client_test.go`, `internal/gateway/gateway_test.go`, `internal/chatcontext/buffer_test.go`) to test `go-openai` structures, standard `OPENAI_*` variables, and backward-compatible `GEMINI_*` fallback behavior.
- Update `cmd/agent/agent_integration_test.go` to mock standard OpenAI chat completion JSON payloads returned by `go-openai`.

## 3. Non-Functional Requirements & Performance
- **Zero Ingress Regression:** Preserves multi-turn ring buffering, mention filtering, paragraph formatting, and WebSocket egress without regressions.
- **Backward Compatibility:** Existing deployment configurations using `GEMINI_*` env vars continue to work without changes.
- **Concurrency & Safety:** All client calls and buffer operations remain race-free and thread-safe (`go test -race ./...`).

## 4. Acceptance Criteria
1. `github.com/sashabaranov/go-openai` is imported and used for all LLM interactions.
2. Configuration supports `OPENAI_*` environment variables with fallback to `GEMINI_*`.
3. Hand-rolled OpenAI request/response structs are removed in favor of `openai` types.
4. Tool definition types and signatures are supported in `internal/llm`.
5. 100% of unit and integration tests pass with `go test -v -race ./...`.

## 5. Out of Scope
- Implementing concrete tool execution handlers (e.g. bash execution, web search) - this track establishes the client and tool definition foundation; concrete tool runtime execution will follow in dedicated feature tracks.
- Streaming responses (`/v1/chat/completions` SSE stream).
