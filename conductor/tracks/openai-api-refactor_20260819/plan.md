# Implementation Plan: Refactor LLM Client to go-openai

## Overview
Refactor LLM communication from hand-rolled HTTP calls to `github.com/sashabaranov/go-openai`, adopt standard `openai.ChatCompletionMessage` structs, support tool definition types, and switch configuration to standard `OPENAI_*` environment variables with backward-compatible `GEMINI_*` fallbacks.

## Phase 1: Dependencies & Configuration Refactor
- [ ] Task: Dependency Management
    - [ ] Add `github.com/sashabaranov/go-openai` dependency to `go.mod` and run `go mod tidy`.
- [ ] Task: Configuration & Backward Compatibility (`internal/config`)
    - [ ] Update `internal/config/config.go` to support `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `OPENAI_MODEL` with fallback to legacy `GEMINI_*` env vars.
    - [ ] Update `internal/config/config_test.go` with unit tests covering `OPENAI_*` variables, `GEMINI_*` fallbacks, and validation rules.

## Phase 2: LLM Client Migration to go-openai (`internal/llm`)
- [ ] Task: Modernize LLM Client
    - [ ] Reimplement `llm.Client` in `internal/llm/client.go` using `openai.Client` with custom base URL, custom HTTP client, and retry logic.
    - [ ] Expose chat completion methods accepting `openai.ChatCompletionMessage` and supporting tool definitions (`openai.Tool`).
    - [ ] Remove deprecated hand-rolled HTTP request/response and message structs.
    - [ ] Update `internal/llm/client_test.go` with unit tests verifying chat completion requests, retries, tool definitions, and error handling.

## Phase 3: Message Struct Standardization & Gateway Integration (`internal/chatcontext` & `internal/gateway`)
- [ ] Task: Context & Gateway Migration
    - [ ] Update `internal/chatcontext/buffer.go` (`ToLLMMessages`, `GetLLMMessages`) to use `openai.ChatCompletionMessage`.
    - [ ] Update `internal/chatcontext/buffer_test.go` to validate `openai.ChatCompletionMessage` output.
    - [ ] Update `internal/gateway/gateway.go` to construct system messages using `openai.ChatMessageRoleSystem` and call `llmClient.GenerateChatResponse`.
    - [ ] Update `internal/gateway/gateway_test.go` to match updated client signatures and types.

## Phase 4: Entrypoint & Integration Verification
- [ ] Task: Entrypoint & End-to-End Tests
    - [ ] Update `cmd/agent/main.go` for updated configuration logging and client setup.
    - [ ] Update `cmd/agent/agent_integration_test.go` to mock OpenAI completion responses for `go-openai`.
- [ ] Task: Quality & Self-Review
    - [ ] Run full test suite: `go test -v -covermode=atomic -race ./...`.
    - [ ] Run linter and formatting checks.
