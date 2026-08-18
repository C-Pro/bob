# Implementation Plan: Simple Context Management

## Overview
Implement per-chat in-memory ring buffer context management, dynamic templated system prompts with bot/user metadata, startup history backfilling, and multi-turn context generation for Gemini via OpenAI-compatible endpoints.

## User Tasks & Milestones

- [x] Task 1: Configuration & Environment Support
  - [x] Add `MsgRingBufferSize` (default 100) to `internal/config/config.go` with parsing and validation.
  - [x] Write unit tests for `MsgRingBufferSize` configuration in `internal/config/config_test.go`.

- [x] Task 2: In-Memory Ring Buffer Module (`internal/context`)
  - [x] Implement thread-safe `RingBuffer` struct and `Entry` model supporting capacity limits, FIFO eviction, and concurrent access.
  - [x] Implement methods to format buffer entries into OpenAI `[]llm.Message` sequences.
  - [x] Write unit tests for `RingBuffer` covering push, eviction at capacity, chronological ordering, and concurrency safety.

- [x] Task 3: Besedka API Client Extensions & User Metadata Caching
  - [x] Add models and client methods for `/api/users`, `/api/chats`, and `/api/chats/{chat_id}/messages` in `internal/models` and `internal/gateway`.
  - [x] Implement thread-safe user metadata cache (`UserCache`) to resolve user IDs to display names and usernames.
  - [x] Write unit tests with HTTP mock server testing API endpoints and caching behavior.

- [x] Task 4: Dynamic Templated System Prompts (`internal/prompt`)
  - [x] Implement template renderers for Townhall (`RenderTownhallPrompt`) and DM (`RenderDMPrompt`) injecting bot identity, user display name, and paragraph recommendations.
  - [x] Write unit tests validating prompt template rendering and parameter injection.

- [x] Task 5: Gateway Ingress, History Backfill & Context Assembly (`internal/gateway`)
  - [x] Integrate user cache, per-chat ring buffers, and prompt templates into `Gateway`.
  - [x] Implement startup/reconnect history backfill (pre-filling ring buffers without triggering LLM responses).
  - [x] Update `ProcessMessage` to handle self-messages (`senderID == botUserID`) in both Townhall and DMs (append with `role: assistant`, no LLM trigger).
  - [x] Update user message handling: append with `role: user` (`<DisplayName>: <Content>`), trigger LLM on `@bot` mention in Townhall and on all user messages in DMs.
  - [x] Update LLM request pipeline to send full multi-turn context `[SystemPrompt, ...RingBufferMessages]`.
  - [x] Append bot replies to the ring buffer as `role: assistant`.
  - [x] Write unit tests for gateway message processing, self-message recognition in DM/Townhall, and multi-turn context generation.

- [x] Task 6: Integration Verification & Quality Checks
  - [x] Run full test suite: `go test -v -covermode=atomic -race ./...`.
  - [x] Verify `make check` / linting passes cleanly.
  - [x] Run local integration tests verifying multi-turn conversational context flow.
