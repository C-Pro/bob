# Implementation Plan: Besedka Progress Reports & User Context Ingestion

## Phase 1: Models & User Context Ingestion (Timezone & Language)
- [x] Task: Update user models and cache
    - [x] Add `TimeZone` and `PreferredLanguage` fields to `models.User`
    - [x] Update `gateway.UserCache` tests and JSON unmarshaling tests for `/api/users`
- [x] Task: Implement once-per-session context injection in gateway
    - [x] Write unit tests for once-per-session timezone/language injection into context ring buffer
    - [x] Implement in-memory session tracking in `gateway` to append system turn `[User context: ...]` on first message

## Phase 2: Progress Protocol Models & Egress API
- [x] Task: Add progress message types to `internal/models`
    - [x] Define `MessageType`, `ProgressStatus`, `ProgressStep`, `ProgressData` structs and constants
    - [x] Update `models.Message` and `models.ClientMessage` with `Type` and `Progress`
    - [x] Add unit tests for serialization and deserialization
- [x] Task: Implement progress message sending in `gateway`
    - [x] Write unit tests for `SendProgressMessage`
    - [x] Implement `SendProgressMessage(ctx, chatID, progress)` via REST `POST /api/chats/{id}/messages` and WebSocket
    - [x] Handle capturing `parentSeq` from root card response

## Phase 3: Retire Legacy Timer-Based Progress
- [x] Task: Remove `tools.ProgressReporter` and legacy timer mechanisms
    - [x] Remove `ProgressReporter`, ticker loops, and `ProgressPrefix` in `internal/tools`
    - [x] Remove `recentProgress` and `isRecentProgress` string deduplication in `gateway`
    - [x] Update message ingress filtering to discard incoming progress messages by `MessageTypeProgress`
    - [x] Update affected existing gateway and tool tests

## Phase 4: Deterministic Step Title & Description Formatter
- [x] Task: Implement tool call progress formatting helper
    - [x] Write unit tests for tool call formatting across `sandbox_exec`, `tavily_search`, `web_fetch`, `recall_memory`, memory/skills tools, and generic fallback
    - [x] Implement `FormatProgressStep(toolCall openai.ToolCall) (title, desc string)` in `internal/tools`

## Phase 5: FSM Tool Loop Progress Card Integration
- [x] Task: Connect FSM step execution with live progress updates
    - [x] Write integration tests for FSM tool loop emitting root card, child step transitions (`running` -> `completed`/`failed`), and root card completion
    - [x] Update `internal/fsm` and `gateway` tool loop runner to emit root card and step events
    - [x] Enforce channel permissions (DMs always enabled for multi-step tools; Townhall checked for bot write permission)

## Phase 6: System Verification & Local Testing
- [x] Task: Run full verification suite
    - [x] Run `go test -v -covermode=atomic -race ./...`
    - [x] Run `make check` (golangci-lint, osv-scanner, semgrep) ensuring 0 errors
