# Implementation Plan: Besedka Progress Reports & User Context Ingestion

## Phase 1: Models & User Context Ingestion (Timezone & Language)
- [ ] Task: Update user models and cache
    - [ ] Add `TimeZone` and `PreferredLanguage` fields to `models.User`
    - [ ] Update `gateway.UserCache` tests and JSON unmarshaling tests for `/api/users`
- [ ] Task: Implement once-per-session context injection in gateway
    - [ ] Write unit tests for once-per-session timezone/language injection into context ring buffer
    - [ ] Implement in-memory session tracking in `gateway` to append system turn `[User context: ...]` on first message

## Phase 2: Progress Protocol Models & Egress API
- [ ] Task: Add progress message types to `internal/models`
    - [ ] Define `MessageType`, `ProgressStatus`, `ProgressStep`, `ProgressData` structs and constants
    - [ ] Update `models.Message` and `models.ClientMessage` with `Type` and `Progress`
    - [ ] Add unit tests for serialization and deserialization
- [ ] Task: Implement progress message sending in `gateway`
    - [ ] Write unit tests for `SendProgressMessage`
    - [ ] Implement `SendProgressMessage(ctx, chatID, progress)` via REST `POST /api/chats/{id}/messages` and WebSocket
    - [ ] Handle capturing `parentSeq` from root card response

## Phase 3: Retire Legacy Timer-Based Progress
- [ ] Task: Remove `tools.ProgressReporter` and legacy timer mechanisms
    - [ ] Remove `ProgressReporter`, ticker loops, and `ProgressPrefix` in `internal/tools`
    - [ ] Remove `recentProgress` and `isRecentProgress` string deduplication in `gateway`
    - [ ] Update message ingress filtering to discard incoming progress messages by `MessageTypeProgress`
    - [ ] Update affected existing gateway and tool tests

## Phase 4: Deterministic Step Title & Description Formatter
- [ ] Task: Implement tool call progress formatting helper
    - [ ] Write unit tests for tool call formatting across `sandbox_exec`, `tavily_search`, `web_fetch`, `recall_memory`, memory/skills tools, and generic fallback
    - [ ] Implement `FormatProgressStep(toolCall openai.ToolCall) (title, desc string)` in `internal/tools`

## Phase 5: FSM Tool Loop Progress Card Integration
- [ ] Task: Connect FSM step execution with live progress updates
    - [ ] Write integration tests for FSM tool loop emitting root card, child step transitions (`running` -> `completed`/`failed`), and root card completion
    - [ ] Update `internal/fsm` and `gateway` tool loop runner to emit root card and step events
    - [ ] Enforce channel permissions (DMs always enabled for multi-step tools; Townhall checked for bot write permission)

## Phase 6: System Verification & Local Testing
- [ ] Task: Run full verification suite
    - [ ] Run `go test -v -covermode=atomic -race ./...`
    - [ ] Run `make check` (golangci-lint, osv-scanner, semgrep) ensuring 0 errors
