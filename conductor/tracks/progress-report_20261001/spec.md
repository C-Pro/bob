# Specification: Besedka Progress Reports & User Context Ingestion

## 1. Overview
Besedka recently introduced user timezone/language reporting (ephemeral fields in `GET /api/users` from WebSocket `userInfo`) and a dedicated rich progress card message type (`type: "progress"`) enabling live, in-place multi-step progress tracking for bot workflows.
This track adds support in Bob for:
1. Ingesting user timezone and preferred language, appending them to the chat context ring buffer as a system turn once per in-memory chat session.
2. Replacing legacy timer-based progress notifications (`tools.ProgressReporter`, `ProgressPrefix`, 30s tickers) with Besedka's rich progress card protocol.
3. Emitting real-time progress card updates (root card creation, step running/completed/failed state transitions, card completion) during multi-step FSM tool loop execution using deterministic title and description generation.
4. Supporting progress reports across both Direct Messages (DMs) and Townhall (with write permission validation).

## 2. Functional Requirements

### 2.1 User Timezone & Preferred Language Context Ingestion
- **Model Extension:** Add `TimeZone` (string) and `PreferredLanguage` (string) to `models.User`.
- **Cache Persistence:** `gateway.UserCache` stores and exposes these fields when populated via `GET /api/users`.
- **Session Injection:**
  - Track session state per chat in `gateway` using an in-memory map (`sync.Map` or per-chat session tracker).
  - When an incoming message from a user is processed, check if their `timeZone` and/or `preferredLanguage` are available in `UserCache`.
  - If available and not yet appended for this chat session, push a system turn to the chat's context ring buffer:
    `[User context: timezone=<timeZone>, language=<preferredLanguage>]` (formatting cleanly if only one is present).
  - Mark user context as injected for this chat session so it is only appended once until bot restart or context clear.

### 2.2 Besedka Progress Protocol Models & REST/WS Egress
- **Model Definitions:**
  - Define `MessageType` (`"text"`, `"progress"`).
  - Define `ProgressStatus` (`"running"`, `"completed"`, `"failed"`).
  - Define `ProgressStep` (`id`, `title`, `description`, `status`).
  - Define `ProgressData` (`parentSeq`, `cardStatus`, `title`, `step`, `steps`).
  - Add `Type MessageType` and `Progress *ProgressData` to `models.Message` and `models.ClientMessage`.
- **Egress API:**
  - Implement `gateway.SendProgressMessage(ctx, chatID, progress)`:
    - For root progress cards (`parentSeq == 0`), send via HTTP `POST /api/chats/{id}/messages` to capture the allocated message `seq` synchronously.
    - For child step updates (`parentSeq > 0`), send via HTTP `POST /api/chats/{id}/messages` or WebSocket.
  - Return the assigned `seq` for root cards to link child step updates.

### 2.3 Retirement of Legacy Timer-Based Progress Reports
- Remove `tools.ProgressReporter`, 30s background ticker goroutines, and `ProgressPrefix = "⏳ "`.
- Remove `gateway.recentProgress` and `gateway.isRecentProgress` string deduplication.
- Update message ingress filters: identify progress messages strictly via `m.Type == models.MessageTypeProgress` (while ignoring legacy prefix if encountered in history).
- Ensure incoming progress messages are dropped before entering ring buffers or vector/FTS long-term memory.

### 2.4 Deterministic Step Title & Description Extraction
- Implement a helper to map `openai.ToolCall` into a human-friendly progress title and description:
  - `sandbox_exec`: title from command name or first token, description from `"description"` argument (falling back to command string).
  - `tavily_search`: title "Web Search", description "Searching for: <query>".
  - `web_fetch`: title "Fetch Web Page", description "Retrieving <url>".
  - `recall_memory`: title "Recall Memory", description "Querying conversation memory for: <query>".
  - Knowledge / skill tools: title e.g. "Discover Skills", description "Loading skill <name>".
  - Fallback: title formatted from tool name, description from summarized JSON arguments.

### 2.5 Multi-Step Workflow & FSM Integration
- Integrate progress reporting with the durable FSM (`internal/fsm`) and tool loop execution:
  - When the FSM identifies tool calls to execute, create a root progress card (e.g. `cardStatus: "running"`, title summarizing the task or step count).
  - For each step executed, emit a child step event with `status: "running"`.
  - On step completion, emit child step event with `status: "completed"` (or `"failed"` on error).
  - When the tool loop finishes (final synthesis reached or all tools executed), update the root card status to `"completed"` (or `"failed"` on abort).
  - The final conversational response is sent as a subsequent regular text message.
- Verify Townhall permissions: only send progress messages in Townhall if the bot has write permissions; in DMs always send if multi-step tools are executed.

## 3. Non-Functional Requirements
- **Thread Safety:** Concurrent access to progress state, user cache, and session trackers must be guarded with mutexes.
- **Resilience:** If Besedka rejects a progress message (e.g. older server version), log a warning and proceed without breaking the core chat completion or tool loop.
- **Zero Memory Leaks:** Clean up progress tracking maps and session states appropriately.
- **Code Quality:** Pass `make check` (`golangci-lint`, `go test -race`, `semgrep`, `osv-scanner`) with 0 errors.

## 4. Out of Scope
- Dynamic progress step generation via additional LLM calls (deterministic extraction is fast, reliable, and zero-cost).
- Long-term local database persistence for progress cards.
