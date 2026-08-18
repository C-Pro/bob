# Track Spec: Simple Context Management

## 1. Overview & Context
This track implements multi-turn conversation context management using per-chat in-memory ring buffers. It enables the AI agent to maintain conversational awareness in both Townhall channels and Direct Messages (DMs), retaining up to $N$ recent message turns (default 100, configurable via `MSG_RING_BUFFER_SIZE`) and sending rich conversational history to Google Gemini via OpenAI-compatible endpoints.

Additionally, system prompts become dynamic templates that incorporate bot metadata, channel-specific response size recommendations, and DM participant display names retrieved from Besedka REST APIs during startup/reconnect.

## 2. Functional Requirements

### 2.1 Metadata Gathering & User Caching on Startup / Reconnect
- On startup and gateway reconnect, the agent queries Besedka REST APIs to gather necessary context:
  - `GET /api/me`: Resolves bot identity (`id`, `userName`, `displayName`).
  - `GET /api/users`: Fetches/caches user profiles (`id`, `userName`, `displayName`) for participant attribution in messages and prompts.
  - `GET /api/chats`: Discovers active chats and DM channels to initialize ring buffers.
  - `GET /api/chats/{chat_id}/messages`: Fetches recent chat history to pre-fill ring buffers up to capacity without triggering LLM responses.

### 2.2 Dynamic Templated System Prompts
- System prompts are rendered dynamically using templates:
  - **Townhall Prompt Template:** Injects the bot's username and display name, Townhall conversation instructions, and response brevity guidelines (`TOWNHALL_MAX_PARAGRAPHS`, default 2 paragraphs).
  - **DM Prompt Template:** Injects the bot's username and display name, the DM user's display name, DM conversation instructions, and response size recommendations (`DM_MAX_PARAGRAPHS`, default 10 paragraphs, brief and helpful).

### 2.3 Per-Chat Ring Buffer Storage
- Maintain a thread-safe circular buffer (`RingBuffer`) per `chatID`.
- Buffer capacity defaults to `100` messages, overridable via the `MSG_RING_BUFFER_SIZE` environment variable.
- Buffer entries store role (`user` vs `assistant`), sender display name/username, and message text content.
- Thread-safe access supporting concurrent message ingestion and retrieval.

### 2.4 Ingress Message Processing & Self-Message Handling
- **Self-Message Handling (Townhall & DM):**
  - When an incoming message has `senderID == botUserID` (self-message from WebSocket stream or history backfill):
    - Appended to that chat's ring buffer with `role: "assistant"`.
    - **NEVER** triggers an LLM response (preventing recursion or echo loops).
- **User Messages in Townhall (`chatID == "townhall"`):**
  - All incoming user messages are appended to Townhall ring buffer with `role: "user"` formatted as `<DisplayName>: <Message Content>` (fallback to username/userID).
  - Preserves full message text (including `@bot` handles) for natural conversational context.
  - **Trigger Condition:** Only user messages directly mentioning the bot (`@bot` / `BOT_HANDLE`) trigger LLM generation. Non-mention messages simply update the ring buffer.
- **User Messages in Direct Messages (`chatID != "townhall"`):**
  - Incoming user messages (`senderID != botUserID`) are appended to the DM's ring buffer with `role: "user"`.
  - Every incoming user message in a DM triggers LLM generation and a bot response.

### 2.5 LLM Request Formatting
- The payload sent to `/v1/chat/completions` consists of:
  1. Rendered System Prompt (Townhall or DM template).
  2. Chronological sequence of up to $N$ latest messages from the ring buffer:
     - `role: "user"`, `content: "<DisplayName>: <Message Content>"`
     - `role: "assistant"`, `content: "<Bot Response Content>"`
  3. The triggering message as the final user message in the context sequence.

### 2.6 Egress Formatting & Message Sending
- Generated LLM responses are formatted using `FormatResponse` (enforcing max paragraph limits) and posted to Besedka.
- Response is appended to the ring buffer with `role: "assistant"`.

## 3. Non-Functional Requirements & Performance
- **Zero-Latency Ingress:** Appending to the in-memory ring buffer is $O(1)$ and lock-contention free under normal chat volume.
- **Deduplication & Safety:** Ensure self-messages received from the WebSocket echo stream or egress do not duplicate and never cause feedback loops.
- **Graceful Bounds:** Fixed memory footprint bounded by `number_of_chats * MSG_RING_BUFFER_SIZE`.
- **Test Coverage:** Unit tests for templated system prompt rendering, user cache/metadata resolution, ring buffer operations (overflow, chronological ordering, concurrency), self-message recognition in DMs & Townhall, and gateway trigger logic.

## 4. Out of Scope
- External vector database or embeddings-based dense search (Phase 2 hybrid retrieval / long-term memory).
- SQLite / PostgreSQL persistent database storage of conversation logs.
- Tool calling or function execution (Phase 3).
