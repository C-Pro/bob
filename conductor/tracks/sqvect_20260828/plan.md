# Implementation Plan: sqvect Long-Term Memory & Isolated RAG

## Phase 1: Dependencies, Configuration & Watermark Schema
- [x] Task: Add `sqvect` dependency and configure `EMBEDDING_MODEL`
    - [x] Add `github.com/liliang-cn/cortexdb/v2` to `go.mod`, run `go mod tidy` and `go mod vendor`
    - [x] Add `EmbeddingModel` field in `internal/config/config.go` (reading `EMBEDDING_MODEL` environment variable, defaulting to empty for FTS5 fallback)
    - [x] Add unit tests in `internal/config/config_test.go` verifying `EMBEDDING_MODEL` loading and defaults
    - [x] Run `go test -race ./internal/config/...`

## Phase 2: Embedding Provider & Vector/FTS Fallback
- [x] Task: Implement embedding generator with graceful FTS5 fallback
    - [x] Implement embedding client in `internal/llm` (or `internal/memory`) using OpenAI-compatible embedding endpoint
    - [x] Support graceful fallback to keyword/FTS5 search when `EMBEDDING_MODEL` is empty or embedding API is unavailable
    - [x] Write unit tests for embedding generation, dimension handling, and fallback behavior
    - [x] Run `go test -race ./internal/llm/...`

## Phase 3: Isolated SQLite Memory Store (`internal/memory`) & Watermark Tracking
- [x] Task: Implement isolated per-chat `sqvect` memory storage and access controller
    - [x] Implement `Store` / `MemoryManager` in `internal/memory` wrapping `sqvect`:
        - Path resolution: `townhall.db` for townhall chat, `dm_<sanitized_chatID>.db` for private user DMs in `cfg.DataDir`
        - Safe file permissions & WAL mode configuration
        - Watermark tracking: `watermark` table storing `last_indexed_seq` for disaster recovery & startup catch-up
        - Message batch indexing (conversation chunks with sender, timestamps, sequence watermark, and dialogue content)
        - Multi-database hybrid retrieval (Townhall queries `townhall.db`; DM queries both `dm_<sanitized_chatID>.db` and `townhall.db`)
        - Strict isolation enforcement preventing cross-DM access
    - [x] Write comprehensive unit tests in `internal/memory/store_test.go`:
        - Single chat index, watermark persistence & query
        - Townhall access (RW to `townhall.db`, no DM DB access)
        - DM access (RW to private DM DB, RO to `townhall.db`, zero cross-DM leakage)
        - FTS5 fallback search without embedding model
    - [x] Run `go test -race ./internal/memory/...`

## Phase 4: Ring Buffer Eviction Hook & Startup Sequence Catch-up
- [x] Task: Connect ring buffer eviction and historical sequence catch-up to asynchronous indexing worker
    - [x] Add eviction callback/hook in `internal/chatcontext/buffer.go` triggering when `Push` drops 1/3 capacity
    - [x] Implement asynchronous indexing worker in gateway / memory coordinator to process evicted batches in background and update sequence watermark
    - [x] Implement startup sequence catch-up / backfill logic in batches of up to 100 messages from `/api/chats/{chat_id}/messages` when `latest_seq > last_indexed_seq`
    - [x] Write unit tests for eviction callback, batching logic, watermark catch-up, and backfill in `internal/chatcontext/buffer_test.go` and `internal/gateway`
    - [x] Run `go test -race ./internal/chatcontext/...` and `go test -race ./internal/gateway/...`

## Phase 5: RAG Tool (`recall_memory`) & Agent Integration
- [x] Task: Implement `recall_memory` tool and connect to chat execution pipeline
    - [x] Add `recall_memory` tool definition in `internal/tools` with parameters `query` (string) and optional `limit` (integer)
    - [x] Wire chat session context into tool executor to query appropriate databases based on chat type (Townhall vs DM)
    - [x] Format retrieved memory passages into clean markdown with timestamps, source labels (`[Townhall]` vs `[Direct Message]`), and sender attribution
    - [x] Write unit and integration tests in `internal/tools/tools_test.go` and `internal/gateway/gateway_test.go`
    - [x] Update `cmd/agent/main.go` to initialize memory store and register tools cleanly

## Phase 6: Verification & Quality Enforcement
- [x] Task: Run full test suite and quality enforcement
    - [x] Run `go test -race ./...` across all packages
    - [x] Run `make check` (`golangci-lint`, `go test -race`, `semgrep`, `osv-scanner`) ensuring 0 errors
    - [x] Run the besedka and bob instance locally (appending EMBEDDING_MODEL=text-embedding-004 to .env). Verify that chats vector search data is backfilled and retrieval works.
    - [x] Update documentation (`GEMINI.md`, `README.md`, `conductor/tech-stack.md`) with `sqvect` memory architecture and `EMBEDDING_MODEL` config (e.g. `text-embedding-004`)

