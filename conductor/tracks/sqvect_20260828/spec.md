# Specification: sqvect Long-Term Memory & Isolated RAG

## 1. Overview
Integrate `github.com/liliang-cn/sqvect/v2/pkg/sqvect` into Bob to provide long-term memory and Retrieval-Augmented Generation (RAG) using isolated SQLite database files in `DATA_DIR`.
Historical messages evicted from in-memory ring buffers are indexed asynchronously in batches into chat-specific SQLite vector databases. Each per-chat database maintains a sequence watermark (`last_indexed_seq`) enabling seamless index recovery and catch-up on service restart in batches of up to 100 messages from the Besedka REST API. Direct Message (DM) conversations have read/write access to their own isolated database and read-only access to the shared townhall database, while the townhall agent has read/write access to the townhall database. A strict privacy boundary guarantees zero cross-DM data sharing. An agent tool `recall_memory` enables the LLM to retrieve past contextual memories on demand.

## 2. Architecture & Data Isolation Model

### 2.1 File & Directory Layout
All SQLite vector databases are stored within the configured `DATA_DIR` (e.g. `./data`):
- `townhall.db`: Shared public townhall database containing townhall discussion history.
- `dm_<sanitized_chatID>.db`: Private database for a specific DM conversation.
- `bob.db`: General-purpose agent metadata/schema database.

### 2.2 Access Control & Permissions Matrix
| Context | Active Database (Read/Write) | Shared Database (Read-Only) | Other DM Databases |
| :--- | :--- | :--- | :--- |
| **Townhall Chat** | `townhall.db` (RW) | N/A | Forbidden (No access) |
| **User DM Chat** | `dm_<sanitized_chatID>.db` (RW) | `townhall.db` (RO) | Forbidden (No access) |

### 2.3 Strict Privacy Guarantee
- Evicted messages from a DM are **strictly written only** to `dm_<sanitized_chatID>.db`.
- When `recall_memory` is invoked in a DM context:
  - Hybrid search is executed against the user's `dm_<sanitized_chatID>.db` and the public `townhall.db`.
  - Results from other DM databases are never accessed or returned.
- When `recall_memory` is invoked in a Townhall context:
  - Hybrid search is executed **only** against `townhall.db`.

## 3. Functional Requirements

### 3.1 Embedding Generation & Fallback
- **OpenAI-Compatible Embeddings**: Support embedding generation using the existing OpenAI/Gemini client endpoint and API key.
- **Configuration**:
  - `EMBEDDING_MODEL`: Configurable model name (e.g., `gemini-embedding-2`, `text-embedding-004`, or `text-embedding-3-small`).
- **Graceful Fallback**: If `EMBEDDING_MODEL` is empty or unset, sqvect vector operations fall back to pure FTS5 keyword retrieval without failing.

### 3.2 Sequence Watermark & Recovery / Backfill
- **Watermark Persistence**: Maintain a `watermark` table / metadata record in each chat database storing `last_indexed_seq`.
- **Catch-up on Startup / Missing DB**:
  - When opening a chat database or starting the service, compare the server's latest message sequence against `last_indexed_seq`.
  - If latest sequence > `last_indexed_seq`, fetch missing historical messages from Besedka REST API (`/api/chats/{id}/messages`) in batches of up to 100 messages, index them into `sqvect`, and update `last_indexed_seq`.
- **Backup & Disaster Recovery**: Enables periodic snapshots / backups of SQLite databases while guaranteeing complete catch-up on service startup.

### 3.3 Asynchronous Batch Eviction Indexing
- **Eviction Hook**: When `RingBuffer.Push` exceeds `MsgRingBufferSize` and evicts a batch of messages (1/3 capacity), the evicted entries are dispatched to an asynchronous background worker queue.
- **Batch Formatting**: The batch of evicted messages is formatted into structured conversation turn chunks (including sender name/ID, timestamp range, and textual dialogue) before being vector-indexed and written into the chat's private SQLite file.
- **Watermark Update**: Update `last_indexed_seq` to the highest sequence number in the indexed batch.
- **Non-blocking**: Gateway real-time message processing remains responsive and non-blocking.

### 3.4 Hybrid Retrieval & RAG Tool
- **`recall_memory` Tool**:
  - Exposed to the LLM via `tools.Registry` and chat session context.
  - Parameters: `query` (string, required), `limit` (integer, optional, default 5).
  - Search execution:
    - If in Townhall: runs hybrid search (or FTS fallback) on `townhall.db`.
    - If in DM: runs hybrid search on `dm_<sanitized_chatID>.db` and `townhall.db`, merging results with source labels (`[Townhall]` vs `[Direct Message]`).
  - Output: Formatted markdown containing matching historical conversation passages with timestamps and sender attribution.

### 3.5 Service Lifecycle & Connection Management
- Storage connections for sqvect are pooled and cached per chat ID.
- Clean shutdown closes all open SQLite sqvect instances.

## 4. Non-Functional Requirements
- **Zero CGO**: Pure Go SQLite and vector operations.
- **Concurrency & Safety**: WAL mode enabled for SQLite databases with proper mutex/connection pool management.
- **Resilience**: Network or embedding errors during background indexing log warnings and do not crash the service or block real-time chat.

## 5. Acceptance Criteria
- Unit & integration tests verifying:
  1. Sequence watermark tracking and catch-up / backfill in batches of up to 100 messages.
  2. `RingBuffer` eviction triggering async batch indexing.
  3. Indexing and hybrid retrieval with `EMBEDDING_MODEL` (e.g. `gemini-embedding-2`).
  4. FTS5 fallback search when `EMBEDDING_MODEL` is not configured.
  5. Isolation verification: DM searches its own DB + Townhall DB; Townhall searches only Townhall DB; DM A cannot read DM B's DB.
  6. `recall_memory` tool execution and output formatting.
  7. All tests pass with `-race` and `make check` passes with 0 errors.

## 6. Out of Scope
- Inter-agent private sharing or arbitrary cross-DM searches.
- Direct synchronization with remote vector databases (e.g. Pinecone, Qdrant).
