# Specification: Local Embeddings & Vector Regeneration CLI

## 1. Overview
Integrate `github.com/C-Pro/go-embed` (`go-embed/pkg/engine`) into Bob to provide high-performance, pure Go (CGO-free), CPU-based local text embeddings when `EMBEDDING_MODEL` is not specified or when local embeddings are desired. Default local model is configured to `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` with weights stored under `<DATA_DIR>/models` and configurable precision (`EMBEDDING_PRECISION`, defaulting to `bf16`).

Additionally, implement a standalone CLI command-line option in `cmd/agent` (`-regenerate-vectors` / `-reembed`) that scans all cortexdb databases (`townhall.db` and all `dm_*.db`) in `DATA_DIR`, fetches all indexed memory chunks, regenerates their vector embeddings using `EmbedBatch` with the currently active embedder (local or remote), updates the stored vectors in SQLite, and exits.

## 2. Architecture & Design

### 2.1 Embedding Provider Hierarchy & Local Fallback
When initializing the agent or memory manager:
1. **Remote LLM Embeddings**: If `EMBEDDING_MODEL` is explicitly configured (e.g. `gemini-embedding-2`, `text-embedding-004`), Bob uses the OpenAI-compatible `/v1/embeddings` API endpoint via `llm.Embedder`.
2. **Local CPU Embeddings (Default Fallback)**: When `EMBEDDING_MODEL` is empty / not specified:
   - Bob initializes the local `go-embed` engine using `engine.NewEngine(...)` with model `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
   - Data directory for model weights is set to `<DATA_DIR>/models`.
   - Precision mode is configured via `EMBEDDING_PRECISION` (options: `fp32`, `bf16`, `int8`, defaulting to `bf16` for optimal memory footprint ~225MB and fidelity).
   - A local embedder implementing cortexdb's `Embedder` interface (`Embed(ctx, text)` and `EmbedBatch(ctx, texts)`) is wired into `memory.Manager` and all opened cortexdb instances.
3. **Pure FTS5 Fallback**: If local engine initialization fails or is explicitly disabled, memory search gracefully falls back to lexical FTS5 search without crashing.

### 2.2 go-embed Multi-Slice & Text-Only Adapter
- **Multi-Window Aggregation**: `go-embed`'s engine supports inputs longer than model max sequence length by returning `[][]float32` for `Embed(text)` and `[][][]float32` for `EmbedBatch(texts)` (one vector per sliding window). The Bob local embedder adapter aggregates multi-window embeddings for a text via mean-pooling and L2 normalization into a single 384-dimensional unit vector conforming to cortexdb's `Embedder` interface.
- **Text-Only Processing**: Only clean text strings (chat dialogue, timestamps, sender attributions) are passed to `go-embed`. Image attachments, base64 data URLs, and binary attachments are strictly omitted.

### 2.3 Dimension Mismatch Graceful Degradation
- When query vector dimension differs from the stored vectors in the database (e.g., if the embedding model was changed without yet running vector regeneration), vector similarity search must gracefully degrade to lexical / FTS5 search instead of failing or throwing an error to the user / LLM.

### 2.4 Vector Regeneration CLI Command (`-regenerate-vectors`)
- **Flag**: Add `-regenerate-vectors` (and `-reembed` alias) boolean flag to `cmd/agent/main.go`.
- **Execution Flow**:
  1. Load runtime configuration and initialize the active embedder (remote if `EMBEDDING_MODEL` set, otherwise local `go-embed`).
  2. Scan `DATA_DIR` for `townhall.db` and all `dm_*.db` SQLite databases.
  3. For each database:
     - Open cortexdb instance with the active embedder.
     - Query all stored memory messages (`session_id LIKE 'memory:%'`).
     - Process memory chunks in batches (batch size 16-32) using `embedder.EmbedBatch(ctx, batchTexts)`.
     - Update the binary vector column in SQLite (`UPDATE messages SET vector = ? WHERE id = ?`).
     - Reconcile collection dimensions if necessary.
     - Log progress (database name, chunk count, duration, errors).
  4. Print summary report and cleanly exit with status code 0 (or 1 on fatal error).

## 3. Functional Requirements

### 3.1 Dependencies & Configuration
- Add `go-embed` dependency (`github.com/C-Pro/go-embed`) to `go.mod`.
- Add `EmbeddingPrecision` field to `internal/config/Config` (reads `EMBEDDING_PRECISION` env var, defaults to `bf16`).
- Model weight download and cache path: `filepath.Join(cfg.DataDir, "models")`.

### 3.2 Local Embedder Implementation (`internal/llm` or `internal/memory`)
- Wrap `engine.Engine` as a cortexdb-compatible Embedder:
  - `Embed(ctx context.Context, text string) ([]float32, error)`
  - `EmbedBatch(ctx context.Context, texts []string) ([][]float32, error)`
  - `Dim() int` (returns vector dimensionality, e.g. 384 for MiniLM-L12-v2)
  - `Model() string`
- Support multi-slice window aggregation via mean pooling + L2 normalization.
- Thread-safe inference execution using `go-embed`'s context pool / model inference.

### 3.3 Memory Search Dimension Mismatch Handling
- In `memory.Manager.Search`: If cortexdb vector search encounters a dimension mismatch error with existing stored vectors, gracefully fall back to lexical text search and log a diagnostic warning recommending `-regenerate-vectors`.

### 3.4 Regeneration Service & CLI Option
- Provide `memory.RegenerateAllVectors(ctx context.Context, cfg *config.Config, embedder cortexdb.Embedder, batchSize int) error` (or equivalent method on `memory.Manager`).
- CLI flag in `cmd/agent`:
  - `go run ./cmd/agent -regenerate-vectors` (or `-reembed`).
  - Supports `--data-dir` or standard `DATA_DIR` env variable.
  - Exits after completion without launching the WebSocket gateway loop.

## 4. Non-Functional Requirements
- **Zero CGO**: Maintains 100% pure Go across all dependencies.
- **Batch Efficiency**: Vector generation uses `EmbedBatch` to maximize vectorization throughput and amortize overhead.
- **Robust Error Handling**: If single chunk embedding fails during CLI re-indexing, log error with chunk ID and continue with remaining batches, reporting final error tally.
- **Memory Safety**: `bf16` precision keeps memory footprint low (~225 MB) during local inference.

## 5. Acceptance Criteria
1. When `EMBEDDING_MODEL` is empty, memory manager automatically uses local `go-embed` engine with `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` and `bf16` precision.
2. Embedding vector dimensionality matches local model (384 dims) and semantic retrieval works locally without API keys.
3. Long texts spanning multiple windows are properly aggregated and normalized.
4. If stored vectors have mismatched dimensions from the query embedder, memory search gracefully falls back to lexical/FTS5 search without error.
5. CLI command `go run ./cmd/agent -regenerate-vectors` successfully iterates over `townhall.db` and all `dm_*.db` files in `DATA_DIR`, recomputing vectors with `EmbedBatch` and updating the database.
6. Unit tests covering:
   - Config loading for `EMBEDDING_PRECISION`.
   - Local embedder implementation, multi-window aggregation, `Embed` and `EmbedBatch`.
   - Dimension mismatch graceful degradation.
   - Vector regeneration across multiple isolated chat databases.
   - CLI flag execution in main/lifecycle tests.
7. All tests pass with `go test -race ./...` and `make check` passes cleanly with 0 errors.

## 6. Out of Scope
- GPU / CUDA acceleration (CPU SIMD and scalar pure Go only).
- Dynamic runtime model swapping while gateway WebSocket loop is active.
