# Technology Stack: Self-Improving General Use AI Agent

## 1. Programming Language & Core Runtime
- **Language:** Go (1.27+) with `GOEXPERIMENT=simd`
- **Concurrency & I/O:** Standard library (`net/http`, `context`, `sync`, `encoding/json`, `time`), Go 1.25+ `sync.WaitGroup.Go`
- **Coding Standards:** Go 1.22+ routing style (`mux.HandleFunc("POST /...", ...)`), idiomatic Go formatting (`gofmt`, `go vet`)

## 2. Ingress & Networking (Besedka Integration)
- **Protocol:** HTTP REST & JSON over WebSocket
- **WebSocket Library:** `github.com/fasthttp/websocket` for high-performance WebSocket client connection and event streaming
- **API Authentication:** HTTP Authorization Bearer token header (`Authorization: Bearer <API_KEY>`) / session tokens for user/bot actions

## 3. LLM Provider & Inference API
- **Endpoint Protocol:** OpenAI-compatible REST API (`/v1/chat/completions`) with multimodal image support (`MultiContent` / `ChatMessagePartTypeImageURL`)
- **Client Library:** `github.com/sashabaranov/go-openai` (built-in support for tool/function definitions)
- **Primary Provider Target (Phase 1):** Google Gemini OpenAI-compatible REST API (`https://generativelanguage.googleapis.com/v1beta/openai/`)
- **Default Model:** `gemini-3.7-flash` (configurable via `OPENAI_MODEL` / `GEMINI_MODEL`)
- **Configurability:** Standard OpenAI environment variables (`OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_BASE_URL`) with backward-compatible fallback for `GEMINI_*` env vars.
- **Future Readiness:** Compatible with local open-weights inference servers (vLLM / SGLang) running local models in future phases.

## 4. Configuration & State Management
- **Configuration:** Environment variables (`env` / `os.Getenv`)
- **Context & State Storage:** In-memory thread-safe per-chat ring buffers (`internal/chatcontext`) bounded by `MSG_RING_BUFFER_SIZE` (default 100) with chunked stepped eviction ($1/3$ capacity batch pruning) to maximize LLM prompt prefix caching, dynamic templated prompt generation (`internal/prompt`), and user metadata caching (`internal/gateway`).
- **Database & Persistent Storage:** SQLite via pure Go `modernc.org/sqlite` (no CGO) in `internal/store`, supporting configurable `DATA_DIR`, automated schema initialization (`schema.tmpl`), single-step version migrations (`migrate.tmpl`), and automated git tagging (`db-release-N`).

## 5. Testing & Code Quality
- **Testing Framework:** Native Go `testing` package, `GOEXPERIMENT=simd go test -race -cover`
- **Integration & E2E Testing:** Go integration tests running against local Besedka instance (`http://localhost:8080`)
- **Static Analysis & Security:** `golangci-lint`, `osv-scanner`, `semgrep`

## 6. Containerization & Deployment Infrastructure
- **Containerization:** Multi-stage Dockerfile based on `golang:1.27-alpine` build step with `GOEXPERIMENT=simd` and minimal runtime image (`scratch` / `alpine:latest`)
- **Deployment Platform:** GCP Spot VM (co-located on the same Spot VM instances as Besedka)
- **Container Registry:** GCP Artifact Registry (`asia-southeast2-docker.pkg.dev`)
- **CI/CD Orchestration:** GitHub Actions (`pipeline.yml` with `workflow_dispatch`, `push` to `main` for test VM, and `v*` tag triggers for prod VM)

## 7. Tools & External Integrations
- **Web Search API:** Tavily REST API (`POST /search`) with native Go HTTP client, exponential retry backoff, and OpenAI function calling integration.
- **Web Extraction & Readability:** `github.com/go-shiori/go-readability` for HTML DOM extraction and content sanitization; Tavily Extract REST API (`POST /extract`) for dynamic client-rendered SPA fallback.
- **Long-Term Memory & Isolated RAG:** Pure-Go SQLite Vector & FTS5 engine via `github.com/liliang-cn/cortexdb/v2` (`internal/memory`), supporting `recall_memory` function calling tool, sequence watermark tracking, asynchronous eviction batch indexing, startup historical backfill catch-up, and strict privacy isolation across DMs and Townhall.
- **Embedding Providers:**
  - **Local Embedded Inference (Default):** Pure-Go transformer embedding engine via `github.com/C-Pro/go-embed` (`internal/llm`) using `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 dimensions, BF16/FP32/INT8 precision mode, ~225MB RAM in BF16 mode), with sliding window mean pooling + L2 normalization and offline model caching under `<DATA_DIR>/models`.
  - **Remote LLM API (Optional):** OpenAI-compatible `/v1/embeddings` endpoint client (`internal/llm`) with exponential retry backoff, active when `EMBEDDING_MODEL` is explicitly configured (e.g. `gemini-embedding-2`, `text-embedding-004`).
  - **Lexical Fallback:** Graceful fallback to pure FTS5 BM25 search when embedding generation fails or when vector dimensions mismatch.
  - **Vector Regeneration CLI:** Standalone vector migration tool via `cmd/agent -regenerate-vectors` / `-reembed` to batch recompute embeddings (`EmbedBatch`) across all chat databases when switching models.
- **Database Backup & Object Storage:** Minimal standard-library AWS SigV4 S3 REST client (`internal/objectstore`), Argon2id + AES-256-GCM encryption pipeline with `BOBB` magic header and streaming gzip compression (`internal/backup`), JSON manifest metadata tracking (`manifest.json`), MinIO CI service integration testing.

