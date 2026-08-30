# Implementation Plan: Local Embeddings & Vector Regeneration CLI

## Phase 1: Dependency & Configuration
- [x] Task: Add `go-embed` dependency and configuration options
    - [x] Add `github.com/C-Pro/go-embed` to `go.mod`, run `go mod tidy` and `go mod vendor`
    - [x] Add `EmbeddingPrecision` field to `internal/config/config.go` reading `EMBEDDING_PRECISION` (defaulting to `bf16`)
    - [x] Write unit tests in `internal/config/config_test.go` for `EmbeddingPrecision` validation and defaults
    - [x] Run `go test -race ./internal/config/...`

## Phase 2: Local Embedder & Multi-Window Aggregation
- [x] Task: Implement local CPU embedder adapter wrapping `go-embed`
    - [x] Implement `LocalEmbedder` in `internal/llm` (or `internal/memory/local_embedder.go`) wrapping `engine.Engine`
    - [x] Implement multi-window aggregation (mean pooling + L2 normalization across returned slices for long texts)
    - [x] Implement `cortexdb.Embedder` interface (`Embed`, `EmbedBatch`, `Dim`, `Model`)
    - [x] Write comprehensive unit tests for multi-window aggregation, batching, dimension reporting, and text-only handling
    - [x] Run `go test -race ./internal/llm/...` and `go test -race ./internal/memory/...`

## Phase 3: Automatic Local Fallback & Graceful Dimension Mismatch Handling
- [x] Task: Wire local embedder into `memory.Manager` when `EMBEDDING_MODEL` is unset and handle dimension mismatches
    - [x] Update `memory.NewManager` / initialization logic to create and use `LocalEmbedder` when `EMBEDDING_MODEL` is empty
    - [x] Ensure graceful fallback to FTS5 lexical search if local engine initialization fails or if search encounters vector dimension mismatch with stored vectors
    - [x] Update memory store tests in `internal/memory/store_test.go` to test local embedder vector generation, retrieval, and dimension mismatch fallback
    - [x] Run `go test -race ./internal/memory/...`

## Phase 4: Vector Regeneration Service & CLI Option
- [x] Task: Implement vector regeneration across all chat databases in `DATA_DIR`
    - [x] Implement `RegenerateAllVectors` in `internal/memory` iterating over `townhall.db` and all `dm_*.db`
    - [x] Query stored memories (`session_id LIKE 'memory:%'`), recompute embeddings in batches with `EmbedBatch`, and update SQLite vectors
    - [x] Add `-regenerate-vectors` and `-reembed` CLI flags in `cmd/agent/main.go`
    - [x] Add unit and integration tests for `RegenerateAllVectors` and CLI flag handling in `internal/memory/store_test.go` and `cmd/agent/lifecycle_test.go`
    - [x] Run `go test -race ./internal/memory/...` and `go test -race ./cmd/agent/...`

## Phase 5: Verification & Quality Enforcement
- [x] Task: Full verification and documentation update
    - [x] Run full test suite with `go test -race ./...`
    - [x] Run `make check` (`golangci-lint`, `go test -race`, `semgrep`, `osv-scanner`) ensuring 0 errors
    - [x] Update `conductor/tech-stack.md`, `README.md`, and `GEMINI.md` with local embeddings and regeneration CLI details
