# Implementation Plan: Images & Attachments in Context with Chunked Cache-Optimized Eviction

## Phase 1: Chunked Context Eviction & Prefix Cache Optimization
- [x] Task: Refactor `RingBuffer.Push` in `internal/chatcontext` for chunked batch eviction
    - [x] Write unit tests in `internal/chatcontext/buffer_test.go` verifying chunked eviction batch sizing (1/3 capacity), prefix invariance between batch prunes, and capacity edge cases (1, 2, 3, 10, 100)
    - [x] Implement chunked eviction logic in `internal/chatcontext/buffer.go`
    - [x] Run `go test -race ./internal/chatcontext/...` and verify tests pass

## Phase 2: Multimodal ChatContext Entry & OpenAI MultiContent Construction
- [x] Task: Extend `chatcontext.Entry` for image attachments and update `ToLLMMessages`
    - [x] Write unit tests in `internal/chatcontext/buffer_test.go` verifying `ToLLMMessages` builds standard text messages (`Content`) when images are absent and `MultiContent` (text part + `image_url` parts) when images are present
    - [x] Define `ImageAttachment` struct and update `Entry` and `ToLLMMessages` in `internal/chatcontext/buffer.go`
    - [x] Run `go test -race ./internal/chatcontext/...` and verify tests pass

## Phase 3: Besedka API Attachment Retrieval
- [x] Task: Implement REST client methods for image thumbnails and text files in `internal/gateway/api.go`
    - [x] Write unit tests in `internal/gateway/api_test.go` testing `FetchImageThumbnail` (`GET /api/images/{fileID}?thumb=1`) and `FetchFileContent` (`GET /api/files/{fileID}` with 16KB cap) with mock HTTP server
    - [x] Implement `FetchImageThumbnail` and `FetchFileContent` on `Gateway` in `internal/gateway/api.go`
    - [x] Run `go test -race ./internal/gateway/...` and verify tests pass

## Phase 4: Message Ingestion Pipeline & Multimodal Context Integration
- [x] Task: Ingest attachments in `ProcessMessage` and `WarmupChat` in `internal/gateway/gateway.go`
    - [x] Write unit/integration tests in `internal/gateway/gateway_test.go` covering image attachment handling, text file attachment inlining, binary fallback annotations, and download error resilience
    - [x] Implement attachment download and context injection in `internal/gateway/gateway.go`
    - [x] Run `go test -race ./internal/gateway/...` and verify tests pass

## Phase 5: Verification & Quality Enforcement
- [x] Task: Full verification and quality checks
    - [x] Run `go test -race ./...` across all packages
    - [x] Run `make check` (`golangci-lint`, `go test -race`, `semgrep`, `osv-scanner`)
    - [x] Verify zero linter or security issues
