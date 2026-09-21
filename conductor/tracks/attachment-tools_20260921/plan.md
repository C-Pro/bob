# Implementation Plan: Attachment Tools for Isolated Sandbox Execution

## Phase 1: Configuration, Besedka API Integration & RAG/Context Metadata Enhancement
- [x] Task: Attachment Upload Size Limit Configuration (`internal/config`)
    - [x] Add `MaxAttachmentSizeBytes int64` (default: 25MB) to `internal/config/config.go` with `MAX_ATTACHMENT_SIZE` env parsing (supporting bytes and units like "25MB", "10MB")
    - [x] Unit tests in `internal/config/config_test.go` verifying parsing defaults and error handling
- [x] Task: Besedka API Upload & Download Client Integration (`internal/gateway/api.go`)
    - [x] Implement `DownloadAttachment(ctx, fileID)` supporting both `/api/files/{fileID}` and `/api/images/{fileID}` fallback
    - [x] Implement `UploadFile(ctx, data, filename, mimeType)` targeting `POST /api/upload/file`
    - [x] Implement `UploadImage(ctx, data, filename, mimeType)` targeting `POST /api/upload/image`
    - [x] Unit tests in `internal/gateway/api_test.go` with mock HTTP server testing downloads, uploads, and error scenarios
- [x] Task: Structured Attachment Metadata in Context & RAG Storage (`internal/gateway`, `internal/chatcontext`, `internal/memory`)
    - [x] Update `processAttachments` in `internal/gateway/gateway.go` to append metadata descriptor `[Attachment: <name> (id: <file_id>, type: <mime_type>)]` for all incoming attachments (images, text files, binary files)
    - [x] Ensure RAG memory indexing (`IndexMessages`, `CatchupChatMemory`) and ring buffer messages include the metadata descriptor so `recall_memory` discovers `file_id`
    - [x] Unit tests in `internal/gateway` and `internal/memory` verifying attachment metadata formatting in context and indexed memory chunks
- [x] Task: Phase 1 Codex Sol Review & Verification
    - [x] Conduct pair programming review with `codex sol` model for Phase 1 changes
    - [x] Run `go test -race ./internal/config/... ./internal/gateway/... ./internal/memory/...`

---

## Phase 2: Sandbox Attachment Tools & Session Lifecycle Guarding (`internal/tools`)
- [x] Task: Attachment Staging in Session Context
    - [x] Extend `ChatSessionContext` in `internal/tools/registry.go` with `StagedAttachments *StagedAttachmentCollector` and thread-safe staging helper
    - [x] Initialize staging container in `NewChatSessionContext`
- [x] Task: Implement `sandbox_download_attachment` Tool
    - [x] Define JSON schema and `SandboxDownloadAttachmentArgs` (`file_id`, `destination_path`)
    - [x] Implement `executeSandboxDownloadAttachment`: verify running sandbox, sanitize and validate destination path within workspace, download via gateway API, write to workspace, return result JSON
- [x] Task: Implement `sandbox_upload_attachment` Tool
    - [x] Define JSON schema and `SandboxUploadAttachmentArgs` (`source_path`, optional `name`) with clear size limits in description
    - [x] Implement `executeSandboxUploadAttachment`: verify running sandbox, sanitize source path, check file size against limits, upload via gateway API, stage in session context, return result JSON
- [x] Task: Dynamic Tool Definition Filtering in `ToolDefinitionsForSession`
    - [x] Add attachment tools to sandbox tool group in `internal/tools/registry.go`
    - [x] Ensure tools are available in DM sessions only when `sbx.Status == sandbox.StatusRunning` and excluded from Townhall
- [x] Task: Unit Tests for Attachment Tools
    - [x] Write unit tests in `internal/tools`: download happy path, missing sandbox, path traversal rejection, upload size limit enforcement, image vs generic file routing, attachment staging
    - [x] Test tool definition availability under different session states
- [x] Task: Phase 2 Codex Sol Review & Verification
    - [x] Conduct pair programming review with `codex sol` model for Phase 2 changes
    - [x] Run `go test -race ./internal/tools/...`

---

## Phase 3: Final Response Attachment Dispatch & End-to-End Integration
- [x] Task: Wire Staged Attachments to Outgoing Chat Messages
    - [x] Add `SendMessageWithAttachments(chatID, content string, attachments []models.Attachment)` in `internal/gateway/gateway.go`
    - [x] In message dispatch & FSM turn completion, collect staged attachments and send via WebSocket `ClientMessage`
- [x] Task: Gateway Integration & E2E Tests
    - [x] Test end-to-end flow: tool loop executing `sandbox_upload_attachment` and final message sending staged attachments
    - [x] Test end-to-end flow: incoming attachment discoverable via memory and downloaded into workspace via `sandbox_download_attachment`
- [x] Task: Phase 3 Codex Sol Review & Verification
    - [x] Conduct pair programming review with `codex sol` model for Phase 3 and full diff
    - [x] Run `make check` (ensure `lint-go`, `test-go -race`, `semgrep`, `osv-scanner` all pass with 0 errors)
