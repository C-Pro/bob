# Specification: Attachment Tools for Isolated Sandbox Execution

## 1. Overview
This track introduces a pair of sandbox-scoped tools—`sandbox_download_attachment` and `sandbox_upload_attachment`—allowing Bob to download chat attachments from Besedka into the active sandbox environment and upload generated files from the sandbox as attachments to chat messages.

Key capabilities and invariants:
- **Sandbox-Coupled Availability:** These tools are available **only** in 1-on-1 Direct Messages (DMs) when an isolated sandbox is actively running (`StatusRunning`). They are excluded from Townhall public chats and will return an error if called without an active sandbox.
- **RAG & Context Discoverability:** Attachments in both live chat context and persistent RAG memory (`internal/memory`) are annotated with structured metadata tags: `[Attachment: <name> (id: <file_id>, type: <mime_type>)]`. This allows Bob to discover file IDs via chat context or `recall_memory` and pass them to the download tool.
- **Safe Upload with Size Limits:** The upload tool schema explicitly specifies upload size limits (default 25MB for generic files via `/api/upload/file`, 10MB for images via `/api/upload/image`, configurable via `MAX_ATTACHMENT_SIZE`), preventing the agent from attempting oversized uploads.
- **Turn-Staged Message Attachments:** Uploaded attachments are staged within the active FSM turn context and sent together with Bob's final response message when the tool loop concludes.
- **Collaborative Review:** Implementation will be executed in agentic-pair-programming mode with `codex sol` model as the peer reviewer.

---

## 2. Functional Requirements

### 2.1 Configuration
- Add `MaxAttachmentSizeBytes int64` to `internal/config/config.go` with default `25 * 1024 * 1024` (25MB), configurable via `MAX_ATTACHMENT_SIZE` environment variable (accepting human-friendly strings like `25MB`, `10MB` or integer bytes).
- Enforce image upload limit: `min(cfg.MaxAttachmentSizeBytes, 10 * 1024 * 1024)` for `/api/upload/image`.

### 2.2 Attachment Representation in Context & RAG Memory
- Update attachment formatting in `internal/gateway/gateway.go` (`processAttachments`, `CatchupChatMemory`, and message ingestion):
  - **Images:** Fetch thumbnail for multimodal LLM processing and append text descriptor: `\n\n[Attachment: <name> (id: <file_id>, type: <mime_type>)]`.
  - **Text Files:** Append snippet along with descriptor: `\n\n[Attachment: <name> (id: <file_id>, type: <mime_type>)]:\n` + "```\n<content>\n```" + `.
  - **Binary Files:** Append descriptor: `\n\n[Attachment: <name> (id: <file_id>, type: <mime_type>)]`.
- Update `internal/chatcontext` and `internal/memory` indexing so that all indexed chunks and ring buffer messages contain the attachment descriptor with `file_id`.
- Ensure `recall_memory` results retain this format so Bob can discover historical attachment IDs.

### 2.3 Besedka API Upload & Download Methods
- Extend `internal/gateway/api.go`:
  - `DownloadAttachment(ctx context.Context, fileID string) ([]byte, string, error)`: Fetches file or image bytes from Besedka `/api/files/{fileID}` (or fallback `/api/images/{fileID}`).
  - `UploadFile(ctx context.Context, data []byte, filename, mimeType string) (string, error)`: Uploads to `/api/upload/file`, returns the returned `fileId`.
  - `UploadImage(ctx context.Context, data []byte, filename, mimeType string) (string, error)`: Uploads to `/api/upload/image`, returns the returned `fileId`.

### 2.4 Tool Definitions & Execution (`internal/tools`)
- **`sandbox_download_attachment`:**
  - Parameters:
    - `file_id` (string, required): The Besedka attachment UUID to download.
    - `destination_path` (string, required): Relative path within the sandbox workspace (e.g., `input/data.csv`).
  - Validation:
    - Verifies active running sandbox exists for the user session.
    - Validates `destination_path` using sandbox path safety routines (preventing `..` or absolute escape outside user workspace).
  - Execution:
    - Downloads file from Besedka.
    - Writes file to `<WorkspaceDir>/<destination_path>` on the host.
    - Returns JSON status with written path, byte size, and filename.
- **`sandbox_upload_attachment`:**
  - Parameters:
    - `source_path` (string, required): Relative path of the file inside sandbox workspace (e.g., `output/report.pdf`).
    - `name` (string, optional): Display name for the chat attachment. Defaults to filename base.
  - Description:
    - Clearly specifies maximum allowed size (e.g., "Maximum file size is 25MB for generic files and 10MB for images").
  - Validation:
    - Verifies active running sandbox exists.
    - Resolves and validates `source_path` within user workspace.
    - Checks file existence and size against configured maximum limits. If exceeded, returns an explicit error explaining the file exceeds size limit.
  - Execution:
    - Reads file content.
    - Detects MIME type.
    - Uploads to `/api/upload/image` (if image and <=10MB) or `/api/upload/file`.
    - Stages attachment record (`models.Attachment`) in the active chat session context.
    - Returns confirmation with file ID, name, size, and staged status.

### 2.5 Staging Attachments & Final Response Transmission
- Extend `ChatSessionContext` in `internal/tools/registry.go` and `internal/fsm`:
  - Provide a thread-safe attachment collector / staging slice `StagedAttachments *[]models.Attachment`.
- Extend `internal/gateway/gateway.go`:
  - When FSM concludes turn execution, retrieve staged attachments.
  - Send message with attachments via WebSocket (`models.ClientMessage{Type: "send", ChatID: chatID, Content: reply, Attachments: stagedAttachments}`).

### 2.6 Dynamic Tool Filtering
- Update `ToolDefinitionsForSession` in `internal/tools/registry.go`:
  - `sandbox_download_attachment` and `sandbox_upload_attachment` are included in DM tools **only** when `sbx.Status == sandbox.StatusRunning`.
  - Excluded from Townhall chats.
  - Excluded when no sandbox is running.

---

## 3. Security & Safety Invariants
1. **Strict Path Containment:** Destination and source paths are resolved strictly within `sbx.WorkspaceDir` using `filepath.Clean` and prefix verification to prevent arbitrary host file reads or writes.
2. **DM & Sandbox Isolation:** Tools fail immediately if called from Townhall or if the session does not own an active sandbox.
3. **Attachment Size Enforcement:** Files exceeding `MAX_ATTACHMENT_SIZE` are rejected before network transmission.
4. **Zero Cross-User Leakage:** A user can only access files within their own sandbox workspace.

---

## 4. Acceptance Criteria
- Unit tests in `internal/gateway/api_test.go` verifying file download and file/image uploads against mock Besedka server.
- Unit tests in `internal/tools` verifying:
  - `sandbox_download_attachment` writes to workspace, rejects path traversal, fails without running sandbox.
  - `sandbox_upload_attachment` reads from workspace, enforces size limits, stages attachment, fails without running sandbox.
- Unit and integration tests verifying context and RAG memory formatting includes `[Attachment: <name> (id: <file_id>, type: <mime_type>)]`.
- Gateway integration test verifying that staged attachments are dispatched in the final `ClientMessage.Attachments` payload.
- Pair programming review with `codex sol` model passes.
- All checks pass cleanly with `make check` (0 linter errors, race detector clean).

---

## 5. Out of Scope
- Direct attachment downloads into ephemeral sandboxes without persistent workspace mounts.
- OCR or binary content parsing within the attachment tools themselves (handled via shell tools inside the sandbox).
