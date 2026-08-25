# Specification: Images & Attachments in Context with Chunked Cache-Optimized Eviction

## 1. Overview
This track introduces two key capabilities to Bob:
1. **Chunked / Stepped Context Eviction:** Refactor `internal/chatcontext/buffer.go` from single-message sliding eviction to batch eviction (pruning 1/3 of buffer capacity at once upon reaching capacity). This maintains a stable prefix across consecutive conversation turns, maximizing LLM prompt/prefix cache hit rates (crucial when large multimodal tokens like images are in the context).
2. **Multimodal & File Attachment Ingestion:** Support images and non-binary text attachments arriving from Besedka chat messages:
   - **Image Attachments:** Fetch thumbnail representation via `GET /api/images/{fileId}?thumb=1` with Bearer auth, convert to base64 data URL, store in ring buffer entries, and format into OpenAI `MultiContent` (`image_url` parts) for multimodal LLM processing.
   - **Text-Based File Attachments:** Fetch via `GET /api/files/{fileId}` with Bearer auth, cap at 16KB, and format cleanly as code blocks within message text content.
   - **Non-Text / Binary Fallback:** Gracefully annotate unsupported binary attachments without failing message processing.

---

## 2. Functional Requirements

### 2.1 Chunked Eviction in RingBuffer (`internal/chatcontext`)
- Update `RingBuffer.Push(entry Entry)`:
  - When `len(entries) > capacity`, evict a batch of oldest messages: `evictCount := capacity / 3; if evictCount < 1 { evictCount = 1 }`.
  - Drop the oldest `evictCount` items: `rb.entries = rb.entries[evictCount:]` (and truncate if still exceeding capacity).
  - Ensure thread-safety with existing mutex protections.
  - Prefix stability: verify that across subsequent pushes up to capacity, existing message items remain untouched.

### 2.2 Multimodal Data Structures & Message Conversion (`internal/chatcontext`)
- Extend `chatcontext.Entry`:
  - Add `Images []ImageAttachment` with `MimeType string` and `Data string` (base64 string or data URL).
- Update `RingBuffer.ToLLMMessages()`:
  - For entries without images: populate `openai.ChatCompletionMessage{Role: role, Content: text}` (with `MultiContent: nil`).
  - For entries with images: populate `openai.ChatCompletionMessage` with `Content: ""` and `MultiContent: []openai.ChatMessagePart`:
    - First part: `ChatMessagePart{Type: ChatMessagePartTypeText, Text: text}`
    - Subsequent parts: `ChatMessagePart{Type: ChatMessagePartTypeImageURL, ImageURL: &ChatMessageImageURL{URL: "data:<mimeType>;base64,<data>", Detail: ImageURLDetailAuto}}`

### 2.3 Attachment Fetching & Processing (`internal/gateway`)
- Add REST client helper methods in `internal/gateway/api.go`:
  - `FetchImageThumbnail(ctx context.Context, fileID string) ([]byte, string, error)`: Calls `GET /api/images/{fileID}?thumb=1` with Bearer auth, returns image bytes and MIME type.
  - `FetchFileContent(ctx context.Context, fileID string, maxBytes int64) ([]byte, string, error)`: Calls `GET /api/files/{fileID}` with Bearer auth, streaming up to 16KB limit.
- Update `ProcessMessage(ctx, msg)` and `WarmupChat(ctx, chatID, lastSeq)`:
  - Iterate over `msg.Attachments`:
    - If image (`AttachmentTypeImage` or `image/*` MIME): fetch thumbnail, base64-encode, and attach to `chatcontext.Entry.Images`.
    - If text-based file (MIME `text/*`, `application/json`, `application/xml`, or common text/code extensions): fetch content up to 16KB, append markdown code block `\n\n[Attachment <name>]:\n```\n<content>\n``` ` to message text.
    - If binary / unsupported file: append placeholder `\n\n[Attachment: <name> (<mimeType>)]`.
    - If fetch fails: append notice `\n\n[Attachment: <name> (failed to download)]` and log warning without dropping the message.

---

## 3. Non-Functional Requirements
- **Performance & Timeouts:** 10s timeout on attachment downloads; stream with 16KB limit for text files.
- **Cache Optimization:** Conversation prefix remains unchanged between batch eviction events.
- **Code Quality:** 100% clean `make check` passing (`lint-go`, `test-go` with `-race`, `semgrep`, `osv-scanner`).

---

## 4. Acceptance Criteria
- Unit tests for chunked/stepped eviction in `internal/chatcontext/buffer_test.go` proving prefix stability and correct eviction batch sizing.
- Unit tests for `ToLLMMessages()` verifying both standard text and multimodal `MultiContent` outputs.
- Unit tests in `internal/gateway` with mocked Besedka HTTP server verifying:
  - Fetching image thumbnail (`/api/images/{id}?thumb=1`) and base64 conversion.
  - Fetching text file (`/api/files/{id}`) with 16KB cap and markdown formatting.
  - Fallback / error handling for failed downloads and binary files.
- Integration tests simulating multimodal multi-turn interaction.
- `make check` passes cleanly with 0 errors.

---

## 5. Out of Scope
- Full-resolution original image downloads (thumbnails only via `&thumb=1`).
- Non-image binary file parsing (e.g., PDF text extraction, OCR, video decoding).
