# Bob (Besedka AI Agent)

Bob is an autonomous AI agent service designed for the [Besedka](https://github.com/c-pro/besedka) chat platform. It connects to a self-hosted Besedka server over WebSocket and REST API, listens for mentions in the public Townhall chat (`@bot`) and private 1-on-1 Direct Messages (DMs), maintains multi-turn conversation memory, reports approximate server location via GEOIP, and generates responses using OpenAI-compatible LLM APIs (defaulting to Google Gemini).

## Architecture & Core Features

- **Ingress & Networking:** Connects via WebSocket (`/api/chat`) with automatic reconnect, ping keepalive, and REST API metadata caching (`/api/me`, `/api/users`, `/api/chats`, `/api/chats/{id}/messages`).
- **Context Management:** In-memory, thread-safe per-chat ring buffers (`internal/chatcontext`) bounded by `MSG_RING_BUFFER_SIZE` with on-demand historical backfill, metadata caching, and chunked batch eviction.
- **Long-Term Memory & Isolated RAG:** Per-chat SQLite vector and FTS5 storage (`internal/memory`) with sequence watermark tracking, asynchronous batch indexing on eviction, startup sequence catch-up, and strict privacy isolation via `recall_memory` tool.
- **Prompt Rendering:** Contextual system prompt formatting for Townhall and DM interactions (`internal/prompt`).
- **GEOIP Location Reporting:** Periodic server location lookup using round-robin querying across public GEOIP providers (`ip-api.com`, `ipapi.co`, `ipinfo.io`) and periodic WebSocket location frame transmission (`internal/geoip`).
- **LLM Provider:** OpenAI-compatible API client (`internal/llm`) with exponential retry backoff, embedding generation, and tool/function support.

## Coding Style

- Use comments only when necessary. Prefer self-documenting code.
- Avoid leaving chain-of-thought style comments in the code.
- Follow Effective Go standards (`gofmt`, `go vet`).
- Idiomatic error handling: explicit error checks without discarding (`_`).
- Concurrency: protect shared resources with mutexes, use contexts for cancellations, avoid goroutine leaks.

## Rules

- **Always run `make check` before finishing any code-related task** to ensure that `lint-go` (`golangci-lint`), `test-go` (`go test -race`), `semgrep`, and `osv-scanner` all pass cleanly with 0 errors. No need to run `make check` when changes are strictly confined to documentation or Conductor plans/specs.
- When implementing a new feature or fixing a bug, always write thorough unit and integration tests (`go test -race ./...`).
- When adding or changing dependencies, always run `go mod tidy` and `go mod vendor` to keep the vendor directory up to date.
- Validate configuration changes against `internal/config`.

## Default operation mode
- Implement changes in separate steps, try to keep steps small while keeping the tests green. Skipping tests for the sake of keeping build green is not a good option. Prefer finish the step with failing test and mentioning this to the user if e.g. fixing everything in one step will make the step/commit too big.
- When step is completed ask user for review. Don't commit or push unless explicitly given permission to work autonomously.
- When explicitly asked to work on the task autonomously or working in goal/autoresearch mode, make sure you are working in a separate branch. When working autonomously you can commit with --no-gpg-sign.

## Running the Agent Locally

1. Ensure the Besedka server is running (e.g. at `http://127.0.0.1:8080`).
2. Configure required environment variables (or create a `.env` file):
   - `BESEDKA_URL`: `http://127.0.0.1:8080`
   - `BESEDKA_API_KEY`: API key or bot session token
   - `OPENAI_API_KEY` (or `GEMINI_API_KEY`): LLM provider API key
   - `OPENAI_MODEL` / `GEMINI_MODEL`: Model identifier (default: `gemini-3.7-flash`)
   - `BOT_HANDLE`: Bot username mention handle (default: `@bot`)
3. Run the service:
   ```bash
   go run ./cmd/agent
   ```

## Temporary Files

When you need to create a temporary file, create it in the project directory or scratch space:
- `mktemp -p .` for an ordinary file
- `mktemp -dp .` for a temporary directory

Always delete temporary files and directories when they are no longer needed.
