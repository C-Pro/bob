# Bob (Besedka AI Agent)

Bob is an autonomous AI agent service designed for the [Besedka](https://github.com/c-pro/besedka) chat platform. It connects to a self-hosted Besedka server over WebSocket and REST API, listens for mentions in the public Townhall chat (`@bot`) and private 1-on-1 Direct Messages (DMs), maintains multi-turn conversation memory, reports approximate server location via GEOIP, and generates responses using OpenAI-compatible LLM APIs (defaulting to Google Gemini).

## Architecture & Core Features

- **Ingress & Networking:** Connects via WebSocket (`/api/chat`) with automatic reconnect, ping keepalive, and REST API metadata caching (`/api/me`, `/api/users`, `/api/chats`, `/api/chats/{id}/messages`).
- **Context Management:** In-memory, thread-safe per-chat ring buffers (`internal/chatcontext`) bounded by `MSG_RING_BUFFER_SIZE` with on-demand historical backfill, metadata caching, and chunked batch eviction.
- **Long-Term Memory & Isolated RAG:** Per-chat SQLite vector and FTS5 storage (`internal/memory`) with sequence watermark tracking, asynchronous batch indexing on eviction, startup sequence catch-up, and strict privacy isolation via `recall_memory` tool.
- **Prompt Rendering:** Contextual system prompt formatting for Townhall and DM interactions (`internal/prompt`).
- **GEOIP Location Reporting:** Periodic server location lookup using round-robin querying across public GEOIP providers (`ip-api.com`, `ipapi.co`, `ipinfo.io`) and periodic WebSocket location frame transmission (`internal/geoip`).
- **LLM Provider:** OpenAI-compatible API client (`internal/llm`) with exponential retry backoff, embedding generation, and tool/function support.
- **Durable FSM & Resilient Tool Loop:** SQLite-backed finite state machine (`internal/fsm`) for executing tool loops with channel-specific iteration budgets (10 in Townhall, 20 in DMs), parallel read-only dispatch, sequential mutating dispatch, per-step timeouts, exponential backoff retries, delayed transitions (`resume_at`), startup crash recovery, and automated retention pruning with incremental vacuuming.

## Evaluation & Training Trajectory Datasets

Bob includes a partitioned benchmarking and trajectory generation dataset infrastructure with strict zero-overlap isolation between evaluation and training:

- **Held-Out Evaluation Suite (`eval/tasks/heldout/heldout_100.json`):** 100 frozen benchmark tasks (35 DeepMind FRAMES, 20 InterCode-Bash, 20 InterCode-CTF, 25 SWE-bench Lite) for zero-data-snooping evaluation. Baseline results on local `Qwen3.6-35B-A3B-MTP-GGUF` achieve a **55.0% end-to-end success rate** (see `eval/results/baseline_100_report.md`).
- **Offline Training Trajectory Pool (`eval/tasks/train/train_pool.json`):** 417 clean tasks across all 4 domains, strictly isolated from held-out tasks (0% overlap guarantee).
- **Recorded Training Trajectories (`eval/results/train/batch_train_pool_1794581821/`):** Full multi-turn rollouts generated using `gemini-3.5-flash-lite` (500 iterations / 1 hour limit), containing **19,706 interaction steps**, **2,004 SelfCompact compaction events** across 152 long-horizon tasks, 417 isolated SQLite database snapshots, and 48 candidate SWE patches.
- **Data Schemas & Usage:** Detailed specifications for task definitions, rollout traces, step structures, compaction payloads, and Python analysis utilities are documented in **[`eval/DATASET.md`](eval/DATASET.md)**.
- **Git Exclusions:** All task manifests (`eval/tasks/**/*.json`) and rollout trace files (`eval/results/**/*.json`, `*.patch`) are strictly excluded from version control via `.gitignore`.

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

### 1. Besedka Chat Daemon
1. Ensure the Besedka server is running (e.g. at `http://127.0.0.1:8080`).
2. Configure required environment variables (or create a `.env` file):
   - `BESEDKA_URL`: `http://127.0.0.1:8080`
   - `BESEDKA_API_KEY`: API key or bot session token
   - `OPENAI_API_KEY` (or `GEMINI_API_KEY`): LLM provider API key
   - `OPENAI_MODEL` / `GEMINI_MODEL`: Model identifier
   - `BOT_HANDLE`: Bot username mention handle (default: `@bot`)
3. Run the service:
   ```bash
   go run ./cmd/agent
   ```

### 2. Standalone CLI One-Shot Frontend (`bob -p`)
1. Build the CLI binary:
   ```bash
   make build-bob
   ```
2. Initialize configuration template (default `~/.config/bob/config.yaml`, or a custom location via `--config`):
   ```bash
   # Custom location (recommended to avoid modifying user home directory):
   ./bin/bob --init-config --config /path/to/custom-config.yaml

   # Or default location:
   ./bin/bob --init-config
   ```
3. Populate `api_key` in the configuration file (ensure mode 0600 permissions).
4. Run one-shot prompt:
   ```bash
   # Using custom config:
   ./bin/bob --config /path/to/custom-config.yaml -p "What is the capital of France?"

   # Or using default config (~/.config/bob/config.yaml):
   ./bin/bob -p "What is the capital of France?"
   ```
   Or mount a directory to execute sandboxed tasks:
   ```bash
   ./bin/bob --config /path/to/custom-config.yaml -p "Run the tests and fix errors" --mount .:rw
   ```

## Temporary Files

When you need to create a temporary file, create it in the project directory or scratch space:
- `mktemp -p .` for an ordinary file
- `mktemp -dp .` for a temporary directory

Always delete temporary files and directories when they are no longer needed.
