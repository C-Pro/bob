# Implementation Plan: Durable FSM for Tool Loops & Agentic Workflows

## Phase 1: Storage Layer, Schema, Retention & Config
- [x] Task: Add `FSM_RETENTION_DAYS` to configuration
    - [x] Add `FSMRetentionDays` field (int, default 7) to `internal/config/config.go`
    - [x] Add unit tests in `internal/config/config_test.go` verifying default and custom env var parsing
    - [x] Run `go test -race ./internal/config/...`
- [x] Task: Implement FSM data models and state enums
    - [x] Create `internal/fsm/models.go` with `FSMRun`, `FSMStep`, `RunStatus`, `StepStatus`, `ExecutionMode`
    - [x] Add serialization helpers for `context_json`, `result_json`, `args_json`
- [x] Task: Implement SQLite schema migrations & auto-vacuum
    - [x] Integrate `fsm_runs`, `fsm_steps`, and indexes into `internal/store/schema.tmpl` and `migrate.tmpl` (schema version 2)
    - [x] Ensure `PRAGMA auto_vacuum = INCREMENTAL;` is executed on DB initialization
    - [x] Implement `EnsureDBSchema` in `internal/store/sqlite.go`
    - [x] Write unit tests for table creation, version 1 -> 2 migration, and foreign key cascade deletion
- [x] Task: Implement isolated per-chat FSM Store
    - [x] Create `internal/fsm/store.go` with CRUD operations: `CreateRun`, `UpdateRunState`, `GetRun`, `CreateSteps`, `GetPendingSteps`, `UpdateStepResult`, `ListActiveRuns`
    - [x] Write unit tests in `internal/fsm/store_test.go` verifying isolated operations across multiple SQLite DBs and transaction rollbacks
- [x] Task: Implement Retention Manager
    - [x] Create `internal/fsm/retention.go` with `PruneTerminalRuns(ctx, cutoffTime)` and `IncrementalVacuum(ctx, pages)`
    - [x] Guarantee active runs (`status IN ('PENDING', 'RUNNING', 'WAITING')`) are never deleted
    - [x] Write unit tests in `internal/fsm/retention_test.go` verifying TTL-based deletion, active run immunity, and cascading step deletion
- [x] Task: Phase 1 Validation & Independent Model Review
    - [x] Run `go test -race ./internal/config/... ./internal/fsm/...`
    - [x] Run independent model verification:
      ```bash
      opencode run --auto -m opencode/muse-spark-1.3-contributor-free "Review the unstaged changes (Phase 1: Storage layer, models, SQLite schema migrations, store operations, and retention manager for durable FSM). Just say LGTM if the changes address the issue and do not introduce new serious one. Don't nitpick"
      ```
    - [x] Address any valid concerns reported by the verifier and re-verify before proceeding

---

## Phase 2: Step Executor (Sequential & Parallel) with Resilience
- [x] Task: Implement tool execution classifier and worker pool
    - [x] Create `internal/fsm/executor.go` with `ClassifyExecutionMode(tools []openai.ToolCall) ExecutionMode`
    - [x] Classify batches as `Parallel` if all tools are read-only (`web_search`, `web_fetch`, `recall_memory`), otherwise `Sequential`
    - [x] Implement parallel dispatch using bounded worker pool (max 4 workers) with `sync.WaitGroup`
    - [x] Implement ordered sequential dispatch for mutating/dependent tools
- [x] Task: Implement tool timeouts and transient retries
    - [x] Enforce per-step timeout using `context.WithTimeout(ctx, step.Timeout)`
    - [x] Implement exponential backoff retry policy for transient errors (network timeouts, HTTP 429 rate limits, 503 errors)
    - [x] Ensure non-retryable/deterministic errors fail immediately without retries
- [x] Task: Write unit tests for step executor
    - [x] Create `internal/fsm/executor_test.go` testing parallel fan-out speed, sequential ordering enforcement, timeout cancellation, and retry backoff calculations
    - [x] Run `go test -race ./internal/fsm/...`
- [x] Task: Phase 2 Validation & Independent Model Review
    - [x] Run `go test -race ./internal/fsm/...`
    - [x] Run independent model verification:
      ```bash
      opencode run --auto -m opencode/muse-spark-1.3-contributor-free "Review the unstaged changes (Phase 2: Step executor with parallel/sequential modes, per-step timeouts, and exponential backoff retries). Just say LGTM if the changes address the issue and do not introduce new serious one. Don't nitpick"
      ```
    - [x] Address any valid concerns reported by the verifier and re-verify before proceeding

---

## Phase 3: Core FSM Engine, State Dispatcher & Simple Tool Loop FSM
- [x] Task: Implement core state dispatcher and delayed transition poller
    - [x] Create `internal/fsm/engine.go` managing workflow run lifecycle
    - [x] Implement in-memory timer channel + background SQLite ticker (2-second interval) to resume runs where `status = 'WAITING' AND resume_at <= now()`
    - [x] Implement crash recovery: on engine startup, query `ListActiveRuns` and resume interrupted executions from their last committed state
- [x] Task: Implement Simple Tool Loop FSM
    - [x] Create `internal/fsm/tool_loop.go` with state handlers:
      - `INIT` -> `LLM_REQUEST`
      - `LLM_REQUEST` -> `COMPLETED` (text response) | `PREPARE_STEPS` (tool calls) | `SYNTHESIS` (iteration cap reached)
      - `PREPARE_STEPS` -> `EXECUTE_STEPS`
      - `EXECUTE_STEPS` -> `WAITING` (delayed retry) | `LLM_REQUEST` (results appended)
      - `SYNTHESIS` -> `COMPLETED` | `FAILED`
- [x] Task: Write unit and integration tests for FSM engine
    - [x] Create `internal/fsm/engine_test.go` and `internal/fsm/tool_loop_test.go` testing full tool loop execution, delayed transitions (`resume_at`), crash recovery, and max iteration synthesis
    - [x] Run `go test -race ./internal/fsm/...`
- [x] Task: Phase 3 Validation & Independent Model Review
    - [x] Run `go test -race ./internal/fsm/...`
    - [x] Run independent model verification:
      ```bash
      opencode run --auto -m opencode/muse-spark-1.3-contributor-free "Review the unstaged changes (Phase 3: Core FSM engine, delayed transitions, crash recovery, and simple tool loop state machine). Just say LGTM if the changes address the issue and do not introduce new serious one. Don't nitpick"
      ```
    - [x] Address any valid concerns reported by the verifier and re-verify before proceeding

---

## Phase 4: Gateway Integration & Channel Iteration Budgets
- [x] Task: Wire FSM engine into Gateway and lifecycle
    - [x] Initialize `fsm.Engine` in `cmd/agent/main.go` and pass to `internal/gateway`
    - [x] Update `internal/gateway/gateway.go` to route tool requests through `fsm.Engine` instead of volatile loop
    - [x] Enforce channel iteration caps: 10 iterations in Townhall, 20 iterations in DMs
    - [x] Connect ephemeral progress reporting (`ProgressReporter`) to emit notifications during FSM transitions
    - [x] Connect periodic retention pruning and incremental vacuum to gateway maintenance ticker
- [x] Task: Write integration tests for Gateway FSM routing
    - [x] Add gateway integration tests in `internal/gateway/gateway_test.go` verifying Townhall iteration limits (10), DM limits (20), and tool execution via FSM
    - [x] Run `go test -race ./internal/gateway/...`
- [x] Task: Phase 4 Validation & Independent Model Review
    - [x] Run `go test -race ./internal/gateway/... ./cmd/agent/...`
    - [x] Run independent model verification:
      ```bash
      opencode run --auto -m opencode/muse-spark-1.3-contributor-free "Review the unstaged changes (Phase 4: Gateway integration with FSM engine, channel iteration limits of 10 for Townhall and 20 for DM, and ephemeral progress reporting). Just say LGTM if the changes address the issue and do not introduce new serious one. Don't nitpick"
      ```
    - [x] Address any valid concerns reported by the verifier and re-verify before proceeding

---

## Phase 5: Verification & Quality Enforcement
- [x] Task: Run full test suite and quality enforcement
    - [x] Run `go test -race ./...` across all packages
    - [x] Run `make check` (`golangci-lint`, `go test -race`, `semgrep`, `osv-scanner`) ensuring 0 errors
    - [x] Run independent model verification on the complete branch diff:
      ```bash
      opencode run --auto -m opencode/muse-spark-1.3-contributor-free "Review the unstaged changes (Phase 5: Full durable FSM implementation, tests, and documentation). Just say LGTM if the changes address the issue and do not introduce new serious one. Don't nitpick"
      ```
    - [x] Update documentation (`conductor/index.md`, `README.md`, `GEMINI.md`)
