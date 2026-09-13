# Specification: Durable FSM for Simple Tool Loops & Agentic Workflows

## 1. Overview
This track introduces a persistent, isolated, pure-Go Finite State Machine (FSM) engine to replace the volatile in-memory tool execution loop in Bob (`internal/llm.GenerateChatResponseWithToolLoop`). The FSM persists execution state directly inside Bob's existing isolated per-chat SQLite databases (`townhall.db` and `dm_<sanitized_chatID>.db`), supports both **sequential** and **parallel** tool execution modes, provides first-class support for **delayed state transitions** (`resume_at`) for future scheduler/timer integration, enforces a configurable retention policy (`FSM_RETENTION_DAYS`, defaulting to 7 days), and establishes the architectural foundation for Phase 16's deep agentic planning and reflection harness.

## 2. Architecture & Data Isolation Model

### 2.1 File & Directory Layout
All FSM tables are maintained directly inside each chat's existing SQLite database file in `DATA_DIR`:
- `townhall.db`: Stores workflows originating in the public townhall chat.
- `dm_<sanitized_chatID>.db`: Stores private workflows for that specific 1-on-1 Direct Message conversation.

### 2.2 Privacy & Security Guarantees
- Zero cross-chat leakage: Townhall cannot query or modify DM workflows, and DM $A$ cannot access DM $B$.
- Off-site backup synergy: Active and completed workflows are automatically captured, defragmented, and encrypted by Bob's existing S3 backup pipeline (`internal/backup`) without requiring new backup logic.

### 2.3 SQLite Schema

#### `fsm_runs`
```sql
CREATE TABLE IF NOT EXISTS fsm_runs (
    id TEXT PRIMARY KEY,                       -- UUID: e.g. "run_01j7abc..."
    chat_id TEXT NOT NULL,                     -- Chat ID
    user_id TEXT NOT NULL,                     -- Initiating User ID
    fsm_type TEXT NOT NULL,                    -- 'tool_loop' or 'agentic_workflow'
    status TEXT NOT NULL,                      -- 'PENDING', 'RUNNING', 'WAITING', 'COMPLETED', 'FAILED', 'TERMINATED'
    current_state TEXT NOT NULL,               -- e.g. 'INIT', 'LLM_REQUEST', 'TOOL_EXECUTION', 'SYNTHESIS'
    iteration INTEGER NOT NULL DEFAULT 0,      -- Current loop iteration
    max_iterations INTEGER NOT NULL DEFAULT 20,-- Channel iteration cap (10 Townhall, 20 DM)
    context_json TEXT NOT NULL,                -- Serialized []openai.ChatCompletionMessage + session state
    result_json TEXT,                          -- Final response text or synthesized output
    error_text TEXT,                           -- Error description if failed/terminated
    resume_at INTEGER,                         -- Unix timestamp (seconds) for delayed transition / timer
    created_at INTEGER NOT NULL,               -- Unix timestamp (seconds)
    updated_at INTEGER NOT NULL                -- Unix timestamp (seconds)
);

CREATE INDEX IF NOT EXISTS idx_fsm_runs_active 
ON fsm_runs(status, resume_at) 
WHERE status IN ('PENDING', 'RUNNING', 'WAITING');
```

#### `fsm_steps`
```sql
CREATE TABLE IF NOT EXISTS fsm_steps (
    id TEXT PRIMARY KEY,                       -- UUID: e.g. "step_01j7xyz..."
    run_id TEXT NOT NULL,                      -- Foreign key referencing fsm_runs(id) ON DELETE CASCADE
    iteration INTEGER NOT NULL,                -- Loop iteration that created this step
    step_index INTEGER NOT NULL,               -- Sequence index within the iteration batch (0-indexed)
    tool_name TEXT NOT NULL,                   -- e.g. 'web_search', 'sandbox_exec'
    tool_call_id TEXT NOT NULL,                -- OpenAI ToolCall ID
    args_json TEXT NOT NULL,                   -- Arguments passed to the tool
    result_json TEXT,                          -- Tool output string or JSON
    execution_mode TEXT NOT NULL,              -- 'sequential' or 'parallel'
    status TEXT NOT NULL,                      -- 'PENDING', 'RUNNING', 'COMPLETED', 'FAILED', 'TIMED_OUT', 'SKIPPED'
    attempt INTEGER NOT NULL DEFAULT 0,        -- Current retry attempt
    max_attempts INTEGER NOT NULL DEFAULT 3,   -- Max retry attempts
    timeout_seconds INTEGER NOT NULL DEFAULT 30-- Per-step timeout deadline
    started_at INTEGER,                        -- Unix timestamp (seconds)
    completed_at INTEGER,                      -- Unix timestamp (seconds)
    error_text TEXT,                           -- Step-level failure error
    FOREIGN KEY(run_id) REFERENCES fsm_runs(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_fsm_steps_run_iter 
ON fsm_steps(run_id, iteration, step_index);
```

## 3. Execution Engine & Tool Classification

### 3.1 Execution Modes
- **Parallel Mode:** Applied when all tool calls in an iteration batch are read-only (`web_search`, `web_fetch`, `recall_memory`). Dispatches steps concurrently using a worker pool bounded by `maxParallelWorkers = 4` and synchronizes with `sync.WaitGroup`.
- **Sequential Mode:** Applied when the batch contains mutating tools (`sandbox_exec`, file writes, sandbox lifecycle) or dependent steps. Steps execute in strict sequential order by `step_index`.

### 3.2 Tool Call Resilience
- **Granular Timeouts:** Enforce explicit per-tool deadlines using Go `context.WithTimeout(ctx, step.Timeout)`.
- **Transient Retries:** Retry transient failures (network timeouts, HTTP 429 rate limits, 503 errors) with exponential backoff and jitter up to `max_attempts`.
- **Deterministic Errors:** Deterministic errors (e.g. invalid JSON args, missing files) fail immediately without wasteful retries.

### 3.3 Delayed Transitions & Scheduler Integration
- State transitions can set a future `resume_at` timestamp and enter `WAITING` status.
- The engine uses an in-memory timer channel for immediate wakeups alongside a background SQLite poller (every 2 seconds) to find runs where `status = 'WAITING' AND resume_at <= now()`.
- On service restart, interrupted runs in `RUNNING` or `WAITING` are recovered and resumed from their last committed state.

## 4. Retention & SQLite Page Management

### 4.1 Configurable Retention (`FSM_RETENTION_DAYS`)
- New configuration variable: `FSM_RETENTION_DAYS` (integer, default `7`).
- Active and scheduled runs (`status IN ('PENDING', 'RUNNING', 'WAITING')`) are **never** pruned regardless of age.
- Expired terminal runs (`status IN ('COMPLETED', 'FAILED', 'TERMINATED')` with `updated_at < now - retention_days`) are deleted directly.
- Cascading deletes (`ON DELETE CASCADE`) atomically remove child `fsm_steps`.

### 4.2 Incremental Vacuuming
- Chat databases set `PRAGMA auto_vacuum = INCREMENTAL;` at creation.
- Periodic maintenance executes `PRAGMA incremental_vacuum(500);` to release freed pages back to the OS without exclusive long-running table locks.

## 5. Channel Iteration Budgets
- **Townhall Chat:** Maximum 10 tool iterations per request.
- **Direct Message (DM):** Maximum 20 tool iterations per request.

## 6. Acceptance Criteria
1. Unit tests for `internal/fsm` storage, models, migrations, and concurrency safety pass with `-race`.
2. Step executor correctly executes independent read-only tools in parallel and mutating tools sequentially.
3. Per-tool timeouts cancel hanging tools cleanly; transient errors trigger exponential backoff retries.
4. Delayed transitions (`resume_at`) suspend runs and resume accurately upon timer expiry or service restart.
5. Retention manager prunes terminal runs past `FSM_RETENTION_DAYS` and executes incremental vacuum while leaving active runs untouched.
6. Gateway integration replaces the volatile loop seamlessly for both Townhall (limit 10) and DM (limit 20).
7. Full test suite and `make check` pass with 0 errors.
