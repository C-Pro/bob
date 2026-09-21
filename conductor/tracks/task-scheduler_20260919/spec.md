# Specification: Task Scheduler & Permission-Guarded Execution

## 1. Overview
This track introduces a background **Task Scheduler** in Bob to support recurring (`cron` / `interval`) and delayed (one-shot future) task executions. The scheduler integrates directly with Bob's **Durable FSM** (`internal/fsm`) and per-chat SQLite databases (`townhall.db`, `dm_<sanitized_chatID>.db`).

A central requirement is strict **Permission Guarding**:
If a scheduled task requires elevated privileges (specifically spawning an isolated environment sandbox with configured drivers, network access modes, allowed external domains, and filesystem mounts), these permissions **must be requested and granted by the human user immediately after scheduling**. 
Once granted, the authorization remains valid for the active lifetime of the schedule across all recurring executions. Permissions can **never** be self-granted or escalated by an agent, and all grants are immediately revoked upon schedule cancellation, completion, or expiration.

---

## 2. Threat Model & Security Invariants

1. **Human-Only Authorization Barrier:**
   - Permissions can ONLY be granted via an explicit user command in chat (`/schedule approve <name>`).
   - The agent tool `schedule_task` can only create a schedule in `PENDING_APPROVAL` status when elevated permissions are requested; it has no mechanism to activate or grant permissions to its own requests.
   - Gateway handlers strictly verify that the approval message sender (`msg.UserID`) is the human user matching the DM session and never matches `botUserID`.

2. **Immutable Capability Envelope & Deterministic Raw JSON Hashing:**
   - Permission requests are encapsulated in an extensible object:
     ```json
     {
       "sandbox": {
         "driver": "docker",
         "image": "alpine:latest",
         "network": "restricted",
         "domains": ["api.github.com"],
         "mounts": [],
         "timeout_seconds": 60
       }
     }
     ```
   - To completely eliminate non-determinism from object deserialization, struct re-marshalling, and map key reordering, `params_hash` is computed as the SHA-256 digest directly on the **exact original raw permission request JSON string**.
   - The exact raw string is stored verbatim in `schedule_grants.permission_request_json`. During execution, the runtime environment is verified against this exact raw specification. Any attempt by the LLM or tool loop to deviate (e.g. requesting unapproved domains, changing container images, or mounting unauthorized directories) is rejected with a fatal authorization error.

3. **Hard Execution Bounds on Single Runs (Mandatory with Configurable Limits):**
   - Every schedule defines hard execution limits:
     - `run_timeout_seconds` (INTEGER): hard deadline enforced by the runner context.
     - `max_turns` (INTEGER): maximum FSM tool loop iterations permitted for a single execution.
   - Both parameters are **mandatory** in `schedule_task` (no tool defaults; the caller must supply them).
   - Server configuration bounds (in `internal/config/config.go`) are customizable via environment variables with defaults:
     - Duration limit: `SCHEDULER_MIN_RUN_TIMEOUT` (default: `1m` / 60s), `SCHEDULER_MAX_RUN_TIMEOUT` (default: `1h` / 3600s).
     - Iteration turns: `SCHEDULER_MIN_MAX_TURNS` (default: `10`), `SCHEDULER_MAX_MAX_TURNS` (default: `100`).
   - The tool strictly validates that `run_timeout_seconds` and `max_turns` fall within these configured bounds and rejects requests outside this envelope.
   - These limits prevent any scheduled task from monopolizing resources or entering runaway execution loops.

4. **Cascade Revocation & Lifetime Coupling:**
   - The `schedule_grants` table maintains a foreign key constraint `REFERENCES schedules(id) ON DELETE CASCADE`.
   - Transitioning a schedule to `CANCELLED`, `COMPLETED`, `EXPIRED`, or `DENIED` immediately invalidates the grant. Any active in-flight ephemeral sandbox associated with that schedule is terminated and destroyed.

5. **Zero Cross-Chat Leakage (Per-Chat SQLite Isolation):**
   - Schedules and permission grants reside strictly inside the SQLite database corresponding to the chat session where they were created.
   - Townhall cannot access or execute DM schedules, and User $A$'s DM database cannot inspect or trigger User $B$'s DM schedules.

6. **Sandbox Scope Restriction:**
   - Consistent with Bob's existing security architecture (`internal/tools/registry.go:708`), scheduled tasks requiring sandbox execution are strictly restricted to 1-on-1 Direct Messages (DMs).

---

## 3. Data Model & Schema

All tables are created in each chat's SQLite database via `internal/scheduler/schema.go` and integrated into `internal/store/sqlite.go`:

### 3.1 `schedules` Table
```sql
CREATE TABLE IF NOT EXISTS schedules (
    id TEXT PRIMARY KEY,                       -- e.g. "sched_01j8abc..."
    name TEXT NOT NULL,                        -- Human-readable schedule identifier (e.g. 'daily_brief')
    chat_id TEXT NOT NULL,                     -- Chat ID context
    user_id TEXT NOT NULL,                     -- Creator user ID
    schedule_type TEXT NOT NULL,               -- 'ONCE', 'CRON', 'INTERVAL'
    cron_expr TEXT,                            -- Standard 5-field cron (e.g. '0 9 * * *')
    interval_seconds INTEGER,                  -- Interval in seconds (minimum 300s in prod)
    instruction TEXT NOT NULL,                 -- Prompt / task instruction for the agent
    status TEXT NOT NULL,                      -- 'PENDING_APPROVAL', 'ACTIVE', 'PAUSED', 'COMPLETED', 'CANCELLED', 'EXPIRED', 'DENIED'
    next_run_at INTEGER NOT NULL,              -- Next execution time (Unix timestamp in seconds)
    last_run_at INTEGER,                       -- Last execution time (Unix timestamp in seconds)
    expires_at INTEGER,                        -- Overall schedule expiration deadline
    max_runs INTEGER NOT NULL DEFAULT 0,       -- Cap on total runs (0 = unlimited until expires_at)
    run_count INTEGER NOT NULL DEFAULT 0,      -- Number of completed executions
    missed_count INTEGER NOT NULL DEFAULT 0,   -- Count of missed runs (downtime or busy skips)
    run_timeout_seconds INTEGER NOT NULL,      -- Hard execution timeout per run (mandatory, validated against config)
    max_turns INTEGER NOT NULL,                -- Hard FSM iteration cap per run (mandatory, validated against config)
    last_status TEXT,                          -- 'SUCCESS', 'FAILED', 'SKIPPED_BUSY'
    created_at INTEGER NOT NULL,               -- Unix timestamp in seconds
    updated_at INTEGER NOT NULL,               -- Unix timestamp in seconds
    CONSTRAINT uq_schedules_chat_name UNIQUE(chat_id, name)
);

CREATE INDEX IF NOT EXISTS idx_schedules_due
ON schedules(status, next_run_at)
WHERE status = 'ACTIVE';

CREATE INDEX IF NOT EXISTS idx_schedules_chat_name
ON schedules(chat_id, name);
```

### 3.2 `schedule_grants` Table
```sql
CREATE TABLE IF NOT EXISTS schedule_grants (
    id TEXT PRIMARY KEY,                       -- e.g. "grant_01j8xyz..."
    schedule_id TEXT NOT NULL UNIQUE,          -- FK to schedules(id) ON DELETE CASCADE
    permission_request_json TEXT NOT NULL,     -- Original raw JSON string of requested permissions
    params_hash TEXT NOT NULL,                 -- SHA-256 digest of exact raw JSON string
    granted_by TEXT NOT NULL,                  -- Human user ID who executed /schedule approve
    granted_at INTEGER NOT NULL,               -- Unix timestamp in seconds
    valid_until INTEGER NOT NULL,              -- Unix timestamp in seconds (matches schedule expires_at)
    FOREIGN KEY(schedule_id) REFERENCES schedules(id) ON DELETE CASCADE
);
```

---

## 4. Scheduler Engine & Polling Architecture

### 4.1 Non-Blocking Poller Tick
- The `scheduler.Engine` runs a background ticker (e.g. every 5 seconds).
- It queries active chat stores for schedules where `status = 'ACTIVE' AND next_run_at <= now()`.
- **Claiming:** Schedules are claimed atomically using an optimistic CAS update (`UPDATE schedules SET next_run_at = ?, updated_at = ? WHERE id = ? AND next_run_at = ?`) to eliminate race conditions across multiple workers.
- The tick loop never executes tasks synchronously; it dispatches execution to an asynchronous worker pool.

### 4.2 Downtime Catch-Up & Anti-Burst Protection
- When the agent restarts or recovers after downtime:
  - If a schedule's `next_run_at` is in the past:
    - If lateness is within allowable threshold (e.g. 15 minutes, matching `recoveryStalenessCutoff` in FSM): trigger a single run (`catchup_once`), then compute next occurrence from cron/interval.
    - If lateness exceeds threshold (e.g. down for 3 days): do NOT burst-fire 100 missed iterations. Increment `missed_count` by the calculated missed intervals, record audit warning, and advance `next_run_at` to the next future occurrence.

### 4.3 Sandbox Concurrency & Bounded Wait
- If a scheduled task requires sandbox execution and the user's interactive sandbox is currently locked:
  - The worker attempts to acquire the user sandbox lock with a **bounded wait of 30 seconds**.
  - If the lock is acquired within 30s: execute normally.
  - If the lock remains busy after 30s (e.g. user executing a heavy interactive build): do not hang the worker or drop the schedule. Record `last_status = 'SKIPPED_BUSY'`, increment `missed_count`, advance `next_run_at` to the next cycle, and emit an informational notice in chat.

---

## 5. Ephemeral Execution & FSM Integration

### 5.1 FSM Run Spawning
When a schedule fires:
1. The scheduler creates a persistent `fsm_runs` entry with:
   - `fsm_type = 'tool_loop'` (or `'agentic_workflow'`)
   - `chat_id = schedule.chat_id`
   - `user_id = schedule.user_id`
   - `max_iterations = schedule.max_turns`
   - Context deadline initialized to `time.Now().Add(time.Duration(schedule.run_timeout_seconds) * time.Second)`
   - Context initialized with task instruction and audit header identifying the schedule name.
   - `context_json` annotated with `schedule_name`, `schedule_id`, and verified `params_hash`.

### 5.2 Ephemeral Sandbox Lifecycle
1. If the schedule has a grant:
   - The worker initializes an ephemeral sandbox session matching the exact granted parameters parsed from `permission_request_json`.
   - The tool registry passed to the FSM hides `sandbox_request` (since configuration is pre-approved) and provides `sandbox_exec` bound directly to the ephemeral sandbox.
2. **Guaranteed Teardown:**
   - An explicit `defer` hook in the execution worker destroys the ephemeral sandbox and cleans up proxy forwarders as soon as the FSM run finishes (`COMPLETED`, `FAILED`, or `TERMINATED`).
   - If a crash occurs mid-execution, Bob's startup reaper identifies and cleans orphaned containers.

### 5.3 Task Completion Notification
- Upon completion of the scheduled FSM run, Bob sends a message into the chat with the results of the task execution (e.g. "📅 **Scheduled Task Executed:** `[daily_brief]` ...").

---

## 6. User Experience & Command Surface

### 6.1 Agent Tool: `schedule_task`
Exposed to the LLM in 1-on-1 DMs:
```json
{
  "name": "schedule_task",
  "description": "Schedule a delayed or recurring background task execution. Both run_timeout_seconds and max_turns are required with no defaults.",
  "parameters": {
    "type": "object",
    "properties": {
      "schedule_name": {
        "type": "string",
        "description": "Human-readable schedule identifier (lowercase, max 4 words separated by underscores, e.g. 'daily_brief')"
      },
      "type": {"type": "string", "enum": ["once", "cron", "interval"]},
      "schedule_spec": {"type": "string", "description": "ISO8601 timestamp, 5-field cron expression, or duration string (e.g. '1h', '30m')"},
      "instruction": {"type": "string", "description": "Detailed task instructions for the agent to execute"},
      "expires_in_hours": {"type": "integer", "description": "Maximum schedule validity duration (default 168 = 7 days, max 720 = 30 days)"},
      "max_runs": {"type": "integer", "description": "Maximum number of executions before schedule auto-completes (0 = unlimited)"},
      "run_timeout_seconds": {
        "type": "integer",
        "description": "Mandatory execution timeout in seconds for a single execution (must be between SCHEDULER_MIN_RUN_TIMEOUT and SCHEDULER_MAX_RUN_TIMEOUT, default config range: 60 to 3600)"
      },
      "max_turns": {
        "type": "integer",
        "description": "Mandatory tool loop iterations for a single execution (must be between SCHEDULER_MIN_MAX_TURNS and SCHEDULER_MAX_MAX_TURNS, default config range: 10 to 100)"
      },
      "permissions": {
        "type": "object",
        "description": "Optional permission requests required for task execution (e.g. sandbox)",
        "properties": {
          "sandbox": {
            "type": "object",
            "properties": {
              "driver": {"type": "string", "enum": ["bwrap", "docker"]},
              "image": {"type": "string"},
              "network": {"type": "string", "enum": ["none", "restricted"]},
              "domains": {"type": "array", "items": {"type": "string"}},
              "mounts": {"type": "array", "items": {"type": "object", "properties": {"path": {"type": "string"}, "read_only": {"type": "boolean"}}}},
              "timeout_seconds": {"type": "integer"}
            }
          }
        }
      }
    },
    "required": ["schedule_name", "type", "schedule_spec", "instruction", "run_timeout_seconds", "max_turns"]
  }
}
```

### 6.2 Gateway Human Commands
All commands operate on `<schedule_name>`:
- `/schedule approve <name>` — Approves pending permissions for schedule `<name>` and activates it. Strictly rejected if sender is `botUserID`.
- `/schedule deny <name>` — Denies pending request and marks schedule as `DENIED`.
- `/schedule cancel <name>` — Cancels an active or pending schedule and immediately revokes all permission grants.
- `/schedule list` — Lists all active, pending, and paused schedules in the current chat.
- `/schedule info <name>` — Displays full schedule configuration, grant metadata, hash, run statistics, and next run time.
- `/schedule pause <name>` / `/schedule resume <name>` — Temporarily pauses or resumes recurring triggers.
