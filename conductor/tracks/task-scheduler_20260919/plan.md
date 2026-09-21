# Implementation Plan: Task Scheduler & Permission-Guarded Execution

## Phase 1: Storage Layer, Data Models, Migrations & Grant Verification
- [x] Task: Add Scheduler configuration to `internal/config/config.go`
    - [x] Add `SchedulerMinRunTimeout` (default `1m`) and `SchedulerMaxRunTimeout` (default `1h`)
    - [x] Add `SchedulerMinMaxTurns` (default `10`) and `SchedulerMaxMaxTurns` (default `100`)
    - [x] Unit tests in `internal/config/config_test.go` verifying env parsing and boundary enforcement
- [x] Task: Define Scheduler data models and enums
    - [x] Create `internal/scheduler/models.go` with `Schedule`, `ScheduleGrant`, `PermissionEnvelope`, `ScheduleType` (`ONCE`, `CRON`, `INTERVAL`), `ScheduleStatus` (`PENDING_APPROVAL`, `ACTIVE`, `PAUSED`, `COMPLETED`, `CANCELLED`, `EXPIRED`, `DENIED`), and `ExecutionResult`
    - [x] Implement `ComputeRawJSONHash(rawJSON string) string` computing SHA-256 directly on the exact raw permissions JSON string to eliminate object serialization non-determinism
    - [x] Implement schedule recurrence parser supporting 5-field cron (e.g. `robfig/cron/v3` or standard cron evaluator), duration intervals (minimum 5m in prod), and ISO8601 timestamps
    - [x] Write unit tests in `internal/scheduler/models_test.go` covering serialization, raw hash determinism, and cron/interval parsing
- [x] Task: Implement SQLite schema migrations for schedules and grants
    - [x] Create `internal/scheduler/schema.go` with `EnsureScheduleSchema(ctx context.Context, db *sql.DB) error`
    - [x] Create `schedules` table with `name`, `chat_id`, `UNIQUE(chat_id, name)`, `run_timeout_seconds` (INTEGER NOT NULL), `max_turns` (INTEGER NOT NULL), and foreign keys
    - [x] Create `schedule_grants` table storing `permission_request_json` (raw string) and `params_hash` with `FOREIGN KEY(schedule_id) REFERENCES schedules(id) ON DELETE CASCADE`
    - [x] Create indexes `idx_schedules_due` and `idx_schedules_chat_name`
    - [x] Integrate `EnsureScheduleSchema` into `internal/store/sqlite.go` and `internal/gateway/store_provider.go`
    - [x] Write schema migration tests in `internal/scheduler/schema_test.go`
- [x] Task: Implement isolated per-chat Scheduler Store
    - [x] Create `internal/scheduler/store.go` with CRUD operations:
      - `CreateSchedule(ctx, schedule, grant)`
      - `GetScheduleByName(ctx, chatID, name)`
      - `GetSchedule(ctx, id)`
      - `GetGrant(ctx, scheduleID)`
      - `ApproveSchedule(ctx, chatID, name, userID, validUntil)`
      - `DenySchedule(ctx, chatID, name)`
      - `CancelSchedule(ctx, chatID, name)`
      - `ListSchedules(ctx, chatID, activeOnly)`
      - `ClaimDueSchedules(ctx, cutoffTime, limit)` (atomic CAS update of `next_run_at`)
      - `RecordExecutionResult(ctx, scheduleID, status, nextRunAt, missedIncr)`
    - [x] Write unit tests in `internal/scheduler/store_test.go` testing CRUD by name and ID, name uniqueness constraint, transactional rollback, cascade deletion of grants, and atomic claiming
- [x] Task: Phase 1 Verification
    - [x] Run `go test -race ./internal/config/... ./internal/scheduler/...`

---

## Phase 2: Gateway Commands & Human Approval Flow
- [x] Task: Implement gateway `/schedule` command dispatcher
    - [x] Add `/schedule` handling in `internal/gateway/gateway.go` (routing to `handleScheduleCommand`)
    - [x] Implement subcommands operating on `<schedule_name>`:
      - `/schedule approve <name>`: strictly requires human sender (`msg.UserID != botID`), transitions status `PENDING_APPROVAL` -> `ACTIVE`, activates grant
      - `/schedule deny <name>`: transitions status `PENDING_APPROVAL` -> `DENIED`
      - `/schedule cancel <name>`: transitions status to `CANCELLED`, cascades deletion of grant, and triggers in-flight cleanup
      - `/schedule list`: outputs markdown table of chat schedules with Name, Type, Next Run, Status, and Permissions indicator
      - `/schedule info <name>`: outputs detailed markdown breakdown of task instruction, authorized permissions, grant metadata, hash, and run metrics
      - `/schedule pause <name>` and `/schedule resume <name>`
- [x] Task: Write Gateway unit tests for schedule commands
    - [x] Create `internal/gateway/schedule_commands_test.go` verifying:
      - Bot attempt to approve a schedule is strictly rejected
      - Non-owner attempt to approve another user's DM schedule is rejected
      - Approve successfully activates schedule by name and creates grant
      - Deny and Cancel properly revoke permissions by name
    - [x] Run `go test -race ./internal/gateway/...`

---

## Phase 3: Agent Tools (`schedule_task`) & Validation
- [x] Task: Implement `schedule_task` tool in tool registry
    - [x] Define `ScheduleTaskArgs` and JSON schema in `internal/tools/registry.go`:
      - `schedule_name`: string, mandatory, lowercase, max 4 words separated by underscores (regex: `^[a-z0-9]+(_[a-z0-9]+){0,3}$`)
      - `type`: string, mandatory, enum `["once", "cron", "interval"]`
      - `schedule_spec`: string, mandatory
      - `instruction`: string, mandatory
      - `run_timeout_seconds`: integer, mandatory (no tool default; strictly validated against `cfg.SchedulerMinRunTimeout` and `cfg.SchedulerMaxRunTimeout`)
      - `max_turns`: integer, mandatory (no tool default; strictly validated against `cfg.SchedulerMinMaxTurns` and `cfg.SchedulerMaxMaxTurns`)
      - `permissions`: object, optional (`{"sandbox": {...}}`)
      - `required`: `["schedule_name", "type", "schedule_spec", "instruction", "run_timeout_seconds", "max_turns"]`
    - [x] Extract raw JSON substring for `permissions` object to pass directly to hashing and storage
    - [x] Enforce DM-only constraint if sandbox permissions are requested (`session.IsDM`)
    - [x] Enforce minimum interval (5 minutes) and maximum expiration duration (30 days)
    - [x] If sandbox requested:
      - Validate driver against `cfg.Drivers`, image against `cfg.AllowedImages`, network against allowed modes
      - Set schedule status to `PENDING_APPROVAL`
      - Format prompt and return approval instructions for the user (`/schedule approve <name>`)
    - [x] If no elevated sandbox requested:
      - Set schedule status directly to `ACTIVE`
- [x] Task: Implement `list_schedules` and `cancel_schedule` tools
    - [x] Expose read-only `list_schedules` to allow agent to inspect active schedules
    - [x] Expose mutating `cancel_schedule` (by `schedule_name`) to allow agent to cancel on user conversational request
- [x] Task: Write tool registry unit tests
    - [x] Add tests in `internal/tools/tools_test.go` verifying name validation regex, mandatory bounds validation, raw hash extraction, permission separation, and anti-self-grant invariants
    - [x] Run `go test -race ./internal/tools/...`

---

## Phase 4: Scheduler Engine, Background Poller & Downtime Recovery
- [x] Task: Implement core `scheduler.Engine`
    - [x] Create `internal/scheduler/engine.go` managing background polling ticker (every 5 seconds)
    - [x] Implement atomic claiming of due tasks via CAS
    - [x] Implement downtime recovery & anti-burst policy:
      - When checking past-due schedules, if overdue by `< 15m`, trigger single run (`catchup_once`)
      - If overdue by `> 15m` (extended downtime), skip missed runs, increment `missed_count`, and recalculate next future occurrence
- [x] Task: Write scheduler engine unit tests
    - [x] Create `internal/scheduler/engine_test.go` testing non-blocking poller, concurrent workers, downtime catchup, and anti-burst skipping
    - [x] Run `go test -race ./internal/scheduler/...`

---

## Phase 5: Ephemeral Sandbox Execution & FSM Integration
- [x] Task: Implement `ScheduleInvoker` and FSM run creation
    - [x] Create `internal/gateway/schedule_invoker.go` bridging the scheduler to `fsm.Engine`
    - [x] Spawn FSM run passing `run_timeout_seconds` as context deadline and `max_turns` as iteration cap
    - [x] Implement bounded 30s lock wait if user interactive sandbox is active:
      - [x] If acquired: create ephemeral sandbox with granted parameters
      - [x] If busy after 30s: record `SKIPPED_BUSY`, increment `missed_count`, reschedule next run without dropping schedule
- [x] Task: Enforce sandbox lifecycle and anti-escalation wrapper
    - [x] In execution worker, ensure `defer sandboxManager.Destroy(...)` guarantees immediate teardown of ephemeral container upon FSM run termination
    - [x] In tool execution for scheduled runs:
      - [x] Validate tool against grant `params_hash` (checked against raw JSON)
      - [x] Hide `sandbox_request` (auto-configured by grant)
      - [x] Expose `sandbox_exec` scoped strictly to the ephemeral sandbox
      - [x] Disallow network policy modifications or unapproved mounts
- [x] Task: Write sandbox scheduler integration tests
    - [x] Create `internal/gateway/schedule_invoker_test.go` verifying ephemeral sandbox creation, bounded wait on busy locks, auto-destroy on completion, parameter validation, and hard turn/timeout enforcement
    - [x] Run `go test -race ./internal/gateway/...`

---

## Phase 6: Lifecycle Wiring, Independent Pair Review & Local Live Verification
- [x] Task: Wire Scheduler into application lifecycle
    - [x] Instantiate `scheduler.Engine` in `cmd/agent/main.go` and start/stop alongside `Gateway` and `FSM`
    - [x] Connect chat notification dispatcher to post scheduled task results to Besedka chats
    - [x] Connect schedule pruning and vacuum to periodic maintenance ticker
- [x] Task: Write end-to-end automated integration tests
    - [x] Add integration tests in `internal/gateway/schedule_integration_test.go` testing full flow:
      `schedule_task` tool -> pending approval -> `/schedule approve <name>` -> scheduled trigger -> ephemeral sandbox exec -> result posted -> sandbox destroyed
- [x] Task: Conduct independent peer model review via `agentic-pair-programming` skill
    - [x] Launch `agent-pair` session using an alternate model (`opencode/big-pickle`)
    - [x] Have peer review all unstaged changes across phases, and address all serious concerns (C1, C2, H1, H2, H3, M1, M3, M7)
    - [x] Cleanly close `agent-pair` session
- [x] Task: Conduct second peer review with `agy` follower using `claude-opus-4-6-thinking` model
    - [x] Launch `agent-pair start --follower agy --model claude-opus-4-6-thinking`
    - [x] Receive detailed critique categorized by severity (H1-H4, M1-M7, L1-L5)
    - [x] Implement and verify fixes:
      - H1: TOCTOU elimination via `querier` interface and tx-scoped reads in `store.go` (`ApproveSchedule`, `DenySchedule`, `CancelSchedule`, `PauseSchedule`, `ResumeSchedule`)
      - H2: Dedicated 5-second `writeCtx` and error logging for `RecordExecutionResult` in `engine.go`
      - M1: Wrapped `PauseSchedule` and `ResumeSchedule` in atomic transactions in `store.go`
      - M2: Split multi-statement SQLite PRAGMA in `schema.go` into individual executions
      - M5: Added schedule creator ownership verification in `cancel_schedule` tool in `scheduler_tools.go`
      - L3: Added "d" (days) duration parsing support in `parseExpiresAt`
      - L4: Rune-aware safe UTF-8 status truncation in `engine.go`
    - [x] Cleanly terminate pair session via `agent-pair stop`
- [x] Task: Local Live Verification with Besedka
    - [x] Launch local Besedka server (`http://localhost:8080`)
    - [x] Run `go run ./cmd/agent` connected to local Besedka
    - [x] Perform live end-to-end interactive test in chat:
      - Mention bot in DM to schedule a recurring sandbox task
      - Verify response contains approval instructions with schedule name
      - Execute `/schedule approve <name>`
      - Observe scheduled FSM execution, sandbox spawn, command execution, and post-task destruction
      - Execute `/schedule cancel <name>` and verify revocation
- [x] Task: Quality & Standards Verification
    - [x] Run `go mod tidy` and `go mod vendor`
    - [x] Run `make check` (must pass with 0 errors across linter, tests, semgrep, and osv-scanner)
