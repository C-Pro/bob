# Specification: Phase 5 Trajectory Logger

## 1. Objective & Background
Phase 5 introduces the **Trajectory Logger** (`internal/trajectory`), establishing the data collection foundation for continuous learning, offline evaluation, and future reinforcement learning (Phase 8 PESO/FST).

A trajectory rollout captures the complete interaction tuple:
$$\tau = (\text{id}, x, \Phi_{\text{task}}, z, a, o, y, R_{\text{eval}})$$
where:
- $x$: user input prompt (raw, stripped)
- $\Phi_{\text{task}}$: active task/memory context and active skills
- $z$: model reasoning trace (Chain of Thought / provider reasoning content)
- $a$: tool action name and parameters
- $o$: raw tool execution observation / output
- $y$: final assistant response
- $R_{\text{eval}}$: scalar execution heuristic evaluation metric $[-1.0, 1.0]$

Because user feedback (a correction on the next turn, a `/good` or `/bad` command, or an explicit rating) arrives *after* turn execution completes, trajectories cannot be flushed directly to cold object storage upon FSM termination.

Bob uses a **2-Stage SQLite Staging + Delayed Revisioned S3 Flush** architecture:
1. **Durable Lifecycle Event Outbox:** Raw lifecycle events (`turn_start`, `tool_attempt`, `llm_response`, `turn_terminal`) are logged to an append-only outbox in the per-chat SQLite database, committing observations atomically with FSM step transitions.
2. **Staged Turn Projections:** On turn completion (or crash recovery reconciliation), a canonical `Trajectory` record is materialized in SQLite with an initial execution heuristic score and an export eligibility deadline.
3. **Revisioned Feedback Mutation:** User feedback during the feedback window updates the audit trail, bumps the trajectory's `revision`, resets status to `staged`, and recalculates $R_{\text{eval}}$.
4. **Asynchronous S3 Export:** A dedicated worker flushes mature, export-approved revisions to S3/MinIO using `internal/objectstore`, acknowledging each revision independently to eliminate export-vs-mutation races.

---

## 2. Data Schema & Domain Models

Domain models reside in `internal/trajectory/models.go`:

```go
package trajectory

import (
	"encoding/json"
	"time"
)

// Event represents an immutable lifecycle event in the execution outbox.
type Event struct {
	ID           string          `json:"id"`            // UUID
	TrajectoryID string          `json:"trajectory_id"` // Trajectory UUID
	RunID        string          `json:"run_id"`        // FSM Run ID
	Iteration    int             `json:"iteration"`     // FSM loop iteration counter
	Phase        string          `json:"phase"`         // "start", "loop", "synthesis", "terminal"
	StepID       string          `json:"step_id"`       // FSM step identifier
	Attempt      int             `json:"attempt"`       // 1-based retry attempt
	EventType    string          `json:"event_type"`    // "turn_start", "tool_attempt", "llm_response", "turn_terminal"
	Payload      json.RawMessage `json:"payload"`       // Type-specific event details
	CreatedAt    time.Time       `json:"created_at"`
}

// Step represents a single tool execution action-observation pair within a turn.
type Step struct {
	Index       int             `json:"index"`
	Iteration   int             `json:"iteration"`             // FSM iteration
	Attempt     int             `json:"attempt"`               // 1-based attempt counter for retries
	Thought     string          `json:"thought,omitempty"`     // z: reasoning trace / provider reasoning content
	Action      string          `json:"action"`                // a: tool name
	ActionArgs  json.RawMessage `json:"action_args"`           // a: tool input arguments JSON
	Observation string          `json:"observation"`           // o: raw untruncated tool output / stdout / stderr
	ExitCode    *int            `json:"exit_code,omitempty"`   // nullable: exit code for execute/sandbox tools
	DurationMs  int64           `json:"duration_ms"`           // tool execution duration
}

// EvalBreakdown decomposes the components contributing to R_eval.
type EvalBreakdown struct {
	TotalToolCalls      int     `json:"total_tool_calls"`      // Total logical tool calls attempted
	SuccessfulToolCalls int     `json:"successful_tool_calls"` // Tool calls completing without error / exit 0
	FailedToolCalls     int     `json:"failed_tool_calls"`     // Tool calls returning errors or non-zero exits
	ToolSuccessRate     float64 `json:"tool_success_rate"`     // [0.0, 1.0] fraction of successful invocations
	BudgetExhausted     bool    `json:"budget_exhausted"`       // true if iteration or token budget was hit
	FSMFailed           bool    `json:"fsm_failed"`             // true if FSM terminated with unhandled error
	TerminationReason   string  `json:"termination_reason"`    // "completed", "budget_exhausted", "fsm_failed", "cancelled", "delivery_failed"
	HasUserFeedback     bool    `json:"has_user_feedback"`      // true if explicit user feedback was received
	UserFeedback        float64 `json:"user_feedback"`          // -1.0 (bad), 0.0 (neutral), +1.0 (good)
	FeedbackNote        string  `json:"feedback_note,omitempty"`
	ImplicitSignal      float64 `json:"implicit_signal"`       // -1.0 to +1.0 derived from follow-up turns
	ImplicitProvenance  string  `json:"implicit_provenance,omitempty"` // explanation/source of implicit score
	RewardVersion       string  `json:"reward_version"`         // e.g. "v1"
}

// Trajectory represents the canonical rollout record for a single logical turn.
type Trajectory struct {
	ID             string        `json:"id"`              // UUID / unique trajectory ID
	RunID          string        `json:"run_id"`          // FSM Run ID or Turn ID
	FrontendID     string        `json:"frontend_id"`     // "besedka", "cli", etc.
	SessionID      string        `json:"session_id"`      // Chat ID or CLI session string
	ScopeID        string        `json:"scope_id"`        // Channel / DM / workspace identifier
	ActorID        string        `json:"actor_id"`        // User ID string
	Model          string        `json:"model"`           // LLM model identifier
	ExecutionKind  string        `json:"execution_kind"`  // "interactive" or "scheduled"
	PrivacyLevel   string        `json:"privacy_level"`   // "public", "private", "internal"
	ExportApproved bool          `json:"export_approved"` // true if approved for objectstore upload
	CreatedAt      time.Time     `json:"created_at"`      // Turn start timestamp
	CompletedAt    time.Time     `json:"completed_at"`    // Turn completion timestamp
	UserInput      string        `json:"x"`               // x: raw stripped user input
	TaskState      string        `json:"phi_task"`        // \Phi_{task}: active memory/task context
	ActiveSkills   []string      `json:"active_skills,omitempty"`
	Steps          []Step        `json:"steps"`           // (z, a, o) sequence
	FinalOutput    string        `json:"y"`               // y: final assistant response
	Reward         float64       `json:"r_eval"`          // R_{eval}: composite execution heuristic [-1.0, 1.0]
	Breakdown      EvalBreakdown `json:"eval_breakdown"`
	Revision       int           `json:"revision"`        // Monotonic revision counter (starts at 1)
}
```

---

## 3. Reward Formulation ($R_{\text{eval}}$)

Calculated in `internal/trajectory/reward.go`.

> [!NOTE]
> $R_{\text{eval}}$ is explicitly defined as an **execution heuristic** over tool behavior and feedback signals, not ground-truth verified task success.

### 3.1 Formulation (Version `v1`)

$$R_{\text{eval}} = \text{clamp}\Big(S_{\text{base}} - 0.5 \cdot \mathbb{I}_{\text{fail}} - 0.3 \cdot \mathbb{I}_{\text{budget}} + 0.6 \cdot F_{\text{user}}, -1.0, 1.0\Big)$$

Where:
- $S_{\text{base}}$:
  - If **0 tool calls** were attempted:
    - If turn completed normally: $S_{\text{base}} = +0.20$ (conversational answers are standard and not penalized).
    - If turn terminated with error/cancellation: $S_{\text{base}} = 0.00$.
  - If **$\ge 1$ tool calls** were attempted:
    $$S_{\text{tool}} = \frac{\text{SuccessfulToolCalls}}{\text{TotalToolCalls}}$$
    $$S_{\text{base}} = 0.4 \cdot (2 \cdot S_{\text{tool}} - 1)$$
    - When $S_{\text{tool}} = 1.0 \implies S_{\text{base}} = +0.40$
    - When $S_{\text{tool}} = 0.5 \implies S_{\text{base}} = 0.00$
    - When $S_{\text{tool}} = 0.0 \implies S_{\text{base}} = -0.40$
- $\mathbb{I}_{\text{fail}} \in \{0, 1\}$: $1$ if FSM failed, crashed, or was unrecovered; $0$ otherwise.
- $\mathbb{I}_{\text{budget}} \in \{0, 1\}$: $1$ if execution budget was exhausted before normal termination; $0$ otherwise.
- $F_{\text{user}} \in [-1.0, 1.0]$: Explicit user feedback. If absent (`HasUserFeedback == false`), the term contributes $0.0$.

### 3.2 Evaluation Table & Acceptance Cases

| Scenario | Tool Count | $S_{\text{tool}}$ | $\mathbb{I}_{\text{fail}}$ | $\mathbb{I}_{\text{budget}}$ | $F_{\text{user}}$ | $R_{\text{eval}}$ | Interpretation |
|---|---|---|---|---|---|---|---|
| Normal conversational turn (0 tools) | 0 | N/A | 0 | 0 | None | **+0.20** | Clean conversational response |
| Conversational turn + positive feedback | 0 | N/A | 0 | 0 | +1.0 | **+0.80** | Positively rated direct answer |
| Conversational turn + negative feedback | 0 | N/A | 0 | 0 | -1.0 | **-0.40** | Rejected direct answer |
| All tools succeed ($1.0$), normal | $\ge 1$ | 1.0 | 0 | 0 | None | **+0.40** | Successful tool execution |
| All tools succeed ($1.0$), positive feedback | $\ge 1$ | 1.0 | 0 | 0 | +1.0 | **+1.00** | Perfect execution & user validation |
| All tools succeed ($1.0$), negative feedback | $\ge 1$ | 1.0 | 0 | 0 | -1.0 | **-0.20** | Tools succeeded but output was rejected |
| Half tools fail ($0.5$), normal | $\ge 2$ | 0.5 | 0 | 0 | None | **0.00** | Partial tool failure |
| All tools fail ($0.0$), normal completion | $\ge 1$ | 0.0 | 0 | 0 | None | **-0.40** | Total tool failure, synthesized reply |
| All tools succeed, budget exhausted | $\ge 1$ | 1.0 | 0 | 1 | None | **+0.10** | High resource burn penalty |
| Zero tools, FSM hard failure | 0 | N/A | 1 | 0 | None | **-0.50** | Hard error before tool execution |
| FSM crash / hard failure with tool failure | $\ge 1$ | 0.0 | 1 | 0 | None | **-0.90** | Unrecovered engine failure |
| Hard failure + negative feedback | $\ge 1$ | 0.0 | 1 | 0 | -1.0 | **-1.00** | Minimum clamp floor |

---

## 4. SQLite Storage & Event Outbox

### 4.1 Storage Architecture
To prevent data loss caused by FSM crashes, step retries, or context truncation (`EncodeStoredToolResult`), storage is divided into two layers:
1. **`agent_trajectory_events` (Immutable Outbox):** Co-located in the per-chat database where FSM state resides. Observation events are committed atomically alongside FSM step state in an explicit SQLite transaction (never holding locks across tool execution).
2. **`agent_trajectories` (Staged Turn Rollout):** Materialized when a turn completes or is reconciled during startup crash recovery.
3. **`agent_trajectory_feedback` (Append-Only Audit Trail):** Records each rating event and note from users with source message idempotency.

### 4.2 Schema Migration (`internal/store/schema.tmpl` / `migrate.tmpl`)

```sql
-- Immutable execution event outbox
CREATE TABLE IF NOT EXISTS agent_trajectory_events (
    id TEXT PRIMARY KEY,
    trajectory_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    iteration INTEGER NOT NULL,
    phase TEXT NOT NULL DEFAULT 'loop', -- 'start', 'loop', 'synthesis', 'terminal'
    step_id TEXT NOT NULL DEFAULT '',
    attempt INTEGER NOT NULL DEFAULT 1,
    event_type TEXT NOT NULL,           -- 'turn_start', 'tool_attempt', 'llm_response', 'turn_terminal'
    payload BLOB NOT NULL,              -- JSON payload with untruncated content
    created_at DATETIME NOT NULL,
    CONSTRAINT uq_trajectory_events UNIQUE (run_id, iteration, phase, step_id, attempt, event_type)
);

CREATE INDEX IF NOT EXISTS idx_trajectory_events_traj
    ON agent_trajectory_events (trajectory_id, created_at ASC);

-- Staged rollouts ready for mutation and export
CREATE TABLE IF NOT EXISTS agent_trajectories (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    frontend_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    actor_id TEXT NOT NULL,
    privacy_level TEXT NOT NULL DEFAULT 'private', -- 'public', 'private', 'internal'
    export_approved INTEGER NOT NULL DEFAULT 0,
    payload BLOB NOT NULL,                         -- JSON-serialized canonical Trajectory
    r_eval REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'staged',         -- 'staged', 'exported', 'failed'
    revision INTEGER NOT NULL DEFAULT 1,
    exported_revision INTEGER NOT NULL DEFAULT 0,
    created_at DATETIME NOT NULL,
    completed_at DATETIME NOT NULL,
    eligible_at DATETIME NOT NULL,
    exported_at DATETIME
);

CREATE INDEX IF NOT EXISTS idx_trajectories_export 
    ON agent_trajectories (status, eligible_at) 
    WHERE status = 'staged';

CREATE INDEX IF NOT EXISTS idx_trajectories_session 
    ON agent_trajectories (frontend_id, session_id, scope_id, completed_at DESC);

-- Append-only user feedback audit table with replay deduplication
CREATE TABLE IF NOT EXISTS agent_trajectory_feedback (
    id TEXT PRIMARY KEY,
    message_id TEXT NOT NULL UNIQUE, -- Source message / idempotency key
    trajectory_id TEXT NOT NULL REFERENCES agent_trajectories(id) ON DELETE CASCADE,
    frontend_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    actor_id TEXT NOT NULL,
    rating REAL NOT NULL, -- +1.0, -1.0, 0.0
    note TEXT NOT NULL DEFAULT '',
    created_at DATETIME NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_trajectory_feedback_target 
    ON agent_trajectory_feedback (trajectory_id, created_at ASC);
```

### 4.3 Store API (`internal/trajectory/store.go`)

```go
type Store interface {
    RecordEvent(ctx context.Context, event *Event) error
    RecordEventTx(ctx context.Context, tx *sql.Tx, event *Event) error
    InsertStaged(ctx context.Context, traj *Trajectory, delay time.Duration) error
    UpdateFeedback(ctx context.Context, req FeedbackRequest) (*Trajectory, error)
    ListReadyForExport(ctx context.Context, now time.Time, limit int) ([]*Trajectory, error)
    MarkExported(ctx context.Context, id string, revision int, exportedAt time.Time) error
    PruneExported(ctx context.Context, olderThan time.Duration) (int64, error)
    ReconcileTerminalRuns(ctx context.Context) error
    Close() error
}

type FeedbackRequest struct {
	MessageID    string  // Source message idempotency key
	FrontendID   string
	SessionID    string
	ScopeID      string
	ActorID      string
	TrajectoryID string  // Optional explicit ID; if empty, targets actor's latest interactive turn
	Rating       float64
	Note         string
}
```

- **Transactional Finalization & Execution Merging:**
  `InsertStaged` executes within an atomic transaction to guarantee idempotent, race-free finalization:
  1. **Stale/Identical Check:** Reject stale execution versions using durable execution ordering (not wall-clock arrival time) so an older FSM finalizer cannot overwrite a newer outcome (such as a volatile fallback run). Otherwise, merge persisted feedback and derive the complete projection. Finalization is a no-op only when all substantive projected fields (steps, output, termination status, budget flags, reward, privacy metadata) are unchanged (excluding generated bookkeeping timestamps).
  2. **Feedback Preservation & Re-evaluation:** If execution data changes (e.g., volatile fallback completing after an FSM failure):
     - Active feedback from `agent_trajectory_feedback` is read and merged into `EvalBreakdown`.
     - $R_{\text{eval}}$ is recomputed incorporating both the updated execution signals and existing user feedback $F_{\text{user}}$.
     - Privacy classification and export approval are re-derived over the updated payload.
  3. **Revision Bumping on Content Change:** If the materialized payload changes:
     - Increment `revision = revision + 1`
     - Reset `status = 'staged'`
     - Preserve `exported_revision`
     - Set `eligible_at = now + delay`
     - Update `payload`, `r_eval`, `privacy_level`, `export_approved`, and `completed_at`.
     This guarantees that if an initial outcome was already exported, a subsequent fallback execution automatically receives revision $r+1$ and is queued for export without mutating historical S3 blobs.

- **Race-Free Revision Acknowledgment:**
  `MarkExported` executes:
  ```sql
  UPDATE agent_trajectories
  SET status = CASE WHEN revision = ? THEN 'exported' ELSE 'staged' END,
      exported_revision = ?,
      exported_at = ?
  WHERE id = ? AND exported_revision < ?;
  ```
  If user feedback arrived while the exporter was uploading revision 1, `revision` is 2. `exported_revision` becomes 1, but `status` stays `staged`, ensuring revision 2 flushes on the subsequent tick!

---

## 5. Execution Recorder & Engine Integration

### 5.1 Pre-Truncation Atomic Hook
- Step observation events and FSM step records are committed together in an atomic transaction:
  1. Execute tool outside any database lock.
  2. Open transaction: commit updated `fsm_steps` and append `agent_trajectory_events` (`event_type = 'tool_attempt'`, with distinct `phase` and `attempt`) containing full untruncated output, exit code, and duration.
  3. Commit transaction.
- Preserves provider `ReasoningContent` (e.g. DeepSeek/Qwen or Gemini thinking text) when returned by `internal/llm/client.go`.

### 5.2 Agent Engine Finalization & Ongoing Recovery
- At the entry of `agent.Engine.Run(ctx, turn)`:
  - Generate `trajectory_id = uuid.NewString()`.
  - Record `turn_start` event with $x$ (user prompt) and $\Phi_{\text{task}}$ context.
- At completion of `Run` (and via `WithRunFinalizer` on all FSM runs, including asynchronous resumed/recovered runs):
  1. Materialize `Trajectory` struct from durable events.
  2. Compute initial $R_{\text{eval}}$ via `CalculateReward(breakdown)`.
  3. Derive `PrivacyLevel`:
     - Inspect prompt, active skills, and context. If retrieved memory contains private data, classify as `private`.
     - Townhall turns without private citations classify as `public`.
     - Set `ExportApproved = (PrivacyLevel == "public")` by default. Unknown or unapproved classifications are non-exportable.
  4. Compute `EligibleAt = completedAt.Add(flushDelay)`.
  5. Store via `InsertStaged(ctx, traj, flushDelay)`.

### 5.3 Startup Recovery Reconciliation
- On daemon startup, `fsm.Engine.Recover` launches background recovery workers.
- Additionally, `ReconcileTerminalRuns(ctx)` queries runs in terminal states (`completed`, `failed`, `terminated`) that lack a staged trajectory, constructing them from durable events before retention cleaners execute.

---

## 6. Asynchronous S3 Exporter

Implemented in `internal/trajectory/exporter.go`:

### 6.1 Configuration Keys (`internal/config/config.go`)
- `TRAJECTORY_ENABLED`: bool (default `false`; must be explicitly enabled to authorize data export).
- `TRAJECTORY_S3_BUCKET`: string (default: falls back to `S3_BUCKET`).
- `TRAJECTORY_S3_PREFIX`: string (default `"trajectories/"`).
- `TRAJECTORY_FLUSH_DELAY`: time.Duration (default `15m`).
- `TRAJECTORY_ALLOW_PRIVATE`: bool (default `false`).

### 6.2 Exporter Eligibility Predicate
A trajectory is eligible for export if and only if all of the following conditions hold:
1. `TRAJECTORY_ENABLED = true`
2. `status = 'staged'` AND `eligible_at <= now`
3. `export_approved = 1`
4. Either:
   - `privacy_level = 'public'`, OR
   - `privacy_level = 'private'` AND `TRAJECTORY_ALLOW_PRIVATE = true`
5. Records with unknown, internal, or unapproved classifications are strictly skipped.

### 6.3 Key Partitioning & Upload
- Format: `s3://<bucket>/<prefix>/YYYY/MM/DD/<trajectory_id>/r<revision>.json`
- Date components YYYY/MM/DD are derived from `created_at.UTC()` to ensure stable paths across revisions.
- Intermediate revisions may be coalesced: the exporter uploads the latest staged revision $r$, satisfying pending exports.
- Uses `internal/objectstore.Client.Put` with stable bytes per revision.
- Calls `MarkExported(ctx, id, revision, now)`.

### 6.4 Lifecycle & Graceful Shutdown
- Background goroutine ticking every 1 minute.
- On SIGTERM / shutdown: runs a bounded `FlushOnce(ctx)` for eligible staged trajectories before exit.

---

## 7. User Feedback Commands & Authorization

### 7.1 Slash Commands
- Supported commands: `/good [note]`, `/bad [note]`, `/rate <+1|-1|0> [note]`.
- Optional explicit target syntax: `/good <trajectory_id> [note]`.
- Handled in `internal/commands/feedback.go` and wired into `internal/gateway/gateway.go`.

### 7.2 Authorization & Deduplication
- **Replay Deduplication:** `MessageID` uniqueness ensures re-delivered webhooks or network retries are no-ops.
- **Targeting Policy:**
  - If `<trajectory_id>` is specified: authorizes actor against the target trajectory's scope.
  - If omitted: targets the most recent interactive trajectory in `(frontend_id, session_id, scope_id)` requested by `actor_id`.
- **Superseding Ratings:** A newly sent rating from the same actor updates their rating, appends to the feedback audit table, atomically increments `revision`, resets status to `'staged'`, recalculates $R_{\text{eval}}$, and advances `eligible_at = now + 1m` (quiescence delay).
- **Post-Pruning Handling:** If feedback targets a trajectory that has already been pruned from local storage, `UpdateFeedback` returns `ErrTrajectoryExpired` and rejects the update (maintaining foreign key integrity).

