# Specification: Phase 1 — Context Assembly, Episodic Index, and Adaptive Task Compaction

## 1. Objective & Background

Bob already supports bounded per-chat conversational ring buffers (`internal/chatcontext`), isolated hybrid FTS5/vector conversational memory (`internal/memory`), user-validated structured memories and skills (`internal/knowledge`), durable SQLite-backed tool loops (`internal/fsm`), and trajectory logging (`internal/trajectory`).

However, two key gaps remain for long-horizon multi-step workflows:

1. **Multi-Source Context Assembly & Episodic Indexing:**
   - Currently, `internal/chatcontext`, `internal/memory`, and `internal/knowledge` are accessed ad-hoc rather than assembled through a unified context builder with explicit per-source token/byte budgets, deduplication, and provenance tracking in `Trajectory.TaskState` ($\Phi_{\text{task}}$) and `Trajectory.ActiveSkills`.
   - Evicted chat messages in `internal/memory` are stored as fixed 33-message chunks (`chunk_<chat>_<start>_<end>`). Long-horizon retrieval needs an **Episodic Index** that segments older history into coherent task/topic episodes with message sequence ranges, participants, timestamps, topic tags, and model-derived navigation summaries—while strictly preserving the distinction between untrusted navigation indexes and user-approved facts.

2. **Active Task State & Adaptive Compaction (SelfCompact):**
   - In `internal/fsm/tool_loop.go`, every iteration appends assistant tool calls and tool observations (`MaxToolResultSizeInContext = 16 KiB`) to `FSMRun.ContextJSON` until `MaxContextJSONBytes` (1 MiB) or `MaxIterations` is hit. Long tool trajectories suffer from context rot and rising token costs.
   - Following the **SelfCompact**[^selfcompact] approach, the FSM runner should periodically probe the model with a lightweight explicit rubric at safe iteration/byte thresholds to decide whether the trajectory has reached a semantic boundary (e.g., a verified subtask completion or converged search). When triggered—or when a hard token/byte backstop is reached—the runner compacts resolved intermediate steps into a **Structured Task State** while pinning system instructions, the original user request, and a short raw tail of recent interactions, and keeping all raw steps/events durably recoverable via a `recall_task_history` tool.

---

## 2. Three-Tier Context Horizons

Bob maintains three distinct context horizons with separate lifecycles:

1. **Recent Chat Context (`internal/chatcontext`):**
   - Bounded ring buffer of recent user/assistant turns for immediate conversational continuity and prefix KV-cache stability.
2. **Active Task Context (`internal/fsm`):**
   - Pinned system instructions, original user objective, latest compacted `StructuredTaskState` (if compacted), active loaded skill/memory references, and the uncompacted raw tail of recent tool iterations.
3. **Long-Term Context (`internal/knowledge` & `internal/memory`):**
   - User-approved structured memories and skills (`internal/knowledge`), episodic navigation summaries, and immutable raw conversation chunks (`internal/memory`).

---

## 3. Component 1: Multi-Source Context Builder (`internal/chatcontext` / `internal/contextbuilder`)

### 3.1 Responsibilities

Introduce a deterministic `Assembler` that builds the turn's initial LLM context and produces an inspectable `AssemblyManifest`:

1. **System Prompt & Pinned Directives:** Static channel system prompt (`internal/prompt`).
2. **Approved Structured Memories (`internal/knowledge`):** Pre-selected or session-loaded active memories within a configurable byte/token budget.
3. **Approved Procedural Skills (`internal/knowledge`):** Active loaded skills within the skill budget (`KnowledgeBudgetTracker`).
4. **Episodic & Historical Context (`internal/memory`):** Relevant episode navigation entries and raw historical chunks when pre-retrieved or carried across turns.
5. **Recent Conversation Turns (`internal/chatcontext`):** Chronological ring-buffer entries, deduplicated against any raw historical chunks whose sequence range `[start_seq, end_seq]` overlaps with buffered `Entry.Seq` values.

### 3.2 Provenance & Trajectory Integration

The `AssemblyManifest` records:
- Injected memory version IDs (`mem_<id>@<rev>`) and byte counts.
- Injected skill version IDs (`skill_<id>@<rev>`) and byte counts.
- Injected episode IDs and raw chunk IDs (`chunk_...`) with sequence ranges.
- Ring buffer sequence range (`[min_seq, max_seq]`) and entry count.
- Total assembled byte/estimated-token count per source.

When `internal/agent/engine.go` records the `TurnStartEvent` and `Trajectory`, `Trajectory.TaskState` ($\Phi_{\text{task}}$) and `Trajectory.ActiveSkills` are populated with both the system prompt and the serialized provenance manifest of active memories/skills loaded during the turn.

---

## 4. Component 2: Episodic Index (`internal/memory`)

### 4.1 Data Model & Storage Schema

Alongside raw conversation chunks in each chat's isolated SQLite database (`townhall.db`, `dm_<chat_id>.db`), add an `episodes` table:

```sql
CREATE TABLE IF NOT EXISTS memory_episodes (
    id TEXT PRIMARY KEY,              -- ep_<chat>_<start_seq>_<end_seq>
    chat_id TEXT NOT NULL,
    chat_type TEXT NOT NULL,          -- 'townhall' or 'dm'
    thread_id TEXT NOT NULL DEFAULT '',
    start_seq INTEGER NOT NULL,
    end_seq INTEGER NOT NULL,
    start_time INTEGER NOT NULL,
    end_time INTEGER NOT NULL,
    participants_json TEXT NOT NULL,  -- JSON array of sender names/IDs
    topic_tags_json TEXT NOT NULL,    -- JSON array of normalized topic tags
    task_status TEXT NOT NULL,        -- 'completed', 'ongoing', 'unresolved', 'discussion'
    summary TEXT NOT NULL,            -- Compact model-derived or heuristic navigation summary
    source_chunk_ids_json TEXT NOT NULL, -- Linked raw chunk IDs
    model_derived INTEGER NOT NULL DEFAULT 1,
    created_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memory_episodes_seq
    ON memory_episodes (chat_id, start_seq ASC, end_seq ASC);
```

Each episode is also indexed into CortexDB under a dedicated namespace/prefix (`episode:`) so hybrid vector + FTS5 search can match episodes alongside raw chunks.

### 4.2 Safety & Trust Invariant

- Episode summaries are **navigation indexes**, never approved user facts.
- Every episode retrieved via search or `recall_memory` is explicitly labeled `[Model-Derived Episode Summary | seq <start>-<end>]` and includes its linked `start_seq..end_seq` and `source_chunk_ids`.
- The system prompt and tool descriptions instruct the agent to inspect the underlying raw chunk/messages (`recall_memory` with sequence range or chunk lookup) before relying on exact code, numbers, commands, or user commitments.

---

## 5. Component 3: Active Task State & Adaptive Compaction (`internal/fsm`)

### 5.1 Structured Task State Schema

When an FSM run compacts its trajectory, it produces a versioned `CompactedTaskState` rather than unconstrained prose:

```go
type VerifiedFinding struct {
	Summary       string   `json:"summary"`
	SourceStepIDs []string `json:"source_step_ids"`
}

type CompactedTaskState struct {
	Version              int               `json:"version"`
	ParentVersion        int               `json:"parent_version"`
	RunID                string            `json:"run_id"`
	CompactedUpToIter    int               `json:"compacted_up_to_iter"`
	CompactedStepIDs     []string          `json:"compacted_step_ids"`
	OriginalObjective    string            `json:"original_objective"`
	ImmutableConstraints []string          `json:"immutable_constraints,omitempty"`
	VerifiedFindings     []VerifiedFinding `json:"verified_findings,omitempty"`
	DecisionsAndRejected []string          `json:"decisions_and_rejected,omitempty"`
	CurrentSubgoal       string            `json:"current_subgoal"`
	NextIntendedActions  []string          `json:"next_intended_actions,omitempty"`
	UnresolvedBlockers   []string          `json:"unresolved_blockers,omitempty"`
	ProducedArtifacts    []string          `json:"produced_artifacts,omitempty"`
	ActiveSkillVersions  []string          `json:"active_skill_versions,omitempty"`
	ActiveMemoryVersions []string          `json:"active_memory_versions,omitempty"`
	RubricVerdict        string            `json:"rubric_verdict"` // "probed_compact" or "backstop_forced"
	Model                string            `json:"model"`
	InputBytes           int               `json:"-"`
	OutputBytes          int               `json:"-"`
	CreatedAt            int64             `json:"created_at"`
}
```

### 5.2 Persistence in SQLite (`fsm_compactions`)

All compaction snapshots are stored in an append-only table in the FSM SQLite database:

```sql
CREATE TABLE IF NOT EXISTS fsm_compactions (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    parent_version INTEGER NOT NULL DEFAULT 0,
    compacted_up_to_iter INTEGER NOT NULL,
    rubric_verdict TEXT NOT NULL,
    state_json TEXT NOT NULL,
    model TEXT NOT NULL DEFAULT '',
    input_bytes INTEGER NOT NULL DEFAULT 0,
    output_bytes INTEGER NOT NULL DEFAULT 0,
    created_at INTEGER NOT NULL,
    FOREIGN KEY(run_id) REFERENCES fsm_runs(id) ON DELETE CASCADE,
     UNIQUE(run_id, version)
);
```

**Immutability Guarantee:**
- Compaction never deletes rows from `fsm_steps` or `agent_trajectory_events`, and never overwrites the original initial conversation messages.
- Instead, `buildChatMessages` constructs the active LLM view by combining:
  1. **Pinned prefix:** System prompt, conversation history prior to the active turn, and the original user request message for the run.
  2. **Compacted state message:** A structured system/developer context block rendering the latest `CompactedTaskState` (`version = k`, covering iterations `1..CompactedUpToIter`).
  3. **Raw tail:** Full assistant tool-call and tool-result messages for iterations `> CompactedUpToIter` (at least the last `TailIterations`, e.g., 2 iterations, are kept raw).

### 5.3 SelfCompact Rubric Trigger & Hard Backstop

After `handleExecuteSteps` finishes an iteration where all steps in that iteration are terminal (`COMPLETED`, `FAILED`, `TIMED_OUT`, `SKIPPED`):

1. **Eligibility Check:**
   - Do **not** probe or compact if any step is non-terminal (`PENDING` or `RUNNING`), if `run.Iteration < MinCompactionIteration` (e.g., 3), or if fewer than `CompactionIntervalIterations` (e.g., 3) have elapsed since `CompactedUpToIter` *and* the active uncompacted message size is below `CompactionSoftThresholdBytes` (e.g., 64 KiB).
   - **Loop/Stuck Guard:** If the last two iterations executed identical `(ToolName, ArgsJSON)` pairs that failed, skip compaction so repetition remains visible to the model.

2. **Hard Byte/Token Backstop:**
   - If the active reconstructed message payload exceeds `CompactionHardThresholdBytes` (e.g., 256 KiB, well below `MaxContextJSONBytes = 1 MiB`), force compaction (`rubric_verdict = "backstop_forced"`) without requiring a separate rubric probe round-trip.

3. **Rubric-Gated Probe (`SelfCompact`):**
   - When the soft iteration/byte trigger fires, issue a bounded rubric prompt via a pluggable `CompactionDecider` / `LLMClient` interface (defaulting to the active LLM client in Phase 1, and offloaded to the dedicated lightweight local decision model in Phase 4) asking for a structured JSON decision:
     - `should_compact: bool`
     - `reason: string`
     - `state: CompactedTaskState` (populated when `should_compact == true`)
   - Compact Criteria:
     - A subtask has completed and its result is verified.
     - A search/investigation has converged on durable findings.
     - An earlier plan or failed approach is no longer active.
     - Large/repeated tool outputs can be distilled into a compact working state.
   - Do-Not-Compact Criteria:
     - The agent is midway through an unresolved derivation or multi-step verification.
     - The most recent tool observation has not yet been interpreted.
     - The agent is stuck or repeating identical failing calls.

4. **Trajectory Outbox Recording:**
   - Every compaction probe and state transition emits a durable event (`event_type = "compaction"`) into `agent_trajectory_events` so offline training and evaluation can inspect compaction boundaries and fidelity.

### 5.4 Raw History Recovery Tool (`recall_task_history`)

Expose a read-only (`ExecutionModeParallel`) tool `recall_task_history` to the agent during FSM runs:

- **Arguments:**
  - `iteration` (optional int): Retrieve all steps and observations from a specific iteration.
  - `step_id` or `tool_call_id` (optional string): Retrieve the full untruncated `ResultJSON` / observation of a specific step.
  - `query` (optional string): Substring/keyword filter across compacted `fsm_steps` args and results for the current `run_id`.
- **Behavior:**
  - Queries `fsm_steps` for the active `run.ID` (strictly scoped to the current run/chat).
  - Returns the raw tool call arguments, status, and output (up to `MaxToolResultSizeInContext` per call, with offset/pagination support `offset_bytes` so even large outputs can be inspected in slices).

---

## 6. Configuration (`internal/config`)

Add configurable parameters with sensible defaults:

- `FSMCompactionEnabled` (`FSM_COMPACTION_ENABLED`, default `true`)
- `FSMCompactionMinIterations` (`FSM_COMPACTION_MIN_ITERATIONS`, default `3`)
- `FSMCompactionIntervalIterations` (`FSM_COMPACTION_INTERVAL_ITERATIONS`, default `3`)
- `FSMCompactionTailIterations` (`FSM_COMPACTION_TAIL_ITERATIONS`, default `1`)
- `FSMCompactionSoftThresholdBytes` (`FSM_COMPACTION_SOFT_THRESHOLD_BYTES`, default `65536` / 64 KiB)
- `FSMCompactionHardThresholdBytes` (`FSM_COMPACTION_HARD_THRESHOLD_BYTES`, default `262144` / 256 KiB)

---

## 7. Acceptance Criteria

1. **Multi-Source Context Assembly & Provenance:**
   - Context assembly enforces per-source byte budgets, deduplicates ring-buffer messages against retrieved raw chunks by sequence number, and records loaded skill/memory version IDs in `Trajectory.ActiveSkills` and `Trajectory.TaskState`.
2. **Episodic Index:**
   - Evicted chat batches generate structured `memory_episodes` records linked to raw chunk IDs and sequence ranges, searchable via hybrid/lexical search and clearly marked as model-derived navigation summaries.
3. **Adaptive Compaction (SelfCompact):**
   - Long FSM tool runs trigger rubric-gated or backstop compaction, persisting `CompactedTaskState` in `fsm_compactions` while keeping pinned messages and the recent raw tail intact.
   - Restarting or recovering an interrupted FSM run restores the latest `CompactedTaskState` seamlessly.
4. **Raw Task History Recovery:**
   - Compacted steps remain in `fsm_steps` and `agent_trajectory_events` and can be retrieved on demand via `recall_task_history` with strict run/session isolation.
5. **Quality Gates:**
   - Unit and race-enabled integration tests cover context assembly deduplication, episode creation/retrieval, compaction trigger rules (including stuck-loop suppression and hard backstop), crash recovery with compacted state, and `recall_task_history`.
   - `make check` passes with 0 errors.

---

## References

[^selfcompact]: Li, T., Zhang, J., Jurayj, W., Wang, X., Jin, C., Farajtabar, M., Nalisnick, E., & Khashabi, D. (2026). *Self-Compacting Language Model Agents* (SelfCompact). [arXiv:2606.23525](https://arxiv.org/abs/2606.23525).
