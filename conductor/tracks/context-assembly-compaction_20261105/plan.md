# Implementation Plan: Phase 1 — Context Assembly, Episodic Index, and Adaptive Task Compaction

## Step 1: Configuration & FSM Compaction Schema (`internal/config`, `internal/fsm`, `internal/trajectory`)
- [x] Task: Add FSM Compaction configuration settings to `internal/config/config.go`
    - [x] Add `FSMCompactionEnabled` (default `true`)
    - [x] Add `FSMCompactionMinIterations` (default `3`), `FSMCompactionIntervalIterations` (default `3`), and `FSMCompactionTailIterations` (default `1`)
    - [x] Add `FSMCompactionSoftThresholdBytes` (default `65536`) and `FSMCompactionHardThresholdBytes` (default `262144`)
    - [x] Add unit tests in `internal/config/config_test.go`
- [x] Task: Define `CompactedTaskState` models and `fsm_compactions` SQLite schema
    - [x] Add `VerifiedFinding`, `CompactedTaskState`, and `CompactionProbeResult` structs to `internal/fsm/models.go`
    - [x] Bump `fsm.SchemaVersion` to `4` in `internal/fsm/schema.go` and create the `fsm_compactions` table + index in `EnsureDBSchema`
    - [x] Implement `SaveCompaction(ctx, compaction)` and `GetLatestCompaction(ctx, runID)` and `ListCompactions(ctx, runID)` in `internal/fsm/store.go`
    - [x] Add `CompactionEventPayload` and `MakeCompactionEvent` to `internal/trajectory` so compactions are recorded in `agent_trajectory_events`
    - [x] Write unit tests in `internal/fsm/schema_test.go` and `internal/fsm/store_test.go`
- [x] Task: Step 1 Verification
    - [x] Run `go test -race ./internal/config/... ./internal/fsm/... ./internal/trajectory/...`

---

## Step 2: SelfCompact Adaptive Task Compaction in FSM Tool Loop (`internal/fsm`)
- [x] Task: Implement compacted view construction in `ToolLoopRunner.buildChatMessages`
    - [x] Load latest `CompactedTaskState` for `run.ID` in `buildChatMessages`
    - [x] Pin initial system prompt, prior conversation turns, and the run's original user request message
    - [x] Render `CompactedTaskState` as a structured context message covering iterations `1..CompactedUpToIter`
    - [x] Append raw assistant tool-call and tool-result messages only for iterations `> CompactedUpToIter`
- [x] Task: Implement SelfCompact trigger logic, loop/stuck guard, and rubric probe
    - [x] Create `internal/fsm/compaction.go` with eligibility evaluation:
      - Check iteration minimum (`MinIterations`), interval since `CompactedUpToIter`, and soft/hard byte thresholds
      - Detect stuck/repeating failing tool calls across consecutive iterations and suppress compaction when stuck
    - [x] Implement rubric-gated compaction probe (`probeAndCompact`) and hard byte-limit backstop (`forceCompact`)
    - [x] Integrate compaction check at the end of `handleExecuteSteps` before transitioning to the next `StateLLMRequest`
    - [x] Record compaction events to `agent_trajectory_events`
- [x] Task: Write unit and recovery tests for FSM compaction
    - [x] Create `internal/fsm/compaction_test.go` testing:
      - Rubric-triggered compaction and structured state rendering
      - Hard byte-threshold backstop compaction
      - Suppression when steps are non-terminal or stuck repeating failing calls
      - Crash recovery resuming a compacted run from `fsm_compactions` + raw tail steps
- [x] Task: Step 2 Verification
    - [x] Run `go test -race ./internal/fsm/...`

---

## Step 3: Raw Task History Recovery Tool (`recall_task_history` in `internal/tools`, `internal/fsm`)
- [x] Task: Implement `recall_task_history` tool
    - [x] Define `RecallTaskHistoryArgs` (`iteration`, `step_id`, `tool_call_id`, `query`, `offset_bytes`, `max_bytes`) and JSON Schema in `internal/tools`
    - [x] Query `fsm_steps` strictly scoped to the active `run_id` and session
    - [x] Support retrieving full raw tool arguments and outputs (with byte-offset slicing for outputs larger than `MaxToolResultSizeInContext`)
    - [x] Classify `recall_task_history` as `ExecutionModeParallel` (read-only)
    - [x] Automatically include `recall_task_history` in the bound toolset for FSM runs when compaction is enabled
- [x] Task: Write unit and integration tests for `recall_task_history`
    - [x] Test retrieving omitted/compacted step results by iteration, `step_id`, and keyword query
    - [x] Verify cross-run and cross-chat isolation (cannot query steps belonging to another `run_id` or chat)
- [x] Task: Step 3 Verification
    - [x] Run `go test -race ./internal/tools/... ./internal/fsm/...`

---

## Step 4: Episodic Index over Evicted Conversation History (`internal/memory`, `internal/tools`)
- [x] Task: Add `memory_episodes` table and episode builder in `internal/memory`
    - [x] Create `internal/memory/episode.go` with `Episode` struct (`ID`, `ChatID`, `ChatType`, `ThreadID`, `StartSeq`, `EndSeq`, `StartTime`, `EndTime`, `Participants`, `TopicTags`, `TaskStatus`, `Summary`, `SourceChunkIDs`, `ModelDerived`)
    - [x] Initialize `memory_episodes` table and indexes in `Manager.GetDB`
    - [x] Generate and persist an `Episode` record alongside raw chunks during `IndexMessages` (extracting participants, sequence/time bounds, topic keywords, and a clearly labeled navigation summary)
    - [x] Index episode summaries in CortexDB (`episode:` prefix) while linking back to `source_chunk_ids` and `[start_seq, end_seq]`
- [x] Task: Enhance `memory.Manager.Search` and `recall_memory` to support episodic navigation
    - [x] Return matching episode navigation entries clearly tagged as `[Model-Derived Episode Summary | seq <start>-<end>]`
    - [x] Allow `recall_memory` to fetch raw chunks by sequence range (`start_seq`, `end_seq`) or `chunk_id` so the agent can drill down from an episode summary into exact raw evidence
- [x] Task: Write unit tests for Episodic Index
    - [x] Create `internal/memory/episode_test.go` verifying episode creation on eviction, hybrid/lexical search, sequence-range raw chunk lookup, and DM/Townhall privacy isolation
- [x] Task: Step 4 Verification
    - [x] Run `go test -race ./internal/memory/... ./internal/tools/...`

---

## Step 5: Multi-Source Context Assembler & Trajectory Provenance (`internal/chatcontext`, `internal/agent`, `internal/trajectory`)
- [ ] Task: Implement Multi-Source Context Assembler
    - [ ] Create `internal/chatcontext/assembler.go` with `Assembler`, `AssemblyInput`, `SourceBudgets`, and `AssemblyManifest`
    - [ ] Enforce per-source byte budgets across:
      1. Pinned system prompt
      2. Approved structured memories (`internal/knowledge`)
      3. Approved procedural skills (`internal/knowledge`)
      4. Episodic/historical passages (`internal/memory`)
      5. Recent conversation ring buffer (`internal/chatcontext`)
    - [ ] Deduplicate overlapping historical chunks whose `[start_seq, end_seq]` overlaps with entries currently in the ring buffer
- [ ] Task: Wire Provenance Manifest into `internal/agent` and `internal/trajectory`
    - [ ] Record loaded skill version IDs (`skill_<id>@<rev>`) and memory version IDs (`mem_<id>@<rev>`) from `KnowledgeBudgetTracker` and `AssemblyManifest` into `Trajectory.ActiveSkills` and `Trajectory.TaskState` ($\Phi_{\text{task}}$)
    - [ ] Ensure `ReconcileTerminalRuns` in `internal/trajectory/store.go` captures skills and memories loaded dynamically via `load_skill` / `load_memory` during the turn
- [ ] Task: Write unit and integration tests for Context Assembly & Provenance
    - [ ] Create `internal/chatcontext/assembler_test.go` testing budget enforcement, sequence overlap deduplication, and manifest generation
    - [ ] Test end-to-end trajectory provenance recording in `internal/agent/engine_test.go` and `internal/trajectory/store_test.go`
- [ ] Task: Step 5 Verification & Full Quality Check
    - [ ] Run `make check` (`lint-go`, `test-go` with `-race`, `semgrep`, `osv-scanner`) and verify 0 errors
