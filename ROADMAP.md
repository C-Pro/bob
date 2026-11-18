# Bob Roadmap

This roadmap starts from Bob's current implementation and outlines the path to a
resource-conscious, continuously improving AI agent.

Bob learns first through fast, non-parametric mechanisms—user-approved memory,
reusable skills, memory consolidation, and evaluated reflective prompt/skill
updates—before consolidating recurring patterns into model weights.

```text
+-----------------------------------------------------------------------------------+
|                                PHASED ARCHITECTURE                                |
+-----------------------------------------------------------------------------------+
| [DONE] Foundations  Core Bot, CLI (`bob -p`), Ring Buffer, CortexDB Hybrid RAG,   |
|                     User-Validated Skills & Structured Memory, Docker & Bwrap     |
|                     Sandboxes, Durable FSM, Cron Scheduler, S3 Backups,           |
|                     Trajectory Logging, Reward Engine (`/feedback`), S3 Exporter  |
| [NEXT] Phase 1      Context Assembly, Episodic Index & Adaptive Task Compaction   |
| [NEXT] Phase 2      Non-Parametric "Sleep": GEPA Skill & Memory Consolidation     |
| [DONE] Phase 3      Evaluation Dataset & Deterministic CLI Gatekeeper Harness     |
| [NEXT] Phase 4      Lightweight Local Control-Plane / Decision Model (Fast/Slow)  |
| [NEXT] Phase 5      Local Open-Weights Actor Backend & Tiered Model Router        |
| [NEXT] Phase 6      Offline Shared-Path PESO & Knowledge Seeding Pipeline         |
| [NEXT] Phase 7      SafeMERGE, Dynamic LoRA Hot-Reload & Adapter Lifecycle        |
| [NEXT] Phase 8      Sleep-Inspired Sandbox "Dreaming" & Synthetic Rehearsal       |
+-----------------------------------------------------------------------------------+
```

## Completed Foundation

The following subsystems are already implemented and tested in the repository:

- **Core Agent, Besedka Gateway & CLI Runner:** Besedka WebSocket/REST gateway
  (`internal/gateway`), OpenAI-compatible LLM client (`internal/llm`), decoupled
  core engine (`internal/agent`, `internal/agentapi`), and standalone one-shot CLI
  frontend (`cmd/bob`, `internal/cli`).
- **Ring Buffer, SQLite Store & Encrypted Backups:** Bounded per-chat ring buffer
  (`internal/chatcontext`), canonical SQLite storage (`internal/store`,
  `internal/agentstore`), conversational memory indexing and regeneration
  (`internal/memory`), local SIMD embeddings (`internal/llm/local_embedder.go`),
  and Argon2id/Blake2b encrypted S3 backups (`internal/backup`).
- **User-Validated Skills & Structured Memory:** Relational + CortexDB hybrid
  knowledge store (`internal/knowledge`), budget-bounded agent discovery/loading/
  proposal tools (`propose_*`, `discover_*`, `load_*` in
  `internal/tools/knowledge_tools.go`), and human-in-the-loop `/skill` and
  `/memory` approval commands (`internal/commands/skill.go`,
  `internal/commands/memory.go`).
- **Tool Framework, Attachments & Web Research:** JSON-Schema tool registry and
  combinators (`internal/tools`), attachment handling, Tavily search, readability
  web fetcher, and per-session token/execution budgets (`internal/tools/budget.go`).
- **Pluggable Sandboxes (Docker & Bubblewrap):** Isolated execution drivers for
  both Docker (`internal/sandbox/docker`) and low-overhead Linux Bubblewrap
  (`internal/sandbox/bwrap`) with restricted network proxy forwarding
  (`cmd/bob-proxy-fwd`).
- **Durable FSM & Cron Task Scheduler:** SQLite-backed finite state machine
  (`internal/fsm`) with crash recovery, parallel/sequential tool dispatch, and
  cron-based out-of-band task execution (`internal/scheduler`,
  `internal/agent/scheduled`).
- **Trajectory Logging, Reward Engine & S3 Exporter:** Turn-by-turn trajectory and
  outbox event capture (`internal/trajectory`), scalar reward calculation
  ($R_{\text{eval}}$ from tool exit codes, completion status, and explicit
  `/feedback` commands in `internal/commands/feedback.go`), privacy/consent
  tracking, and async S3/MinIO JSONL.zst export (`internal/trajectory/exporter.go`,
  `internal/objectstore`).

## Design Principles

- **Fast learning before weight updates:** Prefer structured memory, procedural
  skills, non-parametric sleep consolidation[^sleep], and retrieval improvements
  when they can solve the problem at zero GPU cost.
- **User authority:** Proposed memories and skills remain inactive until the
  affected user explicitly approves them (or, for evolved skill revisions, until
  they pass deterministic sandbox verification and user review).
- **Evidence before promotion:** Changes to skills, prompts, or LoRA adapters must
  pass a deterministic evaluation gatekeeper before deployment.
- **Separate knowledge types:** Raw chat history, durable facts, user preferences,
  procedural skills, active task state, and model-training trajectories have
  different lifecycles and must not be stored as interchangeable text chunks.
- **Privacy by construction:** DM-derived information remains strictly scoped to
  that user. Private facts and unapproved trajectories must never enter global
  skills or training exports without explicit authorization.
- **Reversible evolution:** Prompts, skills, memories, datasets, and LoRA adapters
  are versioned. Every promoted change can be inspected, disabled, or rolled back
  (e.g., via `SafeMERGE`[^safemerge] or adapter hot-switching).
- **Resource- and bandwidth-aware continual learning:** Offload frequent
  control-plane judgments to a tiny local decision model while optimizing the main
  actor for commodity hardware using sparse MoE or compact dense models with
  shared-path LoRA adapters, paired with zero-GPU non-parametric reflection[^gepa]
  and short, inexpensive offline PESO[^peso] training passes.

---

## Phase 1: Context Assembly, Episodic Index, and Adaptive Compaction

While `internal/knowledge` and `internal/tools/knowledge_tools.go` already provide
user-validated skills and structured memories, long-running multi-step workflows
still need unified context assembly, episodic navigation over older history, and
bounded working state.

### 1.1 Context Assembly and Episodic Index

Add a context builder that independently selects and budgets:

1. Recent conversation turns (`internal/chatcontext`).
2. Relevant raw historical passages (`internal/memory`).
3. Approved structured memories (`internal/knowledge`).
4. Approved procedural skills (`internal/knowledge`).

The builder must enforce per-source token budgets, deduplicate overlapping
content, preserve provenance, and expose which items were injected into
`Trajectory.TaskState` ($\Phi_{\text{task}}$) and `Trajectory.ActiveSkills` for
audit and downstream training.

Segment older chat history into task or topic episodes so retrieval does not have
to search only isolated message chunks. Each derived episode should contain a
source message range, participants, time range, topic tags, task status, and a
compact description of what happened. Episode summaries are navigation indexes,
not approved memories: they remain linked to raw messages, are marked as
model-derived, and cannot silently become trusted facts.

### 1.2 Active Task State and Adaptive Compaction

Add adaptive task-context compaction to `internal/fsm` based on the SelfCompact[^selfcompact]
approach: periodically ask the model, using a small explicit rubric, whether the
current tool trajectory has reached a safe semantic boundary for compaction.

Keep three distinct context horizons:

1. **Recent chat context:** The ring buffer for conversational continuity.
2. **Active task context:** Original task, compacted task state, a short raw tail,
   and the tool interactions still needed by the current subtask.
3. **Long-term context:** Approved memories and skills, plus searchable raw history.

Compaction is a view over an immutable trajectory, not deletion. The full messages,
tool calls, tool results, and earlier compacted states remain durably stored in
`internal/fsm` and `internal/trajectory`.

#### Compaction Trigger & Structured Task State

At configurable tool-iteration or token intervals (with a hard token-budget
backstop), compact when a subtask has completed and its result is verified, when a
search has converged on durable findings, or when an earlier failed approach is no
longer active. Do not compact while a tool call is in flight, when the most recent
observation has not yet been interpreted, or when the agent is repeating itself.

The compacted state uses a stable schema:

- Original objective and immutable user constraints.
- Completed and verified findings, with references to source events.
- Decisions made and approaches rejected.
- Current subgoal and next intended actions.
- Unresolved questions, risks, and blockers.
- Produced artifacts, paths, identifiers, and relevant versions.
- Loaded skill IDs and memory IDs that remain applicable.
- A short raw tail containing recent unresolved interactions.

System instructions, the original user request, security constraints, and pending
tool-call protocol messages remain pinned. Provide a `recall_task_history` tool to
narrowly retrieve omitted raw events when the compacted state lacks a needed
detail.

### Acceptance Criteria

- Long FSM workflows can compact and resume after restart without losing their
  original objective, verified findings, active subgoal, or pending tool state.
- Raw task events remain recoverable via `recall_task_history` after any number of
  compactions.
- Context assembly records exact loaded skill and memory versions on recorded
  trajectories.

---

## Phase 2: Non-Parametric "Sleep" — GEPA Skill Evolution & Memory Consolidation

Before spending GPU compute on weight updates, close the Fast-Slow Training (FST)[^fst]
"Fast Loop" and the non-parametric consolidation stage of *Language Models Need
Sleep*[^sleep] in pure Go using the existing `internal/trajectory`,
`internal/knowledge`, `internal/memory`, and `internal/scheduler` packages.

### 2.1 Trajectory Consolidation Tracking

Extend `internal/trajectory` to track which recorded sessions have been processed
by the offline sleep loop:

- Add consolidation tracking metadata (e.g., `consolidated_at` timestamp or status)
  to `trajectories`.
- Provide queries for unconsolidated trajectories prioritized by high surprise or
  failure signal: negative evaluation rewards ($R_{\text{eval}} < 0$), non-zero
  sandbox/tool exit codes, budget exhaustion, or explicit user corrections via
  `/feedback`.

### 2.2 GEPA Reflective Skill Optimizer (Fast Weights)

In Fast-Slow Training[^fst] (combining GEPA[^gepa] and CISPO[^cispo]), declarative
task quirks and tool error recovery rules are absorbed rapidly into non-parametric
prompts and skills rather than forced immediately into slow network weights:

- Schedule a background sleep worker (`internal/sleep`) via `internal/scheduler`
  (plus a manual `/sleep` or CLI trigger).
- Group high-signal unconsolidated trajectories by task or active skill context.
- Pass the execution traces, CoT thoughts, tool arguments, and `stderr`/exit codes
  to an external teacher/proposer model to produce either:
  1. A candidate revision (`ProposeSkill` with `RevisesVersionID`) patching an
     existing skill in `internal/knowledge` that failed or needed user correction.
  2. A new candidate skill distilled from a successful multi-step recovery trace.
- Stage candidate skills as `proposed` revisions in `internal/knowledge` so the
  user can inspect and approve them via `/skill show` and `/skill approve` (and
  optionally pre-verify them with deterministic `bwrap` dry runs).

### 2.3 Archival Memory & Knowledge Garbage Collection

Unchecked RAG accumulation degrades retrieval precision over time due to stale or
contradicted facts (as shown in MemoryAgentBench[^memoryagentbench] and
MemBench[^membench]):

- Run a scheduled maintenance pass across `internal/knowledge`,
  `internal/memory/regenerate.go`, and CortexDB (`graph_cleanup`, `memory_sync`).
- Identify duplicate entities, expire stale episodic chunks, and propose
  superseding revisions or cleanup for contradicted user facts while keeping raw
  chat transcripts immutable.

### Acceptance Criteria

- Unconsolidated high-surprise/negative-reward trajectories are automatically
  analyzed during scheduled sleep passes and marked consolidated idempotently.
- GEPA reflection proposes inspectable skill patches or new draft skills in
  `internal/knowledge` without auto-activating unapproved changes.
- Scheduled memory maintenance prunes stale index entries and reconciles
  `internal/knowledge` with CortexDB without mutating raw chat history.

---

## Phase 3: Evaluation Dataset and Deterministic CLI Gatekeeper Harness [COMPLETED]

A fast, reproducible measurement and gatekeeping system has been established in [`eval/`](eval/README.md) around `cmd/bob` (`internal/cli/runner.go`), `internal/sandbox/bwrap`, and official Docker testbeds:

### 3.1 Bob Evaluation Dataset & Task Pools

Versioned task manifests are committed under [`eval/tasks/`](eval/tasks/):
- **Frozen 100-Task Held-Out Suite (`eval/tasks/heldout_100.json`):** Frozen benchmark suite with verified SHA256 (`7d16e983...`), covering DeepMind FRAMES (35 tasks), InterCode-Bash (20 tasks), InterCode-CTF (20 tasks), and SWE-bench Lite (25 tasks).
- **Smoke Suite (`eval/tasks/smoke_4.json`):** 4-task fast sanity check across all 4 benchmark categories.
- **Training Pools (`eval/tasks/train_pool.json`, `eval/tasks/train_20.json`):** 800+ task pool partitioned from held-out cases, reserved for Phase 4 offline trajectory curation and decision model training.

### 3.2 Deterministic CLI Gatekeeper & Docker Testbed

The evaluation runner ([`eval/harness/runner.py`](eval/harness/runner.py)) provides:
- **Bubblewrap Execution:** Millisecond sandbox isolation with strictly scoped mounts and zero host DB pollution (ephemeral per-task SQLite directories).
- **Docker SWE-bench Evaluation:** Direct integration with `swebench.harness.run_evaluation`, testing generated git diff patches inside official Django containers with full `FAIL_TO_PASS` and `PASS_TO_PASS` test suites.
- **Grader Oracle Verification:** [`eval/harness/test_grader_controls.py`](eval/harness/test_grader_controls.py) verifies positive and negative controls across all 4 grading mechanisms.
- **Replayable Trajectory Archive:** Complete turn-by-turn prompts, reasoning, tool calls, and compaction checkpoints are permanently archived in [`eval/results/`](eval/results/).

### 3.3 Baseline Evaluation Results (Local `Qwen3.6-35B-A3B-MTP-GGUF`)

- **Overall Held-Out Success Rate:** **55.0%** (55 / 100 tasks fully resolved).
- **Standard Graded Tasks:** **66.7%** (50 / 75 tasks: 62.9% FRAMES, 55.0% InterCode-Bash, 85.0% InterCode-CTF).
- **SWE-bench Lite Resolution:** **20.0%** (5 / 25 tasks passed 100% of Django test suites in Docker; 76.0% patch generation rate).
- **Full Report:** See [baseline_100_report.md](eval/results/baseline_100_report.md) for detailed task-by-task results.

---

## Phase 4: Lightweight Local Control-Plane / Decision Model

Running the primary actor or an external cloud API for every internal harness
decision (such as compaction probes, loop detection, or routing) adds unnecessary
latency and token cost. Introduce a dedicated, ultra-lightweight local decision
model (e.g., in the 0.5B–3B parameter class) to handle bounded control-plane tasks
with sub-second latency, and improve it via the same Fast/Slow continual learning
loop[^fst][^peso] before scaling weight updates to the main actor.

### 4.1 Control-Plane Responsibilities

Equip the Go harness (`internal/fsm`, `internal/memory`, `internal/llm`) to delegate
structured classification and small-schema JSON judgments to the local decision
model:

- **SelfCompact Rubric Probing:** Evaluate whether an active FSM trajectory has
  reached a safe semantic boundary for compaction (`should_compact`)[^selfcompact].
- **Stuck / Pathology & Loop Detection:** Inspect recent reasoning traces, tool
  arguments, and `stderr` outputs to detect semantic wheel-spinning, unproductive
  retries, or goal drift beyond exact string-match heuristics.
- **Episodic Segmentation & Tagging:** Generate topic tags, task status labels, and
  compact navigation summaries when chat batches are evicted into
  `memory_episodes`.
- **Implicit User Feedback & Surprise Scoring:** Estimate implicit satisfaction or
  correction signals from follow-up user turns for $R_{\text{eval}}$ calculation in
  `internal/trajectory`.
- **Pre-Routing & Escalation Gate:** Estimate task complexity and decide whether a
  request can be handled locally, requires specific skills/memories, or should
  escalate to the External Teacher tier.

### 4.2 Fast/Slow Continual Learning for the Decision Model

Because the decision model is tiny and emits structured verdicts that can be
checked against downstream trajectory outcomes ($R_{\text{eval}}$, compaction
recovery rate, and teacher labels), it serves as a low-cost proving ground for
Bob's Fast/Slow learning pipeline:

1. **Fast Loop (GEPA Rubric & Few-Shot Evolution)[^gepa][^fst]:** During Phase 2
   sleep passes, compare the decision model's verdicts against actual trajectory
   outcomes (e.g., whether a compaction lost needed context and forced a
   `recall_task_history` call, or whether a stuck loop went undetected). Use the
   teacher model to refine the decision model's classification rubrics and
   few-shot anchors.
2. **Slow Loop (PESO Adapter Consolidation)[^peso]:** Periodically distill
   teacher-labeled control decisions and evolved GEPA rubrics into the decision
   model's LoRA adapter (`FST-distill`[^fst] / `S2L`[^s2l]) anchored by PESO
   Softmax-KL regularization. Training a 0.5B–3B model takes only minutes on
   commodity hardware and validates the entire dataset-building, gatekeeping, and
   adapter hot-reload stack end-to-end.

### Acceptance Criteria

- Control-plane probes (compaction rubric, semantic stuck detection, episode
  tagging) execute locally with minimal memory overhead and zero external API token
  cost.
- Decision model verdicts and features are logged in `internal/trajectory` for
  offline evaluation against teacher ground truth.
- Fast (GEPA rubric) and Slow (PESO LoRA) updates demonstrably improve decision
  agreement with the Phase 3 benchmark and teacher labels without regressions.

---

## Phase 5: Local Open-Weights Actor Backend & Tiered Model Router

Once the evaluation harness (Phase 3) and local control-plane model (Phase 4) are
in place, deploy the primary local open-weights actor model for interactive chat,
coding, and multi-step tool execution.

### 5.1 Sparse MoE & Compact Dense Local Serving

On memory-bandwidth-constrained systems, large dense models are bottlenecked by
streaming all parameters per token, whereas sparse Mixture-of-Experts (MoE) models
activate only a small parameter slice (e.g., 3B–4B active parameters) per token:

- **Primary Local Targets (Sparse MoE):** Sparse MoE coding and tool-use models in
  the ~25B–35B total / 3B–4B active parameter class (4-bit / GGUF / FP8),
  streaming ~2–2.5 GB of active weights per token while retaining strong coding
  and structured tool-calling quality.
- **Compact Dense Alternatives (<=9B–12B):** Dense 4B–12B models for lower-memory
  deployments.
- **Serving Engine:** Run an OpenAI-compatible local inference server
  (`llama-server` or `vLLM`), exposing `/v1/chat/completions` with dynamic LoRA
  adapter mounting (`--lora` / `/v1/load_lora_adapter`).

### 5.2 Three-Tier Harness Model Router (`internal/llm`)

Extend `internal/config` and `internal/llm/client.go` to support transparent
three-tier routing and record the selected tier in `internal/trajectory`:

- **Tier 0 — Local Decision / Control Tier (0.5B–3B + LoRA):** Handles fast rubric
  probes, stuck/loop detection, episode indexing, and complexity routing (Phase 4).
- **Tier 1 — Local Actor Tier (Sparse MoE / Compact Dense + active LoRA):** Handles
  daily chat, structured tool calling, sandbox rollouts, and CLI tasks.
- **Tier 2 — External Teacher / Proposer Tier (Cloud API Models):** Handles Phase 2
  GEPA skill reflection, teacher trajectory generation, and explicit
  high-complexity fallback when the local actor hits confidence or retry limits (no
  tokenizer compatibility required for text reflection or off-policy SFT trace
  generation).

### Acceptance Criteria

- Local actor serving meets interactive latency and memory budgets while passing
  the Phase 3 CLI gatekeeper threshold.
- Three-tier routing in `internal/llm` is recorded in `internal/trajectory` and
  supports privacy-aware fallback rules.

---

## Phase 6: Offline Shared-Path PESO & Knowledge Seeding Pipeline

Scale the slow neocortical consolidation pipeline proven on the Phase 4 decision
model to the primary local actor: internalizing recurring tool-calling habits and
mature skills that cannot be served as efficiently by in-context prompts alone.

### 6.1 Shared-Path LoRA on Sparse MoE

Attaching LoRA adapters to routed MoE FFN experts causes router instability,
expert collapse, and sparse gradient noise. Instead, restrict BF16 LoRA adapters
($r = 32\text{--}64$) strictly to the **100% always-active dense layers**[^peso]:

- Self-attention projections: `q_proj`, `k_proj`, `v_proj`, `o_proj`.
- Always-on shared expert projections: `shared_expert.gate_proj`,
  `shared_expert.up_proj`, `shared_expert.down_proj`.
- All routed FFN experts (`experts.*`) and router gates remain 100% frozen as a
  static knowledge bank.

This keeps gradient covariance and PESO's[^peso] proximal regularization
mathematically identical to a small dense model while cutting optimizer memory by
~85% and producing compact (~50–80 MB) adapter checkpoints. Training runs can
execute either locally or in short rented single-GPU sessions, exporting
GGUF/PEFT adapters back to the inference host.

### 6.2 Dataset Builder, Strict Token Masking & SafeInstruct Anchors

Build a Python training pipeline that ingests exported trajectories from
`internal/trajectory/exporter.go` (S3/MinIO or local export):

- **Strict Binary Token Masking ($M_t \in \{0, 1\}$):** Mask out ($M_t = 0$) user
  prompts $x$, system prompts $\Phi$, and raw tool stdout/stderr observations
  $o_t$. Train ($M_t = 1$) strictly on agent reasoning/CoT traces $z$, tool call
  invocations $a$, and final assistant responses $y$.
- **SafeInstruct & Replay Anchor Mixing:** Interleave a fixed 5–10% buffer of
  safety-aligned refusal QA pairs (`SafeInstruct`) and historical "golden"
  trajectories into every training batch to prevent benign fine-tuning from
  rotating weights out of the low-rank safety refusal subspace.

### 6.3 Two-Gear PESO Optimization & Knowledge Seeding

Maintain $O(1)$ active adapter storage (`lora_fast_active` $v_t$ anchored to the
prior frozen state `lora_fast_prev` $v_{t-1}$)[^peso], plus an optional small bank
of $K$ modular skill/domain LoRA slots (`slot_1..K`)[^sleep][^s2l] when
distribution drift $\delta(t-1, t) = 2(\text{AUC} - 0.5)$ spikes:

$$\mathcal{L}_{\text{PESO}}(v_t) = \mathcal{L}_{\text{task}}(v_t) + \lambda \sum_{g=1}^{G} D_{\text{KL}}\!\left(\text{softmax}(v_t^{(g)}) \,\|\, \text{softmax}(v_{t-1}^{(g)})\right)$$

- **Continual Learning Rate Scaling:** Scale the learning rate down to
  $0.05\times\text{--}0.1\times$ of the base LoRA learning rate ($\sim 5\times
  10^{-6}$ to $1\times 10^{-5}$) with $\lambda \in [2.0, 5.0]$ to prevent
  overfitting on small incremental trajectory batches[^peso].
- **Gear 1 — Masked Off-Policy Distillation & Knowledge Seeding (`FST-distill`[^fst] /
  `S2L`[^s2l], ~90% of updates):** Train on high-reward trajectories ($R_{\text{eval}} >
  0$), including traces solved by the External Teacher tier or guided by approved
  skills in `internal/knowledge`. When a skill is mature and heavily used, strip
  the verbose skill instructions from the student's input context during training
  so the student LoRA internalizes the tool-calling behavior without needing the
  long skill body at runtime[^s2l][^sleep].
- **Gear 2 — Targeted On-Policy CISPO RL (~10% of updates):** Reserved only for
  persistent failure modes where off-policy distillation plateaus. Generate small
  group rollouts ($G = 4$) conditioned on GEPA[^gepa] prompt variants $\Phi$ inside
  fast `bwrap` sandboxes via `internal/cli/runner.go` and optimize with clipped
  importance sampling (CISPO)[^cispo][^fst] + PESO[^peso] regularization.

### Acceptance Criteria

- The dataset builder deterministically converts exported `.jsonl.zst`
  trajectories into token-masked training batches with `SafeInstruct` anchors.
- Shared-path LoRA training leaves routed MoE experts untouched and improves
  target task pass rates on the Phase 3 suite without exceeding single-GPU memory.
- Knowledge Seeding (`S2L`[^s2l]) allows a distilled skill to pass its verification
  tasks without loading the verbose skill Markdown into the prompt.

---

## Phase 7: SafeMERGE, Dynamic LoRA Hot-Reload & Adapter Lifecycle

Automate the verification, safety recovery, and deployment of trained LoRA
adapters (for both the Phase 4 decision model and Phase 5 actor model) without
mutating the base models or restarting the `bob` daemon.

### 7.1 Gatekeeper Verification & SafeMERGE Recovery

Every candidate adapter $v_t$ produced in Phases 4 or 6 is evaluated by the Phase 3
CLI Gatekeeper before promotion:

- Run the Task & Tool Benchmark, Catastrophic Forgetting Check, and Safety
  Alignment Probe via `cmd/bob` + `bwrap`.
- **SafeMERGE Fallback:** If safety refusal probes or general reasoning metrics dip
  slightly below threshold, selectively merge the drifted adapter layers with the
  prior safe adapter $v_{t-1}$ via `SafeMERGE`[^safemerge] (or project updates
  orthogonal to safety gradients) and re-verify before discarding the run.
- **Slot Unmasking on Interference:** If updating an existing slot still causes
  task interference on held-out checks (or if PESO drift score $\delta \ge
  \tau_{\text{drift}}$), revert $v_{t-1}$ and distill the new capability into a
  fresh Slow LoRA slot ($A_k \sim \mathcal{N}(0, \sigma^2), B_k = 0$) followed by
  fast-slot reset (synaptic pruning)[^sleep].

### 7.2 Dynamic LoRA Hot-Reloading

- Keep the foundation model immutable and store versioned adapter checkpoints with
  their training-data manifest, hyperparameters, and gatekeeper report.
- Never overwrite the last known-good adapter (`lora_fast_prev`).
- On gatekeeper `PASS`, convert the PEFT checkpoint to GGUF (for `llama-server`) or
  mount via `/v1/load_lora_adapter` and atomically switch the active adapter in
  `internal/llm` without restarting the running agent service.

### Acceptance Criteria

- Candidate adapters that fail safety or retention gates are automatically
  remediated via `SafeMERGE`[^safemerge] or rejected before reaching production.
- Promoted adapters hot-reload into the local inference server with zero downtime
  and one-command rollback to `lora_fast_prev`.

---

## Phase 8: Sleep-Inspired Sandbox "Dreaming" & Synthetic Rehearsal

Once non-parametric sleep (Phase 2), deterministic evaluation (Phase 3), local
decision & actor serving (Phases 4–5), and PESO consolidation (Phases 6–7) are
reliable, add offline synthetic rehearsal inspired by Stage 2 ("Dreaming" / REM
sleep) of *Language Models Need Sleep*[^sleep].

Rather than building external game engines or separate neural world-model critics,
ground dreaming directly in Bob's real tool and workspace environments (`bwrap`
sandboxes, bash/coding tasks, and file verification).

### 8.1 Synthetic Task Curriculum Generation

During scheduled idle windows (`internal/scheduler`):

- Sample past failed, high-retry, or edge-case coding and bash trajectories from
  `internal/trajectory`.
- Prompt the proposer to synthesize $m$ realistic problem variations ("dreams")
  grounded in available workspace fixtures and tools.
- Optionally inject subspace diversity during rollout generation by pairing the
  active policy with a randomly selected historical skill/LoRA slot or GEPA prompt
  variant to encourage novel tool combinations[^sleep].

### 8.2 Two-Stage Surprise & Gradient-Norm Filtering

Running full sandbox verification or RL updates on every synthetic dream is
computationally wasteful. Filter candidate dreams in two cheap stages:

1. **Zero-Cost Execution RPE (Surprise Heuristic):** Flag rollouts where actual
   sandbox execution outcomes (exit codes, test failures, `stderr`) diverge from
   the model's expected step progression (scored by deterministic exit-code checks
   and the Phase 4 local decision model).
2. **Masked SFT Gradient-Norm Ranking:** Score each candidate dream trajectory
   $\text{DREAM}^{(i)}$ by its Supervised Fine-Tuning gradient norm computed
   strictly over masked agent reasoning and action tokens ($M_t = 1$)[^sleep]:
   $$g_{\text{DR}}^{(i)} = \left\|\nabla_{\theta} \mathcal{L}_{\text{SFT}}\!\left(\text{DREAM}^{(i)}, \theta\right)\right\|$$
   Retain only the Top-$k$ highest-gradient dreams plus $b$ random dreams for
   diversity (~25% retention ratio).

### 8.3 Isolated Trial Verification Before Commit

To prevent iterative self-improvement collapse from noisy synthetic data:

- Execute candidate dreams inside isolated `bwrap` sandboxes to verify
  deterministic task completion ($r = 1$).
- Spin up an isolated trial LoRA update $\theta'^{(i)} \leftarrow
  \text{SFT}(\theta^{(i)}, \text{DREAM}^{(i)})$ and verify positive downstream improvement
  ($\Delta\tau > 0$) on the Phase 3 benchmark before queuing the verified dream
  trajectories into the Phase 6 PESO training buffer[^sleep].

### Acceptance Criteria

- Idle-hour dreaming generates executable coding/tool tasks inside `bwrap` without
  human supervision.
- Gradient-norm + execution filtering discards low-value synthetic rollouts before
  expensive training.
- Only synthetic trajectories that pass isolated trial verification on the Phase 3
  gatekeeper are admitted into the main PESO adapter pipeline.

---

## Explicitly Deferred

- Pretraining or full-model continual weight training.
- Attaching LoRA adapters to routed MoE FFN experts (which causes router drift and
  VRAM bloat).
- Unreviewed automatic activation of generated skills or memories without user
  approval or deterministic gatekeeper verification.
- Training on private DM conversations without explicit user consent and export
  approval (`export_approved = true`).
- Continuous online RL weight updates during live user chat turns.
- Standalone game/gridworld environments or separate neural world-model critics
  that do not transfer directly to Bob's tool and coding workflows.

## Immediate Next Steps

1. **Phase 1 & Phase 2 (Fast-Loop Completion in Go):** Complete active task-state
   compaction (`internal/fsm`) and implement the **Non-Parametric "Sleep" worker**
   (`internal/sleep`) connecting `internal/trajectory`, `internal/knowledge`,
   `internal/memory`, and `internal/scheduler` for automated GEPA skill reflection
   and memory consolidation at zero GPU cost.
2. **Phase 3 & Phase 4 (Deterministic Gatekeeper & Local Decision Model):** Build
   the `cmd/bob` + `bwrap` evaluation suite and deploy the lightweight local
   decision model to offload compaction rubrics and loop detection while testing
   the Fast/Slow training pipeline on a tiny model first.

---

## References

[^selfcompact]: Li, T., Zhang, J., Jurayj, W., Wang, X., Jin, C., Farajtabar, M., Nalisnick, E., & Khashabi, D. (2026). *Self-Compacting Language Model Agents* (SelfCompact). [arXiv:2606.23525](https://arxiv.org/abs/2606.23525).
[^sleep]: Behrouz, A., Hashemi, F., Javanmard, A., & Mirrokni, V. (2026). *Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories*. [arXiv:2606.03979](https://arxiv.org/abs/2606.03979).
[^fst]: Tiwari, R., Sareen, K., Agrawal, L. A., Gonzalez, J. E., Zaharia, M., Keutzer, K., Dhillon, I. S., Agarwal, R., & Khatri, D. (2026). *Learning, Fast and Slow: Towards LLMs That Adapt Continually* (FST: GEPA + CISPO). [arXiv:2605.12484](https://arxiv.org/abs/2605.12484).
[^gepa]: Agrawal, L. A., et al. (2025). *GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning*. [arXiv:2507.19457](https://arxiv.org/abs/2507.19457).
[^cispo]: MiniMax, et al. (2025). *MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention* (introducing Clipped Importance Sampling Policy Optimization, CISPO). [arXiv:2506.13585](https://arxiv.org/abs/2506.13585).
[^peso]: Yoo, H., Li, T.-W., Kang, S., Liu, Z., Xu, C., Qi, Q., & Tong, H. (2025). *Continual Low-Rank Adapters for LLM-based Generative Recommender Systems* (PESO: Proximally rEgularized Single evolving lOra). [arXiv:2510.25093](https://arxiv.org/abs/2510.25093).
[^s2l]: Zhang, T., & Qi, Z. (2026). *Skill-to-LoRA: From Using Skills to Learning Behaviors for Token-Efficient LLM Agents* (S2L). [arXiv:2606.16769](https://arxiv.org/abs/2606.16769).
[^safemerge]: Djuhera, A., Kadhe, S., Ahmed, F., Zawad, S., & Boche, H. (2025). *SafeMERGE: Preserving Safety Alignment in Fine-Tuned Large Language Models via Selective Layer-Wise Model Merging*. [arXiv:2503.17239](https://arxiv.org/abs/2503.17239).
[^memoryagentbench]: Hu, Y., et al. (2025). *Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions* (MemoryAgentBench). [arXiv:2507.05257](https://arxiv.org/abs/2507.05257).
[^membench]: Tan, H., et al. (2025). *MemBench: Towards More Comprehensive Evaluation on the Memory of LLM-based Agents*. [arXiv:2506.21605](https://arxiv.org/abs/2506.21605).
