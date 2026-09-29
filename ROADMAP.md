# Bob Roadmap

This roadmap starts from Bob's current implementation. Completed work such as the
Besedka gateway, chat context, hybrid history retrieval, attachments, isolated
sandboxes, durable tool execution, scheduling, backups, and local embeddings is
intentionally omitted.

The goal is a resource-conscious, continuously improving agent. Bob should first
learn through user-approved memory, reusable skills, and evaluated prompt changes.
Weight updates are a later consolidation mechanism, not the primary source of
day-to-day learning.

## Design Principles

- **Fast learning before weight updates:** Prefer memory, skills, prompt changes,
  and retrieval improvements when they can solve the problem.
- **User authority:** Initially, no proposed memory or skill becomes active until
  the affected user explicitly approves it.
- **Evidence before promotion:** Changes must improve a versioned evaluation suite
  before they can be deployed.
- **Separate knowledge types:** Raw chat history, durable facts, user preferences,
  procedural skills, and model-training trajectories have different lifecycles and
  must not be stored as interchangeable text chunks.
- **Privacy by construction:** DM-derived information remains scoped to that user.
  Private facts and preferences must never enter global skills or training data
  without explicit authorization and redaction.
- **Reversible evolution:** Prompts, skills, memories, datasets, and adapters are
  versioned. Every promoted change can be inspected, disabled, or rolled back.
- **Resource-aware experiments:** Optimize for a small number of informative
  evaluations and short rented-GPU training runs rather than large-scale
  pretraining or continuous online RL.

## Phase 1: Long-Horizon Context, User-Validated Skills, and Memory

Build the first fast-learning and long-horizon context layer on top of Bob's
existing chat history and hybrid retrieval.

### 1.1 Skill Registry

A skill is reusable procedural guidance for performing a class of tasks. It is not
a transcript or a user fact.

Each skill should contain:

- Stable ID, name, description, and version.
- Trigger conditions and task tags used for discovery.
- Focused procedural instructions and optional examples.
- Owner and scope: private user, chat, or explicitly approved global scope.
- Provenance linking the skill to the conversation, correction, or authored source
  that produced it.
- Lifecycle state: `proposed`, `approved`, `rejected`, `disabled`, or `archived`.
- Evaluation metadata recording where the skill helped, had no effect, or caused a
  regression.

Initial operations:

- `propose_skill`: Create an inactive candidate from a conversation or explicit
  user request.
- `discover_skills`: Search skill metadata and contents for the current task.
- `load_skill`: Load a selected approved skill into a bounded context budget.
- `list_skills` and `show_skill`: Make active and proposed skills inspectable.
- `approve_skill`, `reject_skill`, and `disable_skill`: Keep activation under user
  control.

Discovery should use explicit triggers and lexical search first, with embedding
retrieval as a secondary signal. Loading must be selective: inject the smallest
set of relevant skills rather than the entire library.

### 1.2 Structured Memory

Keep the existing raw conversational memory as an immutable source, while adding
typed durable memories for information that should be recalled directly.

Initial memory types:

- Explicit user preferences.
- Durable facts supplied or confirmed by the user.
- Project decisions and constraints.
- Ongoing tasks and unresolved commitments.

Each structured memory should include scope, provenance, status, timestamps,
confidence, and an optional expiration time. Superseding or correcting a memory
must preserve its history rather than silently overwriting it.

Initial operations:

- `propose_memory`: Create an inactive memory candidate with source attribution.
- `discover_memories`: Search approved memory metadata and contents only within
  the caller's permitted scopes.
- `load_memory`: Load selected approved memories into the current task context.
- `list_memories` and `show_memory`: Allow inspection of stored knowledge.
- `approve_memory`, `reject_memory`, `update_memory`, and `forget_memory`: Give the
  user direct control over persistence.

Bob may suggest memories, but initially only explicit approval activates them.
Conflicting, sensitive, or inferred claims must never be auto-approved.

### 1.3 Context Assembly and Episodic Index

Add a context builder that independently selects:

1. Recent conversation turns.
2. Relevant raw historical passages.
3. Approved structured memories.
4. Approved procedural skills.

The builder must enforce per-source token budgets, deduplicate overlapping
content, preserve provenance, and expose which items were injected for debugging.

Segment older chat history into task or topic episodes so retrieval does not have
to search only isolated message chunks. Each derived episode should contain a
source message range, participants, time range, topic tags, task status, and a
compact description of what happened. Episode summaries are navigation indexes,
not approved memories: they remain linked to raw messages, are marked as
model-derived, and cannot silently become trusted facts. Retrieval should use an
episode to locate the relevant raw evidence before relying on important details.

Maintain an explicit task or thread ID where possible so separate goals discussed
in the same chat do not contaminate one another. A task may span multiple chat
turns, process restarts, or scheduled resumptions without requiring the entire chat
history to be replayed.

### 1.4 Active Task State and Adaptive Compaction

The chat ring buffer provides recent conversational continuity, but a long-running
tool workflow needs its own durable working state. Add adaptive task-context
compaction based on the SelfCompact approach: periodically ask the model, using a
small explicit rubric, whether the current trajectory has reached a safe semantic
boundary for compaction.

Keep three distinct context horizons:

1. **Recent chat context:** The existing ring buffer for conversational continuity.
2. **Active task context:** Original task, compacted task state, a short raw tail,
   and the tool interactions still needed by the current subtask.
3. **Long-term context:** Approved memories and skills, plus searchable raw history.

Compaction is a view over an immutable trajectory, not deletion. The full messages,
tool calls, tool results, and earlier compacted states remain durably stored and can
be retrieved for audit or recovery.

#### Compaction Trigger

At configurable tool-iteration or token intervals, append a rubric probe to a copy
of the current trajectory. Compact when:

- A subtask has completed and its result has been verified.
- A search or investigation has converged on durable findings.
- An earlier plan or failed approach is no longer active.
- Repeated tool output has been incorporated into a smaller working state.

Do not compact when:

- A tool call or dependent operation is still in flight.
- The agent is midway through a derivation or unresolved subtask.
- The most recent observation has not yet been interpreted.
- The agent is stuck or repeating itself; that should trigger reflection or
  replanning rather than conceal the failure in a summary.

Use a hard token-budget backstop to prevent overflow if the model repeatedly elects
not to compact. Trigger values must be model-specific and selected through the
evaluation harness rather than embedded as universal constants.

#### Structured Task State

The compacted state should use a stable schema rather than an unconstrained prose
summary:

- Original objective and immutable user constraints.
- Completed and verified findings, with references to their source events.
- Decisions made and approaches rejected.
- Current subgoal and next intended actions.
- Unresolved questions, risks, and blockers.
- Produced artifacts, paths, identifiers, and relevant versions.
- Loaded skill IDs and memory IDs that remain applicable.
- A short raw tail containing recent unresolved interactions.

System instructions, the original user request, security constraints, and pending
tool-call protocol messages are pinned and must not be summarized away. A
`recall_task_history` operation should allow narrowly retrieving omitted raw events
when the compacted state lacks a needed detail.

Each compaction records its input event range, rubric verdict, generated state,
model and prompt versions, token counts, and the compacted state's parent version.
Compacted task state is a derived, replaceable execution view and therefore does
not require user approval. It cannot become a durable user fact or procedural skill
without going through the normal proposal and user-validation workflow.

### Acceptance Criteria

- A user can propose, review, approve, reject, disable, and delete memories and
  skills.
- Proposed items cannot influence responses before approval.
- DM-scoped items cannot be discovered or loaded from another DM or Townhall.
- Skill selection is observable and bounded by a configurable token budget.
- Memory corrections and skill revisions retain an auditable version history.
- Restart and backup/restore operations preserve lifecycle states and provenance.
- Long workflows can compact and resume after restart without losing their original
  objective, verified findings, active subgoal, or pending tool protocol state.
- Raw task events remain recoverable after any number of compactions.

## Phase 2: Evaluation Dataset and Harness

Build the measurement system before introducing automated prompt optimization or
model training. The harness should evaluate the complete agent behavior, not only
the model's final text.

### 2.1 Bob Evaluation Dataset

Create a versioned repository of replayable tasks. Each case should define:

- Input messages and channel context.
- Initial memory, skill, and workspace fixtures.
- Available tools and relevant configuration versions.
- Expected outcomes, prohibited outcomes, and task-specific verifiers.
- Tags for capability, difficulty, privacy scope, and expected execution cost.
- Whether grading is deterministic, rubric-based, or requires user review.

Initial capability groups:

- Skill discovery, selection, loading, and conflicts between skills.
- Memory proposal, approval, correction, retrieval, and forgetting.
- Cross-user and Townhall/DM privacy isolation.
- Tool selection and argument correctness.
- Sandbox tasks with executable or file-state verification.
- Long-running task recovery and scheduled execution.
- Adaptive compaction timing, state fidelity, raw-history recovery, and resistance
  to context rot across long tool trajectories.
- Prompt injection, unsafe tool requests, and secret-handling behavior.
- General response quality and the ability to avoid unnecessary tool calls.

Prefer deterministic verification such as unit tests, JSON Schema validation,
expected database state, file hashes, or explicit tool-call constraints. LLM judges
may supplement these checks but should not be the sole gate for critical behavior.

### 2.2 Harness

The harness should:

- Run the same cases against API and local OpenAI-compatible providers.
- Create isolated, reproducible fixtures for memory, skills, and workspaces.
- Record final results and turn-level actions, observations, errors, and retries.
- Compare no compaction, fixed-threshold compaction, and rubric-gated adaptive
  compaction under matched task and token budgets.
- Measure pass rate, privacy violations, tool-call accuracy, token use, latency, and
  estimated cost.
- Compare a candidate against a named baseline and report paired regressions.
- Support a fast smoke suite and a larger offline suite.
- Emit machine-readable results suitable for future training-data selection.

Maintain separate optimization, development, and held-out test partitions. Do not
use held-out cases to evolve prompts, author skills, or select checkpoints.

### Acceptance Criteria

- The suite produces a reproducible baseline report for the current Gemini-backed
  agent.
- Memory and skill behavior is covered by both positive and isolation tests.
- A candidate change can be rejected automatically for critical regressions.
- Results identify the exact model, prompt, skill set, memory fixture, tool schema,
  and code revision used by each run.

## Phase 3: Versioned Trajectory and Feedback Collection

Collect learning-quality interaction data without treating every successful tool
exit or uncorrected response as a positive example.

Each trajectory should record:

- Model, base revision, adapter revision, decoding configuration, and provider.
- System prompt, tool schema, loaded skills, and retrieved-memory versions.
- State, action, complete next-state observation, and final result for every turn.
- Compaction probes, verdicts, structured state versions, and retrievals from
  compacted task history.
- Deterministic verifier results and the source of any subjective judgment.
- Explicit user feedback, corrections, approvals, and rejected proposals.
- Token use, latency, cost, environment revision, and applicable random seeds.
- Privacy classification, consent, and redaction status.

Raw events remain immutable. Rewards, summaries, skills, and training examples are
derived artifacts that can be regenerated as grading logic changes.

### Acceptance Criteria

- Evaluation and production interactions share a common trajectory schema.
- User corrections can be converted into candidate demonstrations while retaining
  the original failed attempt and next-state feedback.
- Private or unapproved data is excluded from exports by default.
- Dataset generation is reproducible from selected trajectory and artifact
  versions.

## Phase 4: Offline Prompt and Skill Optimization

Use the approved skill library and evaluation harness to improve the fast path
before changing model weights.

### GEPA-Based Optimization

Apply reflective optimization to bounded textual components such as:

- System prompts.
- Tool descriptions and tool-use policies.
- Skill triggers and compact skill instructions.
- Memory retrieval and context-routing rules.
- Compaction rubrics and structured summarization prompts.
- Fast-path versus deep-path routing criteria.

Optimization runs must operate offline on the optimization partition. Proposed
changes are versioned candidates and require held-out evaluation plus user review
before promotion. Production feedback may propose a new optimization run but must
not mutate the active prompt directly.

### Acceptance Criteria

- Every optimized artifact has a baseline comparison and readable change history.
- Improvements survive the held-out suite without critical privacy or safety
  regressions.
- Prompt and skill changes can be canaried and rolled back independently of a model
  deployment.

## Phase 5: Local Open-Weights Inference

Select a local model based on Bob's evaluation suite and measured performance on
the target machine, rather than public benchmark scores alone.

### Model Bake-Off

Start with dense models in approximately the 4B and 8B-9B classes. Measure:

- Bob evaluation pass rate, especially structured tool calling.
- Prompt ingestion and generation speed on the Ryzen 7 PRO 8840U / Radeon 780M.
- Memory use at realistic context sizes.
- Reliability of the OpenAI-compatible server and chat template.
- Quantization-induced regressions.

Retain an API escalation path for tasks the local model cannot reliably complete.
The first local model should be small enough to serve interactively and cheap enough
to fine-tune during short rented-GPU sessions. Large dense and MoE models may be
evaluated as teachers or optional backends, but are not the initial continual
learning target.

### Acceptance Criteria

- The selected local model passes a defined minimum quality threshold on Bob's
  held-out suite.
- Local throughput and memory usage are acceptable at the configured context size.
- Provider routing is transparent to the Go harness and records the selected route
  in trajectories.
- API fallback behavior is explicit, measurable, and privacy-aware.

## Phase 6: Slow-Path LoRA Consolidation

Use weight updates only for recurring, generalizable behavior that cannot be served
adequately by memory, skills, or prompt optimization.

### Initial Experiments

Establish progressively more complex baselines:

1. QLoRA supervised fine-tuning on user-approved, verifier-backed corrections.
2. QLoRA with replay examples and output-KL regularization against a frozen
   reference model.
3. Self-distillation or hindsight-guided on-policy distillation where suitable.
4. PESO as an experimental proximal regularizer for a single evolving adapter.

PESO is not assumed to be superior for agent behavior. Compare it against simpler
baselines using identical data, compute budgets, and evaluation gates.

### Adapter Lifecycle

- Keep the foundation model immutable.
- Bind adapters to an exact base-model and tokenizer revision.
- Store immutable candidate and stable adapter checkpoints.
- Never overwrite the last known-good adapter.
- Track the full training-data manifest, hyperparameters, code revision, and
  evaluation report.
- Promote through offline evaluation, shadow testing, and a limited canary.
- Support immediate rollback.

A previous-adapter proximal term alone does not prevent gradual long-term drift.
Replay data, reference-policy checks, and regression evaluation remain mandatory.

### Acceptance Criteria

- A LoRA candidate improves its target capability beyond the best prompt/skill
  baseline.
- General reasoning, tool use, privacy, and safety regressions remain within
  explicit thresholds.
- Training fits the defined rental budget and can be reproduced from its manifest.
- Deployment and rollback do not require mutating or replacing the base model.

## Phase 7: Verifier-Backed Agentic RL and Self-Play

Consider reinforcement learning only after the evaluation, trajectory, local
inference, and adapter pipelines are reliable.

Start with short-horizon tasks that have deterministic rewards, such as producing a
file that passes tests or issuing a valid sequence of tool calls. Extend to
multi-turn tasks only with turn-level credit assignment and explicit safeguards
against reward hacking and excessive reasoning.

Later, add proposer-solver task generation inside isolated environments:

- Generate tasks grounded in available tools and fixtures.
- Deduplicate and cross-verify generated tasks.
- Keep tasks near the current solver's competence frontier.
- Admit tasks to training only when their verifier is reliable.
- Preserve an untouched external evaluation set.

Generic reward-prediction-error curiosity and unconstrained autonomous "dreaming"
are deferred. Surprising behavior is not necessarily useful behavior, and
verifiable task generation provides a clearer learning signal.

## Explicitly Deferred

- Pretraining or full-model continual training.
- Unreviewed automatic activation of generated skills or memories.
- Training on private conversations without explicit consent and redaction.
- Continuous live weight updates from individual interactions.
- GraphRAG until measured retrieval failures justify its complexity.
- Large-model or MoE adapter training as the first slow-path experiment.
- Autonomous self-modification outside versioned artifacts and evaluation gates.

## Immediate Work

The next two tracks can proceed in sequence or overlap where practical:

1. **User-validated skills, structured memory, and active task state:** schemas,
   scoped storage, proposal/approval tools, discovery, selective loading, adaptive
   compaction, raw-history recovery, and privacy tests.
2. **Evaluation dataset and harness:** case format, deterministic verifiers,
   isolated fixtures, compaction baselines, metrics, and the first Bob capability
   suite.

Automated skill generation, GEPA, local-model selection, and LoRA training should
wait until these foundations produce trustworthy evidence.
