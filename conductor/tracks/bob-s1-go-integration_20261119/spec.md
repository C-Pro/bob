# Specification: Bob Go Runtime Integration for `bob-s1` System-1 Inference Sidecar

## 1. Objective, Process Model & Scope

Integrate Bob's Go runtime (`internal/config`, `internal/llm`, `internal/bootstrap`, `internal/fsm`, `internal/trajectory`, `internal/memory`, `internal/gateway`, `cmd/agent`, `eval/harness`) with the `bob-s1-0.8b` OpenAI-compatible HTTP inference contract frozen under [`services/bob_s1/contracts/`](../../../services/bob_s1/contracts/README.md).

### Process Model
- **Externally Managed Service:** The inference sidecar (`services/bob_s1/`) is started and supervised as a separate external process (matching System-2 `lemond`). Bob never spawns, restarts, or manages Python child processes.
- **Native Server Swappability:** Because Go interacts strictly through the frozen HTTP/JSON contract (`GET /healthz`, `GET /v1/models`, `POST /v1/embeddings`, `POST /v1/chat/completions` carrying `bob-s1.decision.v1`), a future native `lemond`, `vllm`, or `llama.cpp` server satisfying the same contract and conformance fixtures drops in without Go code changes.

### Out of Scope
- Modifying [`services/bob_s1/`](../../../services/bob_s1/README.md) runtime behavior or frozen wire schemas.
- Native C++ `llama.cpp` / `lemond` server patches (deferred to a separate future track).
- Modifying existing Conductor tracks (including [`jev-multimodal-decision-embedder_20261114`](../jev-multimodal-decision-embedder_20261114/spec.md)).
- Wiring `outcome_prefix` into active FSM control-plane branching: per [`eval/BOB_S1.md`](../../../eval/BOB_S1.md) §8, `outcome_prefix` has weak within-domain discrimination (`0.4327` FRAMES / `0.6042` Bash on `cal`), so `SystemOneClient` exposes `DecideOutcomePrefix` as a typed client helper only.

---

## 2. Configuration Contract (`internal/config`, `internal/bootstrap`, `cmd/agent`)

Add an optional **System-1 (`system1`)** configuration block to [`internal/config/config.go`](../../../internal/config/config.go) and [`internal/config/file.go`](../../../internal/config/file.go) (`yaml:"system1"` and `SYSTEM1_*` environment variables). When the System-1 endpoint URL is empty, System-1 integration is disabled and Bob preserves 100% existing behavior.

### 2.1 Configuration Fields & Normalization
- **Endpoint URL (`system1.base_url` / `SYSTEM1_BASE_URL`):**
  - Distinct from System-2 `llm.base_url` (`OPENAI_BASE_URL`).
  - Accepts either a bare server origin (`http://127.0.0.1:8090`) or an OpenAI `/v1` prefix (`http://127.0.0.1:8090/v1`, with or without trailing slash).
  - Normalizes OpenAI endpoint URLs to `<origin>/v1/embeddings`, `<origin>/v1/chat/completions`, and `<origin>/v1/models`, and derives the readiness probe URL `<origin>/healthz`.
- **Connection & Model:**
  - `api_key` (`SYSTEM1_API_KEY`): Optional Bearer token sent to `/v1/*` endpoints.
  - `model` (`SYSTEM1_MODEL`): Model identifier (default `"bob-s1-0.8b"`).
  - `timeout` (`SYSTEM1_TIMEOUT`): Per-request HTTP timeout (default `30s`).
  - `max_seq_len` (`SYSTEM1_MAX_SEQ_LEN`): Optional per-request ceiling (`0` = omit field so the server uses `BOB_S1_MAX_SEQ_LEN`, default `4096`; calibration profiles apply only at `effective_max_seq_len == 4096`).
- **Embeddings:**
  - `embedding_dims` (`SYSTEM1_EMBEDDING_DIMS`): Matryoshka dimension in `{128, 256, 384}` (default `384`).
  - `embeddings_enabled` (`SYSTEM1_EMBEDDINGS_ENABLED`): Default `true` when System-1 URL is configured.
- **Control-Plane Policies:**
  - `compaction_enabled` (`SYSTEM1_COMPACTION_ENABLED`): Default `true` when System-1 URL is configured.
  - `compaction_defer_threshold` (`SYSTEM1_COMPACTION_DEFER_THRESHOLD`): Uncertainty deferral threshold on `max(p_skip, p_compact)` (default `0` = use the sidecar's returned `calibration.risk_threshold_5pct`, `0.8393841862136965`; negative value disables uncertainty deferral).
  - `loop_guard_mode` (`SYSTEM1_LOOP_GUARD_MODE`): `"off" | "shadow" | "hybrid"` (default `"hybrid"` when System-1 URL is configured).
  - `episodes_enabled` (`SYSTEM1_EPISODES_ENABLED`): Default `true` when System-1 URL is configured.
  - `fallback_on_error` (`SYSTEM1_FALLBACK_ON_ERROR`): Default `true` (fail-open to System-2 / deterministic heuristics on sidecar transport errors, `429`, or `503`; set `false` for strict evaluation runs).

### 2.2 Validation & Template Sync
- Validate in both `Config.Validate()` and `Config.ValidateCLI()`:
  - If System-1 URL is non-empty, validate `http`/`https` scheme and non-empty host.
  - `embedding_dims` must be `128`, `256`, or `384`.
  - `max_seq_len` must be `0` or in `256..262144`.
  - `timeout` must be positive.
  - `loop_guard_mode` must be `"off"`, `"shadow"`, or `"hybrid"` (normalized lowercase).
  - `compaction_defer_threshold` must be finite and `<= 1.0` (where `0` means default calibration threshold and `< 0` disables deferral).
- Update `ConfigTemplateContent` in [`internal/config/file.go`](../../../internal/config/file.go) (which enforces `dec.KnownFields(true)`).

### 2.3 Embedder Selection Precedence & Vector-Space Guard (`internal/bootstrap/runtime.go`, `cmd/agent/main.go`)
Preserve a clear, non-breaking embedder resolution order in both [`internal/bootstrap/runtime.go`](../../../internal/bootstrap/runtime.go) and [`cmd/agent/main.go`](../../../cmd/agent/main.go) (`--regenerate-vectors` / `--reembed`):
1. Explicit `opts.Embedder` (unit/integration tests).
2. `embedding_model` in `{"none", "disabled", "off"}` -> `nil` (pure lexical FTS5 mode).
3. Explicit non-empty `embedding_model` -> existing `llm.NewEmbedder(llmClient, cfg.EmbeddingModel)` (System-2 / external embedding API).
4. System-1 URL configured and `system1.embeddings_enabled` -> `llm.NewSystemOneEmbedder(...)`.
5. Fallback default -> existing `llm.NewLazyLocalEmbedder(cfg)` (or `llm.NewLocalEmbedder` in CLI vector regeneration).

- **Vector-Space Compatibility Guard:** Because both legacy `go-embed` (`MiniLM-L12-v2`) and `bob-s1` default to `384` dimensions, matching dimensionality alone cannot detect incompatible embedding spaces. Persist active vector-space metadata (`embedder_id` / `model` and `dims`) in the memory DB metadata table; when a DB created under another embedding model has existing vectors and the active embedder changes (e.g., switching to `bob-s1`), fail or gate vector semantic search to FTS5/BM25 (or fail fast when `fallback_on_error: false`) with an actionable warning until `cmd/agent --regenerate-vectors` / `--reembed` migrates existing vectors and updates the DB vector-space stamp.

---

## 3. `SystemOneClient` & Multimodal `SystemOneEmbedder` (`internal/llm`)

### 3.1 `SystemOneClient` (`internal/llm/systemone.go`)
- Uses a dedicated HTTP client / `go-openai` client **without `geminiTransport`** ([`internal/llm/client.go`](../../../internal/llm/client.go)) so Google Gemini `thought_signature` and `extra_content` transport hooks never mutate `bob-s1` payloads.
- **Strict Wire Schema Compliance (`services/bob_s1/contracts/*.schema.json`):**
  - `schemas.py` enforces `additionalProperties: false` and `_StrictModel` rejects explicit `null` fields; `FORBIDDEN_CHAT_PARAMS` rejects any sampling/generation parameters (`temperature`, `top_p`, `n`, `stop`, `max_tokens`, `max_completion_tokens`, `presence_penalty`, `frequency_penalty`, `logit_bias`, `seed`, `tools`, `tool_choice`, `stream: true`, `logprobs: true`).
  - Request structs must omit unset optional fields (`omitempty` with pointer fields such as `ApplyCalibration *bool`, `RequireCalibrated *bool`, `MaxSeqLen *int`).
  - When images are attached to a decision request (`data:image/(png|jpeg|webp|gif);base64,...`), serialize `messages[0].content` as a part array with all `image_url` parts **strictly before** the single `text` JSON part.
- **Surface:**
  - `Health(ctx) (*SystemOneHealth, error)` (`GET /healthz`).
  - `Decide(ctx, DecisionRequest) (*DecisionResponse, error)` (`POST /v1/chat/completions` using `bob-s1.decision.v1`).
  - Typed helpers: `DecideCompaction`, `DecideLoopGuard`, `DecideOutcomePrefix`, `DecideCustom`.
- **Retry & Error Classification:**
  - Short bounded backoff retries on `429` (`queue_full`), `503` (`model_not_ready`), and transient transport failures.
  - Never retry `400` (`invalid_request_error`, `context_length_exceeded`, `calibration_unavailable`, `calibration_conflict`), `401` (`authentication_error`), or `413` (`payload_too_large`).

### 3.2 `SystemOneEmbedder` (`internal/llm/systemone_embedder.go`)
- Implements `cortexdb.Embedder`:
  - `Dim() int` returns the configured Matryoshka dimension (`128`, `256`, or `384`) immediately without making an RPC.
  - `Embed(ctx, text)` and `EmbedBatch(ctx, texts)` encode requests via `openai.EmbeddingRequest` (not `EmbeddingRequestStrings`) with `Dimensions`, `EncodingFormat: openai.EmbeddingEncodingFormatBase64`, and `ExtraBody: map[string]any{"truncation": "head_tail"}` (+ `"max_seq_len"` when configured). Sending `truncation: "head_tail"` overrides the sidecar's `"error"` default on `/v1/embeddings` so long 33-message chunks never fail with `400 context_length_exceeded`.
  - **Strict Response Validation:** Never rely blindly on `go-openai`'s `Base64Embedding.Decode()` (which silently truncates trailing bytes if byte length is not a multiple of 4). Decode IEEE-754 little-endian `<f4` explicitly and verify: response count equals batch input count, indices are contiguous `0..N-1` (reordered by `Index`), decoded byte length is `4 * dims`, float vector length equals `dims`, and every value is finite (non-NaN, non-Inf). Return a descriptive error on any violation.
- **Batching & Multimodal Caps:**
  - Sub-batch `EmbedBatch` at `<= 16` items per HTTP request (`BOB_S1_MAX_BATCH_SIZE`).
  - Provide `EmbedMultimodal(ctx, text, imageDataURIs)` and context helper `WithEmbeddingImages(ctx, dataURIs)` inspected inside `Embed(ctx, text)`.
  - `Embed` inspects `WithEmbeddingImages` so `cortexdb.DB.SaveMemory` achieves single-RPC multimodal indexing with **zero changes to vendored `cortexdb`**.
  - Filter to valid `data:image/(png|jpeg|webp|gif);base64,...` URIs, cap at `<= 8` images per item (`BOB_S1_MAX_IMAGES`) and `<= 16` images per request (`BOB_S1_MAX_REQUEST_IMAGES`), and gracefully fall back to text-only embedding with a warning log if image validation/payload limits trigger a `400`/`413` (unless running in strict `fallback_on_error: false` mode for multimodal evaluation).
  - Note that vendored `cortexdb` `DB.embedMemoryContent` logs and swallows `Embed()` errors (`vector = NULL`), so `memory.Manager` / strict memory tests must surface or verify non-null vector storage rather than assuming `SaveMemory` propagates embedder failures.

---

## 4. FSM Control-Plane Integration & Trajectory Telemetry (`internal/fsm`, `internal/trajectory`)

Wire `SystemOneClient` into `fsm.Engine` (`WithSystemOneClient`, `WithSystemOneConfig`) and `ToolLoopRunner`.

### 4.1 Train-Serve-Exact State Renderer (`internal/fsm/systemone_state.go`)
Training and calibration (`eval/harness/build_jev_dataset.py` and `build_jev_generic_and_multimodal.py`) used `renderer_version = "go-probe-sanitized-v1"`, which differs from `llmCompactionDecider` in [`internal/fsm/compaction.go`](../../../internal/fsm/compaction.go):
- **Compaction state format (`build_jev_dataset.py:254-301`):**
  - Omit `Run ID: ...` and `StepID=...`.
  - Canonicalize step tool arguments and `Previous Compacted State` using Python `compact_json` semantics: sorted object keys (`encoding/json` map marshaling), compact separators (`,`, `:`) with no extra whitespace, and unescaped UTF-8 (`SetEscapeHTML(false)` / `ensure_ascii=False` parity).
  - Sanitize `Previous Compacted State` to the 10 `STATE_FIELDS` (`original_objective`, `immutable_constraints`, `verified_findings`, `decisions_and_rejected`, `current_subgoal`, `next_intended_actions`, `unresolved_blockers`, `produced_artifacts`, `active_skill_versions`, `active_memory_versions`), stripping `source_step_ids` from `verified_findings` items (`{"summary": ...}`) and omitting `version`, `parent_version`, `run_id`, `rubric_verdict`, `model`, `created_at`.
  - Use identical UTF-8 byte caps (`2048` goal, `8192` previous state, `512` step args, `1024` step output, `32768` total prompt).
- **Loop-guard state format (`build_jev_dataset.py:565-567`, `build_jev_generic_and_multimodal.py:905-922`):**
  - Use trailing window `window = history[-12:]`, `previous = nil`, and append `\nCandidate Tool Calls (not yet executed):\n- Tool=<name> Args=<clipped_args>\n`.

### 4.2 Compaction Gate (`internal/fsm/compaction.go`)
- Keep `compactionTriggerForceBackstop` (`activeBytes >= HardThresholdBytes`) unchanged as the deterministic safety backstop.
- Wrap `SystemOneClient` + System-2 `llmCompactionDecider` in a `systemOneCompactionDecider`:
  1. **Stage 1 (System-1 Gate):** Call `bob-s1` with `task: "compaction"`, `scoring_mode: "joint"`, `prefix_mode: "per_question"`, `apply_calibration: true`, `require_calibrated: true`, and `max_seq_len: 4096` so `cal_v2_4k` calibration is enforced server-side.
  2. **Response Calibration Validation:** Verify `calibration != nil`, `calibration.applied == true`, `calibration.probability_kind == "calibrated_affine"`, and when uncertainty deferral is active (`compaction_defer_threshold >= 0`) verify non-null `calibration.risk_threshold_5pct` if using the sidecar default (`compaction_defer_threshold == 0`). Treat a calibration mismatch as a gate error subject to `fallback_on_error`.
  3. **Uncertainty Deferral:** If `deferThreshold >= 0` (effective default: sidecar's returned `calibration.risk_threshold_5pct`, `0.8393841862136965`) and `max(p_skip, p_compact) < effectiveDeferThreshold`, defer to the full System-2 `llmCompactionDecider`.
  4. **Confident `skip`:** Return `ShouldCompact: false` immediately without calling System-2.
  5. **Confident `compact` (Stage 2 Summarizer):** Invoke System-2 with a summarize-only prompt to populate `CompactedTaskState`. Strictly honor [`jev-multimodal-decision-embedder_20261114/spec.md`](../jev-multimodal-decision-embedder_20261114/spec.md) §4 Experiment 3: **never** silently fall back to `buildDeterministicCompactedState` on System-1 `YES`; if Stage-2 summarization fails or returns `nil` state, return an error (`probe_error` skip) and verify this invariant with a unit test.
  6. **System-1 Error Behavior:** When `fallback_on_error` is `true`, fall back to full System-2 `llmCompactionDecider` (and record `fallback_used: true`); when `fallback_on_error` is `false`, do not invoke System-2 `llmCompactionDecider` and return the error (`probe_error` skip).

### 4.3 Semantic Loop Guard (`internal/fsm/loop_guard.go`, `internal/fsm/tool_loop.go`)
Integrate `bob-s1` `loop_guard` into `handlePrepareSteps` ([`internal/fsm/tool_loop.go`](../../../internal/fsm/tool_loop.go)):
- **Hard 0ms Invariant:** If a candidate call matches a tool call signature previously blocked by `loop_guard_blocked` / `Loop guard notice:`, immediately return `LoopActionForceSynthesis` deterministically without a sidecar call.
- **Minimum History Gate:** Do not call `bob-s1` when `len(historySteps) < 2` (run deterministic `DetectLoop` only).
- **Modes:**
  - `off`: Existing deterministic `DetectLoop`.
  - `shadow`: Enforce deterministic `DetectLoop`, additionally query `bob-s1` `loop_guard` (`window = history[-12:]`) and record agreement/disagreement telemetry without altering execution. When `fallback_on_error` is `false`, surface sidecar probe errors explicitly in telemetry/error reporting instead of ignoring them silently.
  - `hybrid` (default when System-1 enabled): Consult `bob-s1` `loop_guard` (`window = history[-12:]`):
    - Enforce a hard deterministic backstop ceiling (e.g., `>= 5` consecutive identical calls with unchanged observations) against false negatives.
    - Otherwise follow `bob-s1`'s verdict: `none` permits constructive non-zero test iterations, `intercept` attaches `FormatLoopGuardNotice` to candidate calls, and `force_synthesis` stops the loop (downgrading `force_synthesis` -> `intercept` if no prior strike has occurred yet, preserving the 2-strike UX).
    - On sidecar failure: when `fallback_on_error` is `true`, fall back cleanly to deterministic `DetectLoop` (recording `fallback_used: true`); when `fallback_on_error` is `false`, fail/halt cleanly (or propagate the loop-guard error to `Run` failure) rather than silently falling back to deterministic `DetectLoop`.

### 4.4 Trajectory Telemetry (`internal/trajectory/models.go`, `internal/trajectory/recorder.go`)
- Add `SystemOneDecisionPayload` and `MakeSystemOneDecisionEvent` (`event_type: "system1_decision"`), recording: `task`, `choice`, `choice_basis`, `scores`, `calibration_profile`, `calibration_applied`, `mismatch_reason`, `prefix_tokens`, `suffix_tokens`, `truncated`, `latency_ms`, `applied_action`, `heuristic_action`, `fallback_used`, `shadow_mode`, and optional `defer_reason`.
- Add optional `omitempty` fields to `CompactionEventPayload`: `DecisionSource`, `SystemOneLatencyMs`, `SummarizerLatencyMs`.
- **Crash-Replay Identity:** Because `agent_trajectory_events` enforces `UNIQUE (run_id, generation, iteration, phase, step_id, attempt, event_type)` and rejects mismatched JSON payloads on replay, persist `system1_decision` atomically inside the FSM transition transaction or key `attempt`/`step_id` uniquely per probe attempt (e.g., monotonically incrementing attempt / sub-step identity or atomic tx write) so crash-recovery replay never drops telemetry or triggers spurious payload-conflict errors. `ReconcileTerminalRuns` skips unknown event types while `eval/harness/runner.py` exports all `agent_trajectory_events` rows.

---

## 5. Multimodal Memory, Bilingual Episodes & Vector Re-Indexing (`internal/memory`, `internal/gateway`, `cmd/agent`)

### 5.1 Multimodal Chunk Indexing
- Add `Images []string` (data URIs) to `memory.MessageToStore` in [`internal/memory/store.go`](../../../internal/memory/store.go).
- Populate `Images` from `chatcontext.Entry.Images` in `Gateway.handleEvictedBatch` and from `processAttachments` in `Gateway.CatchupChatMemory` ([`internal/gateway/gateway.go`](../../../internal/gateway/gateway.go)).
- In `Manager.IndexMessages`: collect up to 8 valid inline `data:image/(png|jpeg|webp|gif);base64,...` URIs per chunk, attach them via `llm.WithEmbeddingImages(ctx, chunkImages)` when calling `db.SaveMemory`, persist bounded image/attachment references (or mark `has_images: true` & `image_count`) in chunk metadata so vector regeneration can either reconstruct multimodal inputs or safely preserve existing multimodal vectors, and verify non-null vector storage in strict mode (`fallback_on_error: false` / tests).

### 5.2 Bilingual (EN+RU) Episode Classification & Tagging (`internal/memory/episode.go`)
- Replace English-only keyword heuristics in `inferEpisodeTaskStatus` with a 4-way `bob-s1` `DecideCustom` probe over `{"completed", "ongoing", "unresolved", "discussion"}` when System-1 episodes are enabled; fall back to `inferEpisodeTaskStatus` when disabled or when `fallback_on_error: true`, and surface the error when `fallback_on_error: false`.
- Update `extractTopicTags` for Russian and bilingual text:
  - Add common Russian stop-words to `episodeStopWords`.
  - Switch the minimum token filter from `len(tok) < 4` (UTF-8 bytes, which allows 2-letter Cyrillic tokens like `"не"`, `"на"`, `"по"`) to `utf8.RuneCountInString(tok) < 3`.
- Validate EN+RU episode classification and multimodal image+text retrieval on a dedicated fixture/test suite, and verify CortexDB's `memorySemanticFloor = 0.28` behavior with `bob-s1` cosine similarities.

### 5.3 Vector Re-Indexing & Vector-Space Migration (`internal/memory/regenerate.go`, `cmd/agent/main.go`)
- Ensure `cmd/agent --regenerate-vectors` / `--reembed` selects `SystemOneEmbedder` using the same precedence as `bootstrap.New`, respects `<= 16` batch size, and re-embeds chunks, episodes, `dm_memories`, and `dm_skills` (`WHERE session_id LIKE 'memory:%'`).
- Handle multimodal chunks (`has_images: true`) cleanly during regeneration: reconstruct stored image refs via `EmbedMultimodal` when persisted/available, or skip/report rather than overwriting multimodal vectors with text-only `EmbedBatch` embeddings.
- Record and update active vector-space metadata (`embedder_id`, `dims`) in the DB upon successful regeneration.

---

## 6. Controlled Evaluation (`eval/harness`, `smoke_4.json`, `heldout_100.json`)

- Extend [`eval/harness/runner.py`](../../../eval/harness/runner.py) and [`eval/harness/config_template.yaml`](../../../eval/harness/config_template.yaml) with `--system1-url`, `--system1-model`, `--system1-loop-guard-mode`, `--system1-fallback-on-error`, and `BOB_EVAL_SYSTEM1_URL` env support.
- Run [`eval/tasks/smoke_4.json`](../../../eval/tasks/) against a live external `services/bob_s1` sidecar with strict `fallback_on_error: false` to verify end-to-end health, **compaction calibration** (`calibration.applied == true` scoped to `task == "compaction"`), zero unexpected fallbacks, and `system1_decision` trajectory telemetry.
- Run a controlled evaluation on `eval/tasks/heldout/heldout_100.json` against the 55.0% baseline (`Qwen3.6-35B-A3B-MTP-GGUF`, identical budgets and Docker grader), reporting pass rate, per-benchmark breakdown, total compaction latency (`gate + summarizer`), skipped System-2 compaction probes, loop-guard interventions, and fallback/error counts.
