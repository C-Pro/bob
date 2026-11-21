# Implementation Plan: Bob Go Runtime Integration for `bob-s1` System-1 Inference Sidecar

> **Precondition:** `services/bob_s1/` runs as an independent, externally managed HTTP service (matching System-2 `lemond`). All Go changes must keep `make check` (`build-fwd`, `lint-go`, `test-go` with `-race`, `semgrep`, `osv-scanner`) green at the end of each step.

---

## Step 1: System-1 Configuration (`internal/config`)
- [x] Task: Extend `Config` (`internal/config/config.go`) and `fileConfig` / `ConfigTemplateContent` (`internal/config/file.go`) with the `system1:` YAML block and `SYSTEM1_*` env vars:
  - `system1.base_url` / `SYSTEM1_BASE_URL`, `api_key`, `model` (`"bob-s1-0.8b"`), `timeout` (`30s`), `max_seq_len` (`0` = omit), `embedding_dims` (`384`), `embeddings_enabled`, `compaction_enabled`, `compaction_defer_threshold` (`0` = use calibration `risk_threshold_5pct`), `loop_guard_mode` (`"hybrid"`), `episodes_enabled`, `fallback_on_error` (`true`).
  - Add `NormalizeSystemOneURLs` / `SystemOneURLs` accepting `<origin>` or `<origin>/v1` (or reverse-proxy prefix) and deriving OpenAI `/v1` base URL + `/healthz` URL.
  - Add validation in `Validate()` and `ValidateCLI()` (valid URL scheme/host without query/fragment when set, `embedding_dims ∈ {128, 256, 384}`, `max_seq_len == 0 || 256..262144`, valid `loop_guard_mode`, finite `compaction_defer_threshold <= 1.0`, positive/non-negative timeout rules).
- [x] Task: Add comprehensive unit tests in `internal/config/config_test.go` and `internal/config/file_test.go` (env defaults/overrides/errors, URL normalization, validation, YAML loading, unknown-field rejection, env isolation, template check).
- [x] Task: Run `make check` and confirm 0 errors.

---

## Step 2: `SystemOneClient`, Multimodal `SystemOneEmbedder` & Runtime Wiring (`internal/llm`, `internal/bootstrap`, `cmd/agent`)
- [x] Task: Implement `internal/llm/systemone.go` (`SystemOneClient`):
  - Dedicated HTTP / `go-openai` client without `geminiTransport`.
  - Reuse `config.NormalizeSystemOneURLs` to map `/healthz` and `/v1/{models,embeddings,chat/completions}`.
  - Strict schema-compliant `bob-s1.decision.v1` request/response types (no explicit `null`s, no forbidden sampling/generation parameters, `image_url` parts ordered strictly before the JSON `text` part).
  - `Health`, `Decide`, `DecideCompaction`, `DecideLoopGuard`, `DecideOutcomePrefix`, `DecideCustom`, and bounded retry policy (`429`/`503`/transient only; fail immediately on `400`/`401`/`413`).
- [x] Task: Implement `internal/llm/systemone_embedder.go` (`SystemOneEmbedder`):
  - Implement `cortexdb.Embedder` (`Embed`, `EmbedBatch`, `Dim` returning `128|256|384` immediately) using `openai.EmbeddingRequest` with `EncodingFormat: base64` and `ExtraBody: {"truncation": "head_tail"}` (+ optional `max_seq_len`), sub-batching at `<= 16` items.
  - Decode IEEE-754 `<f4` little-endian vectors explicitly and validate exact byte length (`4 * dims`), contiguous `0..N-1` indices, vector count, `dims` match, and finite float values.
  - Implement `EmbedMultimodal` and `WithEmbeddingImages(ctx, dataURIs)` inspected inside `Embed(ctx, text)`, validating `data:image/(png|jpeg|webp|gif);base64,...` URIs, enforcing `<= 8` images/item and `<= 16` images/request, and gracefully falling back to text-only embedding on image error (unless strict mode forbids image fallback).
- [x] Task: Wire `SystemOneClient` and `SystemOneEmbedder` creation into `internal/bootstrap/runtime.go` and `cmd/agent/main.go` (`--regenerate-vectors` / `--reembed`) preserving embedder precedence (`opts.Embedder` > `embedding_model=none` > explicit `embedding_model` > `system1` > `LazyLocalEmbedder`), plus `internal/bootstrap/runtime_test.go` tests.
- [x] Task: Add `httptest.Server` contract tests (`internal/llm/systemone_test.go`, `internal/llm/systemone_embedder_test.go`) asserting schema compliance against `services/bob_s1/contracts/*.schema.json` invariants + opt-in live smoke test gated by `BOB_S1_TEST_URL`.
- [x] Task: Run `make check` and confirm 0 errors.

---

## Step 3: FSM Control-Plane Integration & Trajectory Telemetry (`internal/fsm`, `internal/trajectory`)
- [x] Task: Implement train-serve-exact state renderers (`internal/fsm/systemone_state.go`) matching `go-probe-sanitized-v1` (`eval/harness/build_jev_dataset.py` & `build_jev_generic_and_multimodal.py`, including canonical key-sorted UTF-8 compact JSON args) with golden byte-level tests.
- [x] Task: Implement `systemOneCompactionDecider` (`internal/fsm/compaction.go`) with `require_calibrated: true`, response calibration validation (`calibration.applied`, `probability_kind == "calibrated_affine"`, non-null `risk_threshold_5pct`), 3-way gated deferral, skipping System-2 on confident `skip`, invoking Stage-2 System-2 summarize-only prompt on confident `compact` (with unit tests proving `nil` Stage-2 state never triggers `buildDeterministicCompactedState`), and respecting `fallback_on_error` strictly (`true` -> full System-2 decider; `false` -> `probe_error`).
- [x] Task: Integrate semantic `loop_guard` (`internal/fsm/loop_guard.go`, `internal/fsm/tool_loop.go`) supporting `"off"`, `"shadow"`, and `"hybrid"` modes with 0ms blocked-repeat short-circuiting, `len(historySteps) < 2` gate, `force_synthesis -> intercept` pre-strike downgrade, deterministic ceiling backstop, and strict `fallback_on_error` handling (`false` propagates error instead of silent deterministic fallback).
- [x] Task: Add `SystemOneDecisionPayload` / `MakeSystemOneDecisionEvent` (`event_type: "system1_decision"`) and optional compaction source/latency fields on `CompactionEventPayload` (`internal/trajectory/models.go`, `internal/trajectory/recorder.go`), using crash-replay-safe unique attempt identity or atomic persistence so replay never causes `RecordEventTx` payload conflicts.
- [x] Task: Wire `fsm.WithSystemOneClient` and `fsm.WithSystemOneConfig` in `internal/fsm/engine.go` and `internal/bootstrap/runtime.go`.
- [x] Task: Add thorough unit tests across `internal/fsm` and `internal/trajectory`, then run `make check` and confirm 0 errors.

---

## Step 4: Multimodal Memory, Bilingual Episodes & Vector Re-Indexing (`internal/memory`, `internal/gateway`, `cmd/agent`)
- [x] Task: Persist active DB vector-space metadata (`embedder_id`, `dims`) and gate incompatible vector searches until `--regenerate-vectors` migrates existing vectors (failing explicitly in strict `fallback_on_error: false` mode).
- [x] Task: Add `Images []string` to `memory.MessageToStore` (`internal/memory/store.go`), populate image data URIs in `Gateway.handleEvictedBatch` and `Gateway.CatchupChatMemory` (`internal/gateway/gateway.go`), attach up to 8 images via `llm.WithEmbeddingImages` during `Manager.IndexMessages`, store bounded image/attachment metadata (`has_images: true`), and ensure multimodal chunks store non-null vectors (without silently swallowing errors in strict mode).
- [x] Task: Update `internal/memory/episode.go`:
  - Classify `TaskStatus` (`completed`, `ongoing`, `unresolved`, `discussion`) via `bob-s1` `DecideCustom` with heuristic fallback when disabled or `fallback_on_error: true` (erroring when `fallback_on_error: false`).
  - Add Russian stop-words to `episodeStopWords` and change topic-tag minimum token length from 4 bytes to 3 Unicode runes (`utf8.RuneCountInString`).
- [x] Task: Update `internal/memory/regenerate.go` & `cmd/agent --regenerate-vectors` / `--reembed` to select `SystemOneEmbedder` using the same precedence as `bootstrap.New`, re-embed chunks/episodes/`dm_memories`/`dm_skills` (batch size `<= 16`), reconstruct or safely preserve `has_images: true` multimodal vectors instead of overwriting them with text-only embeddings, and update DB vector-space metadata.
- [x] Task: Add bilingual EN+RU episode classification + multimodal image-chunk retrieval tests (including CortexDB `memorySemanticFloor` validation) across `internal/memory` and `internal/gateway`.
- [x] Task: Run `make check` and confirm 0 errors.

---

## Step 5: Controlled Evaluation Harness & Held-Out Benchmark (`eval/harness`)
- [x] Task: Extend `eval/harness/runner.py` and `eval/harness/config_template.yaml` with `--system1-url`, `--system1-model`, `--system1-loop-guard-mode`, `--system1-fallback-on-error`, and `BOB_EVAL_SYSTEM1_URL` env support.
- [x] Task: Verify end-to-end execution on `eval/tasks/smoke_4.json` against a live external `services/bob_s1` sidecar with `fallback_on_error: false` (checking `/healthz`, `calibration.applied == true` on `task == "compaction"`, zero unexpected fallbacks, and `system1_decision` telemetry in exported trajectories).
- [x] Task: Run controlled evaluation on `eval/tasks/heldout/heldout_100.json` under identical actor model (`Qwen3.6-35B-A3B-MTP-GGUF`), iteration/timeout budgets, and Docker grader against the 55.0% baseline; record pass rate, per-benchmark breakdown, compaction latency (`gate + summarizer`), skipped probes, loop-guard interventions, and fallback/error counts in `eval/results/`.
- [x] Task: Run final `make check` and confirm 0 errors.
