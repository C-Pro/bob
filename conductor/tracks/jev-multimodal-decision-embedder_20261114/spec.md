# Specification: Phase 4 — Unified 0.8B Multimodal Decision & Embedding Model (`Jev`)

## 1. Objective & Donor Selection

Replace expensive generative LLM control-plane calls and external embedding APIs (`gemini-embedding-2` / CPU-only `go-embed`) with a fine-tuned **`0.8B` natively multimodal System-1 decision model with a dedicated embedding output head**, deployed on GPU-accelerated `llama.cpp`.

### Locked Architecture & Donor Choices
- **Primary Donor:** **`jaredpalmer/kev-0.8b`** (Apache-2.0, LoRA `r=16, alpha=32` / `11.3M` params + `PointerHead` on frozen **`Qwen/Qwen3.5-0.8B-Base`**, revision `dc7cdfe2`, shipped temperature `T=2.35`, validated to `8,192` tokens).
  - *Why `kev-0.8b` over `mstrasser/jeff-base` v1.4:* `kev-0.8b` puts `[State 1..S]` **first** and keeps base `Qwen/Qwen3.5-0.8B-Base` weights intact (compatible with prefix embedding & frozen vision weights). By contrast, `mstrasser/jeff-base` v1.4 uses `live-last` (options first, changing state last) + fixed vocabulary codes in `readout.safetensors`, making its final state question-conditioned unless retrained from scratch.
- **Confirmed Backbone Geometry (`Qwen/Qwen3.5-0.8B` / `Qwen3.5-0.8B-Base` `config.json`):**
  - Architecture: `Qwen3_5ForConditionalGeneration`.
  - Text backbone (`qwen3_5_text`): `24` layers (`18` Gated DeltaNet linear attention + `6` full attention via `full_attention_interval: 4`), `hidden_size: 1024`, `max_position_embeddings: 262144`.
  - Vision tower (`qwen3_5_vision`): **`depth: 12`, `hidden_size: 768`, `out_hidden_size: 1024`**, `image_token_id: 248056`, `video_token_id: 248057` — packaged in `llama.cpp` as `mmproj-*.gguf` and kept **100% frozen** during Stage 1/2 fine-tuning.
- **CPU-Only Dropped & Reindexing Approved:** All inference targets GPU / Apple Silicon `llama.cpp`; existing `gemini-embedding-2` SQLite DBs will be re-indexed cleanly via [`RegenerateAllVectors`](file:///home/cpro/work/experiments/bob/internal/memory/regenerate.go#L49).

---

## 2. Model Architecture: Causal State-Prefix Isolation (Independent Question Rows)

### Critical Gated DeltaNet Constraint
Because `Qwen3.5` contains **18 Gated DeltaNet (linear attention + causal conv1d) layers**, attention masking alone **cannot** isolate sibling question branches on a single packed sequence: DeltaNet's recurrent state and short-conv history leak prior tokens across masked branches. Therefore:
1. **Training & Correct Reference Semantics:** Evaluate each `(State 1..S, Question k)` on **independent rows** (matching `jaredpalmer/kev`), guaranteeing zero cross-question state contamination. Implemented in `eval/jev/model.py` using Qwen native FIM/box delimiters: `<|fim_prefix|> [<|vision_start|> <|image_pad|>×N <|vision_end|>]* STATE` followed by `<|fim_middle|> Q (<|box_start|> opt_k: desc <|box_end|>)+ <|fim_suffix|>`.
2. **State-Prefix Invariance:** Within every row `[Prefix 0..S-1] [Suffix S..L-1]`, causal attention and DeltaNet recurrence satisfy $\partial H_{S-1} / \partial x_{\ge S} = 0$. Reading the last prefix hidden state $H_{S-1}$ yields a **question-invariant** state representation in the exact same forward pass as answering Question `k`.
3. **Inference Optimization (Python / Forked `llama-server` Cache):** Compute `(KV, Gated DeltaNet recurrent_state, conv_state)` once for the prefix `[0..S-1]`, extract the embedding at $H_{S-1}$, then **clone** (never mutate in-place across siblings) the prefix cache per question branch with matching `position_ids` (`question_from_prefix(..., isolate=True)`).

### Dual-Head Design & Anti-Collapse Controls
1. **Dedicated Multimodal Embedding Head (`W_emb`):**
   - Projects $H_{S-1} \in \mathbb{R}^{1024}$ to Matryoshka dims `[128, 256, 384]` (`L2`-normalized **per sliced dimension** independently).
   - Accepts text-only states, images, or images-followed-by-text states in one space via frozen `qwen3_5_vision` (`prepare_state` prepends all image spans before text tokens; arbitrary text-image-text interleaving is not currently implemented).
2. **Decision / Pointer Head (`PointerHead`):**
   - Two bias-enabled linear projections without `LayerNorm` (matching `kev-0.8b`): `q = Linear(1024, 256, bias=True)` at `<|fim_suffix|>` + `k = Linear(1024, 256, bias=True)` at candidate `<|box_end|>` tokens -> `(q · k) / sqrt(256)`.
   - Primitives: `noul` (boolean), `choice` (`2..255` options + abstention threshold selected by risk-vs-coverage / asymmetric cost curves, never assuming rescaled `p_max` equals calibrated probability without domain calibration), and `score` (`2..10` levels -> $E[\text{score}]$).
3. **Empirically Verified Head-Interference Prevention:**
   - **Baseline first:** Freeze the backbone and train `W_emb` + `PointerHead` linear/MLP heads only to measure separation without risking backbone drift.
   - **Controlled adapter intervention:** Add adapter tuning (`LoRA_emb` / `LoRA_dec` or `kl_frozen_reference`) only after measuring before/after decision Brier/ECE AND retrieval `Recall@1/5` + Alignment/Uniformity. Never assume named LoRAs prevent prefix representation collapse by naming alone.

---

## 3. Data Audit & Boundary-Exact Extraction Contract

Audited directly across all `417` rollout files in [`eval/results/train/batch_train_pool_1794581821/`](file:///home/cpro/work/experiments/bob/eval/results/train/batch_train_pool_1794581821/) (`model: gemini-3.5-flash-lite`, zero held-out overlap with [`eval/tasks/heldout/heldout_100.json`](file:///home/cpro/work/experiments/bob/eval/tasks/heldout/heldout_100.json)):

### Ground-Truth Label Audit & Corrections
1. **Compaction Events (`2,332` events across `events[]`, deduplicated from `compactions[]` snapshots):**
   - `2,004` `probed_compact` vs `328` `probed_skip` = `318` rubric skips + **`10` `probe_error` failures** (e.g., timeouts/errors recorded at `internal/fsm/compaction.go:775` — **must be excluded** from semantic `NO` labels!).
   - **Severe domain/class skew:** `1,669/2,004` compacts and `286/328` skips are SWE (`FRAMES` and `Bash` have zero skips); **"Always compact" scores `85.9%` accuracy** on raw events. Models must be judged on **task-macro balanced accuracy, per-class PR, Brier/ECE, and per-benchmark breakdown** against Always-Compact & Current-Policy baselines.
   - `24` `recall_task_history` calls occur across **only 7 tasks** (e.g., pagination / step drill-down) and are valid agent retrieval, NOT automatic compaction failure. Retain as auxiliary metadata only; do NOT flip `probed_compact` labels automatically.
   - **Token budgets:** `input_bytes` measures active context, whereas [`DecideCompaction`](file:///home/cpro/work/experiments/bob/internal/fsm/compaction.go#L488-L545) truncates probe prompts to `maxProbePromptBytes = 32768`. Record exact Qwen3.5 token counts and evaluate metrics bucketed by token length (`<=1k`, `1k–4k`, `4k–8k`, `>8k`).
2. **Outcome Supervision (`323` clean tasks vs `94` invalid grades):**
   - `EvalBreakdown.ImplicitSignal` is **`0.0` across all 417 tasks** -> no step-level `ImplicitSignal` labels exist in this batch.
   - Among `371` rows marked `graded`, **`48` generated SWE patches failed to produce Docker `report.json`** (`eval/harness/runner.py:157`) and were recorded as `reward=0.0`, alongside `46` Bash `oracle_error` runs. Excluding infrastructure failures leaves **`323` clean task outcomes (`120` passed, `203` failed)** usable only as **prefix outcome prediction** (if trained), never as step-level `ImplicitSignal`.
3. **Loop Guard (`loop_guard.go:87`):**
   - `37` FSM `FAILED` steps come from `load_memory` (`26`), `load_skill` (`7`), `web_search` (`4`) — while `execute` has **`7,377` non-zero exit codes** (many of which are constructive failures like a failing reproduction test). Extract loop state using canonicalized + raw args, result/error deltas, and strike counts (`DetectLoop`), separating true stagnation from polling or failing-test progress.
4. **Out-of-Corpus Modalities (Separate Manifests Required):**
   - Coding/Bash/CTF/FRAMES rollouts do not provide EN+RU Besedka conversation statuses or image-screenshot pairs. Stage 1 EN+RU retrieval, episode-status, and image-text pairs must come from dedicated separate manifests (never fabricated from task pass/fail outcomes).

### Strict Decision-Boundary & 3-Way Task-Group Split Contract
- Reconstruct inputs **strictly from state observable at decision time `t`** (`OriginalUserGoal`, prior `compacted_up_to_iter < t` summary, candidate steps `<= compacted_up_to_iter`, tail steps, canonicalized history). Exclude current `state_json`, `verdict`/`reason`, future steps, final answers, terminal status, and final grades.
- Partition **by task / challenge / normalized instruction group** into **three** isolated splits: **`train`**, **`dev`** (architecture & hyperparameter selection), and **`cal`** (post-training temperature & threshold fitting), while leaving `eval/tasks/heldout/heldout_100.json` **100% untouched** until final end-to-end controller comparison under identical executor/budgets/grader.

---

## 4. Phased Verification & Runtime Architecture

### Experiment Progression (Local AMD Radeon 780M ROCm 6.4)
1. **Experiment 0 (Hardware & Leakage Sanity):** Measure forward+backward memory, kernel behavior (Gated DeltaNet ROCm fallback profiling), finite gradients, and tok/s at batch `1`, lengths `512–1024`, gradient checkpointing, without full-vocab logits.
2. **Experiment 1 (Text-Only Compaction Gate & Outcome Baseline):** Benchmark frozen-backbone head vs controlled LoRA against Always-Compact (`85.9%`) and Current-Policy on `dev` / `cal`, measuring balanced accuracy, Brier/ECE, and length curves.
3. **Experiment 2 (Dual-Head + `W_emb` Offline Teacher Distillation):** Cache `Qwen/Qwen3-VL-Embedding-8B` / `SemIf-OpenJev` outputs offline (never co-loading 8B+4B teachers during student training on unified RAM), evaluate retrieval `Recall@1/5` + Alignment/Uniformity alongside decision metrics, and run all **4** prefix/cache invariance tests.
4. **Experiment 3 (End-to-End Controller Eval):** **Compaction Gate Policy Explicit Spec:** When System-1 `ShouldCompact == YES`, Bob **must still invoke the structured summarizer LLM** to populate `CompactedTaskState` (`internal/fsm/compaction.go:785`), never silently falling back to `buildDeterministicCompactedState`. End-to-end latency savings come from skipping unnecessary summary calls (`NO`), and total latency must measure `gate + invoked summarizer`.

### Sidecar -> Phase B (`llama-server` / `lemond` / `vllm`) Deployment
- **Measured Serving Gap (`eval/BOB_S1.md` §6.2):** Stock `llama-server`, `lemond`, and `Ollama` do not currently support multimodal `/embeddings` (`mmproj` hidden-state extraction) or `--pooling none` `PointerHead` scoring over `/v1/embeddings`.
- **Interim Sidecar ([`bob-s1-inference-sidecar_20261118`](../bob-s1-inference-sidecar_20261118/spec.md)):** Temporary OpenAI-compatible Python inference sidecar under `services/bob_s1/` wrapping `DualHeadDecisionEmbedder` for multimodal embeddings (`POST /v1/embeddings`), `PointerHead` decisions (`POST /v1/chat/completions` with `bob-s1.decision.v1`), and single-pass joint output.
- **Phase B (Native Inference Server Patches — Out of Sidecar Scope):** Patch `llama-server` / `lemond` / `vllm` to natively execute `W_emb` + `PointerHead` + `mmproj` and clone `(kv, recurrent_state, conv_state)` from the shared prefix while honoring the frozen OpenAI-compatible wire contract and conformance fixtures established by the sidecar track.
