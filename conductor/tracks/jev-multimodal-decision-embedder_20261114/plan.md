# Implementation Plan: Phase 4 — Unified 0.8B Multimodal Decision & Embedding Model (`Jev`)

## Step 1: Boundary-Exact, Group-Split Dataset Extractor & Leakage Tests (`eval/harness/`)
- [x] Task: Ignore generated dataset/checkpoint artifacts in `.gitignore`
    - [x] Add `eval/results/**/*.jsonl`, `eval/results/jev_dataset/`, and `eval/jev/runs/` / model checkpoint binaries (`*.safetensors`, `*.pt`, `*.gguf`) to `.gitignore` while keeping Python source code in `eval/jev/` tracked.
- [x] Task: Build leak-free dataset extractor `eval/harness/build_jev_dataset.py`
    - [x] Read all `417` rollout JSONs in `eval/results/train/batch_train_pool_1794581821/`, deduplicating `events[]` and `compactions[]` snapshots.
    - [x] Enforce hard task-ID AND normalized-instruction deduplication / 0% overlap check against `eval/tasks/heldout/heldout_100.json`.
    - [x] Partition by task group into 3 stratified splits (`train` ~70%, `dev` ~15%, `cal` ~15%).
    - [x] Implement decision-boundary time-`t` state reconstruction matching `internal/fsm/compaction.go:488` (excluding current `state_json`, verdicts/reasons, future steps, terminal status, and final grades) and compute exact `Qwen/Qwen3.5-0.8B` token counts + truncation flags.
    - [x] Emit clean labeled views with explicit provenance & quality flags:
      - `compaction`: `2,004` compact (`YES`) vs `318` rubric skips (`NO`), explicitly excluding/tagging the `10` `probe_error` skips (`compaction.go:775`), keeping `24` `recall_task_history` calls (across `7` tasks) strictly as auxiliary metadata, and saving natural prevalence + task-balanced weights. (Actual after quarantine + metadata dedup: `2,317` rows = `2,003` compact vs `314` skip; `415` clean + `2` heldout-overlap quarantined.)
      - `outcome_prefix`: `323` clean graded tasks (`120` passed, `203` failed), tagging `46` Bash oracle errors and `48` SWE missing-`report.json` infra failures (`runner.py:157`) as excluded `infra_error`/`unknown` (`implicit_signal` recorded as unavailable/zero). (Actual after quarantine: `321` clean graded tasks = `119` passed, `202` failed.)
      - `loop_guard`: trajectory windows with canonicalized + raw args, result/error deltas, and strike counts.
      - Contrastive text state pairs (same-task prefix pairs vs different-task hard negatives) for Step 2 embedding baseline/distillation.
- [x] Task: Step 1 Verification
    - [x] Create `eval/harness/test_build_jev_dataset.py` testing: future-data/post-boundary exclusion, nested JSON & event/snapshot deduplication, `probe_error` filtering, `recall_task_history` pagination/metadata tracking, SWE missing-`report.json` infra failure filtering, and zero task-group leakage across `train` / `dev` / `cal` / `heldout_100`.
    - [x] Run Python unit tests (`pytest` / `unittest`) and execute `build_jev_dataset.py` to generate `eval/results/jev_dataset/{train,dev,cal}.jsonl` and `stats.json`.

---

## Step 2: Small-Scale Local AMD GPU (`780M` ROCm) Architecture, Invariance Tests & Training Experiment (`eval/jev/`)
- [x] Task: Implement `eval/jev/model.py` & ROCm profiling (profiled inline on ROCm; standalone `eval/jev/profile_rocm.py` not retained)
    - [x] Support `Qwen/Qwen3.5-0.8B-Base` / `jaredpalmer/kev-0.8b` with **independent per-question rows** (NOT block-diagonal masking on DeltaNet!) and prefix `(kv, recurrent_state, conv_state)` cache cloning for fast inference.
    - [x] Implement `W_emb` (Matryoshka `[128, 256, 384]` with separate per-slice L2 normalization at `<|state_end|>`) + `PointerHead` (`q`/`k` matching `kev-0.8b` `head.pt` without `LayerNorm`, scaled dot-product, `noul`/`choice`/`score`, calibrated `T`).
    - [x] Profile ROCm 6.4 (`HSA_OVERRIDE_GFX_VERSION=11.0.2`) forward + backward pass (`batch=1`, `seq=512..1024`, gradient checkpointing, finite gradients, VRAM & DeltaNet kernel timing) before launching multi-step training.
- [x] Task: Create strict invariance & unit test suite `eval/jev/test_jev_model.py`
    - [x] Verify `<|state_end|>` embedding invariance when question/options change (`max_diff < 1e-5`).
    - [x] Verify Question A logits remain identical when Question B changes or moves across independent/cloned-cache paths.
    - [x] Verify cached `(kv, recurrent, conv)` vs uncached outputs match; right-padded `batch=8` benchmarked inline and intentionally kept at exact unpadded `batch=1` (`1038.0ms/item` vs `1204.5ms/item` with `0.0235` `bf16` padding drift).
    - [x] Verify Matryoshka per-slice L2 normalization and temperature/Brier/ECE helpers.
- [x] Task: Run controlled small-scale GPU experiments (`eval/jev/train_experiment.py` & `eval/jev/eval_experiment.py`)
    - [x] **Experiment A (Frozen-Backbone Baseline):** Train linear `PointerHead` + `W_emb` on frozen `Qwen/Qwen3.5-0.8B-Base` (and benchmark `kev-0.8b` adapter baseline) on GPU.
    - [x] **Experiment B (Controlled LoRA Intervention):** Fine-tune with task-balanced compaction + contrastive state InfoNCE (`LoRA` with `kl_frozen_reference`) and measure whether decision tuning degrades retrieval `Recall@1/5` or Alignment/Uniformity.
    - [x] **Calibration & Thresholding:** Fit temperatures and risk-vs-coverage abstention thresholds strictly on `cal.jsonl`; evaluate on untouched `dev.jsonl` against **Always-Compact** (`84.30%` on clean dev / `85.9%` overall) and **Current-Policy** baselines, reporting balanced accuracy, task-macro F1, per-class PR, Brier, NLL, ECE, token-length buckets, and single-pass GPU latency (`experiment_report.json` / `.md`).

---

## Step 3: Bob Configuration & Phase A `llama-server` Client (`internal/config`, `internal/llm`)
- [ ] Task: Add Phase 4 configuration fields to `internal/config/config.go` & `internal/config/file.go` + tests
- [ ] Task: Implement `SystemOneClient` (`internal/llm/systemone.go`) & multimodal `CreateEmbeddings` extension (`internal/llm/embedding.go`) + mock server tests
- [ ] Task: Run `make check`

---

## Step 4: Bob FSM Control-Plane Integration (`internal/fsm`, `internal/trajectory`)
- [ ] Task: Wire `SystemOneClient` into `CompactionDecider` (explicitly preserving structured summarizer invocation on confident `YES` so deterministic fallback is not silently triggered) and `loop_guard.go`, recording telemetry in `internal/trajectory`
- [ ] Task: Add/update unit tests and run `make check`

---

## Step 5: Multimodal Memory, Episode Classification & Re-Index (`internal/memory`)
- [ ] Task: Build separate EN+RU episode & multimodal image+text evaluation manifest; update `internal/memory/episode.go`, `internal/memory/store.go` (`AttachmentTypeImage`), and `internal/memory/regenerate.go` (`RegenerateAllVectors`) + tests
- [ ] Task: Run `make check`

---

## Step 6: Phase B `llama.cpp` Fork (`tools/server`) & End-to-End Controlled Evaluation
- [ ] Task: Implement forked `llama-server` unified endpoint cloning `(kv, recurrent, conv)` state from `<|state_end|>` and returning `{decisions, embedding}` in one pass
- [ ] Task: Run controlled end-to-end comparison on `eval/tasks/heldout/heldout_100.json` (identical executor model, budgets, and grader vs baseline controller) + `make check`
