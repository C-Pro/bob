# Implementation Plan: Comprehensive Multimodal Dataset, Zero-Shot Retention & Iterative Tuning Loop (`Jev`)

## Phase 1: Frozen `qwen3_5_vision` Integration & Multimodal Unit/Invariance Tests (`eval/jev/`)
- [x] Task 1.1: Extend `eval/jev/model.py` for multimodal states (`Qwen3_5ForConditionalGeneration`)
    - [x] Load `Qwen3_5ForConditionalGeneration` (retaining frozen `model.visual` alongside `PeftModel` on `model.language_model`) when `include_vision=True` or images are passed.
    - [x] Add deterministic image preprocessing helper (converting PIL images / tensors into `pixel_values`, `image_grid_thw`, and `<|vision_start|><|image_pad|>...<|vision_end|>` token spans inside `[State 1..S]`) without external network calls.
    - [x] Update `encode_prefix`, `forward_prompt`, `equivariant_scores`, `state_hidden`, and `embed_state` to accept optional `images` while keeping text-only calls backward-compatible and preserving `save_checkpoint` / `load_checkpoint` (`model.visual` stays frozen and excluded from adapter checkpoints).
    - [x] Extend `synthetic_model()` in `eval/jev/model.py` with a tiny synthetic vision encoder stub so fast CPU unit tests verify multimodal codepaths without loading the 0.8B checkpoint.
- [x] Task 1.2: Add multimodal unit & real-checkpoint smoke tests in `eval/jev/test_jev_model.py`
    - [x] Test multimodal `<|state_end|>` embedding causal invariance to suffix questions/options (`max_diff < 1e-5`).
    - [x] Test multimodal option-permutation equivariance and `(kv, recurrent_state, conv_state)` prefix cache cloning across multiple visual questions.
    - [x] Test image sensitivity: changing the image in `[State]` changes both `embed_state` (`128d/256d/384d`) and visual decision logits while `model.visual` parameters have `requires_grad == False`.
    - [x] Run `pytest` on `eval/jev/test_jev_model.py` + `make check`.

---

## Phase 2: Generic Out-of-Domain Decision, Positive `loop_guard`, Bilingual & Multimodal Dataset Builder (`eval/harness/`)
- [x] Task 2.1: Build generic out-of-domain decision dataset generator (`eval/harness/build_jev_generic_and_multimodal.py`)
    - [x] Generate/extract `generic_decisions_{train,cal,dev}.jsonl` with group-isolated splits (`0%` overlap):
        - Programmatic policy/rule/date/trade-off questions + `"none_of_the_above"` / missing-evidence abstention minimal pairs.
        - General EN + RU NLI, boolean QA, and topic/urgency classification from local cached corpora.
        - Candidate future Bob control questions (`confirmation_gate`, `prompt_injection`, `memory_relevance`, `verification_readiness`).
- [x] Task 2.2: Build balanced 3-class `loop_guard` and bilingual EN+RU retrieval splits
    - [x] Augment `loop_guard_{train,cal,dev}.jsonl` with group-isolated positive `"intercept"` and `"force_synthesis"` windows paired with constructive-failure `"none"` hard negatives.
    - [x] Generate `emb_pairs_bilingual_{train,cal,dev}.jsonl` (EN, RU, and cross-lingual EN<->RU episode/chat retrieval triplets).
- [x] Task 2.3: Build deterministic multimodal image+text dataset (`multimodal_emb_pairs` & `multimodal_decisions`)
    - [x] Render grouped `train / cal / dev` visual assets (terminal/log screenshots, metric charts, UI dialogs/status cards, architecture diagrams) with exact ground-truth labels and hard visual negatives.
    - [x] Emit `multimodal_emb_pairs_{train,cal,dev}.jsonl` and `multimodal_decisions_{train,cal,dev}.jsonl`.
- [x] Task 2.4: Add dataset integrity & zero-leakage tests (`eval/harness/test_build_jev_generic_and_multimodal.py`)
    - [x] Verify `0%` group overlap across `train / cal / dev` and `heldout_100.json`, image file readability, class balance, and schema compatibility.
    - [x] Run `pytest` and generate dataset artifacts + summary stats in `eval/results/jev_dataset/`.

---

## Phase 3: Multi-Suite Evaluation & Training Loop Extensions (`eval/jev/`)
- [x] Task 3.1: Extend `eval/jev/eval_experiment.py` for generic, 3-class loop, bilingual, and multimodal benchmarks
    - [x] Add `generic_decisions` evaluation (overall & per-family accuracy, raw ROC-AUC, Brier, ECE, and mean KL divergence against `zero_shot_kev`).
    - [x] Add 3-class `loop_guard` metrics (macro-F1, per-class recall for `none` / `intercept` / `force_synthesis`, and `none` FPR).
    - [x] Add `emb_pairs_bilingual`, `multimodal_emb_pairs`, and `multimodal_decisions` evaluation to `--cal-cv`.
    - [x] Run baseline audit on `cal`: benchmark `zero_shot_kev` vs `exp_b7_final` across all new suites to quantify exact zero-shot retention and multimodal baseline scores before any new training.
- [x] Task 3.2: Extend `eval/jev/train_experiment.py` for multi-domain replay & multimodal batches
    - [x] Add configurable task-mixing schedule supporting `generic_decisions` replay + pristine-Kev KL anchoring, 3-class `loop_guard` class-balanced sampling, `emb_pairs_bilingual`, `multimodal_emb_pairs`, and `multimodal_decisions`.
    - [x] Add unit/smoke tests in `eval/jev/test_jev_model.py` and run `pytest` + `make check`.

---

## Phase 4: Small-Scale Training / Evaluation / Improvement Loop (Strictly on `train` + `cal` CV)
- [x] Task 4.1: Iteration 1 — Baseline retention & multimodal alignment pilot (`~30–40` updates on ROCm)
    - [x] Train small-scale candidate from `exp_b7_final` (or `zero_shot_kev` warm-start comparison) with generic replay + 3-class `loop_guard` + bilingual/multimodal batches (`--emb-batch-size 3`, `--save-every 5`).
    - [x] Evaluate on `cal` 5-fold CV against all 4 Go/No-Go gates (`generic_decisions`, `in-domain decisions + 3-class loop_guard`, `text + bilingual embeddings`, `multimodal embeddings + visual decisions`).
- [x] Task 4.2: Iteration 2..N — Targeted diagnostics & refinement loop (consult `gpt-6-astra` via `agent-pair` as needed)
    - [x] Adjust sampling ratios, per-family KL weights, or Stage-B `W_emb` multimodal projection refit to resolve any gate regressions without touching `dev`.
    - [x] Repeat bounded small-scale runs until all Go/No-Go gates pass simultaneously on `cal`.
- [x] Task 4.3: Readiness Review for Full-Scale Training
    - [x] Present complete `cal` 5-fold CV comparison table (`zero_shot_kev` vs `exp_b7_final` vs winning small-scale candidate) across all text, generic, bilingual, and multimodal suites, and get user sign-off on the final full-scale training recipe.

---

## Phase 5: 4k Full Training, GGUF Release (`bob-s1-0.8b`) & Track Wrap-Up
- [x] Task 5.1: Full-context `4096`-token Stage 1 (`exp_b_v2_4k` → `u0048`) + Stage 2 (`exp_b_v2_4k_emb_onlymm`) training, `cal` gate sign-off (`cal_cv_v2_4k_u48_emb`), and locked V2 `dev` evaluation (`dev_final_v2_4k`)
- [x] Task 5.2: Export `bob-s1-0.8b` GGUF artifacts (`bob-s1-0.8b-bf16.gguf`, `bob-s1-0.8b-q4_k_m.gguf`, `bob-s1-0.8b-mmproj-bf16.gguf`, `bob-s1-0.8b-heads.pt`, `bob-s1-0.8b-preprocess.json`), add symlink alias `eval/jev/checkpoints/bob-s1-0.8b -> exp_b_v2_4k_emb_onlymm`, and formalize + unit-test the export pipeline in `eval/jev/export_gguf.py` + `eval/jev/test_export_gguf.py`
- [x] Task 5.3: Publish Model Card & Reproducible Technical Report ([`eval/BOB_S1.md`](../../../eval/BOB_S1.md)), critique via `agent-pair` (`gpt-6-astra`), and link from `README.md`, `eval/README.md`, `eval/DATASET.md`, and `conductor/index.md`
