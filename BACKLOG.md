# Bob Backlog

Tracked follow-ups, technical debt, and sidequest items discovered during implementation, debugging, and benchmark analysis. Keep [ROADMAP.md](ROADMAP.md) for the high-level architectural phases and use this backlog for concrete queued tasks not yet promoted to an active track in [conductor/index.md](conductor/index.md).

## Guidelines

- **Self-Contained & Mnemonic-Tagged:** Always pair numbered identifiers (phases, tracks, gaps, backlog IDs) with a short mnemonic and either define the item inline here or link directly to its canonical definition so entries remain unambiguous in fresh context.
- **Sidequests:** When you discover an out-of-scope bug, UX friction, performance opportunity, or cleanup item while working on another task, log it here with reproduction context and acceptance criteria instead of derailing the current task.
- **Data Isolation:** Any item touching `bob-s1` training, mining, calibration, or threshold selection must operate strictly on `eval/tasks/train/train_pool.json` and certified train rollouts ([`eval/harness/train_provenance.py`](eval/harness/train_provenance.py)), with zero reads from `eval/tasks/heldout/*` or `eval/results/heldout/*`.

---

## High Priority

### `[B-001]` Re-train & Re-calibrate `bob-s1` for Compaction Selectivity and Semantic Stall Loops (`s1-retraining`)
- **Problem Statement:** In the hardened 100-task held-out evaluation ([hardened evaluation report, *Multi-compaction semantic stall loops*](eval/results/system1_hardened_heldout_100_report.md#L81-L83)), nine long tasks (`ic_ctf_070` plus eight SWE-bench Lite tasks including `swe_django__django-12470`, `13033`, `13265`) still hit the 3,600 s wall-clock ceiling due to semantic repetition across compactions. While the Go/Python control plane has already been hardened in [the control-plane hardening track](conductor/tracks/bob-s1-control-plane-hardening_20261121/plan.md), the served checkpoint (`eval/jev/checkpoints/bob-s1-0.8b/model.pt`) still uses the pre-hardening weights that rarely fire `skip` on compaction or catch non-identical semantic loops.
- **Available Upstream Pipeline (Hardening Track Phases Already Implemented):**
  - Hardening Phase 0 (*Train-Only Provenance Certificate & 4-Fold Partitioning* — [`eval/harness/train_provenance.py`](eval/harness/train_provenance.py)): builds a certified allowlist (`train_fit`, `train_dev`, `train_calibration`, `train_acceptance`) strictly from `eval/tasks/train/train_pool.json`.
  - Hardening Phase 2 (*Truncation-Aware Sectioned State Rendering* — [`eval/harness/build_jev_dataset.py`](eval/harness/build_jev_dataset.py)): matches `go-probe-sanitized-v2` head/tail rendering so training inputs align with runtime Go probes.
  - Hardening Phase 3 (*Selectivity & Contrastive Stall Mining + Tri-Zone Calibration* — [`eval/harness/build_jev_dataset.py`](eval/harness/build_jev_dataset.py), [`eval/harness/build_jev_generic_and_multimodal.py`](eval/harness/build_jev_generic_and_multimodal.py), [`services/bob_s1/contracts/calibration_profiles.json`](services/bob_s1/contracts/calibration_profiles.json)): mines interval-10 `skip` compaction probes and contrastive semantic-stall vs. productive-progression loop states strictly from `train_pool` rollouts.
- **Concrete Deliverables:**
  1. Rebuild the Jev training, calibration, and acceptance datasets strictly from certified `train_pool` trajectories via [`eval/harness/train_provenance.py`](eval/harness/train_provenance.py).
  2. Re-train and export updated `bob-s1` weights/LoRA checkpoint under `eval/jev/` and re-fit the `bob-s1-0.8b/compaction/cal_v2_4k` and `loop_guard` calibration profiles exclusively on the `train_calibration` fold.
  3. Verify `skip` precision/recall, semantic-stall loop detection vs. progressing counterexamples, and end-to-end resolution/latency on `train_acceptance` before running a locked held-out re-evaluation.
