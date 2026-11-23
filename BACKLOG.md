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

## Medium Priority

### `[B-002]` Sync Tracker Completion Markers (`tracker-sync`)
- **Problem Statement:** Several roadmap and conductor index entries lag behind completed implementation commits on `feat/bob-s1-go-integration`.
- **Specific Drift:**
  - [`conductor/index.md`](conductor/index.md#L76-L92): still marks `jev-multimodal-decision-embedder_20261114` (`Unified 0.8B Multimodal Decision & Embedding Model`) as `[~]` and `bob-s1-go-integration_20261119` (`Go Integration for bob-s1`) as `[ ]`, and omits the completed [`bob-s1-control-plane-hardening_20261121`](conductor/tracks/bob-s1-control-plane-hardening_20261121/plan.md) track.
  - [`ROADMAP.md`](ROADMAP.md#L18-L21): status overview table still marks Phase 1 (*Context Assembly, Episodic Index & Adaptive Task Compaction*) and Phase 4 (*Lightweight Local Control-Plane / Decision Model*) as `[NEXT]` despite shipped implementations.
- **Concrete Deliverables:** Update status badges and add missing track links across [`conductor/index.md`](conductor/index.md) and [`ROADMAP.md`](ROADMAP.md).

### `[B-003]` Isolate `swebench` Evaluation Scratch Artifacts from Repo Root (`eval-cwd-isolation`)
- **Problem Statement:** During Docker-validated SWE-bench grading, `evaluate_swe_patch_docker()` in [`eval/harness/runner.py`](eval/harness/runner.py#L148-L193) invokes `python -m swebench.harness.run_evaluation` with the repository root as its working directory, leaving untracked `bob.eval_<instance>_<ts>.json` summaries and a `logs/run_evaluation/` tree (~37 MB per 25-task batch) in the repo root even though pass/fail metrics are already saved into each batch's `eval/results/.../` JSONs.
- **Concrete Deliverables:** Execute `swebench.harness.run_evaluation` inside a dedicated gitignored output/scratch directory (or set `cwd` / `--report_dir` to a temp/batch subdirectory cleanly) and clean up/ignore intermediate report files so `git status` remains clean after benchmark runs.

### `[B-004]` Push Pending Assistant Entries Before WebSocket Send (`gateway-send-push-ordering`)
- **Problem Statement:** Outgoing assistant message delivery in [`deliverAgentReply`](internal/gateway/ui.go) (and previously in [`Gateway.Deliver`](internal/gateway/gateway.go) / [`Gateway.generateAndSendAgentReply`](internal/gateway/gateway.go)) sends each message over WebSocket via [`SendMessageWithAttachmentsAndUI`](internal/gateway/gateway.go) before calling `g.contextManager.Push`. If Besedka echoes the bot's own WebSocket frame before `Push` stores the pending (`Seq == 0`) assistant entry in [`chatcontext.RingBuffer`](internal/chatcontext/buffer.go), `ProcessMessage` cannot reconcile the self-echo against a pending entry via [`ReconcileAssistantEcho`](internal/chatcontext/buffer.go) and a duplicate entry is appended right before `Push` runs.
- **Concrete Deliverables:** Push the pending (`Seq == 0`) assistant entry into `g.contextManager` (or pre-register an in-flight reconciliation token) before transmitting over WebSocket, rollback/remove on WebSocket send failure, and add a race-reproduction test simulating an immediate concurrent WebSocket echo prior to `Push`.
