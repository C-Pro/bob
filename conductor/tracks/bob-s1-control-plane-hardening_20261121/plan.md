# Implementation Plan: System-1 Control-Plane Hardening, Train-Only Calibration & Runtime Optimization

## Phase 0: Enforceable Data Provenance & Certified Train-Only Partitioning (`eval/harness`)
- [x] Task: Build deterministic train-only allowlist & 4-fold group-disjoint partition (`train_fit`, `train_dev`, `train_calibration`, `train_acceptance`) over `eval/tasks/train/train_pool.json` with transitive grouping and row-level lineage metadata.
- [x] Task: Update `eval/harness/build_jev_dataset.py` and `eval/harness/build_jev_generic_and_multimodal.py` (plus unit tests) to enforce the certified train allowlist without requiring `eval/tasks/heldout/*` access at build time.

## Phase 1: Go Control-Plane & FSM Hardening (`internal/fsm`)
- [x] Task: Replace global `hasPriorStrike` in `internal/fsm/loop_guard.go` and `internal/fsm/tool_loop.go` with candidate-scoped, execution-backed stall evidence, non-destructive sidecar advisories when executed evidence is absent, explicit blocked step disposition, mixed-batch scoping, and state-change repetition reset.
- [x] Task: Harden Stage-2 summarizer parsing in `internal/fsm/models.go` and `internal/fsm/compaction.go` (provenance-checked string-to-`VerifiedFinding` normalization, single shared-deadline repair retry, independent probe cooldown tracking, and hard-byte backstop precedence) with comprehensive unit tests.

## Phase 2: Event-Triggered Probing & Truncation-Aware State Rendering (`internal/fsm`, `services/bob_s1`)
- [x] Task: Implement progress ledger, event-triggered `loop_guard` invocation gating, cooldowns, and exact-state decision cache in `internal/fsm`.
- [x] Task: Implement section-aware head/tail state rendering (`internal/fsm/systemone_state.go`) with reserved candidate/recent-step budgets, structured truncation telemetry, and Go/Python parity tests.

## Phase 3: Train-Only Dataset Rebuilding & Action-Policy Calibration (`eval/harness`, `services/bob_s1`)
- [x] Task: Mine interval-10 compaction selectivity states and contrastive semantic-stall vs. productive-progression loop states strictly from `train_pool` rollouts.
- [x] Task: Extend `services/bob_s1/bob_s1/calibration.py` and `contracts/calibration_profiles.json` with cost-sensitive compaction action thresholds (`compact`/`skip`/`defer`) and calibrated `loop_guard` thresholds fitted exclusively on `train_calibration` and bound to runtime config metadata.

## Phase 4: Sidecar & Runtime Performance (`services/bob_s1`, `internal/config`)
- [x] Task: Eliminate duplicate tokenization in `services/bob_s1/bob_s1/worker.py`, add bounded prefix cache & worker progress/latency telemetry in `/healthz`, and set APU-safe CPU default optimization flags.

## Phase 5: Train-Acceptance Gates, Repository Verification & Blind Held-Out Re-Evaluation
- [x] Task: Run full verification gates (`make check`, `eval/harness` pytest suite, `eval/jev` pytest suite, `services/bob_s1` pytest suite) and validate on `train_acceptance` / smoke suite.
- [x] Task: Run locked blind re-evaluation on `eval/tasks/heldout/heldout_100.json` and update `eval/results/system1_heldout_100_report.md`.
