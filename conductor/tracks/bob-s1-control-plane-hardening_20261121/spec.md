# Specification: System-1 Control-Plane Hardening, Train-Only Calibration & Runtime Optimization

## 1. Objective & Zero-Data-Snooping Invariant

Address the control-plane, dataset/calibration, and sidecar latency findings identified in [`eval/results/system1_heldout_100_report.md`](../../../eval/results/system1_heldout_100_report.md) across six rigorous phases while strictly enforcing **train-only data isolation**:

- **Zero Held-Out Leakage Invariant:** No fine-tuning, threshold selection, calibration fitting, prompt/rule tuning, or example/fixture mining may touch `eval/tasks/heldout/*` or `eval/results/heldout/*`. All dataset builders, mining scripts, calibration folds, and acceptance tests must operate exclusively on certified training manifests (`eval/tasks/train/train_pool.json`) and training rollouts (`eval/results/train/*`), using an explicit train allowlist certificate that succeeds even when held-out directories are absent.

## 2. Six-Phase Architecture

### Phase 0: Enforceable Data Provenance & Certified Train-Only Partitioning
- Generate a certified train-only allowlist and deterministic 4-fold group-disjoint partition (`train_fit`, `train_dev`, `train_calibration`, `train_acceptance`) over the clean training pool (`eval/tasks/train/train_pool.json`), grouping transitively by task ID, normalized instruction hash, gold target, and repo/base commit.
- Update [`eval/harness/build_jev_dataset.py`](../../../eval/harness/build_jev_dataset.py) and [`eval/harness/build_jev_generic_and_multimodal.py`](../../../eval/harness/build_jev_generic_and_multimodal.py) to enforce the certified train allowlist and emit full row-level lineage metadata (`source_task_id`, `group_id`, `partition`, `rollout_sha256`, `renderer_version`, `label_policy`) without requiring access to `eval/tasks/heldout/*` during dataset builds.

### Phase 1: Go Control-Plane & FSM Hardening (`internal/fsm`)
- **Evidence-Bound Loop Guard ([`internal/fsm/loop_guard.go`](../../../internal/fsm/loop_guard.go), [`internal/fsm/tool_loop.go`](../../../internal/fsm/tool_loop.go)):**
  - Replace the global `hasPriorStrike` boolean with structured, candidate-scoped stall evidence requiring executed step outcomes and decoded observation fingerprints.
  - Prevent sidecar-only `force_synthesis` or `intercept` predictions without qualifying executed stall evidence from terminating runs or blocking unexecuted commands; convert unsupported sidecar warnings into non-destructive advisories while allowing pending authorized tool calls to execute.
  - Record an explicit intercepted/blocked disposition so blocked calls are never persisted as `COMPLETED` exit-0 executions.
  - Scope interception to offending candidate indices in mixed tool batches and reset repetition counts across state-mutating edits.
- **Stage-2 Summarizer Resilience ([`internal/fsm/models.go`](../../../internal/fsm/models.go), [`internal/fsm/compaction.go`](../../../internal/fsm/compaction.go)):**
  - Normalize string elements in `verified_findings` when deterministic provenance exists in eligible executed steps or prior compacted state (never fabricating step IDs), and perform at most one bounded repair retry sharing the parent context deadline.
- **Independent Compaction Probe Cooldown & Hard-Byte Backstop:**
  - Track `last_probe_iter` independently of `last_compacted_iter` so skipped/failed probes do not re-probe on every subsequent iteration, while keeping the hard-byte backstop above repeated-failure suppression.

### Phase 2: Event-Triggered Probing & Truncation-Aware State Rendering (`internal/fsm`, `services/bob_s1`)
- **Cross-Compaction Progress Ledger & Event-Triggered `loop_guard`:**
  - Track lightweight progress signals (inspected targets, observation hashes, artifact edits, failure signatures) and gate semantic `loop_guard` sidecar calls behind train-validated stall triggers, cooldowns, exact-state caching, and periodic audit cadence.
- **Section-Aware Head/Tail Renderer:**
  - Upgrade [`internal/fsm/systemone_state.go`](../../../internal/fsm/systemone_state.go) and Python/sidecar preparation so objective, candidate tool calls, recent executed observations, strike evidence, and compact progress history each have reserved budgets without mid-JSON slicing or dropping recent evidence before the sidecar's 4,096-token window.
  - Record structured truncation telemetry (`retained_step_range`, `omitted_sections`, `essential_evidence_present`).

### Phase 3: Train-Only Dataset Rebuilding & Action-Policy Calibration (`eval/harness`, `services/bob_s1`)
- Mine interval-10 compaction decision states (including true `skip` states where derivations are mid-flight or recent compaction suffices) and contrastive semantic-stall vs. productive-progression loop states strictly from `train_pool` rollouts.
- Separate probability calibration from cost-sensitive action policies in [`services/bob_s1/bob_s1/calibration.py`](../../../services/bob_s1/bob_s1/calibration.py) (`compact` / `skip` / `defer_to_system2` operating thresholds and calibrated multiclass `loop_guard` thresholds), fitted exclusively on `train_calibration` and bound to the full runtime configuration signature.

### Phase 4: Sidecar & Runtime Performance (`services/bob_s1`, `internal/config`)
- Eliminate duplicate tokenization in [`services/bob_s1/bob_s1/worker.py`](../../../services/bob_s1/bob_s1/worker.py), add branch-safe prefix encoding reuse and stage-level latency breakdown in `/healthz` and decision responses, and default to explicit CPU/APU-safe device ownership (`bf16` + `sdpa` + `merge_lora` + pinned physical cores).

### Phase 5: Train-Pool Acceptance Verification & Blind Held-Out Re-Evaluation
- Verify all unit, contract, and train-acceptance gates (`make check`, Python dataset/model/sidecar pytest suites, and train-split evaluation).
- Execute one locked blind re-evaluation on `eval/tasks/heldout/heldout_100.json` after all code, weights, calibration profiles, and thresholds are frozen.
