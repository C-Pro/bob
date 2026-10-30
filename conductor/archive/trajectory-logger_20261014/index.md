# Track Index: Trajectory Logger (Phase 5)

## Summary
- **Track ID:** `trajectory-logger_20261014`
- **Type:** Feature
- **Status:** In Progress
- **Created:** 2026-10-14
- **Depends on:** `core-engine-decoupling_20261006`
- **Description:** Implement Phase 5 Trajectory Logger (`internal/trajectory`): capture execution rollouts $\tau = (x, \Phi_{\text{task}}, z, a, o, y, R_{\text{eval}})$, 2-stage SQLite staging with revisioned feedback mutation, durable FSM/engine hooks with pre-truncation capture and recovery reconciliation, asynchronous S3/MinIO export worker, and user feedback commands (`/good`, `/bad`).

## Key Artifacts
- **Specification:** [spec.md](spec.md)
- **Implementation Plan:** [plan.md](plan.md)
- **Metadata:** [metadata.json](metadata.json)

## Key Decisions
- **String Identity Throughout:** Use string types for all identifiers (`trajectory_id`, `run_id`, `frontend_id`, `session_id`, `scope_id`, `actor_id`, `execution_kind`) to support heterogeneous frontends (Townhall, DMs, CLI, scheduled tasks).
- **Immutable Events & Versioned Projections:** Execution steps and user feedback events are preserved as immutable records; materialized payloads and composite scalar rewards $R_{\text{eval}}$ are versioned projections (`reward_version = "v1"`).
- **Revisioned Staging & Race Elimination:** Staging table tracks monotonic `revision` and `exported_revision`. User feedback increments `revision` and updates `eligible_at`. Worker uploads to `s3://<bucket>/<prefix>/YYYY/MM/DD/<trajectory_id>/r<revision>.json` with stable bytes and acknowledges only the uploaded revision.
- **Completion-Driven Delay Window:** The feedback collection delay (default 15 minutes) begins at `completed_at`, never at creation time. Trajectories with explicit feedback can flush after a short quiescence or immediate qualification.
- **Pre-Truncation FSM & Engine Recording:** Tool observations and reasoning traces ($z$) are captured before `EncodeStoredToolResult` truncates tool outputs for context budgets.
- **Crash Recovery Reconciliation:** Startup reconciles runs that terminated during execution or crash recovery before retention policies prune underlying rows.
- **Privacy By Default:** Trajectory collection adheres to privacy boundaries; export configuration flags distinguish public Townhall from private/unapproved scopes.

## Scope & Non-Goals
- **In Scope:**
  - `internal/trajectory` domain models, SQLite storage, reward engine, execution recorder, and S3 exporter.
  - Integration with `agent.Engine` and `fsm.Engine`.
  - Slash commands `/good` and `/bad` (with optional note and target).
  - Background exporter worker with exponential backoff and graceful shutdown flush.
  - Comprehensive unit and integration test suites.
- **Out of Scope:**
  - Online reinforcement learning / fine-tuning pipelines (Phase 8 PESO/FST).
  - Web UI for inspecting trajectory datasets.

## Progress Log
- **2026-10-14:** Track initialized in pair-programming session with Codex Astra. Completed architectural review on identity generalization, revisioned export semantics, reward heuristics, pre-truncation FSM hooks, and crash recovery reconciliation. Spec and plan drafted.
