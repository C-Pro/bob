# Implementation Plan: Phase 5 Trajectory Logger

## Phase 0: Baseline & Dependency Verification
- [x] Task: Document and verify baseline test suite
    - [x] Run `make check` to ensure all tests, linting, semgrep, and osv-scanner pass cleanly with 0 errors
    - [x] Inspect existing `internal/store`, `internal/objectstore`, and `internal/agent/engine.go` hooks

## Phase 1: SQLite Storage, Outbox & Migrations
- [x] Task: Storage Schema & Migrations
    - [x] Define canonical domain models in `internal/trajectory/models.go` (`Event`, `Step`, `EvalBreakdown`, `Trajectory`, `FeedbackRequest`)
    - [x] Add `agent_trajectory_events`, `agent_trajectories`, and `agent_trajectory_feedback` tables in `internal/store/schema.tmpl` and `migrate.tmpl`
    - [x] Bump store schema version in `internal/store/version.go`
    - [x] Implement `internal/trajectory/schema.go` with standalone database bootstrapping helper for tests and decoupled stores
- [x] Task: Store Implementation
    - [x] Implement `Store` interface in `internal/trajectory/store.go`
    - [x] Implement `RecordEvent(ctx, event)` and `RecordEventTx(ctx, tx, event)` to append pre-truncation attempts to `agent_trajectory_events`
    - [x] Implement `InsertStaged(ctx, traj, delay)` with initial revision = 1, status = 'staged', and calculated `eligible_at`
    - [x] Implement `UpdateFeedback(ctx, req)` with `MessageID` replay deduplication, actor-session validation, superseding rating updates, revision bump, and status reset to `'staged'`
    - [x] Implement `ListReadyForExport(ctx, now, limit)` filtering staged rows where `eligible_at <= now`, `export_approved = 1`, and privacy constraints are met
    - [x] Implement `MarkExported(ctx, id, revision, exportedAt)` with revision-safe conditional status updates
    - [x] Implement `PruneExported(ctx, olderThan)` retention pruner
    - [x] Implement `ReconcileTerminalRuns(ctx)` to materialize trajectories from durable events for all terminal states (`completed`, `failed`, `terminated`)
    - [x] Write unit tests in `internal/trajectory/store_test.go` covering outbox atomic transactions, phase discriminator collision avoidance, idempotent finalization, feedback & export preceding fallback finalization (verifying feedback survival and revision bump), concurrent feedback vs export, revision increments, late feedback after export, post-pruning ErrTrajectoryExpired, and replay deduplication

## Phase 2: Reward Engine ($R_{\text{eval}}$)
- [x] Task: Implement Reward Formulation
    - [x] Create `internal/trajectory/reward.go` with `CalculateReward(b EvalBreakdown) float64`
    - [x] Handle 0-tool conversational completion baseline (+0.20), 0-tool error baseline (0.00), 100% tool success (+0.40), user feedback weight (+0.60), budget exhaustion penalty (-0.30), and hard failure penalty (-0.50)
    - [x] Bound all returns in $[-1.0, 1.0]$ via `clamp`
    - [x] Set `reward_version = "v1"`
- [x] Task: Unit Testing Reward Calculation
    - [x] Write comprehensive table-driven tests in `internal/trajectory/reward_test.go` covering all 12 acceptance scenarios in the spec matrix

## Phase 3: FSM & Engine Recorder Hooks
- [x] Task: Reasoning Content Preservation
    - [x] Ensure `internal/llm/client.go` captures provider-returned `ReasoningContent` on assistant tool call messages
- [x] Task: Pre-Truncation Atomic Hook
    - [x] Implement `internal/trajectory/recorder.go`
    - [x] Extend `internal/fsm/executor.go` and `internal/fsm/observer.go` to invoke `RecordEventTx` within the step transaction, capturing full tool observations, raw stdout/stderr, and exit codes prior to `EncodeStoredToolResult` compression
- [x] Task: Engine Integration & Run Finalizers
    - [x] Capture `turnStartTime = time.Now()`, generate `trajectory_id`, and record `turn_start` event at entry of `agent.Engine.Run` in `internal/agent/engine.go`
    - [x] Attach `WithRunFinalizer` on all FSM runs (fresh, resumed, and recovered) to materialize and stage `Trajectory` upon termination
    - [x] Wire `ReconcileTerminalRuns` into startup recovery after `fsm.Engine.Recover`

## Phase 4: Async S3 Exporter Worker
- [x] Task: Configuration Keys
    - [x] Add `TRAJECTORY_ENABLED`, `TRAJECTORY_S3_BUCKET`, `TRAJECTORY_S3_PREFIX`, `TRAJECTORY_FLUSH_DELAY`, `TRAJECTORY_ALLOW_PRIVATE` in `internal/config/config.go`
- [x] Task: Background Exporter Implementation
    - [x] Create `internal/trajectory/exporter.go` with background polling loop and exponential backoff
    - [x] Partition object keys as `s3://<bucket>/<prefix>/YYYY/MM/DD/<trajectory_id>/r<revision>.json` using stable UTC `created_at` dates
    - [x] Support intermediate revision coalescing (uploading latest revision $r$)
    - [x] Upload using `internal/objectstore.Client.Put` with stable bytes per revision
    - [x] Call `MarkExported` to acknowledge only the specific revision uploaded
    - [x] Implement `FlushOnce(ctx)` and graceful shutdown flushing
- [x] Task: Unit & Mock Testing
    - [x] Write `internal/trajectory/exporter_test.go` with mock SigV4 objectstore server testing partition structure, payload bytes, privacy predicates, revision acknowledgment crash recovery, and retry logic

## Phase 5: Feedback Slash Commands & Service Lifecycle
- [x] Task: Commands Implementation
    - [x] Implement `/good [note]`, `/bad [note]`, and `/rate <+1|-1|0> [note]` in `internal/commands/feedback.go`
    - [x] Support optional explicit target `<trajectory_id>`; fallback to latest interactive turn by actor in session
    - [x] Enforce authorization and handle repeated ratings via superseding updates with `MessageID` deduplication
    - [x] Register commands in `internal/gateway/gateway.go`
- [x] Task: Daemon Lifecycle Wiring
    - [x] Wire `trajectory.Store` and `trajectory.Exporter` into `cmd/agent/main.go`
    - [x] Hook into graceful shutdown context to run bounded final flush

## Phase 6: End-to-End Integration & Verification
- [x] Task: Integration Test
    - [x] Add integration test in `cmd/agent/agent_integration_test.go` running a full tool-assisted turn through `agent.Engine`, triggering `/good`, running `Exporter.FlushOnce`, and asserting JSON payload and $R_{\text{eval}} > 0.8$ in mock object store
- [x] Task: Full Repository Verification
    - [x] Run `make check` (lint-go, test-go, semgrep, osv-scanner) to verify 0 errors
