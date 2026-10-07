# Implementation Plan: Core Agent Engine Decoupling

## Phase 0: Baseline & Dependency Characterization
- [x] Task: Document and verify baseline test suite
    - [x] Run `make check` to ensure all tests, linting, semgrep, and osv-scanner pass with 0 errors
    - [x] Characterize existing tool loop fallback, session restoration, and crash recovery paths in `internal/gateway` and `internal/fsm`

## Phase 1: Independent Subsystem Relocations
- [x] Task: Relocate `KnowledgeWorker` to `internal/knowledge`
    - [x] Move `knowledge_worker.go` and `knowledge_worker_test.go` from `internal/gateway` to `internal/knowledge`
    - [x] Update imports across `gateway` and tests; verify tests pass
- [x] Task: Relocate `MemoryStoreProvider` to `internal/agentstore`
    - [x] Create `internal/agentstore/provider.go` from `internal/gateway/store_provider.go`
    - [x] Move `store_provider_test.go` to `internal/agentstore`
    - [x] Provide transitional type aliases in `internal/gateway` to preserve backward compatibility during migration
    - [x] Verify database paths, table schemas, and tests pass cleanly


## Phase 2: Neutral Contracts (`internal/agentapi`)
- [x] Task: Define shared contracts package `internal/agentapi`
    - [x] Create `internal/agentapi/session.go` (`SessionRef`, `Actor`, `RunDescriptor`, `ExecutionKind`)
    - [x] Create `internal/agentapi/turn.go` (`Turn`, `Result`, `Completion`, `Action`)
    - [x] Create `internal/agentapi/bindings.go` (`Frontend`, `Bindings`, `NotificationSink`)
    - [x] Create `internal/agentapi/tools.go` (`Toolset`, `ToolDefinition`, `ToolCall`, `ToolResult`)
    - [x] Create `internal/agentapi/attachments.go` (`Attachment`, `AttachmentHandler`, `AttachmentContent`)
    - [x] Create `internal/agentapi/progress.go` (`ProgressObserver`, `ProgressEvent`, `StepProgress`)
- [x] Task: Implement Besedka model conversion adapters
    - [x] Write converters between `agentapi.Attachment` and `models.Attachment`
    - [x] Add unit tests verifying serialization and round-trip conversion

## Phase 3: Execution-Scoped FSM Tool Bindings
- [x] Task: Adapt `internal/fsm` to accept per-run tool bindings
    - [x] Update `fsm.ToolLoopRequest` to accept per-run `agentapi.Toolset`
    - [x] Update `fsm.StepExecutor` to dispatch through per-run `Toolset`
    - [x] Enforce strict tool authority: nil or empty tool definitions execute 0 tools without falling back to global registry
    - [x] Update FSM unit and integration tests; verify tests pass

## Phase 4: Durable Frontend Identity & Recovery
- [x] Task: Add frontend descriptor persistence in FSM runs
    - [x] Extend `fsm.FSMRun` schema and model to persist `FrontendID` and session metadata (with backward compatibility for existing runs)
    - [x] Implement legacy Besedka recovery mapping for pre-existing runs without explicit `FrontendID`
    - [x] Update crash recovery in `fsm.Engine` to rebind tools and dependencies via `Frontend.Bind`
    - [x] Route recovered completion deliveries through `Frontend.Deliver`
    - [x] Add unit tests for recovery with registered, unregistered, and legacy frontends

## Phase 5: Core Engine Extraction (`internal/agent`)
- [x] Task: Implement `agent.Engine`
    - [x] Create `internal/agent/engine.go` implementing `TurnRunner`
    - [x] Port execution turn coordination and fallback volatile tool loop from `gateway.generateAndSendAgentReply`
    - [x] Move per-session concurrency lock (`ChatLocker` → `SessionLocker`) into `internal/agent`
    - [x] Write unit tests for `agent.Engine` using a test mock frontend (verifying execution without Besedka dependencies)

## Phase 6: Subsystem Extraction & Scheduled Invocation
- [x] Task: Relocate and decouple slash command handlers
    - [x] Create `internal/commands` with neutral `Request` and `Result` types
    - [x] Extract `/memory` command handler; write unit tests
    - [x] Extract `/skill` command handler; write unit tests
    - [x] Extract `/schedule` command handler; write unit tests
    - [x] Extract `/sandbox` command handler; write unit tests
- [x] Task: Relocate `ScheduleInvoker` to `internal/agent/scheduled`
    - [x] Decouple schedule execution from Besedka-specific types
    - [x] Route scheduled turn execution through `agent.Engine` and `Frontend.Deliver`
    - [x] Verify ephemeral sandbox cleanup and tool exclusions for scheduled runs

## Phase 7: Configuration Decoupling & Composition Root
- [x] Task: Decouple configuration validation
    - [x] Update `config.Config.Validate(requireAPIKey bool)`: make `BESEDKA_URL` optional
    - [x] Add `config.Config.ValidateBesedka()` requiring `BESEDKA_URL` for daemon mode
    - [x] Add tests for core-only and Besedka-specific configuration validation
- [x] Task: Refactor Besedka Gateway into a pure frontend adapter
    - [x] Refactor `internal/gateway/gateway.go` to implement `agentapi.Frontend`
    - [x] Rewire `cmd/agent/main.go` composition root: instantiate `agent.Engine`, register Besedka frontend, start gateway
    - [x] Remove obsolete transitional aliases and deprecated helper functions

## Phase 8: System Verification & Regression Testing
- [x] Task: Full verification and linting
    - [x] Run full test suite: `go test -v -covermode=atomic -race ./...`
    - [x] Run `make check` (golangci-lint, semgrep, osv-scanner) ensuring 0 errors
    - [x] Verify that no core packages (`internal/agent`, `internal/agentapi`, `internal/fsm`, `internal/tools`) import `internal/gateway`
