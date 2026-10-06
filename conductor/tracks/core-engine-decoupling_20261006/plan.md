# Implementation Plan: Core Agent Engine Decoupling

## Phase 0: Baseline & Dependency Characterization
- [ ] Task: Document and verify baseline test suite
    - [ ] Run `make check` to ensure all tests, linting, semgrep, and osv-scanner pass with 0 errors
    - [ ] Characterize existing tool loop fallback, session restoration, and crash recovery paths in `internal/gateway` and `internal/fsm`

## Phase 1: Independent Subsystem Relocations
- [ ] Task: Relocate `KnowledgeWorker` to `internal/knowledge`
    - [ ] Move `knowledge_worker.go` and `knowledge_worker_test.go` from `internal/gateway` to `internal/knowledge`
    - [ ] Update imports across `gateway` and tests; verify tests pass
- [ ] Task: Relocate `MemoryStoreProvider` to `internal/agentstore`
    - [ ] Create `internal/agentstore/provider.go` from `internal/gateway/store_provider.go`
    - [ ] Move `store_provider_test.go` to `internal/agentstore`
    - [ ] Provide transitional type aliases in `internal/gateway` to preserve backward compatibility during migration
    - [ ] Verify database paths, table schemas, and tests pass cleanly

## Phase 2: Neutral Contracts (`internal/agentapi`)
- [ ] Task: Define shared contracts package `internal/agentapi`
    - [ ] Create `internal/agentapi/session.go` (`SessionRef`, `Actor`, `RunDescriptor`, `ExecutionKind`)
    - [ ] Create `internal/agentapi/turn.go` (`Turn`, `Result`, `Completion`, `Action`)
    - [ ] Create `internal/agentapi/bindings.go` (`Frontend`, `Bindings`, `NotificationSink`)
    - [ ] Create `internal/agentapi/tools.go` (`Toolset`, `ToolDefinition`, `ToolCall`, `ToolResult`)
    - [ ] Create `internal/agentapi/attachments.go` (`Attachment`, `AttachmentHandler`, `AttachmentContent`)
    - [ ] Create `internal/agentapi/progress.go` (`ProgressObserver`, `ProgressEvent`, `StepProgress`)
- [ ] Task: Implement Besedka model conversion adapters
    - [ ] Write converters between `agentapi.Attachment` and `models.Attachment`
    - [ ] Add unit tests verifying serialization and round-trip conversion

## Phase 3: Execution-Scoped FSM Tool Bindings
- [ ] Task: Adapt `internal/fsm` to accept per-run tool bindings
    - [ ] Update `fsm.ToolLoopRequest` to accept per-run `agentapi.Toolset`
    - [ ] Update `fsm.StepExecutor` to dispatch through per-run `Toolset`
    - [ ] Enforce strict tool authority: nil or empty tool definitions execute 0 tools without falling back to global registry
    - [ ] Update FSM unit and integration tests; verify tests pass

## Phase 4: Durable Frontend Identity & Recovery
- [ ] Task: Add frontend descriptor persistence in FSM runs
    - [ ] Extend `fsm.FSMRun` schema and model to persist `FrontendID` and session metadata (with backward compatibility for existing runs)
    - [ ] Implement legacy Besedka recovery mapping for pre-existing runs without explicit `FrontendID`
    - [ ] Update crash recovery in `fsm.Engine` to rebind tools and dependencies via `Frontend.Bind`
    - [ ] Route recovered completion deliveries through `Frontend.Deliver`
    - [ ] Add unit tests for recovery with registered, unregistered, and legacy frontends

## Phase 5: Core Engine Extraction (`internal/agent`)
- [ ] Task: Implement `agent.Engine`
    - [ ] Create `internal/agent/engine.go` implementing `TurnRunner`
    - [ ] Port execution turn coordination and fallback volatile tool loop from `gateway.generateAndSendAgentReply`
    - [ ] Move per-session concurrency lock (`ChatLocker` → `SessionLocker`) into `internal/agent`
    - [ ] Write unit tests for `agent.Engine` using a test mock frontend (verifying execution without Besedka dependencies)

## Phase 6: Subsystem Extraction & Scheduled Invocation
- [ ] Task: Relocate and decouple slash command handlers
    - [ ] Create `internal/commands` with neutral `Request` and `Result` types
    - [ ] Extract `/memory` command handler; write unit tests
    - [ ] Extract `/skill` command handler; write unit tests
    - [ ] Extract `/schedule` command handler; write unit tests
    - [ ] Extract `/sandbox` command handler; write unit tests
- [ ] Task: Relocate `ScheduleInvoker` to `internal/agent/scheduled`
    - [ ] Decouple schedule execution from Besedka-specific types
    - [ ] Route scheduled turn execution through `agent.Engine` and `Frontend.Deliver`
    - [ ] Verify ephemeral sandbox cleanup and tool exclusions for scheduled runs

## Phase 7: Configuration Decoupling & Composition Root
- [ ] Task: Decouple configuration validation
    - [ ] Update `config.Config.Validate(requireAPIKey bool)`: make `BESEDKA_URL` optional
    - [ ] Add `config.Config.ValidateBesedka()` requiring `BESEDKA_URL` for daemon mode
    - [ ] Add tests for core-only and Besedka-specific configuration validation
- [ ] Task: Refactor Besedka Gateway into a pure frontend adapter
    - [ ] Refactor `internal/gateway/gateway.go` to implement `agentapi.Frontend`
    - [ ] Rewire `cmd/agent/main.go` composition root: instantiate `agent.Engine`, register Besedka frontend, start gateway
    - [ ] Remove obsolete transitional aliases and deprecated helper functions

## Phase 8: System Verification & Regression Testing
- [ ] Task: Full verification and linting
    - [ ] Run full test suite: `go test -v -covermode=atomic -race ./...`
    - [ ] Run `make check` (golangci-lint, semgrep, osv-scanner) ensuring 0 errors
    - [ ] Verify that no core packages (`internal/agent`, `internal/agentapi`, `internal/fsm`, `internal/tools`) import `internal/gateway`
