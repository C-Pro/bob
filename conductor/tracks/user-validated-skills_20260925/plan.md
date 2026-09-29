# Implementation Plan: User-Validated Skills and Structured Memory

## Phase 1: Configuration, Data Models, SQLite Schema & Relational Store (`internal/config`, `internal/knowledge`)
- [x] Task: Add Knowledge configuration to `internal/config/config.go`
    - [x] Add `KnowledgeMaxLoadedMemories` (default `8`) and `KnowledgeMaxLoadedMemoryBytes` (default `16384` / 16 KiB)
    - [x] Add `KnowledgeMaxLoadedSkills` (default `3`) and `KnowledgeMaxLoadedSkillBytes` (default `24576` / 24 KiB)
    - [x] Add `KnowledgeDefaultDiscoveryLimit` (default `5`) and `KnowledgeMaxDiscoveryLimit` (default `10`)
    - [x] Unit tests in `internal/config/config_test.go` verifying env parsing and boundary enforcement
- [x] Task: Define Knowledge data models, enums and helper utilities
    - [x] Create `internal/knowledge/models.go` with `KnowledgeItem`, `MemoryVersion`, `SkillVersion`, `Provenance`, `Tombstone`
    - [x] Define item statuses (`active`, `disabled`, `archived`, `forgotten`) and version statuses (`proposed`, `approved`, `rejected`, `superseded`, `expired`)
    - [x] Define memory types (`preference`, `fact`, `decision`, `ongoing_task`) and indexing statuses (`pending`, `ready`, `error`)
    - [x] Implement deterministic ID generation (`mem_<random>`, `skill_<random>`, `<item_id>@<revision>`) and version reference parsing
    - [x] Implement SHA-256 content hashing for tamper-proof provenance recording
    - [x] Write unit tests in `internal/knowledge/models_test.go`
- [x] Task: Implement SQLite schema migrations for knowledge storage
    - [x] Add knowledge and scheduler DDL to `internal/store/schema.tmpl` (v2) and `internal/store/migrate.tmpl` (v1 -> v2)
    - [x] Bump store schema version to 2 in `internal/store/version.go`
    - [x] Create `knowledge_items`, `memory_versions`, `skill_versions`, and `knowledge_tombstones` tables with foreign keys and cascade rules
    - [x] Create indexes for item status lookup, version ordering, and indexing status reconciliation
    - [x] Write schema migration tests in `internal/store/migration_lifecycle_test.go` and `internal/store/sqlite_test.go`
- [x] Task: Implement isolated per-DM relational Knowledge Store
    - [x] Create `internal/knowledge/store.go` with CRUD operations:
      - `ProposeMemory(ctx, item, version)`
      - `ProposeSkill(ctx, item, version)`
      - `GetItem(ctx, id)`
      - `GetVersion(ctx, versionID)`
      - `ListItems(ctx, chatID, kind, statusFilter)`
      - `ListVersions(ctx, itemID)`
      - `ApproveVersion(ctx, versionID, userID)` (atomically transitions proposed -> approved, supersedes active version, sets item active pointer)
      - `RejectVersion(ctx, versionID, userID)`
      - `DisableSkill(ctx, itemID, userID)`, `EnableSkill(ctx, itemID, userID)`, `ArchiveSkill(ctx, itemID, userID)`
      - `ForgetItem(ctx, itemID, userID)` and `DeleteSkill(ctx, itemID, userID)` (erases content rows, creates tombstone)
      - `GetPendingIndexVersions(ctx, limit)` and `UpdateIndexStatus(ctx, versionID, status)`
    - [x] Write unit tests in `internal/knowledge/store_test.go` covering full lifecycle transitions, atomic supersession, and tombstone erasure
- [x] Task: Phase 1 Verification
    - [x] Run `go test -race ./internal/config/... ./internal/knowledge/...`

---

## Phase 2: CortexDB Hybrid Indexing, Retrieval & Reconciliation (`internal/knowledge`, `internal/memory`)
- [x] Task: Implement CortexDB indexing adapter for structured memories and skills
    - [x] Create `internal/knowledge/indexer.go` managing indexing into dedicated namespaces (`dm_memories`, `dm_skills`)
    - [x] Format searchable payloads: combined metadata + procedural instructions for skills; text content + type for memories
    - [x] Connect with existing embedding provider and CortexDB instance in `internal/memory`
    - [x] Retain pure lexical FTS5 fallback when embeddings are unavailable
- [x] Task: Implement post-retrieval relational revalidation
    - [x] Create `internal/knowledge/search.go` with `SearchMemories(ctx, query, types, limit)` and `SearchSkills(ctx, query, tags, limit)`
    - [x] Query CortexDB for candidate IDs and revalidate against relational SQLite store:
      - Verify `chat_id` and `user_id` match session
      - Verify item status is `active` and version matches `active_version_id`
      - Filter out expired items (`now() >= expires_at`) and discard stale index hits
- [x] Task: Implement index reconciliation and background synchronization
    - [x] Immediate indexing attempt on version approval
    - [x] Reconcile `pending` and `error` indexing records upon store opening and before discovery queries
    - [x] Remove CortexDB index entries when items are disabled, archived, superseded, or forgotten
- [x] Task: Write indexing and retrieval unit tests
    - [x] Create `internal/knowledge/indexer_test.go` and `internal/knowledge/search_test.go`
    - [x] Test hybrid search, lexical fallback, stale-index filtering, and reconciliation of pending records
- [x] Task: Phase 2 Verification
    - [x] Run `go test -race ./internal/knowledge/...`

---

## Phase 3: Gateway Commands & Human Lifecycle Management (`internal/gateway`)
- [x] Task: Implement `/memory` slash command dispatcher
    - [x] Add `/memory` handling in `internal/gateway/gateway.go` (routing to `handleMemoryCommand`)
    - [x] Implement subcommands:
      - `/memory list [status]`: list memories by status (default `active`)
      - `/memory show <id>`: display full item or version details with provenance
      - `/memory approve <version-id>`: human-only approval, activates version, triggers indexing
      - `/memory reject <version-id>`: marks version as rejected
      - `/memory forget <item-id>`: erases content, creates tombstone, returns explicit message that raw transcript history remains unchanged
- [x] Task: Implement `/skill` slash command dispatcher
    - [x] Add `/skill` handling in `internal/gateway/gateway.go` (routing to `handleSkillCommand`)
    - [x] Implement subcommands:
      - `/skill list [status]`: list skills by status (default `active`)
      - `/skill show <id>`: display full skill details and Markdown instructions
      - `/skill approve <version-id>`: human-only approval, activates version, triggers indexing
      - `/skill reject <version-id>`: marks version as rejected
      - `/skill disable <item-id>` and `/skill enable <item-id>`
      - `/skill archive <item-id>`
      - `/skill delete <item-id>`: erases content, creates tombstone
- [x] Task: Enforce authorization barriers and command normalization
    - [x] Reject commands if sender is bot (`msg.UserID == botID`)
    - [x] Reject commands if sender is not the DM owner
    - [x] Strictly reject commands in public Townhall chat
    - [x] Normalize command inputs (case insensitivity, whitespace trimming) matching existing sandbox/schedule commands
- [x] Task: Write Gateway unit tests for knowledge commands
    - [x] Create `internal/gateway/knowledge_commands_test.go`
    - [x] Test authorization enforcement, approval/rejection flows, forget/delete erasure, and transcript separation notice
- [x] Task: Phase 3 Verification
    - [x] Run `go test -race ./internal/gateway/...`

---

## Phase 4: Agent Tools, Context Budget Accounting & FSM Provenance (`internal/tools`, `internal/fsm`)
- [ ] Task: Extend FSM provenance and Session Context accounting
    - [ ] Extend `fsm_runs` schema and execution context with `source_message_seq`
    - [ ] Extend `ChatSessionContext` with `SourceMessageSeq`, `FSMRunID`
    - [ ] Implement thread-safe `KnowledgeBudgetTracker` in `ChatSessionContext`:
      - Track loaded memory count (max 8) and loaded memory bytes (max 16 KiB)
      - Track loaded skill count (max 3) and loaded skill bytes (max 24 KiB)
      - Track loaded IDs to prevent duplicate loads from consuming budget twice
    - [ ] Restore budget accounting from completed `load_memory` and `load_skill` steps in `fsm_steps` on FSM run recovery
- [ ] Task: Implement Memory Agent Tools
    - [ ] Define JSON schemas and implementations in `internal/tools`:
      - `propose_memory(type, content, confidence?, ttl_days?, revises_memory_id?)`: sequential execution mode, creates inactive version
      - `discover_memories(query, types?, limit?)`: read-only execution mode, returns bounded summaries
      - `load_memory(memory_id)`: sequential execution mode, returns full content, enforces budget
- [ ] Task: Implement Skill Agent Tools
    - [ ] Define JSON schemas and implementations in `internal/tools`:
      - `propose_skill(name, description, triggers, tags, instructions_markdown, revises_skill_id?)`: sequential execution mode, creates inactive version
      - `discover_skills(query, tags?, limit?)`: read-only execution mode, returns bounded metadata
      - `load_skill(skill_id)`: sequential execution mode, returns complete Markdown instructions, enforces budget
- [ ] Task: Tool definition filtering and FSM classification
    - [ ] Expose knowledge tools strictly in interactive DM sessions (`session.IsDM && !session.IsScheduled`)
    - [ ] Exclude knowledge tools from Townhall and scheduled background tasks
    - [ ] Classify `discover_*` as `ExecutionModeParallel` and `propose_*`/`load_*` as `ExecutionModeSequential` in `internal/fsm/executor.go`
- [ ] Task: Write tool registry unit tests
    - [ ] Create `internal/tools/knowledge_tools_test.go`
    - [ ] Test argument validation, budget enforcement, duplicate load caching, and execution mode classification
- [ ] Task: Phase 4 Verification
    - [ ] Run `go test -race ./internal/tools/... ./internal/fsm/...`

---

## Phase 5: Store Provider Wiring, Prompt Guidance & Integration Flow (`internal/gateway`, `internal/prompt`)
- [ ] Task: Wire Knowledge Store into Store Provider
    - [ ] Extend `internal/gateway/store_provider.go` with `GetKnowledgeStore(ctx, chatID, isDM)`
    - [ ] Integrate knowledge schema initialization alongside FSM and scheduler schemas on DM store open
    - [ ] Inject knowledge store and budget limits into `internal/tools/registry.go`
- [ ] Task: Update DM System Prompt in `internal/prompt`
    - [ ] Clarify three-tier knowledge architecture: raw history (`recall_memory`), structured knowledge (`discover_memories`/`load_memory`), and procedural skills (`discover_skills`/`load_skill`)
    - [ ] Instruct model on conservative proposal conditions (explicit request or durable correction)
    - [ ] Instruct model to inform user that proposals require slash-command approval
- [ ] Task: Write end-to-end automated integration tests
    - [ ] Create `internal/gateway/knowledge_integration_test.go` testing full flows:
      - Memory proposal -> discovery invisibility -> `/memory approve` -> discover & load -> revision -> active version stability -> approval -> `/memory forget` with transcript preservation
      - Skill proposal -> approval -> discover & load -> disable -> enable -> delete
      - Townhall and cross-user authorization rejection
- [ ] Task: Phase 5 Verification
    - [ ] Run `go test -race ./internal/gateway/...`

---

## Phase 6: Peer Review, Local Live Verification & Quality Checks
- [ ] Task: Conduct independent peer review via `agentic-pair-programming` skill
    - [ ] Launch `agent-pair` session with peer reviewer model
    - [ ] Review implementation across phases and address findings
    - [ ] Cleanly close `agent-pair` session
- [ ] Task: Local Live Verification with Besedka
    - [ ] Launch local Besedka server (`http://localhost:8080`)
    - [ ] Run `go run ./cmd/agent` connected to local Besedka
    - [ ] Interact with bot in DM: propose memory and skill, verify slash commands, and test discovery/load in conversation
- [ ] Task: Quality & Standards Verification
    - [ ] Run `go mod tidy` and `go mod vendor`
    - [ ] Run `make check` (ensure `lint-go`, `test-go -race`, `semgrep`, and `osv-scanner` all pass with 0 errors)
