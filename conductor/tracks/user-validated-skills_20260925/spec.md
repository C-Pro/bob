# Specification: User-Validated Skills and Structured Memory

## 1. Overview

This track introduces private, per-DM procedural skills and typed structured memories to Bob, layered on top of the agent's existing SQLite per-chat storage and CortexDB hybrid vector/lexical retrieval engine.

The core design principle is **User Authority via Explicit Human Validation**:
- The model may propose candidate memories or skills, search for available items via discovery tools, and selectively load them into a bounded per-request context budget.
- **Only slash commands from the owning human user in the specific Direct Message (DM) chat can activate, modify, or remove them.** Conversational LLM responses can never grant approval or alter active knowledge state.
- Existing raw conversation transcript storage and historical `recall_memory` retrieval remain strictly unchanged and authoritative for raw conversational history.
- Future capabilities—including automated evaluation harnesses, unreviewed automatic context injection, episodic summaries, and adaptive compaction—are explicitly deferred to follow-up tracks.

---

## 2. Security, Privacy & Threat Model

### 2.1 DM-Exclusive Scope & Privacy by Construction
- Structured memories and skills are strictly scoped to 1-on-1 Direct Messages (DMs).
- Storage resides exclusively within each chat's isolated SQLite database (`dm_<sanitized_chatID>.db`).
- **Zero cross-chat leakage:** Townhall chats have no access to knowledge tools or commands. User $A$'s DM database cannot inspect, query, or execute User $B$'s private skills or memories.
- In v1, there is no global sharing or cross-DM promotion of knowledge items.

### 2.2 Human-Only Approval Barrier
- Agent tools can only propose candidate versions in the `proposed` state. They have no mechanism to self-approve or activate knowledge.
- Approval requires an explicit slash command (`/memory approve <version-id>` or `/skill approve <version-id>`) sent by the human DM owner.
- Gateway command handlers strictly verify that the sender (`msg.UserID`) is the human user matching the DM session and never matches `botUserID`.

### 2.3 Autonomous & Scheduled Run Exclusion
- Proposal tools (`propose_memory`, `propose_skill`) require an interactive source message from a human user.
- Scheduled tasks (`schedule_task`) and autonomous background runs are prohibited from proposing memories or skills.
- Tool descriptions and the DM system prompt instruct the agent to propose items only after an explicit human request or an unambiguous durable correction.

### 2.4 Trusted Runtime Provenance & Tamper Resistance
- Provenance metadata is captured from trusted runtime context rather than user- or model-supplied parameters:
  - `chat_id`: Current DM chat identifier.
  - `user_id`: Owning human user identifier.
  - `source_message_seq`: Sequence number of the triggering chat message.
  - `fsm_run_id`: Active FSM execution run ID.
  - `created_at`: Unix timestamp in seconds.
  - `content_hash`: Deterministic SHA-256 digest of the item content for deduplication and audit tracking.

### 2.5 Erasure, Tombstones & Transcript Separation
- Forgetting a memory (`/memory forget <item-id>`) or deleting a skill (`/skill delete <item-id>`) permanently removes all content-bearing version records and corresponding CortexDB index entries.
- A minimal tombstone is retained in the relational database containing only non-content audit metadata: item ID, item kind, actor user ID, and deletion timestamp.
- **Transcript Independence:** Forgetting structured memory does not alter the source Besedka chat transcript or existing raw-history chunks indexed by `recall_memory`. Command responses from `/memory forget` must explicitly inform the user of this distinction.

---

## 3. Data Model & Relational Schema

All tables are maintained inside each DM's SQLite database (`dm_<sanitized_chatID>.db`) in `DATA_DIR`, isolated from other chats. Migrations are managed by `internal/knowledge/schema.go`.

### 3.1 Namespaced Schema Versioning
```sql
CREATE TABLE IF NOT EXISTS knowledge_schema_version (
    version INTEGER PRIMARY KEY,
    description TEXT NOT NULL,
    applied_at INTEGER NOT NULL
);
```

### 3.2 `knowledge_items` Table
Represents the stable item identity across revisions and lifecycle state.
```sql
CREATE TABLE IF NOT EXISTS knowledge_items (
    id TEXT PRIMARY KEY,                       -- e.g. "mem_01j8xyz..." or "skill_01j8abc..."
    chat_id TEXT NOT NULL,                     -- Owning DM chat ID
    user_id TEXT NOT NULL,                     -- Owning human user ID
    kind TEXT NOT NULL,                        -- 'memory' or 'skill'
    status TEXT NOT NULL,                      -- 'active', 'disabled', 'archived', 'forgotten'
    active_version_id TEXT,                    -- Reference to current approved version (e.g. "mem_...#1")
    created_at INTEGER NOT NULL,               -- Unix timestamp (seconds)
    updated_at INTEGER NOT NULL                -- Unix timestamp (seconds)
);

CREATE INDEX IF NOT EXISTS idx_knowledge_items_lookup
ON knowledge_items(chat_id, kind, status);
```

### 3.3 `memory_versions` Table
Maintains immutable versions of structured memories.
```sql
CREATE TABLE IF NOT EXISTS memory_versions (
    id TEXT PRIMARY KEY,                       -- e.g. "mem_01j8xyz@1" (<item_id>@<revision>)
    item_id TEXT NOT NULL,                     -- Foreign key to knowledge_items(id) ON DELETE CASCADE
    revision INTEGER NOT NULL,                 -- Monotonically increasing revision number (1, 2, ...)
    type TEXT NOT NULL,                        -- 'preference', 'fact', 'decision', 'ongoing_task'
    content TEXT NOT NULL,                     -- Memory text content
    confidence REAL NOT NULL DEFAULT 1.0,      -- Confidence score (0.0 - 1.0)
    ttl_days INTEGER,                          -- Optional TTL in days
    expires_at INTEGER,                        -- Unix timestamp (seconds) when memory expires
    status TEXT NOT NULL,                      -- 'proposed', 'approved', 'rejected', 'superseded', 'expired'
    chat_id TEXT NOT NULL,                     -- Provenance: DM chat ID
    user_id TEXT NOT NULL,                     -- Provenance: human user ID
    source_message_seq INTEGER,                -- Provenance: triggering message sequence
    fsm_run_id TEXT,                           -- Provenance: triggering FSM run ID
    content_hash TEXT NOT NULL,                -- Provenance: SHA-256 digest of content
    created_at INTEGER NOT NULL,               -- Proposal timestamp (seconds)
    reviewed_at INTEGER,                       -- Review timestamp (seconds)
    reviewed_by TEXT,                          -- Human user ID who approved/rejected
    index_status TEXT NOT NULL DEFAULT 'pending', -- 'pending', 'ready', 'error'
    FOREIGN KEY(item_id) REFERENCES knowledge_items(id) ON DELETE CASCADE,
    CONSTRAINT uq_memory_versions_item_rev UNIQUE(item_id, revision)
);

CREATE INDEX IF NOT EXISTS idx_memory_versions_item
ON memory_versions(item_id, revision);

CREATE INDEX IF NOT EXISTS idx_memory_versions_index_sync
ON memory_versions(status, index_status);
```

### 3.4 `skill_versions` Table
Maintains immutable versions of procedural skills.
```sql
CREATE TABLE IF NOT EXISTS skill_versions (
    id TEXT PRIMARY KEY,                       -- e.g. "skill_01j8abc@1" (<item_id>@<revision>)
    item_id TEXT NOT NULL,                     -- Foreign key to knowledge_items(id) ON DELETE CASCADE
    revision INTEGER NOT NULL,                 -- Monotonically increasing revision number (1, 2, ...)
    name TEXT NOT NULL,                        -- Human-readable skill name (e.g. 'deploy_service')
    description TEXT NOT NULL,                 -- Short purpose and scope description
    triggers_json TEXT NOT NULL,               -- JSON array of activation triggers (strings)
    tags_json TEXT NOT NULL,                   -- JSON array of categorization tags (strings)
    instructions_markdown TEXT NOT NULL,       -- Procedural Markdown instructions (exportable)
    status TEXT NOT NULL,                      -- 'proposed', 'approved', 'rejected', 'superseded', 'archived'
    chat_id TEXT NOT NULL,                     -- Provenance: DM chat ID
    user_id TEXT NOT NULL,                     -- Provenance: human user ID
    source_message_seq INTEGER,                -- Provenance: triggering message sequence
    fsm_run_id TEXT,                           -- Provenance: triggering FSM run ID
    content_hash TEXT NOT NULL,                -- Provenance: SHA-256 digest of instructions
    created_at INTEGER NOT NULL,               -- Proposal timestamp (seconds)
    reviewed_at INTEGER,                       -- Review timestamp (seconds)
    reviewed_by TEXT,                          -- Human user ID who approved/rejected
    index_status TEXT NOT NULL DEFAULT 'pending', -- 'pending', 'ready', 'error'
    FOREIGN KEY(item_id) REFERENCES knowledge_items(id) ON DELETE CASCADE,
    CONSTRAINT uq_skill_versions_item_rev UNIQUE(item_id, revision)
);

CREATE INDEX IF NOT EXISTS idx_skill_versions_item
ON skill_versions(item_id, revision);

CREATE INDEX IF NOT EXISTS idx_skill_versions_index_sync
ON skill_versions(status, index_status);
```

### 3.5 `knowledge_tombstones` Table
Minimal audit tombstone retained after content erasure.
```sql
CREATE TABLE IF NOT EXISTS knowledge_tombstones (
    item_id TEXT PRIMARY KEY,                  -- Original item ID
    kind TEXT NOT NULL,                        -- 'memory' or 'skill'
    chat_id TEXT NOT NULL,                     -- DM chat ID
    user_id TEXT NOT NULL,                     -- Owning user ID
    deleted_by TEXT NOT NULL,                  -- User ID who executed forget/delete command
    deleted_at INTEGER NOT NULL,               -- Unix timestamp (seconds)
    reason TEXT                                -- Optional reason
);
```

---

## 4. Retrieval & CortexDB Integration

### 4.1 Authoritative Relational Records & Isolated Namespaces
- Relational SQLite tables remain authoritative for all lifecycle state, ownership, and content.
- CortexDB is used strictly as an accelerated index. Only active, approved, non-expired, and enabled versions are indexed.
- CortexDB indexes are isolated in dedicated per-chat namespaces:
  - `dm_memories`: Vector and FTS5 index for approved structured memories.
  - `dm_skills`: Vector and FTS5 index for approved skills.
  - Raw conversation history remains in the existing `messages` namespace.

### 4.2 Searchable Metadata & Content Indexing
- **Memories:** Indexed with their type, text content, and confidence.
- **Skills:** Indexed by concatenating metadata (`name`, `description`, `triggers`, `tags`) with procedural instructions so discovery queries can match on intended workflow patterns or explicit tool tags. Full canonical instructions are fetched from SQLite on explicit load.

### 4.3 Hybrid Retrieval & Lexical Fallback
- Retrieval executes hybrid search (dense embedding vector cosine similarity combined with SQLite FTS5 BM25 scoring).
- When embeddings are unavailable (local embedding model offline or provider rate-limited), retrieval falls back cleanly to lexical-only FTS5 search without failing.

### 4.4 Strict Post-Retrieval Revalidation
CortexDB search results return candidate item IDs. Before any item metadata or content is returned to the agent, the retrieval layer **must revalidate**:
1. `chat_id` and `user_id` match the active session.
2. Item `status` is `active` (and not `disabled`, `archived`, or `forgotten`).
3. Returned version matches `active_version_id` and its version status is `approved`.
4. If `expires_at` is set, `now() < expires_at`. Stale index entries that have expired or been superseded are discarded immediately and scheduled for index cleanup.

### 4.5 Indexing Reconciliation & Lifecycle Consistency
- Version approval triggers an immediate synchronous or background indexing task in CortexDB.
- `index_status` is tracked as `pending`, `ready`, or `error`.
- When the knowledge store is initialized, and prior to executing discovery queries, any items with `pending` or `error` indexing state are reconciled.
- When an item is disabled, archived, superseded, or forgotten, the relational update immediately marks it unavailable within the database transaction *before* triggering best-effort index deletion.

---

## 5. Agent Tools & Context Budgeting

Six tools are registered in `internal/tools`, exposed **strictly in interactive 1-on-1 DMs**.

### 5.1 Tool Definitions

#### 1. `propose_memory`
- **Purpose:** Propose a new durable memory or revise an existing memory.
- **Classification:** `ExecutionModeSequential` (persists candidate version).
- **Parameters:**
  - `type` (string, required): Enum `["preference", "fact", "decision", "ongoing_task"]`.
  - `content` (string, required): Specific memory statement.
  - `confidence` (number, optional): Value between 0.0 and 1.0 (default 1.0).
  - `ttl_days` (integer, optional): Retention duration in days before automatic expiration.
  - `revises_memory_id` (string, optional): Existing `mem_<id>` if this updates a prior memory.
- **Behavior:**
  - Rejected if executed in scheduled runs or Townhall.
  - Creates a candidate version in `proposed` state.
  - Generates a stable ID (`mem_<random>`) and version reference (`<item_id>@<rev>`).
  - Returns confirmation message instructing the user how to approve it (`/memory approve <version-id>`).

#### 2. `discover_memories`
- **Purpose:** Search approved memories relevant to a query.
- **Classification:** `ExecutionModeParallel` (read-only retrieval).
- **Parameters:**
  - `query` (string, required): Natural language search query.
  - `types` (array of strings, optional): Filter by memory type (`["preference", "fact", ...]`).
  - `limit` (integer, optional): Maximum results (default 5, maximum 10).
- **Behavior:**
  - Searches CortexDB `dm_memories` namespace with hybrid retrieval.
  - Revalidates all candidates against the relational store.
  - Returns compact summaries: `item_id`, `type`, truncated snippet (max 150 characters), and confidence.

#### 3. `load_memory`
- **Purpose:** Load the full canonical content of an approved memory into the active turn context.
- **Classification:** `ExecutionModeSequential` (mutates turn budget).
- **Parameters:**
  - `memory_id` (string, required): Stable memory identifier (`mem_<id>`).
- **Behavior:**
  - Revalidates item status and resolves the current `active_version_id`.
  - Checks request-scoped budget: max 8 memories and 16 KiB total loaded memory content per run.
  - Deduplication: Loading an already loaded memory in the same run returns cached content without consuming budget twice.
  - If budget is exceeded, returns an explicit budget exhaustion error.

#### 4. `propose_skill`
- **Purpose:** Propose a new reusable procedural skill or revise an existing skill.
- **Classification:** `ExecutionModeSequential` (persists candidate version).
- **Parameters:**
  - `name` (string, required): Concise skill identifier (e.g. `format_changelog`).
  - `description` (string, required): Purpose and scenario description.
  - `triggers` (array of strings, required): Activation phrases or conditions.
  - `tags` (array of strings, required): Workflow categorization tags.
  - `instructions_markdown` (string, required): Procedural steps and guidance in Markdown.
  - `revises_skill_id` (string, optional): Existing `skill_<id>` if revising.
- **Behavior:**
  - Rejected if executed in scheduled runs or Townhall.
  - Creates candidate version in `proposed` state.
  - Returns confirmation message with approval instructions (`/skill approve <version-id>`).

#### 5. `discover_skills`
- **Purpose:** Search approved procedural skills relevant to a task.
- **Classification:** `ExecutionModeParallel` (read-only retrieval).
- **Parameters:**
  - `query` (string, required): Task description or query.
  - `tags` (array of strings, optional): Filter by tags.
  - `limit` (integer, optional): Maximum results (default 5, maximum 10).
- **Behavior:**
  - Searches CortexDB `dm_skills` namespace.
  - Revalidates candidates against relational store.
  - Returns compact metadata: `skill_id`, `name`, `description`, `triggers`, and `tags`. Does NOT return full instructions markdown.

#### 6. `load_skill`
- **Purpose:** Load full procedural Markdown instructions of an approved skill into active turn context.
- **Classification:** `ExecutionModeSequential` (mutates turn budget).
- **Parameters:**
  - `skill_id` (string, required): Stable skill identifier (`skill_<id>`).
- **Behavior:**
  - Revalidates item status and resolves `active_version_id`.
  - Checks request-scoped budget: max 3 skills and 24 KiB total loaded skill content per run.
  - Deduplication: Idempotent if re-loaded in the same run.
  - Returns complete `instructions_markdown`.

### 5.2 Context Budget & Turn Accounting
- `ChatSessionContext` tracks thread-safe accounting:
  - `loadedMemories map[string]int` (tracks IDs and byte sizes; limit: 8 items, 16 KiB total).
  - `loadedSkills map[string]int` (tracks IDs and byte sizes; limit: 3 items, 24 KiB total).
- **Crash Recovery & FSM Resume:** On engine restart or run recovery, the accounting tracker inspects completed `load_memory` and `load_skill` steps in `fsm_steps` and restores the exact loaded state.

### 5.3 FSM Provenance Propagation
- `fsm_runs` schema and runtime context are extended with `source_message_seq` (the sequence number of the triggering chat message).
- When a proposal tool is invoked, this sequence number is extracted directly from the session context for tamper-proof provenance recording.

---

## 6. Gateway Human Slash Commands

Slash commands are intercepted by `internal/gateway` before LLM processing. They are restricted strictly to the human owner in the exact DM session (`msg.UserID != botUserID`).

### 6.1 `/memory` Commands
- `/memory list [status]`: Lists memories in the chat. Optional status filter: `pending`, `active` (default), `rejected`, `expired`, `forgotten`, `all`. Outputs Markdown table with ID, Type, Snippet, Revision, and Status.
- `/memory show <id>`: Shows complete details of an item or specific version (`<item-id>` or `<item-id>@<rev>`), including full content, type, confidence, expiry, and provenance (source message sequence, timestamps, FSM run ID).
- `/memory approve <version-id>`: Approves a proposed version (`<item-id>@<rev>`).
  - Transitions version status from `proposed` to `approved`.
  - Atomically supersedes any previous approved version (`superseded`).
  - Sets item `active_version_id` to this version and item status to `active`.
  - Queues CortexDB indexing.
- `/memory reject <version-id>`: Rejects a proposed version. Transitions version status to `rejected`.
- `/memory forget <item-id>`: Permanently erases all content-bearing version rows and CortexDB index entries for `<item-id>`. Inserts a minimal tombstone record.
  - **Explicit Notice:** Command response must explicitly state:
    > "Memory `<item-id>` has been forgotten and erased from structured memory. *Note: Past conversation transcripts and raw message history remain unchanged.*"

### 6.2 `/skill` Commands
- `/skill list [status]`: Lists skills in the chat. Optional status: `pending`, `active` (default), `disabled`, `rejected`, `archived`, `all`. Outputs Markdown table with ID, Name, Description, Tags, and Status.
- `/skill show <id>`: Shows full skill details or specific version (`<item-id>` or `<item-id>@<rev>`), including Markdown instructions, triggers, tags, and provenance.
- `/skill approve <version-id>`: Approves a proposed version (`<item-id>@<rev>`).
  - Transitions version status from `proposed` to `approved`.
  - Atomically supersedes any previous approved version.
  - Sets item `active_version_id` to this version and item status to `active`.
  - Queues CortexDB indexing.
- `/skill reject <version-id>`: Rejects a proposed version (`rejected`).
- `/skill disable <item-id>`: Transitions item status from `active` to `disabled`. Immediately removes it from discovery and loading without deleting content.
- `/skill enable <item-id>`: Transitions item status from `disabled` back to `active`.
- `/skill archive <item-id>`: Transitions item status to `archived`. Preserved for audit but excluded from active use.
- `/skill delete <item-id>`: Permanently deletes all content-bearing versions and index entries, retaining an audit tombstone.

---

## 7. System Prompt Integration & Guidance

The DM system prompt (`internal/prompt`) is updated to guide model behavior:
1. **Three-Tier Knowledge Architecture:**
   - **Tier 1 (Raw History):** Use `recall_memory` to search past chat conversation transcripts.
   - **Tier 2 (Structured Knowledge):** Use `discover_memories` to identify durable user preferences, facts, decisions, and ongoing tasks, then invoke `load_memory` to fetch full content within context budgets.
   - **Tier 3 (Procedural Skills):** Use `discover_skills` to find relevant procedures, then invoke `load_skill` to load actionable instructions.
2. **Conservative Proposal Policy:**
   - Propose memories or skills **only** when the user explicitly requests them (e.g. *"Remember that I prefer..."*, *"Save this workflow as a skill"*) or provides an unambiguous durable correction.
   - Do NOT propose items for transient conversation topics or speculative preferences.
   - Clearly notify the user in conversational output that a proposal was created and requires approval via the corresponding slash command.
