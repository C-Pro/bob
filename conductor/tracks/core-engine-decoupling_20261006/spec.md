# Specification: Core Agent Engine Decoupling

## 1. Objective & Background
Bob currently operates solely as a Besedka bot daemon. The `internal/gateway` package combines Besedka WebSocket/REST transport concerns with core agent intelligence (turn execution, FSM tool loop orchestration, context ring buffers, command routing, and storage management). Furthermore, configuration validation and system prompts hardcode Besedka assumptions.

The objective of this track is **strictly splitting and decoupling**: extract a reusable, transport-agnostic core agent engine while preserving 100% of existing Besedka features, behaviors, and regression tests. No new frontends (such as CLI `-p` or TUI) will be built in this track; however, the resulting architecture must allow any connecting frontend to plug into the core engine with full authority over its execution environment.

## 2. Core Architectural Principle: Frontend Authority
Different frontends (e.g., Besedka chat vs. local terminal CLI vs. interactive TUI) possess radically different capabilities, security boundaries, and user experience requirements. Therefore, the connecting frontend must hold explicit authority over:
1. **System Prompt**: Supplied per turn (e.g., Besedka formatting guidelines vs. clean CLI stdout).
2. **Tool Schemas & Execution (`Toolset`)**: The frontend controls which tools are visible and executable for each session/turn. An empty or nil toolset means no tools are run (no implicit fallback to global tools).
3. **Attachments**: Handled via frontend-specific storage (Besedka REST download/upload vs. local workspace filesystem).
4. **Progress Observation**: Abstracted event stream (`ProgressObserver`) allowing Besedka to send progress cards and future frontends to render terminal spinners.
5. **Recovery & Delivery**: During crash recovery or detached execution, results are routed to the originating frontend via `Frontend.Deliver`.

## 3. Package Layout & Boundaries

```
internal/
├── agentapi/          # Shared contracts & neutral domain types (no gateway or engine imports)
│   ├── session.go     # SessionRef, Actor, RunDescriptor, ExecutionKind
│   ├── turn.go        # Turn, Result, Completion, Action
│   ├── bindings.go    # Frontend, Bindings, NotificationSink
│   ├── tools.go       # Toolset, ToolDefinition, ToolCall, ToolResult
│   ├── attachments.go # Attachment, AttachmentHandler, AttachmentContent
│   └── progress.go    # ProgressObserver, ProgressEvent, StepProgress
├── agent/             # Core execution orchestrator
│   ├── engine.go      # concrete Engine implementing TurnRunner
│   └── scheduled/     # Schedule execution coordinator (bridges scheduler & engine)
├── agentstore/        # Store composition (bridges memory.Manager to FSM/scheduler/knowledge)
│   └── provider.go    # Relocated and cleaned MemoryStoreProvider
├── commands/          # Transport-agnostic slash command services (/memory, /skill, /schedule, /sandbox)
├── fsm/               # Durable FSM execution (updated with per-run execution bindings)
├── gateway/           # Besedka transport adapter (consumes agent.Engine, manages WS/REST)
└── config/            # Decoupled validation (BESEDKA_URL optional for non-gateway modes)
```

## 4. Interface Definitions

### 4.1 Neutral Contracts (`internal/agentapi`)

```go
type SessionRef struct {
    FrontendID string
    SessionID  string
    ScopeID    string
}

type Actor struct {
    ID string
}

type ExecutionKind string
const (
    Interactive ExecutionKind = "interactive"
    Scheduled   ExecutionKind = "scheduled"
)

type RunDescriptor struct {
    RunID         string
    Session       SessionRef
    Actor         Actor
    Kind          ExecutionKind
    Model         string
    MaxIterations int
}

type Turn struct {
    Run          RunDescriptor
    SystemPrompt string
    Messages     []openai.ChatCompletionMessage
}

type Result struct {
    RunID       string
    Content     string
    Iterations  int
    Attachments []Attachment
    Actions     []Action
}

type Completion struct {
    Run    RunDescriptor
    Result Result
    Status string
    Error  string
}

type Frontend interface {
    Bind(context.Context, RunDescriptor) (Bindings, error)
    Deliver(context.Context, Completion) error
}

type Bindings struct {
    Tools         Toolset
    Attachments   AttachmentHandler
    Progress      ProgressObserver
    Notifications NotificationSink
}

type Toolset interface {
    Definitions(context.Context) ([]ToolDefinition, error)
    Execute(context.Context, ToolCall) (ToolResult, error)
}

type ToolDefinition struct {
    Schema   openai.Tool
    ReadOnly bool
}

type ToolCall struct {
    ID        string
    Name      string
    Arguments string
}

type ToolResult struct {
    Content     string
    Attachments []Attachment
    Actions     []Action
}
```

### 4.2 Engine Interface (`internal/agent`)

```go
type TurnRunner interface {
    Run(context.Context, agentapi.Turn) (agentapi.Result, error)
}

type Engine struct {
    // LLM client, FSM engine, session locks, registered frontends
}

func NewEngine(deps Dependencies) (*Engine, error)
func (e *Engine) Run(ctx context.Context, turn agentapi.Turn) (agentapi.Result, error)
func (e *Engine) Start(ctx context.Context) error
func (e *Engine) Close() error
```

## 5. Execution Flow & Crash Recovery

### 5.1 Normal Interactive Execution
1. Frontend receives an input message/event, updates local conversation history, and prepares a `Turn`.
2. Engine acquires a session execution lock (preventing concurrent turns on the same session).
3. Engine resolves the registered frontend via `Turn.Run.Session.FrontendID` and calls `Frontend.Bind(ctx, desc)`.
4. Engine executes the turn via `fsm.Engine` using the session's active `Bindings` (tools, attachments, progress observer).
5. Synchronous `Result` is returned to the frontend caller for immediate rendering/transmission.

### 5.2 Crash Recovery
1. The FSM engine scans SQLite stores for active/waiting runs upon startup.
2. The run descriptor includes persisted metadata (`FrontendID`, `SessionID`, `ScopeID`, `ActorID`, `Kind`, `Model`).
3. For interrupted runs, the FSM resolves the originating frontend via `FrontendID` and calls `Frontend.Bind(ctx, desc)` to rebind tools and dependencies.
4. If a frontend cannot be resolved or rejects binding, the run fails safely without executing arbitrary tools.
5. On recovered run completion, the result is delivered out-of-band via `Frontend.Deliver(ctx, completion)`.

## 6. Subsystem Relocations & Cleanups
1. **`MemoryStoreProvider` → `internal/agentstore`**: Relocated from `internal/gateway`. Retains per-chat SQLite database mapping and isolation without depending on Besedka models.
2. **`KnowledgeWorker` → `internal/knowledge`**: Relocated from `internal/gateway`. Owns periodic background reconciliation of knowledge embeddings.
3. **`ScheduleInvoker` → `internal/agent/scheduled`**: Broken out of `internal/gateway` to avoid circular package dependencies between `internal/tools` and `internal/scheduler`.
4. **Slash Commands → `internal/commands`**: Logic for `/memory`, `/skill`, `/schedule`, and `/sandbox` extracted into standalone services returning structured text/actions, independent of Besedka `models.Message` and WebSocket `SendMessage`.

## 7. Non-Goals
- No CLI `-p` command or TUI interface implementation in this track (deferred to subsequent tracks).
- No changes to LLM provider integrations (OpenAI/Gemini client remains as-is).
- No database schema migrations or changes to existing `.db` file locations.
- No changes to existing Besedka user-facing behavior or protocol syntax.
