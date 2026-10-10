# Specification: CLI One-Shot Frontend (`bob -p`)

## 1. Objective & Background
The `core-engine-decoupling_20261006` track extracted a transport-agnostic `agent.Engine` with frontend authority over system prompts, tools, attachments and progress (`agentapi.Frontend`). Besedka is currently the only frontend.

This track adds a **local command-line frontend**: a new `bob` binary that, when invoked with `-p <prompt>`, executes exactly one agent run and prints the answer to stdout. It also introduces a **file-based user configuration** (`~/.config/bob/config.yaml`) that will be shared with the future interactive TUI mode (invoked as `bob` without `-p`).

Design principles:
- **Safe by default:** no sandbox and no local file access unless the user explicitly mounts a directory.
- **Explicit consent through flags:** flags passed on the command line act as the approval. There is no `sandbox_request` / `/sandbox approve` flow in CLI mode.
- **Shared core wiring:** `cmd/agent` (Besedka daemon) and `cmd/bob` (CLI/TUI) build the core runtime (memory, stores, tools registry, FSM, agent engine) through a single shared bootstrap package.

## 2. User-Facing Behaviour

### 2.1 Invocation
```bash
bob -p "What changed in Go 1.26?"                       # web + knowledge tools, no sandbox
bob -p "Summarize the TODOs in this repo" --mount .      # CWD mounted read-only at /workspace, bwrap
bob -p "Run the tests and fix failures" --mount .:rw     # CWD mounted read-write
bob -p "Build it" --mount ~/src/app:rw --sandbox docker --image golang:1.26
bob -p "pip install and run" --mount . --net restricted --allow-domain pypi.org --allow-domain files.pythonhosted.org
git diff | bob -p "Write a commit message for this diff"  # piped stdin appended as context
```

Running `bob` without `-p` is reserved for the future TUI. In this track it prints usage plus "interactive mode is not implemented yet" and exits with code 2.

### 2.2 Flags
| Flag | Default | Constraint |
|------|---------|------------|
| `-p`, `--prompt STRING` | — | Required in this track. `-p -` reads the prompt from stdin. |
| `--config PATH` | `$XDG_CONFIG_HOME/bob/config.yaml` (fallback `~/.config/bob/config.yaml`) | — |
| `--mount PATH[:ro\|:rw]` | none (no sandbox) | Single workspace. `.` = CWD. Access defaults to `:ro`. |
| `--sandbox bwrap\|docker` | `bwrap` (or `sandbox.default_driver`) | Requires `--mount`. |
| `--net none\|restricted\|full` | `none` | Requires `--mount`. |
| `--allow-domain HOST` (repeatable) | — | Requires `--net restricted`. |
| `--image IMAGE` | `sandbox.docker_image` | Requires `--sandbox docker`; must be in `sandbox.allowed_images`. |
| `--model NAME` | `llm.model` | — |
| `--max-iterations N` | `cli.max_iterations` (20) | `N > 0` |
| `-q`, `--quiet` | false | Suppresses intermediate step output; stdout contains only the final answer. |
| `-v`, `--verbose` | false | Debug-level structured logs on stderr. |
| `--init-config` | — | Writes a commented template config (mode 0600, dir 0700), refuses to overwrite, then exits. |

Invalid combinations (for example `--sandbox` without `--mount`, or `--allow-domain` without `--net restricted`) are usage errors that exit with code 2 before any network or LLM activity.

### 2.3 Output & Exit Codes
- **stdout (default):** intermediate steps are streamed as they happen, followed by the final assistant answer. Steps are compact lines: tool name, short argument summary, duration, success/failure, and the exit code for `execute`. They're printed regardless of whether stdout is a TTY. Colour/styling is used only when stdout is a TTY. A blank line separates the step log from the final answer (Markdown, newline-terminated).
- **stdout (`-q`/`--quiet`):** only the final answer, with no step lines. This is the mode for `bob -q -p ... > out.md` and shell pipelines.
- **stderr:** structured logs only (WARN by default, DEBUG with `--verbose`) and fatal error messages. Progress never goes to stderr.
- **Piped stdin:** when stdin is not a TTY and the prompt is not `-`, stdin (capped at `MaxAttachmentSizeBytes`) is appended to the user message as a fenced block.
- **Exit codes:** `0` success · `1` runtime failure (LLM, tool loop, sandbox creation) · `2` usage/config error · `130` interrupted (SIGINT/SIGTERM).
- **Interruption:** Ctrl-C cancels the run context, destroys the sandbox, marks the FSM run as cancelled/failed (no orphaned `running` rows), closes stores and exits 130.

## 3. Configuration (`config.yaml`)

### 3.1 Location, Format & Security
- Path resolution: `--config` flag, else `os.UserConfigDir()/bob/config.yaml`. No environment-variable overrides and no `${VAR}` expansion; the YAML file is the single source of truth for CLI/TUI.
- **Permissions enforced:** loading fails if the file is accessible to group/other (`mode & 0o077 != 0`) or is not owned by the current user. The error message suggests `chmod 600 <path>`.
- **Strict decoding:** unknown keys are an error (`yaml.v3` `KnownFields(true)`), so typos are caught early.
- `~/` prefixes in path values (`data_dir`) are expanded.
- Missing file: error code 2 with a hint to run `bob --init-config`.

### 3.2 Schema
```yaml
llm:
  api_key: "..."                  # required
  base_url: "https://generativelanguage.googleapis.com/v1beta/openai/"
  model: "gemini-3.8-flash"
  embedding_model: ""             # empty/none = existing local/disabled behaviour
  embedding_precision: "bf16"

web:
  tavily_api_key: ""              # web_search omitted when empty; web_fetch always available
  tavily_base_url: "https://api.tavily.com"

data_dir: "~/.local/share/bob"    # default: $XDG_DATA_HOME/bob

cli:
  max_iterations: 20

knowledge:
  max_loaded_memories: 8
  max_loaded_memory_bytes: 16384
  max_loaded_skills: 3
  max_loaded_skill_bytes: 24576
  default_discovery_limit: 5
  max_discovery_limit: 10

sandbox:
  default_driver: "bwrap"
  docker_socket: "/var/run/docker.sock"
  docker_image: "python:3.12-slim"    # default: first of allowed_images
  allowed_images: [...]               # default: sandbox.DefaultSandboxAllowedImages
  allowed_network_modes: ["none", "restricted", "full"]
  default_exec_timeout: "1m"
  max_exec_timeout: "10m"
  max_lifetime: "30m"
  cpu_limit: 1.0
  memory_limit_mb: 512
```

### 3.3 Mapping to `config.Config`
- `LoadFromEnv` is refactored so that defaults come from a shared `config.Defaults()`. `config.LoadFromFile(path)` starts from the same defaults, overlays the YAML values and returns the existing `*config.Config`. This lets `Validate`, `SandboxConfig()` and all downstream constructors be reused unchanged.
- Daemon-only fields (`BesedkaURL`, `BotHandle`, S3, GEOIP, scheduler) are not part of the YAML schema.
- `ValidateCLI()` = `Validate(true)` plus CLI-specific checks (`data_dir` is writable, `default_driver` is a known driver, `cli.max_iterations > 0`).
- In CLI mode, flags are explicit operator consent. `allowed_network_modes` therefore defaults to all three modes, unlike the daemon default that excludes `full`. Users can still restrict it in YAML.

## 4. Tools & Authority

### 4.1 Default Toolset (no mount)
| Tool | Condition |
|------|-----------|
| `web_search` | `web.tavily_api_key` set |
| `web_fetch` | always |
| `discover_memories`, `load_memory`, `discover_skills`, `load_skill` | always (read-only knowledge) |

Explicitly **excluded**: `propose_memory`, `propose_skill` (no approval channel in one-shot mode), `recall_memory` (no chat history in CLI), scheduler tools, `sandbox_request`/`sandbox_exec`/`sandbox_destroy`, and attachment tools.

### 4.2 With `--mount`
The CLI frontend provisions the sandbox **before** the run starts, so failures surface as exit code 1 before any LLM call. It then adds:

| Tool | Description |
|------|-------------|
| `execute` | Runs `sh -c <command>` inside the pre-provisioned sandbox. Args: `command` (required), `description`, `timeout_seconds` (clamped to `max_exec_timeout`). `ReadOnly: false` so dispatch is sequential. The tool description states driver, `/workspace` access (ro/rw) and network mode. Output format matches `sandbox_exec` through a shared formatter. |

### 4.3 Enforcement
- `tools.AllowlistToolset(base, names...)` filters both `Definitions` and `Execute`. Calls to tools outside the allowlist are rejected and never reach the registry, which preserves the strict tool authority from the decoupling track.
- `tools.CompositeToolset(sets...)` merges toolsets and errors on duplicate tool names.
- These helpers live in `internal/tools` so the future TUI can reuse them.

## 5. Sandbox Provisioning

### 5.1 Mount Specification
`ParseMountSpec(spec, cwd)`:
1. Split an optional trailing `:ro` / `:rw` suffix (only these exact suffixes, so paths containing `:` still work). Default is `ro`.
2. Expand `.` and relative paths against CWD, plus leading `~/`. Resolve with `filepath.Abs` + `filepath.EvalSymlinks`.
3. The path must exist and be a directory.
4. **Reject** `/`, and any path that is equal to, an ancestor of, or a descendant of the resolved **config directory** or **data directory**. This prevents leaking `config.yaml` (API keys) or SQLite databases into the sandbox. Example: mounting `$HOME` is rejected while `~/.config/bob` lives under it, with an explanatory error.

### 5.2 Pre-Approved Provisioning
New `sandbox.Manager.Provision(ctx, userID, chatID string, p ProvisionParams) (*UserSandbox, error)`:
- `ProvisionParams{Driver, DockerImage, NetworkMode, AllowedDomains, WorkspaceDir, ReadOnly, Lifetime}`.
- Performs the same driver, image and network-policy validation as `RequestSandbox`, including the mandatory blocked CIDRs.
- Uses the **explicit host `WorkspaceDir`** instead of `UserWorkspaceDir(userID)` and a single whole-workspace mount (`RelativePath: "."`). The workspace is mounted at `/workspace`, which is also the working directory.
- Skips `StatusPendingApproval` and creates the sandbox synchronously (→ `StatusRunning`), respecting the 1-sandbox-per-user invariant.
- Bwrap: must not `MkdirAll` or otherwise modify a pre-existing host workspace beyond what `--bind`/`--ro-bind` requires.
- Docker: binds the absolute host path directly. `HostDataDir` translation applies only to paths under `DataDir`.
- The sandbox is destroyed on process exit (normal, error or signal) via `Manager.Close()`.

### 5.3 CLI Identity
- `FrontendID = "cli"`, `SessionID = ScopeID = "cli"`, `Actor.ID = "local"` (matches `validUserIDRegex`). `IsDM = true`, `IsScheduled = false` so knowledge tools are authorized.
- Each invocation gets a fresh `RunID`. Runs are stateless: no prior conversation history is loaded.
- Knowledge and FSM data live in `data_dir` through the existing `agentstore` layout (`dm_cli` store). They are fully separate from any Besedka daemon data dir.

## 6. Package Layout
```
cmd/
├── agent/                # Besedka daemon (env config) — rewired onto internal/bootstrap
└── bob/                  # NEW: CLI one-shot (and future TUI) entrypoint
internal/
├── bootstrap/            # NEW: shared core composition (memory.Manager, agentstore, tools.Registry,
│                         #      optional sandbox.Manager, fsm.Engine, agent.Engine)
├── cli/                  # NEW: CLI frontend (agentapi.Frontend), mount parsing, progress renderer, runner
├── config/               # + Defaults(), LoadFromFile(), DefaultPaths(), ValidateCLI(), template
├── prompt/               # + RenderCLI()
├── sandbox/              # + Manager.Provision(), explicit-workspace support in drivers
└── tools/                # + AllowlistToolset, CompositeToolset, ExecTool
```

### 6.1 CLI Frontend Contract
```go
type Options struct {
    Prompt        string
    Stdin         string
    Model         string
    MaxIterations int
    Sandbox       *SandboxSpec // nil = no sandbox, no execute tool
    Quiet         bool
}

type Frontend struct { /* registry, sandbox manager, progress writer */ }

func (f *Frontend) Bind(ctx context.Context, run agentapi.RunDescriptor) (agentapi.Bindings, error)
func (f *Frontend) Deliver(ctx context.Context, c agentapi.Completion) error // one-shot: returns ErrDetachedDeliveryUnsupported

func Run(ctx context.Context, rt *bootstrap.Runtime, opts Options, stdout, stderr io.Writer) (exitCode int)
```
- `Bind` returns: Tools = allowlisted registry toolset (+ `ExecTool` when a sandbox exists); Attachments = nil; Progress = stdout step renderer (no-op when `Quiet`); Notifications = stdout step line (dropped when `Quiet`). Step lines and the final answer share one writer guarded by a mutex, so parallel read-only tool dispatch can't interleave partial lines.
- `cmd/bob/main.go` is a thin shim: `os.Exit(run(os.Args[1:], os.Stdin, os.Stdout, os.Stderr))`, so `run` is unit-testable.

### 6.2 System Prompt (`prompt.RenderCLI`)
Terminal-oriented: current date/time and timezone, OS, concise Markdown output, no Besedka paragraph limits or mention syntax. Includes a tool summary and, when sandboxed, the host path ↔ `/workspace` mapping, the access mode (ro/rw) and the network mode. Without a sandbox it explicitly states that local files cannot be accessed.

## 7. Lifecycle (One-Shot)
1. Parse flags → validate the flag matrix (exit 2 on error).
2. Resolve paths → load and validate `config.yaml` (exit 2 on error).
3. `bootstrap.New(cfg, opts)` builds the runtime. The FSM engine is created **without** calling `Start`/`Recover`, so interrupted runs from other invocations or processes are not resumed. Knowledge worker, GEOIP and S3 backup are not started.
4. If `--mount` is set: `ParseMountSpec` → `Manager.Provision` (exit 1 on failure).
5. Register the `cli` frontend → `agent.Engine.Run(Turn{Deliver: false})`.
6. Print `Result.Content` to stdout → teardown (sandbox destroy, engine/stores close) → exit 0.

## 8. Non-Goals
- Interactive TUI and multi-turn sessions/history persistence (future track; this track only reserves the entrypoint).
- `propose_memory` / `propose_skill`, approval subcommands, `recall_memory` and indexing of CLI runs.
- Scheduler tools, attachment upload/download, S3 backup, GEOIP.
- Crash recovery or resumption of CLI runs.
- Multiple mounts, and masking the config/data dir inside a broader mount (rejected instead).
- Environment-variable configuration for CLI mode, and non-Linux sandbox support.
- Any change to Besedka daemon behaviour or env configuration.

## 9. Risks & Open Points
- **Startup latency:** local embedder initialization on every one-shot run may be slow. The embedder should be initialized lazily on the first knowledge search; measure this in Phase 5.
- **Empty knowledge base:** with `propose_*` excluded, the CLI knowledge store stays empty until the TUI or approval commands exist, or until `data_dir` points at an existing store. This is documented in README.
- **Concurrent invocations:** parallel `bob -p` processes share the `dm_cli` SQLite store. In-process session locks don't cover this; we rely on SQLite busy timeouts. Recovery is disabled, so one process never adopts another's run.
- **Docker driver** currently assumes workspaces under `DataDir` for `HostDataDir` translation. Phase 4 must verify and adapt this.
