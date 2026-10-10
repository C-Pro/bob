# Implementation Plan: CLI One-Shot Frontend (`bob -p`)

Each task is sized to be one reviewable step that keeps `make check` green.

## Phase 0: Baseline
- [x] Task: Verify baseline
    - [x] Create branch `feature/cli-oneshot` from the merged `feature/split-core`
    - [x] Run `make check` and record a clean baseline

## Phase 1: Shared Core Bootstrap (`internal/bootstrap`)
- [x] Task: Extract core composition out of `gateway.NewGateway`
    - [x] Create `bootstrap.Runtime` holding `memory.Manager`, `agentstore.MemoryStoreProvider`, `tools.Registry`, optional `sandbox.Manager`, `fsm.Engine`, `agent.Engine`, `llm.Client`
    - [x] `bootstrap.New(cfg, Options{EnableSandbox, EnableScheduler, Frontends...})` with an ordered `Close()` (engine → FSM → sandbox → stores)
    - [x] Embedder selection logic moved verbatim. Make local embedder initialization lazy (first use), measuring before and after
    - [x] Unit tests: runtime builds with and without sandbox/tavily, `Close` is idempotent, no goroutine leaks
- [x] Task: Rewire Besedka onto bootstrap
    - [x] `gateway.NewGateway` accepts prebuilt components from `bootstrap.Runtime`; Besedka-only wiring (attachment client, DM authorizer, scheduler, knowledge worker, GEOIP) stays in gateway/`cmd/agent`
    - [x] Update `cmd/agent/main.go`; all existing gateway/FSM regression tests pass unchanged

## Phase 2: File-Based Configuration
- [x] Task: Shared defaults
    - [x] Introduce `config.Defaults()` and refactor `LoadFromEnv` to overlay env vars on it (behaviour unchanged, covered by existing tests)
- [x] Task: YAML loader
    - [x] Promote `gopkg.in/yaml.v3` to a direct dependency; `go mod tidy && go mod vendor`
    - [x] `config.DefaultPaths()` → config path (`os.UserConfigDir()/bob/config.yaml`) and data dir (`$XDG_DATA_HOME/bob` or `~/.local/share/bob`)
    - [x] `config.LoadFromFile(path)`: strict decoding (`KnownFields`), `~/` expansion, duration parsing, mapping onto `config.Config` (OpenAI* and legacy Gemini* fields)
    - [x] Permission check: reject `mode & 0o077 != 0` or foreign owner
    - [x] `config.ValidateCLI()` and CLI default `allowed_network_modes = [none, restricted, full]`
    - [x] `config.WriteTemplate(path)`: commented template, dir 0700, file 0600, refuses to overwrite
    - [x] Tests: valid file, unknown key, bad duration, missing api_key, insecure perms, missing file, `~` expansion, template round-trip (template + api_key loads cleanly)

## Phase 3: Tool Authority Helpers & `execute` Tool
- [x] Task: Toolset combinators in `internal/tools`
    - [x] `AllowlistToolset(base, names...)`: filters `Definitions`, rejects `Execute` for non-allowlisted names without calling base
    - [x] `CompositeToolset(sets...)`: merged definitions, routing by name, error on duplicate names
    - [x] Tests: filtering, rejection path, duplicate detection, `ReadOnly` flags preserved
- [x] Task: `ExecTool`
    - [x] Extract the shared exec-result formatter from `executeSandboxExec`; `sandbox_exec` output stays byte-identical (golden test)
    - [x] `tools.NewExecTool(mgr, userID, ExecToolInfo{Driver, ReadOnly, Network})` implementing `agentapi.Toolset` with the single `execute` tool (`ReadOnly: false`), timeout clamped to `MaxExecTimeout`
    - [x] Tests with a fake `sandbox.Driver`: success, non-zero exit, timeout, empty command, missing sandbox

## Phase 4: Pre-Approved Sandbox with Explicit Workspace
- [x] Task: Mount spec parsing (`internal/cli/mount.go`)
    - [x] `ParseMountSpec(spec, cwd, protected []string)`: `:ro`/`:rw` suffix, `.`/relative/`~/` resolution, `EvalSymlinks`, must be an existing dir
    - [x] Reject `/`, and any path equal to, an ancestor of or a descendant of the config/data dirs
    - [x] Table tests including symlinks pointing into protected dirs and paths containing `:`
- [x] Task: `sandbox.Manager.Provision`
    - [x] `ProvisionParams` + validation shared with `RequestSandbox` (driver availability, image allowlist, network policy, mandatory CIDRs)
    - [x] Explicit `WorkspaceDir` with a single whole-workspace mount; synchronous creation → `StatusRunning`; 1-per-user invariant
    - [x] Unit tests with a fake driver (no approval state, invalid driver/image/net, double provision, Close destroys)
- [x] Task: Driver support for external workspaces
    - [x] Bwrap: don't create or chmod a pre-existing host workspace; whole-workspace `--ro-bind`/`--bind` at `/workspace`
    - [x] Docker: bind an absolute host path directly; apply `HostDataDir` translation only under `DataDir`
    - [x] Integration tests (skipped when bwrap/docker are unavailable): ro mount write fails, rw mount write persists on host, `net none` has no egress

## Phase 5: CLI Frontend (`internal/cli`)
- [x] Task: CLI system prompt
    - [x] `prompt.RenderCLI(CLIPromptContext{Now, TZ, OS, Tools, Sandbox *SandboxInfo})`; golden tests for sandboxed and unsandboxed variants
- [x] Task: Progress renderer
    - [x] Implement the progress observer interface consumed by `fsm` (via `agentapi.ProgressObserver`) and write compact step lines to stdout through a mutex-guarded writer shared with the final answer; no-op variant when `Quiet`
    - [x] Styling only when stdout is a TTY; plain lines otherwise
    - [x] Tests with a buffer writer: step lines then a blank separator then the answer; quiet → answer only; concurrent read-only steps produce whole lines
- [x] Task: Frontend & runner
    - [x] `cli.Frontend` (`Bind`/`Deliver`) composing `AllowlistToolset` (web_search*, web_fetch, discover/load memories and skills) + optional `ExecTool`
    - [x] `cli.Run(ctx, rt, opts, stdout, stderr)`: build the `Turn` (identity `cli`/`local`, fresh RunID, stdin block), call `agent.Engine.Run`, print to stdout, map errors to exit codes
    - [x] Ensure cancellation marks the FSM run cancelled/failed (no lingering `running` rows)
    - [x] Tests with a fake LLM client: plain answer, tool call to `web_fetch`, rejected non-allowlisted tool, `execute` via fake driver, ctx cancel → 130

## Phase 6: `cmd/bob` Entrypoint
- [x] Task: Flag parsing and validation
    - [x] `run(args, stdin, stdout, stderr) int` using `flag.FlagSet`: short and long aliases (`-p/--prompt`, `-q/--quiet`, `-v/--verbose`), repeatable `--allow-domain`, `--init-config`
    - [x] Flag-matrix validation (sandbox/net/image require mount, allow-domain requires restricted, image requires docker); no `-p` → "interactive mode not implemented yet", exit 2
    - [x] `-p -` and piped-stdin handling with size cap; stdout TTY detection used for styling only (`-q` alone controls step visibility)
    - [x] Tests: default run prints steps + answer on stdout; `-q` prints only the answer; logs never appear on stdout
    - [x] Table tests for every usage error and the success path (with an injected runtime factory)
- [x] Task: Lifecycle wiring
    - [x] Signal handling (SIGINT/SIGTERM → cancel → teardown → 130); slog handler on stderr (WARN / DEBUG with `-v`)
    - [x] No FSM `Start`/`Recover`, knowledge worker, GEOIP or backup in one-shot mode
    - [x] Makefile: `build-bob` target; check whether the Dockerfile needs changes (expected: none)

## Phase 7: Documentation & Verification
- [x] Task: Docs
    - [x] README: CLI usage, config reference, security notes (0600, mount guard, flags = consent), knowledge-store caveat
    - [x] Update `GEMINI.md`/`AGENTS.md` "Running the Agent Locally" and `conductor/tech-stack.md` (yaml.v3)
- [x] Task: Full verification
    - [x] `go test -race ./...` and `make check` clean (lint, semgrep, osv-scanner)
    - [x] Verify `internal/agent`, `internal/agentapi`, `internal/fsm`, `internal/tools`, `internal/cli` don't import `internal/gateway`
    - [x] Manual smoke tests: unsandboxed web question; `--mount .` read-only listing; `--mount .:rw` file creation; `--net restricted --allow-domain`; Ctrl-C during a tool call; `--init-config` + insecure-perms rejection
