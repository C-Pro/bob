# Track Index: CLI One-Shot Frontend (`bob -p`)

## Summary
- **Track ID:** `cli-oneshot-frontend_20261008`
- **Type:** Feature
- **Status:** Complete
- **Created:** 2026-10-08
- **Depends on:** `core-engine-decoupling_20261006`
- **Description:** New `bob` binary that executes a single agent run for `-p <prompt>` and prints the answer to stdout. Configuration comes from `~/.config/bob/config.yaml` (shared with the future TUI). Web and read-only knowledge tools are on by default with no sandbox. `--mount PATH[:ro|:rw]` provisions a pre-approved bwrap (default) or docker sandbox and enables the `execute` tool.

## Key Artifacts
- **Specification:** [spec.md](spec.md)
- **Implementation Plan:** [plan.md](plan.md)
- **Metadata:** [metadata.json](metadata.json)

## Key Decisions
- Separate `cmd/bob` binary. `cmd/agent` stays the env-configured Besedka daemon. `bob` without `-p` is reserved for the TUI.
- YAML config only (no env overrides or expansion); file mode 0600 and ownership enforced; strict keys.
- Single workspace mount `--mount PATH[:ro|:rw]` (`.` = CWD, default `ro`) at `/workspace`. Mounts that overlap the config/data dirs are rejected.
- Network: `--net none|restricted|full` (default `none`), `--allow-domain` repeatable for `restricted`.
- Memory tools: read-only knowledge tools only (`discover_*`, `load_*`). No `propose_*` and no `recall_memory`.
- Shared `internal/bootstrap` composition root for daemon and CLI.
- Intermediate steps are streamed to stdout by default; `-q/--quiet` leaves only the final answer on stdout. Logs always go to stderr.

## Scope & Non-Goals
- **In Scope:** `internal/bootstrap`, YAML config loader, toolset combinators + `execute` tool, `sandbox.Manager.Provision` with explicit workspace, `internal/cli` frontend, `cmd/bob`.
- **Out of Scope:** TUI and multi-turn sessions, approvals for proposed knowledge, CLI run recovery, multiple mounts, env-based CLI config.

## Progress Log
- **2026-10-08:** Track initialized. Spec and plan drafted after design Q&A (mount syntax, tool set, network flags, binary layout, config security).
