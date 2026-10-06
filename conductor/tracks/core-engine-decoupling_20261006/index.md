# Track Index: Core Agent Engine Decoupling

## Summary
- **Track ID:** `core-engine-decoupling_20261006`
- **Type:** Refactor
- **Status:** New
- **Created:** 2026-10-06
- **Description:** Decouple Bob's core agent engine from the Besedka chat transport, establishing clear contracts for frontend authority over system prompts, tools, attachments, and progress reporting.

## Key Artifacts
- **Specification:** [spec.md](spec.md)
- **Implementation Plan:** [plan.md](plan.md)
- **Metadata:** [metadata.json](metadata.json)

## Scope & Non-Goals
- **In Scope:** Strictly splitting and decoupling the existing architecture. Extracting `internal/agent`, `internal/agentapi`, `internal/agentstore`, and `internal/commands`. Refactoring `internal/gateway` to become a consumer of `agent.Engine`.
- **Out of Scope:** Implementing new frontends (such as CLI `-p` or TUI). These will be developed in separate follow-up tracks once the core engine is decoupled and stabilized.

## Progress Log
- **2026-10-06:** Track initialized following pair-programming review with Codex Astra. Drafted neutral contracts, subsystem relocations, and execution recovery designs.
