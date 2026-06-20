# Intergrax — Agent Instructions (Cursor stub)

> **Full reference:** [`docs/guides/AGENT_INSTRUCTIONS.md`](docs/guides/AGENT_INSTRUCTIONS.md) — use `@docs/guides/AGENT_INSTRUCTIONS.md` for task routing, verification commands, ADR workflow, anti-patterns.
> **Always-on workflow:** [`.cursor/rules/intergrax-iteration.mdc`](.cursor/rules/intergrax-iteration.mdc). **Token setup:** [`docs/guides/CURSOR_TOKEN_SETUP.md`](docs/guides/CURSOR_TOKEN_SETUP.md)

**Intergrax** is a four-tier **Agent OS / Harness AI** runtime (Python 3.12, `uv`):

```text
Tier-0 intergrax/  · Tier-1 intergrax/runtime/  · Tier-2 agents/  · Tier-3 applications/
```

**Hard boundaries (never violate):**

```text
intergrax/ MUST NOT import from agents/ or applications/
agents/ MUST NOT import from applications/
applications/ MAY import from agents/ and intergrax/
```

**Session (F3):** one domain = one new chat. **Output (O1):** terse by default; full report only on request — [`AGENT_INSTRUCTIONS.md`](docs/guides/AGENT_INSTRUCTIONS.md) § Operator communication. **Docs hub:** [`docs/intergrax_runtime_architecture.md`](docs/intergrax_runtime_architecture.md). **Invariants:** [`docs/guides/SYSTEM_INVARIANTS.md`](docs/guides/SYSTEM_INVARIANTS.md). **Maturity:** [`docs/guides/MATURITY_TAXONOMY.md`](docs/guides/MATURITY_TAXONOMY.md). Default scope: gate maintenance in [`docs/plan/PLATFORM_FOUNDATION.md`](docs/plan/PLATFORM_FOUNDATION.md) hub; Phase K / §6.3 only with explicit operator approval.

**Do not delete this file via Cursor Settings** — the UI entry is the file itself; use the stub + full reference split instead (see CURSOR_TOKEN_SETUP F2).
