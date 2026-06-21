# Intergrax — Agent Instructions (Cursor stub)

> **Doc map:** [`docs/DOCUMENTATION_MAP.md`](docs/DOCUMENTATION_MAP.md) — roles of all instructional artifacts (human · operator · agent).  
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

**Token budget (I1/O1):** always-on [`.cursor/rules/intergrax-token-budget.mdc`](.cursor/rules/intergrax-token-budget.mdc). **Session (F3):** one domain = one new chat; HEP steps → [`docs/bootstrap/hep_step.txt`](docs/bootstrap/hep_step.txt). Full O1 → [`AGENT_INSTRUCTIONS.md`](docs/guides/AGENT_INSTRUCTIONS.md) § Operator communication. **Canon routing:** load domain docs on demand via [`AGENT_INSTRUCTIONS.md`](docs/guides/AGENT_INSTRUCTIONS.md) — not bulk guides (invariants, maturity, strategy are `.cursorignore`; grep SYS-INV IDs or `@` explicit). Default scope: gate maintenance in [`docs/plan/PLATFORM_FOUNDATION.md`](docs/plan/PLATFORM_FOUNDATION.md) hub read-scope; Phase K / §6.3 only with explicit operator approval.

**Do not delete this file via Cursor Settings** — the UI entry is the file itself; use the stub + full reference split instead (see CURSOR_TOKEN_SETUP F2).
