# Intergrax - Agent Instructions (stub)

> **Doc map:** [`docs/project/technical/DOCUMENTATION_MAP.md`](docs/project/technical/DOCUMENTATION_MAP.md) - roles of instructional artifacts (human · operator · agent).

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

**Documentation pairs:** domain-layer `docs/project/architecture/<DOMAIN>.md` ↔ `docs/project/maintainers/plans/<DOMAIN>.md` (+ `satellites/` per tier, `.cursorignore`); multi-layer feature `docs/project/capabilities/architecture/<FEATURE>.md` ↔ `docs/project/capabilities/plan/<FEATURE>.md` (+ matching `satellites/` under each tier) - hub [`docs/project/capabilities/README.md`](docs/project/capabilities/README.md). Feature docs coordinate cross-layer delivery; domain ownership remains authoritative.

## Rule loading (Cursor · Claude Code)

**Single source of truth:** [`agent-env/`](agent-env/README.md) - rules, skills, plans. `.cursor/` and `.claude/` hold only generated adapters (`uv run python scripts/governance/sync_agent_env.py`). **Always-on:** [`intergrax-token-budget.md`](agent-env/rules/intergrax-token-budget.md) - I1/O1 token budget, mandatory preflight (+ other `loading: always` rules, see README).

**On demand only** - load only when explicitly referenced (Cursor `@file`, Claude Code `/<rule-id>`) or operator asks for that mode:

| Artifact | When |
|----------|------|
| [`intergrax-iteration.md`](agent-env/rules/intergrax-iteration.md) | New domain session, F3 workflow, iteration closeout |
| [`intergrax-hep-step.md`](agent-env/rules/intergrax-hep-step.md) | HEP / EVID implementation step |
| [`intergrax-ci-hotfix.md`](agent-env/rules/intergrax-ci-hotfix.md) | CI/test hotfix - new chat; no docs/arch |
| [`AGENT_INSTRUCTIONS.md`](docs/project/technical/guides/AGENT_INSTRUCTIONS.md) | Full routing, verification, ADR, Full O1 report |
| [`hep_step.txt`](docs/project/maintainers/bootstrap/hep_step.txt) | HEP step bootstrap (with operator step prompt) |

**Default behavior:** do not auto-load iteration rule, AGENT_INSTRUCTIONS, plan/arch hubs, or domain guides. If read scope exceeds operator-listed files → **STOP** and ask. **Quality workflow (hooks + CI):** [`ENTERPRISE_QUALITY_WORKFLOW.md`](docs/project/technical/guides/ENTERPRISE_QUALITY_WORKFLOW.md) - `uv run python scripts/dev/install_git_hooks.py`. **Cursor token setup:** [`CURSOR_TOKEN_SETUP.md`](docs/project/technical/guides/CURSOR_TOKEN_SETUP.md). **Do not delete this file via Cursor Settings.**

## Delivery DevOS (ticket-first)

- **Tickets SSOT:** [`governance/tickets/`](governance/tickets/) — every product-path batch needs `T-IGX-<NN>-<L>.md`.
- **Commit trailer:** `Ref: governance/tickets/T-IGX-012-A.md` on commits touching `intergrax/`, `agents/`, `applications/`, `scripts/`.
- **Default agent role:** `orchestrator` — see [`governance/agent-control-plane/role_registry.yaml`](governance/agent-control-plane/role_registry.yaml).
- **Quality before push:** `uv run python -m scripts.governance.igx_gov ticket` (wraps ticket Ref check + regression `fast`).
- **Hub:** [`docs/project/technical/devos/README.md`](docs/project/technical/devos/README.md).
