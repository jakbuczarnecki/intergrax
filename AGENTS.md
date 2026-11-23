# Intergrax - Agent Instructions (stub)

> **Doc map:** [`DOCUMENTATION_MAP.md`](docs/project/technical/DOCUMENTATION_MAP.md).

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

**Doc pairs:** `docs/project/architecture/<DOMAIN>.md` ↔ `docs/project/maintainers/plans/<DOMAIN>.md`; capability features under `docs/project/capabilities/` (hub [`README.md`](docs/project/capabilities/README.md)).

## Rule loading (Cursor · Claude Code)

**Single source of truth:** [`agent-env/`](agent-env/README.md) - rules, skills, plans. `.cursor/` and `.claude/` hold only generated adapters (`uv run python scripts/governance/sync_agent_env.py`). **Always-on:** [`intergrax-core.md`](agent-env/rules/intergrax-core.md), [`intergrax-english-only-commits.md`](agent-env/rules/intergrax-english-only-commits.md).

**On demand only** - load only when explicitly referenced (Cursor `@file`, Claude Code `/<rule-id>`) or operator asks for that mode:

| Artifact | When |
|----------|------|
| [`intergrax-iteration.md`](agent-env/rules/intergrax-iteration.md) | New domain session, F3 workflow, iteration closeout |
| [`intergrax-hep-step.md`](agent-env/rules/intergrax-hep-step.md) | HEP / EVID implementation step |
| [`intergrax-ci-hotfix.md`](agent-env/rules/intergrax-ci-hotfix.md) | CI/test hotfix - new chat; no docs/arch |
| [`AGENT_INSTRUCTIONS.md`](docs/project/technical/guides/AGENT_INSTRUCTIONS.md) | Full routing, verification, ADR, Full O1 report |
| [`hep_step.txt`](docs/project/maintainers/bootstrap/hep_step.txt) | HEP step bootstrap (with operator step prompt) |

**Default:** load on-demand rules only when referenced; max 3 extra reads with reason; out-of-scope edit = escalate (`intergrax-core.md`). Hooks/CI: [`ENTERPRISE_QUALITY_WORKFLOW.md`](docs/project/technical/guides/ENTERPRISE_QUALITY_WORKFLOW.md). Token setup: [`CURSOR_TOKEN_SETUP.md`](docs/project/technical/guides/CURSOR_TOKEN_SETUP.md).

## Delivery DevOS (ticket-first)

- **Tickets SSOT:** [`governance/tickets/`](governance/tickets/) — every product-path batch needs `T-IGX-<NN>-<L>.md`.
- **Commit trailer:** `Ref: governance/tickets/T-IGX-012-A.md` on commits touching `intergrax/`, `agents/`, `applications/`, `scripts/`.
- **Default agent role:** `orchestrator` — see [`governance/agent-control-plane/role_registry.yaml`](governance/agent-control-plane/role_registry.yaml).
- **Quality before push:** `uv run python -m scripts.governance.igx_gov ticket` (wraps ticket Ref check + regression `fast`).
- **Hub:** [`docs/project/technical/devos/README.md`](docs/project/technical/devos/README.md).
