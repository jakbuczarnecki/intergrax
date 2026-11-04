# Intergrax - Agent Instructions (Cursor stub)

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

## Cursor rule loading

**Always-on:** [`.cursor/rules/intergrax-token-budget.mdc`](.cursor/rules/intergrax-token-budget.mdc) - I1/O1 token budget; mandatory preflight before implementation.

**On demand only** - load only when explicitly `@`-referenced or operator asks for that mode:

| Artifact | When |
|----------|------|
| [`.cursor/rules/intergrax-iteration.mdc`](.cursor/rules/intergrax-iteration.mdc) | New domain session, F3 workflow, iteration closeout |
| [`.cursor/rules/intergrax-hep-step.mdc`](.cursor/rules/intergrax-hep-step.mdc) | HEP / EVID implementation step |
| [`.cursor/rules/intergrax-ci-hotfix.mdc`](.cursor/rules/intergrax-ci-hotfix.mdc) | CI/test hotfix - new chat; no docs/arch |
| [`docs/project/technical/guides/AGENT_INSTRUCTIONS.md`](docs/project/technical/guides/AGENT_INSTRUCTIONS.md) | Full routing, verification, ADR, Full O1 report |
| [`docs/project/maintainers/bootstrap/hep_step.txt`](docs/project/maintainers/bootstrap/hep_step.txt) | HEP step bootstrap (with operator step prompt) |

**Default behavior:** do not auto-load iteration rule, AGENT_INSTRUCTIONS, plan/arch hubs, or domain guides. If read scope exceeds operator-listed files → **STOP** and ask.

**Enterprise quality workflow (hooks + CI):** [`docs/project/technical/guides/ENTERPRISE_QUALITY_WORKFLOW.md`](docs/project/technical/guides/ENTERPRISE_QUALITY_WORKFLOW.md) — `uv run python scripts/dev/install_git_hooks.py`.

**Token setup:** [`docs/project/technical/guides/CURSOR_TOKEN_SETUP.md`](docs/project/technical/guides/CURSOR_TOKEN_SETUP.md)

**Do not delete this file via Cursor Settings** - use stub + full reference split (CURSOR_TOKEN_SETUP F2).

## Delivery DevOS (ticket-first)

- **Tickets SSOT:** [`governance/tickets/`](governance/tickets/) — every product-path batch needs `T-IGX-<NN>-<L>.md`.
- **Commit trailer:** `Ref: governance/tickets/T-IGX-012-A.md` on commits touching `intergrax/`, `agents/`, `applications/`, `scripts/`.
- **Default agent role:** `orchestrator` — see [`governance/agent-control-plane/role_registry.yaml`](governance/agent-control-plane/role_registry.yaml).
- **Quality before push:** `uv run python -m scripts.governance.igx_gov ticket` (wraps ticket Ref check + regression `fast`).
- **Hub:** [`docs/project/technical/devos/README.md`](docs/project/technical/devos/README.md).

## Delivery DevOS (ticket-first)

- **Tickets SSOT:** [`governance/tickets/`](governance/tickets/) — every product-path batch needs `T-IGX-<NN>-<L>.md`.
- **Commit trailer:** `Ref: governance/tickets/T-IGX-012-A.md` on commits touching `intergrax/`, `agents/`, `applications/`, `scripts/`.
- **Default agent role:** `orchestrator` — see [`governance/agent-control-plane/role_registry.yaml`](governance/agent-control-plane/role_registry.yaml).
- **Quality before push:** `uv run python -m scripts.governance.igx_gov ticket` (wraps ticket Ref check + regression `fast`).
- **Hub:** [`docs/project/technical/devos/README.md`](docs/project/technical/devos/README.md).
