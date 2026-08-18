---
id: IJ-2026-06-13-001
date: 2026-06-13
tiers:
  - tier-0
  - tier-1
scope: CODE_CRAFT
plan_ref:
  - ECC-1
  - GAP-ECC-03
  - GAP-ECC-04
  - GAP-ECC-09
status: completed
commit: b79cb177
adr: none — extends ADR-CODECRAFT-001 per plan ECC-1
---

# ECC-1 — single-shot codecraft.run with StaticCodeGate and trace

## Operator request

Bring Ephemeral Code Craft from architecture-only (Planned) to first production slice: governed single-shot execution with static L0 gate, fail-closed policy, and CODECRAFT trace steps.

## Summary

Implemented Tier-0 `StaticCodeGate`, `CodeCraftProfile`, and `CraftResult` contracts; Tier-1 `CodeCraftTraceEmitter` with `TraceComponent.CODECRAFT`; catalog tool `codecraft.run` registered via `CodeCraftToolPlugin`. Tool routes through existing sandbox `code.exec` substrate; `codecraft.run` added to `SANDBOX_REQUIRED_TOOLS`.

## Project impact

Agents and hosts can invoke a harness-owned ephemeral code path with AST/import/size gates and typed results instead of ad-hoc Tier-2 loops. Foundation for ECC-2 session orchestration and ECC-3 Tier-3 profile wiring.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/project/architecture/CODE_CRAFT.md` §6, §16 |
| Plan | `docs/project/maintainers/plans/CODE_CRAFT.md` — Phase ECC-1 |
| ADR | `docs/project/technical/adr/entries/2026-06-10/ADR-CODECRAFT-001.md` |
| Audit / gap | GAP-ECC-03 (partial), GAP-ECC-04, GAP-ECC-09 (partial) |

## Changed artifacts

- `intergrax/codecraft/` — contracts, profile, static gate
- `intergrax/runtime/codecraft/trace.py` — CodeCraftTraceEmitter
- `intergrax/tools/providers/codecraft/` — codecraft.run tool bundle
- `intergrax/tools/registry/shipped_plugins.py` — CodeCraftToolPlugin
- `intergrax/runtime/sandbox/sandbox_runtime.py` — SANDBOX_REQUIRED_TOOLS
- `intergrax/runtime/nexus/tracing/trace_models.py` — TraceComponent.CODECRAFT
- `tests/unit/codecraft/`, `tests/unit/tools/providers/codecraft/` — gate and tool tests

## Verification

```bash
uv run pytest tests/unit/codecraft/ tests/unit/tools/providers/codecraft/ -q
uv run pytest -m gate -q
python scripts/maintenance/check_harness_no_getattr.py
python scripts/docs/check_docs_domain_pairs.py
python scripts/maintenance/check_harness_adr.py
```

Result: 17 codecraft tests pass; full gate **1980 passed** (739s); harness doc/ADR/getattr checks pass.

## Risks and follow-ups

- `CodeCraftProfile` passed via `ToolWiringContext.extras` until ECC-3 `wire_application_codecraft()`.
- ECC-2 orchestrator + iteration tools; ECC-3 HITL and promotion; full gate suite on CI recommended.
