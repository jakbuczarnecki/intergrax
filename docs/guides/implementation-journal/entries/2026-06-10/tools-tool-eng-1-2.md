---
id: IJ-2026-06-10-004
date: 2026-06-10
tiers:
  - tier-1
scope: TOOLS
plan_ref:
  - TOOL-ENG-1
  - TOOL-ENG-2
status: completed
commit: aa0a3e61
adr: docs/adr/entries/2026-06-10/ADR-TOOL-001.md
---

# Catalog dispatch and full-gateway tool routing

## Operator request

Complete catalog-native tool dispatch: tools selected from the harness catalog must route through the full gateway path with explicit contracts, replacing legacy shortcut behaviour.

## Summary

Implemented catalog dispatch and full-gateway routing per TOOL-ENG-1/2. Documented decision in ADR-TOOL-001. Updated `tool_runtime`, planner services, and architecture/plan TOOLS pair.

## Project impact

Harness tool invocations follow one canonical gateway path — improves observability, policy enforcement, and auditability of tool calls across Tier-2 agents and Tier-3 hosts.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/TOOLS.md` |
| Plan | `docs/plan/TOOLS.md` TOOL-ENG-1, TOOL-ENG-2 |
| ADR | `docs/adr/entries/2026-06-10/ADR-TOOL-001.md` |

## Changed artifacts

- `intergrax/runtime/nexus/tools/tool_runtime.py`
- `docs/adr/entries/2026-06-10/ADR-TOOL-001.md`
- `docs/architecture/TOOLS.md`, `docs/plan/TOOLS.md`

## Verification

```bash
uv run pytest -m gate -q
```

Result: pass.

## Risks and follow-ups

- TOOL-ENG-4/11 (plan `tool_ids` constraints, context scope) — next wave on same branch series.
