---
id: IJ-2026-06-12-022
date: 2026-06-12
tiers:
  - tier-1
scope: TOOLS
plan_ref:
  - TOOL-ENG-7
  - TOOL-ENG-10
status: completed
commit: c63dd381
adr: none — extends existing governance and AHI routing patterns
---

# TOOLS S8 closeout — HIGH+ verify enforcement and AHI tool mode hook

## Operator request

Continue Tools layer completion iteratively: close remaining S8 rows TOOL-ENG-7 and TOOL-ENG-10.

## Summary

Extended `run_post_tool_verify` to raise `ToolVerificationRequiredError` when `enforce_high_risk_tool_verify` is set (default follows `production_mode`). Added `RuntimeState.high_risk_tool_approvals` bypass. Shipped AHI `ToolEngineHook`, `recommend_tool_modes` per-run resolver, `ToolEngineSelectionEngine` L4 proposals, and `check_tool_engine_ahi_hook.py`.

## Project impact

HIGH/CRITICAL tool invocations can be blocked pending explicit approval. Product hosts with adaptive routing tuning get automatic L6/L2a mode selection by catalog scale.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/project/architecture/TOOLS.md` §post-invoke · §AHI dynamic mode |
| Plan | `docs/project/maintainers/plans/TOOLS.md` S8 — Phase TOOL-ENG closed |

## Changed artifacts

- `intergrax/runtime/nexus/tools/tool_verify_hooks.py`
- `intergrax/runtime/nexus/tools/adaptive_tool_mode_resolver.py`
- `intergrax/runtime/adaptive/tool_engine_selection_engine.py`
- `intergrax/applications/_shared/tool_engine_wiring.py`
- `scripts/maintenance/check_tool_engine_ahi_hook.py`

## Verification

```bash
uv run pytest tests/unit/runtime/nexus/tools/ -q
uv run python scripts/maintenance/check_tool_engine_ahi_hook.py
```

Result: 58 passed · AHI hook gate OK.

## Risks and follow-ups

- Optional L1 critic (`eval.judge`) on tool output remains a separate CVL scope — not wired into default tool loop.
- `recommend_tool_modes` is rule-based v1; richer AHI signals can extend the resolver without changing the hook contract.
