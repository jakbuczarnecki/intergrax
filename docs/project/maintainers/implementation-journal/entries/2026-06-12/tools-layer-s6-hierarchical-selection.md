---
id: IJ-2026-06-12-020
date: 2026-06-12
tiers:
  - tier-1
scope: TOOLS
plan_ref:
  - TOOL-ENG-14
  - TOOL-ENG-15
  - TOOL-ENG-26
  - TOOL-ENG-31
  - TOOL-ENG-32
  - TOOL-ENG-7
  - TOOL-ENG-8
  - TOOL-ENG-12
status: completed
commit: 818bd174
adr: docs/project/technical/adr/entries/2026-06-12/ADR-TOOL-005.md
---

# TOOLS S6 + S8 partial — hierarchical selection, plugins, governance hooks

## Operator request

Continue Tools layer completion from S5: close selection plugin surfaces, hierarchical mode, selection telemetry, and governance rows TOOL-ENG-7/8/12.

## Summary

Shipped deterministic hierarchical selection (`HierarchicalToolSelectionStrategy`, `hierarchical_tool_selector.py`), `RuntimeConfig.tool_selection_strategy` override, entry-point loader `tool_selection_registry.py`, and `ToolSelectionDiagV1` trace emission. Completed `keyword_top_k` alias tests. Added governance: `ToolsRequiredError` on empty required-mode runs, `tool_choice_for_mode` wiring in patterns, `emit_high_risk_tool_verify_signal` with typed `ToolVerifyRequiredDiagV1`.

## Project impact

Hosts can narrow tools by category tree, inject custom selection strategies, and observe selection candidates in traces. Required tools_mode fails hard; high-risk invokes emit verify signals for CVL follow-up.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/project/architecture/TOOLS.md` §hierarchical · §selection plugin |
| Plan | `docs/project/maintainers/plans/TOOLS.md` S6, S8 partial |
| ADR | `docs/project/technical/adr/entries/2026-06-12/ADR-TOOL-005.md` |

## Changed artifacts

- `intergrax/runtime/nexus/tools/hierarchical_tool_selector.py`
- `intergrax/runtime/nexus/tools/tool_selection_registry.py`
- `intergrax/runtime/nexus/tracing/tools/tool_selection.py`
- `intergrax/runtime/nexus/tools/tool_verify_hooks.py`
- `intergrax/runtime/nexus/errors/tools_required_error.py`
- `intergrax/runtime/nexus/tools/tool_planning_policy.py`
- `tests/unit/runtime/nexus/tools/test_hierarchical_tool_selector.py`
- `tests/unit/runtime/nexus/tools/test_tool_selection_registry.py`

## Verification

- `uv run pytest tests/unit/runtime/nexus/tools/ -q` — 49 passed

## Risks and follow-ups

- TOOL-ENG-10 (AHI dynamic mode) and full CVL approval gate for TOOL-ENG-7 remain open.
- Hierarchical v1 is deterministic keyword rank — LLM category pass deferred per ADR-TOOL-005.
- Next sprint: S7 (`DeterministicChainPattern`, invocation EP registry, CI gate).
