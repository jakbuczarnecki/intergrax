---
id: IJ-2026-06-12-021
date: 2026-06-12
tiers:
  - tier-1
  - tier-3
scope: TOOLS
plan_ref:
  - TOOL-ENG-20
  - TOOL-ENG-24
  - TOOL-ENG-27
  - TOOL-ENG-28
  - TOOL-ENG-30
status: completed
commit: c7ef461e
adr: none — extends ADR-TOOL-003 plugin model; no new Tier boundary
---

# TOOLS S7 — deterministic chain pattern and invocation plugin closeout

## Operator request

Commit S5–S6 work and continue Tools layer completion with S7: chain pattern, invocation entry points, pattern telemetry, CI gate, and lab DX.

## Summary

Shipped `ToolChainSpec` with field mapping, `DeterministicChainPattern`, invocation pattern entry-point loader, `ToolsSummaryDiagV1` pattern fields, `check_tool_invocation_patterns.py`, and `LAB_TOOL_INVOCATION_MODE` on lab host profile.

## Project impact

Hosts can run fixed tool pipelines without LLM between steps, load custom invocation patterns via entry points, observe pattern_id in tools trace, and validate shipped patterns in CI. Lab application documents all invocation modes via env.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/TOOLS.md` §Pattern 4 |
| Plan | `docs/plan/TOOLS.md` S7 |

## Changed artifacts

- `intergrax/runtime/nexus/tools/tool_chain_spec.py`
- `intergrax/runtime/nexus/tools/patterns/deterministic_chain.py`
- `intergrax/runtime/nexus/tools/tool_invocation_registry.py`
- `scripts/check_tool_invocation_patterns.py`
- `applications/lab_application/host/settings.py`

## Verification

```bash
uv run pytest tests/unit/runtime/nexus/tools/ applications/lab_application/lab_application_tests/host/test_lab_tool_invocation_mode.py -q
uv run python scripts/check_tool_invocation_patterns.py
```

Result: 59 passed · gate script OK.

## Risks and follow-ups

- TOOL-ENG-10 (AHI dynamic mode) and TOOL-ENG-7 CVL approval block remain open for full layer closeout.
