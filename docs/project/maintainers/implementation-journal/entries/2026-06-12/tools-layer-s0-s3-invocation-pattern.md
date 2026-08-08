---
id: IJ-2026-06-12-017
date: 2026-06-12
tiers:
  - tier-0
  - tier-1
scope: TOOLS
plan_ref:
  - TOOL-ENG-16
  - TOOL-ENG-17
  - TOOL-ENG-18
  - TOOL-ENG-21
  - TOOL-ENG-22
  - TOOL-ENG-23
  - TOOL-ENG-TEST.1
status: completed
commit: pending
adr: docs/project/technical/adr/entries/2026-06-12/ADR-TOOL-003.md
---

# TOOLS layer completion S0–S3 — ToolInvocationPattern protocol and shipped patterns

## Operator request

Run Layer Completion Mode on the Tools domain: audit Tier-0 catalog and Tier-1 tool engine, update documentation first, then implement sprint plan closing orchestration gaps (TOOL-ENG-16–23).

## Summary

Delivered layer-completion audit + sprint register in plan/architecture docs (commit `e5ffb7db`). Implemented `ToolInvocationPattern` protocol (`tool_invocation_pattern.py`), shipped `SinglePassPattern` and `BoundedReactPattern`, `ToolInvocationMode` enum + `pattern_for_mode()` factory, `run_bounded_tool_loop` delegation, and `ApplicationEnvironmentProfile.tool_invocation_mode` bridge. ADR-TOOL-003 accepted. Fixed invoker unit test regression (FakeRegistry handler) and added missing `test_tool_selection_strategy.py`.

## Project impact

Hosts can configure tool orchestration mode (`single_pass` / `bounded_react`) via environment profile; multi-call execution is no longer monolithic in `tool_loop.py`. Atomic invoke path unchanged. Foundation for parallel batch, chain, and semantic composite patterns (S4–S7).

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/project/architecture/TOOLS.md` §Invocation patterns · engine gap register |
| Plan | `docs/project/maintainers/plans/TOOLS.md` Phase TOOL-ENG · §Layer completion sprints |
| ADR | `docs/project/technical/adr/entries/2026-06-12/ADR-TOOL-003.md` |

## Changed artifacts

- `intergrax/runtime/nexus/tools/tool_invocation_pattern.py` — protocol + factory
- `intergrax/runtime/nexus/tools/patterns/` — SinglePass, BoundedReact
- `intergrax/runtime/nexus/tools/tool_loop.py` — pattern delegation
- `intergrax/runtime/nexus/config_types.py` — `ToolInvocationMode`
- `intergrax/runtime/nexus/config.py` — `tool_invocation_mode`
- `intergrax/applications/contracts/environment_profile.py` — host field
- `intergrax/applications/_shared/catalog_runtime_bridge.py` — bridge
- `intergrax/runtime/nexus/tools/plan_context_invocation.py` — wiring
- `tests/unit/runtime/nexus/tools/` — conformance + selection + invoker fixes
- `tests/unit/applications/test_catalog_runtime_bridge_tool_invocation.py` — bridge test
- `docs/project/maintainers/plans/TOOLS.md`, `docs/project/architecture/TOOLS.md` — status sync

## Verification

```bash
uv run pytest tests/unit/runtime/nexus/tools/ tests/integration/runtime/test_tool_loop_integration.py -q
uv run pytest -m gate -q
uv run python scripts/maintenance/check_harness_adr.py
```

## Risks and follow-ups

- S4–S8 still open: parallel batch, semantic index, governance (TOOL-ENG-7/8/12).
- `ParallelBatchPattern` / `DeterministicChainPattern` factory raises `NotImplementedError` until S4–S7.
