---
id: IJ-2026-06-13-003
date: 2026-06-13
tiers:
  - tier-0
  - tier-1
scope: CODE_CRAFT
plan_ref:
  - ECC-7
  - ECC-8
  - ECC-9
  - S7
  - S8
  - S9
  - S10
status: completed
commit: 784d9dc9
adr: none — extends ADR-CODECRAFT-001
---

# S7–S10 — Code Craft post-closeout layer completion

## Operator request

Run Layer Completion Mode on CODE_CRAFT: audit, doc-first updates, sprint plan, iterative delivery with commits per sprint.

## Summary

Post ECC-0…ECC-6 audit found four actionable gaps (trace taxonomy, single-shot sandbox routing, health probe registration, CI gate). S7 updated domain pair with gap register; S8–S10 closed implementation gaps. Depth backlog remains: metrics dashboards, container isolation tier, dedicated codegen LLM wiring.

## Project impact

Code Craft layer reaches L3+ production parity for harness operators: full trace taxonomy on orchestration paths, consistent isolation routing on `codecraft.run`, registered `health.check_codecraft` probe, and `check_codecraft_layer.py` gate for regression protection.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/CODE_CRAFT.md` |
| Plan | `docs/plan/CODE_CRAFT.md` — S7–S10, GAP-ECC-16…19 |
| ADR | ADR-CODECRAFT-001 (no new ADR) |

## Changed artifacts

- `intergrax/runtime/codecraft/trace.py` — generation, test, verdict, HITL, promote steps
- `intergrax/runtime/codecraft/orchestrator.py` — trace hooks
- `intergrax/tools/providers/codecraft/service.py` — `resolve_craft_sandbox_session` on run path
- `intergrax/tools/providers/health/` — `health.check_codecraft` registration + mode probe fix
- `scripts/maintenance/check_codecraft_layer.py` — new CI gate
- `tests/unit/` — orchestrator trace test, cloud fallback test, health probe tests

## Verification

- `uv run pytest tests/unit/codecraft tests/unit/tools/providers/codecraft tests/unit/runtime/codecraft tests/unit/tools/providers/health/test_codecraft_probe.py` — 26 passed
- `uv run python scripts/maintenance/check_codecraft_layer.py` — OK

## Risks and follow-ups

- `codegen_llm_profile_ref` still uses template adapter (GAP-ECC-20)
- `container` isolation tier not implemented (GAP-ECC-21)
- Metrics dashboards §10.2 deferred (GAP-ECC-22)
