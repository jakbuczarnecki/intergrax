---
id: IJ-2026-06-13-010
date: 2026-06-13
tiers:
  - tier-0
  - tier-1
scope: CODE_CRAFT
plan_ref:
  - S11
  - GAP-ECC-23
  - RUN-ECC-01
status: completed
commit: 08a0c59f
adr: none — doc sync and exec budget enforcement; no contract change
---

# S11 — Code Craft layer completion II (audit sync + exec budget)

## Operator request

Close the gap where Ephemeral Code Craft reads as architecture-only in audit prompts while `codecraft.*` tools and `wire_application_codecraft()` are already shipped runtime.

## Summary

Regenerated `docs/audit_results/CODE_CRAFT.md` from updated `generate_domain_audit_prompts.py` (ECC-0…S11 Done, depth backlog only). Enforced `max_total_exec_time_s` fail-closed in `CodeCraftOrchestrator.iterate` and capped single-shot `codecraft.run` timeout. Registered GAP-ECC-23 for per-task `Task.metadata.codecraft_mode` override.

## Project impact

Audit agents and operators now see honest L3+ runtime status. Cumulative sandbox exec budget is enforced on iteration paths, closing RUN-ECC-01 production gap.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/project/architecture/CODE_CRAFT.md` §6.3 GAP-ECC-23 note |
| Plan | `docs/project/maintainers/plans/CODE_CRAFT.md` — layer completion audit II, S11 sprint |
| ADR | ADR-CODECRAFT-001 — no amendment |
| Audit / gap | DOC-ECC-01/02, RUN-ECC-01, GAP-ECC-23 |

## Changed artifacts

- `scripts/audit/generate_domain_audit_prompts.py` — CODE_CRAFT domain Done status
- `docs/audit_results/CODE_CRAFT.md` — regenerated audit prompt
- `intergrax/codecraft/profile.py` — exec budget helpers
- `intergrax/runtime/codecraft/orchestrator.py` — budget deny + timeout cap
- `intergrax/tools/providers/codecraft/service.py` — single-shot timeout cap
- `tests/unit/runtime/codecraft/test_orchestrator.py` — budget test
- `docs/project/maintainers/plans/CODE_CRAFT.md`, `docs/project/architecture/CODE_CRAFT.md` — audit II register

## Verification

```bash
uv run python scripts/maintenance/check_codecraft_layer.py
uv run pytest tests/unit/codecraft/ tests/unit/tools/providers/codecraft/ tests/unit/runtime/codecraft/ -q
```

Result: pass — 25 tests, gate OK.

## Risks and follow-ups

- GAP-ECC-20 codegen LLM profile wiring (P3)
- GAP-ECC-21 container isolation tier (P3)
- GAP-ECC-22 metrics dashboards (P3)
- GAP-ECC-23 Task.metadata.codecraft_mode override (P2)
