---
id: IJ-2026-06-18-012
date: 2026-06-18
tiers:
  - tier-0
  - tier-1
scope: MODALITY
plan_ref:
  - MOD-MAINT-01
  - MOD-MAINT-02
  - MOD-MAINT-03
  - MOD-MAINT-04
status: completed
commit: 0b4c7543
adr: none — audit maintenance register only; test-fix scope tracked in MOD-MAINT-01/02
---

# MOD-MAINT-01..04 — Interactive layer 15 audit plan registration

## Operator request

Interactive layer-by-layer harness audit (Mode A2): confirm MODALITY verdict, register maintenance proposals including explicit failing test fixes, commit, advance to OBSERVABILITY.

## Summary

Layer 15 revalidation confirmed L3 architecture (W-ML Done, modality CI scripts green) with **two failing unit tests** in `tests/unit/model_inference/`. Registered MOD-MAINT-01/02 as **P2 test/code fixes** (not environment waivers) plus docs/depth rows in `docs/project/maintainers/plans/MODALITY.md` §6.1av.

## Project impact

Modality test hygiene backlog is actionable — failing tests must pass in standard dev/CI before layer closeout claims full green.

## Traceability

| Link | Target |
|------|--------|
| Plan | `docs/project/maintainers/plans/MODALITY.md` §6.1av |
| Audit result | `docs/audit_results/2026-06-18/MODALITY.md` |
| Failing tests | `test_opencv_vision.py`, `test_celery_modality_execution.py` |

## Verification

Doc-only iteration; modality CI gate scripts green; 2 unit tests red (tracked MOD-MAINT-01/02).
