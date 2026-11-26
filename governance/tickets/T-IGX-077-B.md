# T-IGX-077-B — closed-world unknown dependency and frozen CW-12 baseline

## Scope

PT-G01 non-product gate. An in-universe import with no classified target fails the gate. No test-only prefix exception. CW-12 reads the frozen qualification baseline. Operator accepted this behavior on 2026-11-26 (OD-CW). This session implements the ticket and therefore does not write the `Approved:` line (ticket README R-3).

## Plan digest

Plan digest: 9dd36e8b1b446a933f22c94636b824394ace0db9ebde8633109ea0cd87e28514

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-077-B.md`](plans/PLAN_T-IGX-077-B.md).

## Test contract

`uv run pytest tests/unit/architecture/test_closed_world_tier_gate.py -q --tb=line`

Include one unknown-import failure and one proof that the live exclusion register does not change the CW-12 baseline. Do not edit the live register to hide a regression.

## Out of scope

Product trees, seam ledger, roadmap, FRZ checklist.

