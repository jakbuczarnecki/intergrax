# T-IGX-052-H — ARCH-FREEZE W6 M3 pin prep (PT-Q11 scaffold)

## Scope

Governance refinement only @ `e69e1ff58`. SSOT for **PT-Q11** (EBH-7 final pin + independent B6), the last W6 event before W7 `LOCK`. Read-only maps to [`M3_PREP_CHECKLIST.md`](../../docs/project/maintainers/qualification/arch_freeze_program/continuation/M3_PREP_CHECKLIST.md) and MEP-AF2 W6 exit criteria. **This batch edits only this ticket and its plan.** Does **not** pin M3, does not set `FREEZE_CERTIFICATION_SHA`, does not run the EBH-7 pin commit, does not edit qualification docs.

Per MEP §K / M3P-Q11-4: F/G/H governance scaffolds should be refined and approved **before** any M3 pin batch.

## Plan digest

Plan digest: e847c96a597ae116912b63725e6b7a2a415bbc2549438110a1d7d49bce255350

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-052-H.md`](plans/PLAN_T-IGX-052-H.md). Recompute with `scripts/governance/plan_digest.py` after any plan edit.

## Write set (this batch)

```text
governance/tickets/T-IGX-052-H.md
governance/tickets/plans/PLAN_T-IGX-052-H.md
```

## Approved

_(awaiting operator pre-pin packet; OD-AF-17 — governance refine may proceed in W6 exec lane without OD-AF-17 answer)_

## Test contract

- `uv run python -m scripts.governance.igx_gov ticket`

The identity probe cited in `M3_PREP_CHECKLIST.md` (MEP PD3-05) is for a future pin batch, not this refinement.

## Out of scope

Actual pin merge; W7 PT-Q13; `governance/tickets/` edits after M3; `ARCH_FREEZE_CLOSED`; product paths; pytest. No M3 PASS or LOCK claim.

## Lifecycle state

plan_pending