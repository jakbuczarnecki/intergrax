# T-IGX-052-F — ARCH-FREEZE W6 Campaign B omega prep (PT-B1W, PT-B2W, PT-BA, PT-D13)

## Scope

Governance refinement only @ `e69e1ff58`. Defines Delivery DevOS SSOT for MEP-AF2 **W6** campaign nodes **PT-B1W**, **PT-B2W**, **PT-BA**, and **PT-D13** on the `development` tor after **FCS**. Read-only maps to [`W6_CAMPAIGN_PREP_CHECKLIST.md`](../../docs/project/maintainers/qualification/arch_freeze_program/continuation/W6_CAMPAIGN_PREP_CHECKLIST.md) and [`W6_PREP_DOSSIER.md`](../../docs/project/maintainers/qualification/arch_freeze_program/continuation/W6_PREP_DOSSIER.md). **This batch edits only this ticket and its plan** — no campaign execution, no auditor session, no omega @ FCS evidence, no qualification doc changes.

Sibling scaffolds: T-IGX-052-G (requal + PT-C01), T-IGX-052-H (M3 pin prep). Letters `T-IGX-052-A`..`E` remain W5 finals per EDER-N-0004.

## Plan digest

Plan digest: 83de1dbde3ede67b3310e57e1fd89b2c9b2b4ebc9ddf441856f51119eddb0b88

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-052-F.md`](plans/PLAN_T-IGX-052-F.md). Recompute with `scripts/governance/plan_digest.py` after any plan edit.

## Write set (this batch)

```text
governance/tickets/T-IGX-052-F.md
governance/tickets/plans/PLAN_T-IGX-052-F.md
```

## Approved

_(awaiting operator wave packet for W6 governance refinement)_

## Test contract

Governance gate only (no pytest, no docs pair checker at refine time):

- `uv run python -m scripts.governance.igx_gov ticket`

## Out of scope

Campaign runs; `tests/` at FCS; FRZ promotion; M3 pin; W7 lock; product paths; `scripts/` changes; edits to `W6_CAMPAIGN_PREP_CHECKLIST.md`; `PT-C01` and `PT-Q*` nodes (T-IGX-052-G/H). No PASS, certification, or independent-verification claim.

## Lifecycle state

plan_pending