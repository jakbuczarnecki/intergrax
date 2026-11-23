# T-IGX-052-G — ARCH-FREEZE W6 Campaign C sample and family requal prep (PT-C01, PT-Q05..Q12)

## Scope

Governance refinement only @ `e69e1ff58`. SSOT for MEP-AF2 **W6** nodes **PT-C01**, **PT-Q05**, **PT-Q06**, **PT-Q07**, **PT-Q08**, **PT-Q09**, optional **PT-Q10**, and **PT-Q12**. Read-only maps to [`W6_REQUAL_PREP_CHECKLIST.md`](../../docs/project/maintainers/qualification/arch_freeze_program/continuation/W6_REQUAL_PREP_CHECKLIST.md), Campaign C rows in [`W6_CAMPAIGN_PREP_CHECKLIST.md`](../../docs/project/maintainers/qualification/arch_freeze_program/continuation/W6_CAMPAIGN_PREP_CHECKLIST.md), and [`W6_PREP_DOSSIER.md`](../../docs/project/maintainers/qualification/arch_freeze_program/continuation/W6_PREP_DOSSIER.md). **This batch edits only this ticket and its plan** — no independent audit PASS, no family replay, no qualification doc changes.

Campaign B omega nodes remain T-IGX-052-F; M3 pin prep remains T-IGX-052-H.

## Plan digest

Plan digest: 29ee86a4d9fbd5516e78a64b9d92e5f6b96b440d46808fa4b4172280328797e4

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-052-G.md`](plans/PLAN_T-IGX-052-G.md). Recompute with `scripts/governance/plan_digest.py` after any plan edit.

## Write set (this batch)

```text
governance/tickets/T-IGX-052-G.md
governance/tickets/plans/PLAN_T-IGX-052-G.md
```

## Approved

_(awaiting operator wave packet for W6 governance refinement)_

## Test contract

- `uv run python -m scripts.governance.igx_gov ticket`

## Out of scope

`PT-Q11` / M3 (T-IGX-052-H); W7/W8 nodes; product edits; IIIC certifier sessions; FRZ promotion; checklist file edits in this batch. No PASS or CLOSED language.

## Lifecycle state

plan_pending