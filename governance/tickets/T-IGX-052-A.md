# T-IGX-052-A — ARCH-FREEZE W5 ENT baseline pre-freeze (T-AF2-ENT-BASELINE-PREFREEZE)

## Scope

Material governance batch for MEP node **T-AF2-ENT-BASELINE-PREFREEZE** after **FCS** and **PT-S06**: freeze delegated ENT baseline chain checkpoints at tree **G2** without product commits on `intergrax/`, `agents/`, or `applications/`. Maps to [`WP12_W5_PT_TICKETS_PREP.md`](../../docs/project/maintainers/qualification/arch_freeze_program/continuation/WP12_W5_PT_TICKETS_PREP.md).

## Plan digest

Plan digest: pending

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-052-A.md`](plans/PLAN_T-IGX-052-A.md).

## Approved

_(awaiting operator wave packet after FCS)_

## Test contract

`uv run python -m scripts.governance.igx_gov ticket` (push-fast only). Scoped `verify_iiac_*` only when activation paths change — **no** full governance-fast locally.

## Out of scope

Product paths before G2 lock assertion; M3 pin; IIIC PASS without certifier session; FC-3 260 / trust 80 bundles.

## Lifecycle state

draft