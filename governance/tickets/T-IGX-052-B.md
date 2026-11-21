# T-IGX-052-B — ARCH-FREEZE W5 freeze manifests final (T-AF2-MANIFESTS)

## Scope

Documentation and maintenance batch regenerating FRZ-FRZ-05..08 manifests from the **G2** tree (successor to **T-IGX-050-G** drafts), plus drift-gated maintenance under `scripts/maintenance/` when authorized. Governance tickets and manifest JSON under `docs/project/maintainers/qualification/arch_freeze_program/manifests/`.

## Plan digest

Plan digest: pending

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-052-B.md`](plans/PLAN_T-IGX-052-B.md).

## Approved

_(awaiting operator wave packet after FCS)_

## Test contract

`uv run python -m scripts.governance.igx_gov ticket` (push-fast). If `scripts/maintenance` changes: targeted manifest gate tests named in plan at approval — **not** FC-3 260.

## Out of scope

FRZ promotion; accepting manifests (W7); product refactors for manifest rows; trust 80 bundle.

## Lifecycle state

draft