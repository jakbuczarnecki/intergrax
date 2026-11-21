# T-IGX-043-N — FC-1 residual: human decision store → ToolWiringContext (NT-FC1-26 case 2)

## Scope

Delivery ticket for master-plan node **PT-FC1** follow-on (**standing F-4** in continuation plan A.4): close **NT-FC1-26 case 2** — wire human decision store through `ToolWiringContext` without a second authority store on the host path.

Prerequisite: **T-IGX-043-A** (FC-1) landed or reviewed on `development` (PR **#680** / digest @ `a6999e97`); do not duplicate FC-1 Part B work already in 043-A.

Tor: `development` only. Overlaps **T-IGX-043-J** and **T-IGX-051-A** on `harness_host_runtime.py` — serialize per [`WP12_W3_PT_TICKETS_PREP.md`](../../docs/project/maintainers/qualification/arch_freeze_program/continuation/WP12_W3_PT_TICKETS_PREP.md).

Scaffold only — no PASS, FRZ promotion, or independent-verification claim.

## Plan digest

PENDING — plan file not attached at scaffold time (`plan_pending`).

## Approved

PENDING — requires wave-packet approval after **T-IGX-043-A** review closes.

## Test contract

To be defined with plan: red-first **NT-FC1-26** case 2 at baseline SHA; regression nodes from FC-1 bundle; `uv run python -m scripts.governance.igx_gov ticket` before push when product paths change.

## Out of scope

Full FC-1 rewrite, TENANT **051-A**, train merge, FRZ promotion, FCS, `ARCH_FREEZE_CLOSED`.

## Lifecycle state

draft