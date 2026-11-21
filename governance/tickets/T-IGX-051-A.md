# T-IGX-051-A — ARCH-FREEZE T-AF2-RW-TENANT (G-09 / tenant fail-closed wave)

## Scope

Delivery ticket for master-plan node **T-AF2-RW-TENANT** / residuals **EX-L-02**, **EX-L-09**, **EX-D4-04**, **EX-D5-05**, **EX-D8-03**, **EX-D8-04** (tenant-bound fixes; **G-09** primary path coordinated with **T-IGX-043-J**).

Primary write surface includes `intergrax/applications/_shared/harness_host_runtime.py` — **serializes after** **T-IGX-043-A**, **043-N**, and **043-J** G-09 slice per [`WP12_W3_PT_TICKETS_PREP.md`](../../docs/project/maintainers/qualification/arch_freeze_program/continuation/WP12_W3_PT_TICKETS_PREP.md).

**Operator gate:** OD-AF-AUTO-04 / OD-AF-AUTO-05 — **BLOCKED** at MEP W3 prep; no product commits without explicit operator grant.

## Plan digest

PENDING — plan file to be attached from RW-PLAN / FRZ matrix row.

## Approved

PENDING — blocked on operator autonomy decisions for TENANT wave.

## Test contract

To be defined: tenant propagation negatives, harness host STRICT/PRODUCT refusals, regression aligned with **PT-WF** bundle; `uv run python -m scripts.governance.igx_gov ticket` before push.

## Out of scope

Trust **051-C** (merged off-pin), SUPPLY **051-N**, train merge, RW milestone claim, FCS, `ARCH_FREEZE_CLOSED`.

## Lifecycle state

draft