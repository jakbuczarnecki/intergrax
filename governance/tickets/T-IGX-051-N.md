# T-IGX-051-N — ARCH-FREEZE T-AF2-RW-SUPPLY Parts A/B (supply strictness)

## Scope

Delivery ticket for **T-AF2-RW-SUPPLY** Parts **A/B** on the `development` tor (master-plan A.4). Implementation exists on branch @ `d5d54710` — **review digest only**, do not rewrite from scratch.

Covers supply-chain residuals **EX-D1-01**, **EX-D1-02**, **EX-D1-04**, **EX-D6-01**, **EX-D6-03**, **EX-D8-01** (see continuation plan matrix). Parts C–I already merged on train (**#676**); this ticket lands A/B on primary tor before G2 requal.

Write set (from existing branch review): `trust.py`, `admin_service.py`, launchers — **not** `harness_host_runtime.py` (disjoint from FC-1 / TENANT).

## Plan digest

PENDING — attach `PLAN_T-IGX-051-N.md` and compute digest after review import from branch `d5d54710`.

## Approved

PENDING — implementation review session; not granted at MEP W3 scaffold.

## Test contract

Defined in imported plan: supply negative tests, strict admission paths, `uv run python -m scripts.governance.igx_gov ticket` before push.

## Out of scope

New supply implementation from blank tree, **051-A** TENANT, merge to `arch-freeze/rw-train` without green `governance-fast`, FRZ promotion, FCS.

## Lifecycle state

draft