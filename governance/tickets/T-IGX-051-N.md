# T-IGX-051-N — ARCH-FREEZE T-AF2-RW-SUPPLY Parts A/B (supply strictness, `development` tor)

## Scope

Delivery ticket for **T-AF2-RW-SUPPLY** Parts **A/B** on the primary **`development`** tor (master-plan A.4; node **T-AF2-RW-SUPPLY-AB**). Land attestation-on-install/upgrade, mandatory `package_trust` composition, revocation serving stop, and lifecycle evidence consumer per [`PLAN_T-IGX-051-N.md`](plans/PLAN_T-IGX-051-N.md).

Findings (titles only): **EX-D1-01**, **EX-D1-02**, **EX-D1-04**, **PDC-D1-R017** — DS-EXC disposition `REMEDIATION_REQUIRED_BEFORE_FREEZE`. Parts C–I merged on train via PR **#676**; this ticket closes A/B on `development` before G2 requal.

**Review import (do not rewrite):** implementation draft on `origin/af2/rw-supply-ab` @ `d5d5471029b1530fe2847d66f4502db4332effb3` — cherry-pick or diff-import; plan revision 2 copied into `governance/tickets/plans/PLAN_T-IGX-051-N.md` at Runda 1 harden.

Write set (plan §5): `intergrax/agent_distribution/trust.py`, `intergrax/agent_distribution/admin_service.py`, `intergrax/applications/_shared/production_agent_capability_runtime.py`, `intergrax/applications/_shared/deploy_activate_pipeline.py` (state rename/docstring), `applications/governed_contractor_application/host/single_node_profile_v1_strict_startup.py`, plus P1–P3 admin config constructors and `tests/unit/agent_distribution/test_rw_supply_attestation_and_revocation.py` — **not** `harness_host_runtime.py` (disjoint from FC-1 / TENANT).

**Runda 1 operator defaults:** **D5** (**051-A** TENANT) **BLOCKED** (OD-AF-AUTO-04) — this ticket does not implement TENANT.

Scaffold hardened from MEP W3 prep — no PASS, FRZ promotion, or independent-verification claim.

## Plan digest

Plan digest: 9915282df991e6e7325dee8226ec0ae0eba59e53a64748371cb9610a31037356

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-051-N.md`](plans/PLAN_T-IGX-051-N.md) (imported from `af2/rw-supply-ab` @ `d5d54710`; recompute digest after any plan edit).

## Approved

PENDING — implementation review / wave packet; standing delegate approval on train branch **does not** authorize `development` product commits without a new approval line.

## Test contract

Defined in imported plan §6–7: supply negative tests **NT-RW-SUP-01..04**, red-first/revert-red discipline, pytest lock under `D:/Projekty/_wt/_locks/pytest.lock`; `uv run python -m scripts.governance.igx_gov ticket` before push when product paths change.

## Out of scope

Parts C–I of **051-B** (already on train), **051-A** TENANT (D5 BLOCKED), merge to `arch-freeze/rw-train` without green `governance-fast`, new implementation from blank tree when review branch is importable, FRZ promotion, FCS.

## Lifecycle state

`plan_pending`