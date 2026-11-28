# T-IGX-093-B — AHE W1-B01 cross-program integration

## Scope

Canonical integration plan: [`plans/PLAN_T-IGX-093-B.md`](plans/PLAN_T-IGX-093-B.md).

Coordination batch **BATCH-ENT-AHE-W1-B01-CROSS-01** integrates the independently qualified W1-B01-P AHE product candidate at commit `88b85392559783666c97878d6b620de7dc66517f` (qualified tree `95047e21166073d1023db334d7b5768087e291f7`) via draft PR **#814**, together with the exact future EBH-5 registry correction in `tests/qualification/ebh_5/_sqlite_touchpoint_registry.py` (one new `SqliteTouchpointRecord` for `intergrax/runtime/adaptive/agent_design_search/sqlite_archive.py`, classification INTERNAL).

Incorporation of the seven qualified AHE runtime and test files requires external integration branch authorization and an authorized integration mechanism; this preparation batch registers governance only.

## Plan digest

Plan digest: 5746462840ea01fe7a9e5acb2d96363ef72d0533c66a4f7acd35e809e21353b4

## Approved

Approved: operator @ 2026-11-28T17:24:20+01:00

OD-AHE-05 - APPROVED, variant A.

Approved plan digest: 5746462840ea01fe7a9e5acb2d96363ef72d0533c66a4f7acd35e809e21353b4

EBH-5 owner acceptance: INTERNAL classification for ADAS W1 LAB SQLite archive.

Authorized: preparation, publication and qualification of the integration candidate.

Historical exception: preserve b53b74a755395812b9d4f2f93754ca1b669e8eae without history rewriting.

All independent certification and final integration controls remain mandatory.

Timestamp records repository registration of the previously granted Operator decision.

## Test contract

**Preparation-only (this batch):**

- `uv run --no-sync python scripts/governance/plan_digest.py governance/tickets/plans/PLAN_T-IGX-093-B.md`
- `uv run --no-sync python -m scripts.governance.igx_gov ticket`

**Future integration (after Operator approval, EBH-5 owner acceptance, and integration authorization):**

- Focused ADAS regression: `uv run --no-sync pytest -q tests/unit/runtime/adaptive/agent_design_search/`
- EBH-5 SQLite accountability: `uv run --no-sync pytest -q tests/qualification/ebh_5/test_ebh_5_l2_integrity_gates.py::test_ebh5_q09_sqlite_touchpoints_fully_accounted`
- Adjacent adaptive regression: `uv run --no-sync pytest -q tests/unit/runtime/adaptive/ --ignore=tests/unit/runtime/adaptive/agent_design_search/`
- Ruff and Pyright where applicable on touched product paths
- Governance ticket gate: `uv run --no-sync python -m scripts.governance.igx_gov ticket`
- Applicable independent IIAC qualification per canonical protocol
- EVOL-X and Enterprise Audit per owner-controlled scopes
- Mandatory integration CI on the integration candidate

## Out of scope

Product redesign; production activation; governance gate or hook changes; automatic IIAC activation; unauthorized repairs in ENT-A, TEN, VBN, or EVOL-X external domains; unauthorized merge to development; W1-B01-D documentation implementation; W1-B02 implementation; modification of T-IGX-093-A or its approved plan; bypass of integrator target/kind allowlist.

## Lifecycle state

plan_approved