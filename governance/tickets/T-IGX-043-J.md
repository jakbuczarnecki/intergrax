# T-IGX-043-J — ARCH-FREEZE remediation PT-WF (W-F harness and evaluation wave)

## Scope

Delivery ticket for master-plan node **PT-WF** (WP-12 W3, AR-44): tenant **G-09** fail-closed on harness evaluation wiring (**EX-L-09**), owner-repair batch for FRZ rows in WP-4 section C (OWN/CTR/DOC-05/DEBT/PLG/PRD/SEC-04/TEN families), and Campaign-B-conditional fixes **E2–E4** when B-1 confirms.

Primary write set (from `WP12_TICKETS_DRAFT.md` row PT-WF): `intergrax/applications/_shared/harness_host_runtime.py`, `evaluation_wiring.py`, `intergrax/runtime/execution/runtime.py`, `intergrax/compat/langchain/documents.py`, `intergrax/experiments/workflow.py`, `intergrax/experiments/store.py`, `intergrax/debug/router.py`, plus negative tests and regression nodes defined in the plan.

**Serialization:** overlaps `harness_host_runtime.py` with **PT-FC1** and **T-IGX-051-A (TENANT)** — do not edit that file in parallel with TENANT; coordinate merge order on `development`.

**Train reference:** implementation may mirror `arch-freeze/rw-train` when PT-WF commits land on train; do not merge `rw-train` into `development` without operator mandate.

This scaffold records scope only. No PASS, FRZ promotion, or independent-verification claim.

## Plan digest

Plan digest: 3f0265cdddf89229ec4921274e4cce4356574aba9b54cb887a5faf2bf786277d

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-043-J.md`](plans/PLAN_T-IGX-043-J.md).

## Approved

Approved: operator @ 2026-11-21T12:00:00+01:00

Approval source: OD-AF-AUTO-03 (autonomy session 2026-11-21); digest verified against canonical plan file.



## Test contract

Defined in the plan (red-first negative tests, regression node list, `uv run python -m scripts.governance.igx_gov ticket` before push). Harness qualification re-run: plan §8.1 — **12 passed** on `test_harness_evaluation_wiring.py` (2026-11-21, `af-dev-pt-wf` @ `bab8114d8`).

## Implementation record (fala B2)

| Step | Evidence |
|------|----------|
| G-09 harness noop | plan §8.1 — **12 passed**; no E2–E4 product paths touched |
| G-09 fala 5 close | plan §8.1 re-run @ `e61e5136a` — **12 passed** (development tor; E2–E4 untouched) |
| E2–E4 | **BLOCKED** on Campaign B-1 (continuation cross-ref plan §8.2) |
| wf-exec doc slice @ `e69e1ff58` | plan §9 — provisional `NT-WF-NEG-E2/3/4-01` anchors in `WF_E2_E4_STATUS.md`; **no** E2–E4 product diff; harness 12 **not** re-run |


## P-R0-6 checkpoint (part 1 closed; part 2 open)

| Slice | Status | Evidence |
|-------|--------|----------|
| **PT-WF F01–F04** (part 1) | **Published** on development @ 2c6d7af1a | Same four paths as candidate 216570370: intergrax/debug/router.py, intergrax/experiments/workflow.py, 	ests/unit/debug/test_wf_debug_tenant_authority.py, 	ests/unit/experiments/test_experiment_workflow.py — not a PASS or FRZ promotion |
| **Part 2** (E2 + owner-repair batch) | **OPEN** | MEP §F E2–E4 and WP-4 owner batch per plan §3; tracks WF_E2_E4_STATUS.md; Campaign B-1 still gates product E2 |

## Out of scope

TENANT **T-IGX-051-A**, FC-1/FC-3 file overlap without declared merge order, `scripts/` and `.github/` unless a later plan amendment names them, FRZ promotion, FCS, merge to `arch-freeze/rw-train`, `ARCH_FREEZE_CLOSED`.

## Lifecycle state

in_progress — part 1 (F01–F04) published @ `2c6d7af1a`; part 2 (E2 + owner batch) open; no ticket closure

## RW traceability (GA.2R)

Traceability only (RW-PLAN-AUDIT finding M7, ticket T-IGX-078-B). This section grants no write authority and changes no scope, plan, plan digest, approval or lifecycle state.

| FRZ row of the RW CODE_FIX inventory | Negative test ids in the RW plan |
|---|---|
| FRZ-BND-06 | NT-BND-06-1 |
| FRZ-OWN-02 | NT-OWN-02-P, NT-OWN-02-1 |
| FRZ-PLG-01 | NT-PLG-01-P, NT-PLG-01-1 |
| FRZ-PLG-02 | NT-PLG-02-P, NT-PLG-02-1, NT-PLG-02-2 |
| FRZ-PLG-06 | NT-PLG-06-P, NT-PLG-06-1 |
| FRZ-PLG-07 | NT-PLG-07-P, NT-PLG-07-1 |
| FRZ-PRD-08 | NT-PRD-08-P, NT-PRD-08-1, NT-PRD-08-2 |
| FRZ-TEN-01 | NT-TEN-01-1, NT-TEN-01-2, NT-WF-A01 |

Planned test files that the write set extracted from this ticket's plan does not contain: `tests/unit/applications/test_wf_tenant_evaluation_fail_closed.py`, `tests/unit/architecture/test_rw_gates_contract_census.py`, `tests/unit/architecture/test_rw_gates_plugin_contracts.py`, `tests/unit/architecture/test_rw_gates_production_only_bypass.py`, `tests/unit/architecture/test_rw_gates_transport_semantics.py`. The plan does not cover them yet.

FIND-AF2 mapping that names this ticket: FIND-AF2-09.

Binding: `docs/project/maintainers/qualification/arch_freeze_program/rw_plan_bindings.v1.json`. Lane, hard predecessors, merge position and mandatory tests of this ticket are in the live plan `docs/project/maintainers/qualification/arch_freeze_program/RW_PLAN.md`.
