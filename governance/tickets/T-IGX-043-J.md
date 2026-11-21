# T-IGX-043-J — ARCH-FREEZE remediation PT-WF (W-F harness and evaluation wave)

## Scope

Delivery ticket for master-plan node **PT-WF** (WP-12 W3, AR-44): tenant **G-09** fail-closed on harness evaluation wiring (**EX-L-09**), owner-repair batch for FRZ rows in WP-4 section C (OWN/CTR/DOC-05/DEBT/PLG/PRD/SEC-04/TEN families), and Campaign-B-conditional fixes **E2–E4** when B-1 confirms.

Primary write set (from `WP12_TICKETS_DRAFT.md` row PT-WF): `intergrax/applications/_shared/harness_host_runtime.py`, `evaluation_wiring.py`, `intergrax/runtime/execution/runtime.py`, `intergrax/compat/langchain/documents.py`, `intergrax/experiments/workflow.py`, `intergrax/experiments/store.py`, `intergrax/debug/router.py`, plus negative tests and regression nodes defined in the plan.

**Serialization:** overlaps `harness_host_runtime.py` with **PT-FC1** and **T-IGX-051-A (TENANT)** — do not edit that file in parallel with TENANT; coordinate merge order on `development`.

**Train reference:** implementation may mirror `arch-freeze/rw-train` when PT-WF commits land on train; do not merge `rw-train` into `development` without operator mandate.

This scaffold records scope only. No PASS, FRZ promotion, or independent-verification claim.

## Plan digest

Plan digest: 5b6a2fb26889aaadbc763ff10f66247b1076ab372b223d59ca5742f9f32559c8

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-043-J.md`](plans/PLAN_T-IGX-043-J.md).

## Approved

Approved: operator (OD-AF-AUTO-03, autonomy session 2026-11-21; digest verified against canonical plan file).

## Test contract

Defined in the plan (red-first negative tests, regression node list, `uv run python -m scripts.governance.igx_gov ticket` before push). No test run at scaffold time.

## Out of scope

TENANT **T-IGX-051-A**, FC-1/FC-3 file overlap without declared merge order, `scripts/` and `.github/` unless a later plan amendment names them, FRZ promotion, FCS, merge to `arch-freeze/rw-train`, `ARCH_FREEZE_CLOSED`.

## Lifecycle state

`plan_approved` (§7 filled fala 3; G-09 green; E2–E4 BLOCKED on B-1; owner batch not started)