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

Approved: operator (OD-AF-AUTO-03, autonomy session 2026-11-21; digest verified against canonical plan file).

## Test contract

Defined in the plan (red-first negative tests, regression node list, `uv run python -m scripts.governance.igx_gov ticket` before push). Harness qualification re-run: plan §8.1 — **12 passed** on `test_harness_evaluation_wiring.py` (2026-11-21, `af-dev-pt-wf` @ `bab8114d8`).

## Implementation record (fala B2)

| Step | Evidence |
|------|----------|
| G-09 harness noop | plan §8.1 — **12 passed**; no E2–E4 product paths touched |
| G-09 fala 5 close | plan §8.1 re-run @ `e61e5136a` — **12 passed** (development tor; E2–E4 untouched) |
| E2–E4 | **BLOCKED** on Campaign B-1 (continuation cross-ref plan §8.2) |
| wf-exec doc slice @ `e69e1ff58` | plan §9 — provisional `NT-WF-NEG-E2/3/4-01` anchors in `WF_E2_E4_STATUS.md`; **no** E2–E4 product diff; harness 12 **not** re-run |

## Out of scope

TENANT **T-IGX-051-A**, FC-1/FC-3 file overlap without declared merge order, `scripts/` and `.github/` unless a later plan amendment names them, FRZ promotion, FCS, merge to `arch-freeze/rw-train`, `ARCH_FREEZE_CLOSED`.

## Lifecycle state

`in_progress` (§7 G-09 green; wf-exec §9 documents E2–E4 anchors; product E2–E4 BLOCKED on B-1; owner batch not started)