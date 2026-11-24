# PT-WF — E2, E3, E4 (Campaign B-1) — status on the development train

**Plan SSOT:** [governance/tickets/plans/PLAN_T-IGX-043-J.md](../../../../../../governance/tickets/plans/PLAN_T-IGX-043-J.md) §7.1  
**Ticket:** T-IGX-043-J (plan_approved; G-09 product slice **done** on earlier SHAs)

## Summary

| Layer | Paths (plan §3) | Node ids | Wave status |
|-------|-----------------|----------|---------------|
| **E2** | intergrax/runtime/execution/runtime.py | NT-WF-NEG-E2-01 — provisional anchor (below) | **BLOCKED** (Campaign B-1 not confirmed for slice) |
| **E3** | intergrax/compat/langchain/documents.py | NT-WF-NEG-E3-01 — provisional anchor (below) | **BLOCKED** |
| **E4** | intergrax/experiments/store.py, debug/router.py, experiments/workflow.py | NT-WF-NEG-E4-01 — provisional anchor (below) | **PARTIAL** — F01–F04 product delta @ 2c6d7af1a on workflow.py / 
outer.py only; **E2** and owner batch still **OPEN** (P-R0-6 checkpoint) |

**G-09** (separate PT-WF row): **DONE** on development — tests NT-WF-G09-01..03 green (	est_harness_evaluation_wiring.py).

## P-R0-6 checkpoint (T-IGX-043-J)

- **Part 1 (PT-WF F01–F04):** published on development @ 2c6d7af1a (four paths; see ticket implementation record). Not independent verification or FRZ promotion.
- **Part 2:** **OPEN** — MEP §F **E2** (
untime.py) plus the WP-4 owner-repair batch (	ests/unit/applications/test_wf_fail_closed.py and related FRZ rows per plan §3). Owner batch **not started**.

## wf-exec wave (operator UNBLOCK, pin e69e1ff58)

The plan does **not** allow the E2–E4 product slice without Campaign B-1 output (PLAN_T-IGX-043-J §8.2; prerequisite §2 SCOPE / PT-B01). That wave **only documents** provisional pytest node candidates at the pin — it does **not** replace a plan rev 2 or mark slices green.

### Provisional anchors (to confirm after B-1)

| Node id | Slice | Proposed pytest node id (pin e69e1ff58) | Rationale |
|---------|-------|-------------------------------------------|-----------|
| NT-WF-NEG-E2-01 | E2 | *(no negative — TBD)*; regression anchor: 	ests/unit/runtime/execution/test_execution_runtime.py::test_inference_root_runtime_binds_identity_authority_budget | MEP §F E2 covers 
untime.py; B-1 must name a fail-closed violation before a red-first negative exists |
| NT-WF-NEG-E3-01 | E3 | 	ests/unit/compat/langchain/test_documents.py::test_from_langchain_rejects_missing_tenant | fail-closed on documents.py seam (tenant in metadata transport) |
| NT-WF-NEG-E4-01 | E4 | 	ests/unit/experiments/test_experiment_workflow.py::test_experiment_session_echo_workflow | F-D5-07 (PLAN_T-IGX-041-E): governance identity at ExperimentSession.run |

**E4 related files (plan §3):** intergrax/experiments/workflow.py (primary), intergrax/experiments/store.py, intergrax/debug/router.py — cross-file fix only after B-1 + plan rev 2.

## Why BLOCKED (product E2 / owner batch)

1. MEP-AF2 §F E2–E4 are **Campaign B-1 conditional** — negative nodes need B-1 before the architect assigns full pytest paths.
2. Plan §7.1 rows E2–E4 remain **BLOCKED**; provisional anchors above are **not** slice closure.
3. Owner-repair batch (	ests/unit/applications/test_wf_fail_closed.py) still **not started** (SCOPE / FRZ owners).

## Next step (outside this batch)

After B-1: plan rev 2 binding NT-WF-NEG-* → red-first / green product on §3 paths.

## STOP

No 
w-train merge, no FRZ promotion, no SCOPE closure claim (**RW-PLAN-AUDIT** remains **BLOCKED** in [RW_PLAN_AUDIT_STATUS.md](RW_PLAN_AUDIT_STATUS.md); independent re-run required).
