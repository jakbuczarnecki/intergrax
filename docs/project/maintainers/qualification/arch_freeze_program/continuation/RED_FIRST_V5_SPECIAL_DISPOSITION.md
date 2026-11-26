# RED_FIRST V5 — special disposition (FRZ-PRD-08, FRZ-FRZ-10)

**Worktree / branch:** `D:/Projekty/_wt/intergrax/af-v5-ev` · `af-v5/evidence-integrity`  
**Machine result (unchanged rows):** [`RED_FIRST_V5_RESULT.json`](RED_FIRST_V5_RESULT.json)  
**Inventory reconciliation:** same file, top-level `inventory_reconciliation` and `remeasurements`  
**Companion disposition:** [`RED_FIRST_V5_DISPOSITION.md`](RED_FIRST_V5_DISPOSITION.md)

## Operator status

No **T-AF2-RED-CLASSES** row is marked **DECIDED** in this document. Recommendations are planning input only.

## FRZ-PRD-08 (`NT-PRD-08`)

| Field | Value |
| --- | --- |
| `red_first_prefix_sha` | `754ee000675d71c1734656702beb0aef8ba58fd8` |
| Nominated path | `tests/unit/runtime/architecture/test_ee_a1_execution_engine_ownership_certification_gate.py` |
| Recorded measurement | **`RED_FIRST_UNREPRODUCIBLE`** |
| Logged reason | Production-only bypass count already 0 at prefix (ee_a1 inventory gate); `pytest_exit_code` **0** on inventory node |

At prefix `754ee000675d71c1734656702beb0aef8ba58fd8`, `test_ee_a1_p0_inventory_reports_zero_production_bypass` already reports **zero** production bypass count (**exit 0**). No new historical red was manufactured for this row.

**Assessment:** A **CODE_FIX** obligation is **not** demonstrated by a red-at-prefix defect on the nominated inventory gate. The row does not supply positive red-first mechanism evidence.

**T-AF2-RED-CLASSES recommendation:** **OUT** (recommendation only — operator **not DECIDED**).

**Row state:** Stays **OPEN** as **`RED_FIRST_UNREPRODUCIBLE`**.

**Validator effect:** `verify_rw_plan_phase_check3.py` still rejects **`GREEN_AT_PREFIX`** for this inventory outcome and still requires the full **39**-row CODE_FIX set. This disposition does **not** remove or relabel the row in machine inventory.

## FRZ-FRZ-10 (`NT-FRZ-10`)

| Field | Value |
| --- | --- |
| `red_first_prefix_sha` | `332c5217c390202a911ad5592e88a8f366845587` |
| Nominated path | `tests/unit/governance/test_arch_freeze_plan_traceability.py` |
| Recorded measurement | **`RED_FIRST_UNREPRODUCIBLE`** |
| Logged reason | `AttributeError`: module `scripts.governance.verify_arch_freeze_plan_traceability` has no attribute `_self_test_cases` |

**Owning mechanisms (future wave):** **T-B7-PFP-ACTIVATE** (W5) and **T-B7-PFP-VERIFY** (W8).

The logged **`AttributeError` on `_self_test_cases`** is a missing historical self-test hook on the governance script surface. It is **not** the policy-activation mechanism the FRZ matrix names. Historical regression evidence at the prefix and future policy activation under W5/W8 are **different** obligations. **Do not implement W5** from this document.

**Circularity (phase-check):** Phase-check requires every **CODE_FIX** row, including **FRZ-FRZ-10**, to be **`RED_AT_PREFIX`** before **`RED_FIRST_EVIDENCE=COMPLETE`**, while the nominated policy-activation mechanism does not exist as a pre-freeze product defect at this prefix. Canonical resolution is an **operator disposition** under **T-AF2-RED-CLASSES**, not a validator edit.

**Recommended resolution (not DECIDED):** Keep the row **`RED_FIRST_UNREPRODUCIBLE`** and **deferred** to W5/W8 activation/verify. It **blocks COMPLETE** red-first evidence and therefore **SCOPE** close on that gate; it does **not** block execution of the other residual probes.

**Operator decision:** **not DECIDED** — do not sign the card.

## Explicit non-effects

This document does **not**:

- set **SCOPE**, **G2**, **FCS**, or **M3**;
- edit `scripts/governance/verify_rw_plan_phase_check3.py`;
- change `red_first_status` values in `rw_plan.v1.json`;
- mark any operator card **DECIDED**.

**Recorded:** 2026-11-26 · ARCH-FREEZE V5 evidence-integrity pass (`af-v5/evidence-integrity`).