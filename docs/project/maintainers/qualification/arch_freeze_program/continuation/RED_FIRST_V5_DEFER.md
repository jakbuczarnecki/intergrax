# RED_FIRST V5 — selective defer record (two rows)

**Branch / worktree:** `af-v5/residual-docs` · `D:/Projekty/_wt/intergrax/af-v5-doc`  
**Document role:** Records operator-planning deferrals for two `RED_FIRST_UNREPRODUCIBLE` rows. Does **not** drop rows, does **not** promote `GREEN_AT_PREFIX` as positive evidence, and does **not** mark **T-AF2-RED-CLASSES** as **DECIDED**.

**Canonical basis:** MEP-AF2 **T-AF2-RED-CLASSES**; `scripts/governance/verify_rw_plan_phase_check3.py` rejects **`GREEN_AT_PREFIX`** as positive red-first evidence and requires the **39-row `CODE_FIX`** set for phase advancement. This file is narrative only; it does **not** edit `rw_plan.v1.json` statuses.

## FRZ-PRD-08 — `prefix_already_green`

| Field | Value |
| --- | --- |
| `frz_id` | `FRZ-PRD-08` |
| Recorded prefix | `754ee000675d71c1734656702beb0aef8ba58fd8` |
| Cause class | `prefix_already_green` |
| Logged outcome @ prefix | Production-only bypass inventory probe **exit 0**; bypass count already **0** |
| RW plan disposition | Row stays **`OPEN`** and **`RED_FIRST_UNREPRODUCIBLE`** |
| T-AF2-RED-CLASSES recommendation | **OUT** (recommendation only — not an operator citation) |
| Operator decision | **not DECIDED** |

**Rationale:** At the nominated prefix the negative path is already satisfied. There is no reproducible red mechanism to replay under strict red-first rules. The validator must not treat this as `GREEN_AT_PREFIX` evidence toward closure. **Do not drop the row** from `rw_plan.v1.json`; historical non-evidence remains on the matrix until an explicit operator path retires or re-homes the concern.

## FRZ-FRZ-10 — `historical_self_test` (owner deferral)

| Field | Value |
| --- | --- |
| `frz_id` | `FRZ-FRZ-10` |
| Recorded prefix | `332c5217c390202a911ad5592e88a8f366845587` |
| Cause class | `historical_self_test` |
| Logged symptom (non-mechanism) | `AttributeError`: governance traceability helper lacks `_self_test_cases` at prefix |
| Owning mechanism (post-freeze policy) | **T-B7-PFP-ACTIVATE** (W5) · **T-B7-PFP-VERIFY** (W8) |
| RW plan disposition | Row stays **`RED_FIRST_UNREPRODUCIBLE`** |
| T-AF2-RED-CLASSES | Deferred to W5/W8 owners — **not** satisfied by fixing `_self_test_cases` on the historical script |
| Operator decision | **not DECIDED** |

**Rationale:** Post-freeze policy activation is owned by the W5 activation node and W8 verification node, not by a missing self-test hook on `verify_arch_freeze_plan_traceability` at the prefix SHA. Treating the `AttributeError` as the FRZ-10 mechanism would mis-attribute the matrix row.

## Explicit non-effects

This document does **not**:

- set **SCOPE**, **G2**, **R**, **FCS**, **M3**, **ENTRY**, or **ARCH-FREEZE CLOSED**;
- change any RW row status in `rw_plan.v1.json`;
- mark any operator card **DECIDED**.

**Recorded:** 2026-11-26 · ARCH-FREEZE V5 residual docs pass.