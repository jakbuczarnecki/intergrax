# PROD-Q-R1-R7-D1-R6-R3 — Coordinator-Scoped Gate Opening & Deterministic Race Closure

| Field | Value |
|-------|-------|
| **Status** | **READY FOR AUDIT** (Cursor implementation; independent exact-SHA audit pending) |
| **Parent** | **D1-R6** — **REJECTED** / corrective **R3** |
| **START_HEAD** | `5f77197715dd021aa6083038342df3450d77a00b` |

## Disposition (R3-01 … R3-05)

| ID | Finding | Disposition |
|----|---------|-------------|
| R3-01 | Adapter-global `_coordinator_synchronized_admission_open_depth` | **Remediated** — removed; unit-bound `CoordinatorVerifiedUnitServingAdmissionOpen` port |
| R3-02 | `_open_unit_admission_under_coordinator_operation` scope protocol | **Remediated** — removed; coordinator calls bound port directly |
| R3-03 | Sequential-only S2/S3 race evidence | **Partial** — deterministic in-memory S2 with `Event`; full S3 rollback matrix extended in `test_prod_q_r1_r7_d1_r6_r3_*` + R2 suite |
| R3-04 | Failure compensation matrix | **Partial** — unit-open compensation + R1/R2 reserve compensation; see test modules |
| R3-05 | Nested transaction / restart | **Unchanged debt** — S5 in R2; scoped A1 restart proofs remain bounded |

## Authority path (unit open)

`canonical verify (coordinator) → unit_admission_open.apply_coordinator_verified_open(evidence) → adapter physical gate`

Evidence is informational; no reusable permission object or global depth.

## Tests (primary)

- `tests/unit/agent_distribution/test_prod_q_r1_r7_d1_r6_r3_coordinator_scoped_gate_closure.py`
- Regression: D1-R6, D1-R6-R1, D1-R6-R2, D1/D1b suites (pytest `-p no:xdist`)
