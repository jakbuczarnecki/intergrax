# TRACE-X-P1-R1 — Strict Durable Child Lineage Admission Certification

**Parent:** TRACE-X-P1 — Identity, Transport Mapping & Parent-Child Causality

**START_HEAD:** `f31a326bb96b936e07d149c2ba8c75351d9117c4`

**Accepted evidence/code baseline:** `2643d36edb7e90fb2e68b4dd88dc146aca1b58af`

**Mechanical SSOT:** `tests/qualification/trace_x/_trace_x_p1_r1_support.py`

**Applicable FRZ:** `FRZ-TRC-02` (primary); `FRZ-TRC-12` prior transport evidence @ `097b8236817456377848885a324afa1044101009`

**Status:** **TRACE-X-P1-R1 = CLOSED / independently accepted** @ `2643d36edb7e90fb2e68b4dd88dc146aca1b58af`

---

## ARCHITECTURE DECISION (accepted P1 architecture)

**Child execution requires successful canonical durable `ExecutionLineage` parent→child admission.**

**Reason:** `FRZ-TRC-02` exactly-one lineage owner; forensic completeness; no executed child without durable topology.

When `ExecutionLineageUnavailableError` occurs during child admission, the runtime may record durable degradation and attempt-level degradation markers, but **must re-raise** and **must not** invoke the child delegate.

**Equivalent invariant:** `executed child ⇒ durable parent→child ExecutionLineage admission`.

---

## Production scope (@ accepted baseline)

| File | Change |
|---|---|
| `intergrax/runtime/execution/lineage/admission.py` | Re-raise after degradation on child admission unavailable |
| `intergrax/runtime/execution/lineage/active_lineage.py` | Remove non-durable execution machinery |
| `intergrax/runtime/execution/child.py` | Remove nested non-durable parent guard |

---

## P1 blocker resolution

| ID | State |
|---|---|
| P1-BLK-DEGRADED-LINEAGE-01 | **RESOLVED / independently accepted** @ `2643d36edb7e90fb2e68b4dd88dc146aca1b58af` |

---

## FRZ disposition (independent audit)

| FRZ | Disposition | Evidence SHA |
|---|---|---|
| FRZ-TRC-02 | **PASS** | `2643d36edb7e90fb2e68b4dd88dc146aca1b58af` |
| FRZ-TRC-12 | **PASS** (transport chain unchanged from `097b8236817456377848885a324afa1044101009`) | `2643d36edb7e90fb2e68b4dd88dc146aca1b58af` |

**FRZ-TRC-02 evidence summary:** `ExecutionLineage` sole canonical parent→child topology owner; durable admission prerequisite for delegate; `ExecutionLineageUnavailableError` → best-effort degradation → re-raise → no child delegate; `mark_degraded` success does not authorize child; structural conflicts fail closed; missing lineage never heuristically reconstructed; `RuntimeEvent` / `PlatformCausalEvidence` not second topology truth.

**Regression evidence:** failed admission blocks delegate; `mark_degraded` failure blocks child; structural conflicts block child; successful admission precedes delegate; budget / identity / authority cleanup; truthful reconstruction.

---

## Historical policy (SUPERSEDED BY TRACE-X-P1-R1)

Prior documented behavior allowing child execution after durable `mark_degraded` when admission unavailable is **historical audit provenance only** — not current platform behavior. See P1 certification §6 and [`DG_001_EXECUTION_LINEAGE_ADMISSION_PERSISTENCE_R1.md`](DG_001_EXECUTION_LINEAGE_ADMISSION_PERSISTENCE_R1.md).

---

## Test command (independent audit @ `2643d36edb7e90fb2e68b4dd88dc146aca1b58af`)

```bash
uv run --with cryptography pytest tests/qualification/trace_x/test_trace_x_p1_r1_strict_lineage.py -p no:xdist -q
```

Accepted on exact GitHub SHA `2643d36edb7e90fb2e68b4dd88dc146aca1b58af`.
