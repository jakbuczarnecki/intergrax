# TRACE-X-P1-R1 — Strict Durable Child Lineage Admission Certification

**Parent:** TRACE-X-P1 — Identity, Transport Mapping & Parent-Child Causality

**START_HEAD:** `f31a326bb96b936e07d149c2ba8c75351d9117c4`

**Mechanical SSOT:** `tests/qualification/trace_x/_trace_x_p1_r1_support.py`

**Applicable FRZ:** `FRZ-TRC-02` (primary); `FRZ-TRC-12` prior evidence preserved @ `097b8236817456377848885a324afa1044101009`

**Status:** **TRACE-X-P1-R1 = READY FOR AUDIT** (not CLOSED)

---

## ARCHITECTURE DECISION

**child execution requires successful canonical durable ExecutionLineage parent→child admission.**

**Reason:** `FRZ-TRC-02` exactly-one lineage owner; forensic completeness; no executed child without durable topology.

When `ExecutionLineageUnavailableError` occurs during child admission, the runtime may record durable degradation and attempt-level degradation markers, but **must re-raise** and **must not** invoke the child delegate.

---

## Production scope

| File | Change |
|---|---|
| `intergrax/runtime/execution/lineage/admission.py` | Re-raise after degradation on child admission unavailable |
| `intergrax/runtime/execution/lineage/active_lineage.py` | Remove non-durable execution machinery |
| `intergrax/runtime/execution/child.py` | Remove nested non-durable parent guard |

---

## P1 blocker resolution

| ID | State |
|---|---|
| P1-BLK-DEGRADED-LINEAGE-01 | **RESOLVED PENDING INDEPENDENT AUDIT** |

---

## FRZ disposition (Cursor recommendation only)

| FRZ | Disposition |
|---|---|
| FRZ-TRC-02 | READY FOR INDEPENDENT CLOSURE REVIEW |
| FRZ-TRC-12 | Prior independently audited transport evidence preserved |

---

## Test command

```bash
uv run --with cryptography pytest tests/qualification/trace_x/test_trace_x_p1_r1_strict_lineage.py -p no:xdist -q
```

Record `FINAL_COMMIT` in the independent audit after push.
