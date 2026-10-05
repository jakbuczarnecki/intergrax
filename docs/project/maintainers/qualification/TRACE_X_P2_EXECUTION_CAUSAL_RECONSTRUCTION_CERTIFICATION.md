# TRACE-X-P2 — Execution Causal Reconstruction Certification

**Parent:** TRACE-X — End-to-End Traceability & Evidence Certification

**START_HEAD:** `f85ec3697742d41a9beb476e7cb9cc0aeae956d4`

**Mechanical SSOT:** `tests/qualification/trace_x/_trace_x_p2_support.py`

**Applicable FRZ (P2 only):** `FRZ-TRC-01`

**Production delta:** local completeness-aware cross-source `ExecutionId` coherence inside `ExecutionReconstructor` (no new owner)

**Status:** **TRACE-X-P2 = READY FOR AUDIT** (pending independent SHA audit)

---

## 1. Scope

Prove `RuntimeEvent` + `PlatformCausalEvidence` + `ExecutionLineage` compose via **exactly-one** `ExecutionReconstructor` into a deterministic, integrity-checked `ExecutionReconstruction` read model. Does not enter P3–P6 or CERT.

---

## 2. Ownership

| Concern | Owner | Persistence | Reader |
|---|---|---|---|
| Runtime facts | `RuntimeEvent` | `EvidencePersistencePort` | `ExecutionReconstructor` |
| Transport→runtime | `PlatformCausalEvidence` | `CausalEvidencePersistence` | `ExecutionReconstructor` |
| Parent topology | `ExecutionLineage` | `ExecutionLineagePersistence` | `ExecutionLineageReader` |
| Derived reconstruction | `ExecutionReconstructor` | **none** | `ExecutionReconstructionReader` consumers |

---

## 3. Reconstruction chain

```text
MessageBusTaskRef
  → PlatformCausalEvidence.target (RuntimeExecutionRef)
  → AttemptId / ExecutionId
  → RuntimeEvent history (position-ordered)
  → ExecutionLineage admissions / segments
  → ExecutionReconstruction (derived, non-persisted)
```

---

## 4. Cross-source execution coherence

When lineage is **AVAILABLE** with **OPEN** or **COMPLETE** completeness, `PlatformCausalEvidence.target.execution_id` and `RuntimeEvent.execution_id` must belong to the attempt lineage admission set; otherwise **fail closed**.

When lineage is **PARTIAL**, **TRUNCATED**, **UNAVAILABLE**, or **ABSENT**, reconstruction preserves truthful incompleteness (no fabricated joins).

---

## 5. Adversarial matrix (A–F)

| Case | Expected |
|---|---|
| A — causal in lineage | PASS |
| B — causal outside complete topology | `ExecutionReconstructionIntegrityError` |
| C — runtime contradicts complete topology | `ExecutionReconstructionIntegrityError` |
| D — PARTIAL lineage | no auto-corruption on missing membership |
| E — lineage UNAVAILABLE | `read_status=UNAVAILABLE` |
| F — runtime TRUNCATED | `RuntimeHistoryCompleteness.TRUNCATED` |

---

## 6. FRZ disposition (recommendation only)

**FRZ-TRC-01:** **READY FOR INDEPENDENT CLOSURE REVIEW** (Cursor recommendation — not PASS).

---

## 7. Verification commands

```bash
uv run --with cryptography pytest tests/qualification/trace_x/test_trace_x_p2_execution_causal_reconstruction.py -p no:xdist -q
uv run --with cryptography pytest tests/unit/runtime/observability/reconstruction/test_execution_reconstruction.py tests/unit/runtime/observability/reconstruction/test_execution_lineage_reconstruction.py -p no:xdist -q
uv run --with cryptography pytest tests/unit/runtime/observability/reconstruction/test_obs_asof_rebase_r1_lineage_integrity.py tests/unit/contracts/test_execution_reconstruction_reader.py -p no:xdist -q
```
