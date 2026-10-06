# TRACE-X-P2 — Execution Causal Reconstruction Certification

**Parent:** TRACE-X — End-to-End Traceability & Evidence Certification

**START_HEAD:** `f85ec3697742d41a9beb476e7cb9cc0aeae956d4`

**Accepted evidence/code baseline:** `4c6b7d05e2e45048c2a5e0cf609b910339d2dcb6`

**P1 final accepted baseline (preserved):** `2643d36edb7e90fb2e68b4dd88dc146aca1b58af`

**Mechanical SSOT:** `tests/qualification/trace_x/_trace_x_p2_support.py`

**Applicable FRZ (P2 only):** `FRZ-TRC-01`

**Production delta @ accepted baseline:** completeness-aware cross-source `ExecutionId` coherence inside `ExecutionReconstructor` (no new owner; no ownership change)

**Status:** **TRACE-X-P2 = CLOSED / INDEPENDENTLY ACCEPTED** @ `4c6b7d05e2e45048c2a5e0cf609b910339d2dcb6`

---

## 1. Scope

Prove `RuntimeEvent` + `PlatformCausalEvidence` + `ExecutionLineage` compose via **exactly-one** factual reconstruction owner — `ExecutionReconstructor` — into a deterministic, integrity-checked `ExecutionReconstruction` read model. Does not enter P3–P6 or CERT.

---

## 2. Ownership

| Concern | Owner | Persistence | Reader |
|---|---|---|---|
| Runtime facts | `RuntimeEvent` | `EvidencePersistencePort` | `ExecutionReconstructor` |
| Transport→runtime | `PlatformCausalEvidence` | `CausalEvidencePersistence` | `ExecutionReconstructor` |
| Parent topology | `ExecutionLineage` | `ExecutionLineagePersistence` | `ExecutionLineageReader` |
| Derived reconstruction | `ExecutionReconstructor` | **none** | `ExecutionReconstructionReader` consumers |

`ExecutionReconstructor` composes `RuntimeEvent` + `PlatformCausalEvidence` + `ExecutionLineage` into `ExecutionReconstruction`, which remains **derived**, **non-persisted**, and **non-authoritative**.

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

## 5. Accepted integrity guarantees (@ `4c6b7d05…`)

- tenant / task / run scope validated
- facts partitioned by `AttemptId`
- runtime ordering follows execution position, not timestamp
- `PlatformCausalEvidence.target.execution_id` and `RuntimeEvent.execution_id` must belong to canonical lineage topology when lineage membership is provable
- cross-source contradiction → `ExecutionReconstructionIntegrityError`
- **PARTIAL** → remains incomplete
- **TRUNCATED** → remains incomplete
- **UNAVAILABLE** → remains unavailable
- missing evidence → never guessed
- no `correlation_id` heuristic
- no timestamp heuristic
- no second causal graph

---

## 6. Architecture invariant (accepted)

> Complete canonical evidence that claims to describe the same execution chain must be execution-ID coherent; incomplete evidence remains explicitly incomplete and is never repaired heuristically.

---

## 7. Adversarial matrix (A–F)

| Case | Expected |
|---|---|
| A — causal in lineage | PASS |
| B — causal outside complete topology | `ExecutionReconstructionIntegrityError` |
| C — runtime contradicts complete topology | `ExecutionReconstructionIntegrityError` |
| D — PARTIAL lineage | no auto-corruption on missing membership |
| E — lineage UNAVAILABLE | `read_status=UNAVAILABLE` |
| F — runtime TRUNCATED | `RuntimeHistoryCompleteness.TRUNCATED` |

---

## 8. FRZ disposition

**FRZ-TRC-01:** **PASS** (independently accepted @ `4c6b7d05e2e45048c2a5e0cf609b910339d2dcb6`).

---

## 9. Verification commands (accepted baseline evidence)

```bash
uv run --with cryptography pytest tests/qualification/trace_x/test_trace_x_p2_execution_causal_reconstruction.py -p no:xdist -q
uv run --with cryptography pytest tests/unit/runtime/observability/reconstruction/test_execution_reconstruction.py tests/unit/runtime/observability/reconstruction/test_execution_lineage_reconstruction.py -p no:xdist -q
uv run --with cryptography pytest tests/unit/runtime/observability/reconstruction/test_obs_asof_rebase_r1_lineage_integrity.py tests/unit/contracts/test_execution_reconstruction_reader.py -p no:xdist -q
```

P2 closure bookkeeping on `development` after `4c6b7d05…` is docs-only — not P2 evidence baseline.
