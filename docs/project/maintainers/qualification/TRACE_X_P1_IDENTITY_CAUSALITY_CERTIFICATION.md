# TRACE-X-P1 — Identity, Transport Mapping & Parent-Child Causality Certification

**Parent:** TRACE-X — End-to-End Traceability & Evidence Certification

**START_HEAD:** `19eda895d7db3c11758d77aaf8a96d746fa7c304`

**FINAL_COMMIT / AUDITED_HEAD:** `097b8236817456377848885a324afa1044101009`

**Mechanical SSOT:** `tests/qualification/trace_x/_trace_x_p1_support.py`

**Applicable FRZ (P1 only):** `FRZ-TRC-02`, `FRZ-TRC-12`

**Production delta:** **0** (certification + qualification evidence only)

**Status:** **TRACE-X-P1 = READY FOR AUDIT** (post-R1; not CLOSED)

**Child:** `TRACE-X-P1-R1` strict durable child lineage @ START_HEAD `f31a326bb96b936e07d149c2ba8c75351d9117c4`

---

## 1. Scope

Prove transport→runtime mapping via `PlatformCausalEvidence` and parent→child topology via `ExecutionLineage`, with adversarial tenant/identity gates. Does not enter P2–P6 or CERT.

---

## 2. Ownership matrix

| Concern | Semantic owner | Writer | Persistence | Reader |
|---|---|---|---|---|
| transport→runtime | `PlatformCausalEvidence` | `admit_background_execution_handler` | `CausalEvidencePersistence` | `ExecutionReconstructor` |
| parent→child | `ExecutionLineage` | lineage admission hooks | `ExecutionLineagePersistence` | `ExecutionLineageReader` / reconstructor |

Duplicate owner = 0 · shadow writer = 0.

---

## 3. Transport entrypoint inventory (closed-world @ HEAD)

| ID | Module | Gate |
|---|---|---|
| TXP1-T01 | `intergrax/queueing/worker/dispatcher.py` | `admit_background_execution_handler` |
| TXP1-T02 | `intergrax/background_tasks/worker_runtime.py` | same |
| TXP1-T03 | `intergrax/queueing/providers/broker_worker_base.py` | same |
| TXP1-T04 | `intergrax/queueing/providers/document_store/colocated_worker.py` | same |

`build_transport_triggered_execution_evidence` production definition: `required_audit_evidence.py` only.

---

## 4. Degraded-lineage verdict (post TRACE-X-P1-R1)

**Can an executed child exist without reconstructable canonical parent edge?** **NO**

After `TRACE-X-P1-R1`, `ExecutionLineageChildAdmissionHook` re-raises on `ExecutionLineageUnavailableError` after best-effort `mark_degraded`; the child delegate **does not** run. Failed admissions may leave the attempt `PARTIAL`/degraded without fabricating a parent edge for a non-executed child.

No alternate canonical owner stores equivalent parent→child topology (`delegated_execution` bindings are delegation correlation, not lineage truth).

**FRZ-TRC-02:** **READY FOR INDEPENDENT CLOSURE REVIEW** (Cursor recommendation only; not PASS).

**FRZ-TRC-12:** **READY FOR INDEPENDENT CLOSURE REVIEW** — prior evidence @ `097b8236817456377848885a324afa1044101009` preserved.

---

## 5. Tenant Isolation Audit (P1)

| Field | Value |
|---|---|
| tenant scope applicable | YES |
| canonical tenant identity | `tenant_id` |
| transport propagation | `MessageBusTaskRef` → `PlatformCausalEvidence` → `RuntimeExecutionRef` |
| lineage propagation | `ExecutionLineageAttemptScope` |
| cross-tenant negative tests | `test_tenant_mismatch_fails_closed`, `test_list_segments_tenant_isolation`, `test_tenant_isolation_for_same_transport_id` |
| result | PASS (P1 scope; does not close TENANT-X) |

---

## 6. Test commands

```bash
uv run --with cryptography pytest tests/qualification/trace_x/test_trace_x_p1_identity_causality.py -p no:xdist -q
uv run --with cryptography pytest tests/unit/runtime/background_execution/test_background_causal_evidence_admission_paths.py tests/unit/runtime/background_execution/test_required_audit_evidence_admission.py tests/unit/runtime/observability/test_causal_evidence_contract.py tests/unit/runtime/observability/test_durable_causal_evidence_persistence.py -p no:xdist -q
uv run --with cryptography pytest tests/unit/runtime/execution/lineage/test_execution_lineage_admission_order.py tests/unit/runtime/execution/lineage/test_execution_lineage_persistence_conformance.py tests/unit/runtime/execution/lineage/test_child_lineage_admission_failure.py tests/unit/runtime/observability/reconstruction/test_execution_lineage_reconstruction.py -p no:xdist -q
```

**Pass 1:** 32 passed (`test_trace_x_p1_identity_causality.py`). **Pass 2:** 49 passed (requires `uv run --extra dev` for Celery path tests). **Pass 3:** 31 passed.

Record `FINAL_COMMIT` / `AUDITED_HEAD` in the independent audit after push.

---

## 7. Findings

| ID | Classification |
|---|---|
| P1-BLK-DEGRADED-LINEAGE-01 | RESOLVED PENDING INDEPENDENT AUDIT (TRACE-X-P1-R1) |

---

## 8. Recommended roadmap

```text
TRACE-X = CURRENT / MANDATORY
TRACE-X-P1 = READY FOR AUDIT
TRACE-X-P1-R1 = READY FOR AUDIT
TRACE-X-P2 = NOT ENTERED
```
