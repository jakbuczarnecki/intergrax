# TRACE-X-P1 — Identity, Transport Mapping & Parent-Child Causality Certification

**Parent:** TRACE-X — End-to-End Traceability & Evidence Certification

**START_HEAD:** `19eda895d7db3c11758d77aaf8a96d746fa7c304`

**Initial transport qualification evidence:** `097b8236817456377848885a324afa1044101009`

**Final accepted P1 evidence/code baseline:** `2643d36edb7e90fb2e68b4dd88dc146aca1b58af`

**Mechanical SSOT:** `tests/qualification/trace_x/_trace_x_p1_support.py`

**Applicable FRZ (P1 only):** `FRZ-TRC-02`, `FRZ-TRC-12`

**Production delta:** **0** (certification + qualification evidence only)

**Status:** **TRACE-X-P1 = CLOSED / independently accepted** @ `2643d36edb7e90fb2e68b4dd88dc146aca1b58af`

**Child:** `TRACE-X-P1-R1` strict durable child lineage — **CLOSED / independently accepted** @ same baseline

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

**Fail-closed transport path:** transport task → establish/recover runtime identity → persist `PlatformCausalEvidence` → handler executes; required evidence persistence failure → handler does not execute.

---

## 4. Identity domains (FRZ-TRC-12)

| Domain | Type | Notes |
|---|---|---|
| Transport / provider | `MessageBusTaskRef` | Not runtime identity |
| Platform runtime | `RuntimeExecutionRef` | `TaskId`, `RunId`, `AttemptId`, `ExecutionId`, `tenant_id` |
| Transport→runtime mapping | `PlatformCausalEvidence` | Canonical relation owner |

Equal textual values ≠ semantic identity-domain equivalence.

Primary independently audited transport evidence: `097b8236817456377848885a324afa1044101009`. Final P1 baseline `2643d36edb7e90fb2e68b4dd88dc146aca1b58af` — no production transport semantics changed between these evidence points.

---

## 5. Degraded-lineage verdict (current — post independent P1-R1 audit)

**Can an executed child exist without reconstructable canonical parent edge?** **NO**

**Frozen P1 invariant:** No child execution may cross the execution boundary unless its canonical parent→child `ExecutionLineage` admission has been durably accepted (`executed child ⇒ durable canonical parent edge exists`).

`ExecutionLineageChildAdmissionHook` re-raises on `ExecutionLineageUnavailableError` after best-effort `mark_degraded`; the child delegate **does not** run. Failed admissions may leave the attempt `PARTIAL`/degraded without fabricating a parent edge for a non-executed child.

**FRZ-TRC-02:** **PASS** @ `2643d36edb7e90fb2e68b4dd88dc146aca1b58af`.

**FRZ-TRC-12:** **PASS** @ `2643d36edb7e90fb2e68b4dd88dc146aca1b58af` (transport evidence chain above).

No alternate canonical owner stores equivalent parent→child topology (`delegated_execution` bindings are delegation correlation, not lineage truth). **`RuntimeEvent`** and **`PlatformCausalEvidence`** are not second parent topology truth.

---

## 6. Historical policy (audit provenance — not current behavior)

**SUPERSEDED BY TRACE-X-P1-R1** (independently accepted @ `2643d36edb7e90fb2e68b4dd88dc146aca1b58af`):

```text
child admission unavailable + durable mark_degraded → child may continue
```

Preserved in [`DG_001_EXECUTION_LINEAGE_ADMISSION_PERSISTENCE_R1.md`](DG_001_EXECUTION_LINEAGE_ADMISSION_PERSISTENCE_R1.md) for provenance only.

---

## 7. Tenant Isolation Audit (P1)

| Field | Value |
|---|---|
| tenant scope applicable | YES |
| canonical tenant identity | `tenant_id` |
| transport propagation | `MessageBusTaskRef` → `PlatformCausalEvidence` → `RuntimeExecutionRef` |
| lineage propagation | `ExecutionLineageAttemptScope` |
| cross-tenant negative tests | `test_tenant_mismatch_fails_closed`, `test_list_segments_tenant_isolation`, `test_tenant_isolation_for_same_transport_id` |
| result | **PASS** (P1 scope only; does **not** close **TENANT-X** or **FRZ-TEN-***) |

---

## 8. Test commands (independent audit @ `2643d36edb7e90fb2e68b4dd88dc146aca1b58af`)

```bash
uv run --with cryptography pytest tests/qualification/trace_x/test_trace_x_p1_identity_causality.py -p no:xdist -q
uv run --with cryptography pytest tests/unit/runtime/background_execution/test_background_causal_evidence_admission_paths.py tests/unit/runtime/background_execution/test_required_audit_evidence_admission.py tests/unit/runtime/observability/test_causal_evidence_contract.py tests/unit/runtime/observability/test_durable_causal_evidence_persistence.py -p no:xdist -q
uv run --with cryptography pytest tests/unit/runtime/execution/lineage/test_execution_lineage_admission_order.py tests/unit/runtime/execution/lineage/test_execution_lineage_persistence_conformance.py tests/unit/runtime/execution/lineage/test_child_lineage_admission_failure.py tests/unit/runtime/observability/reconstruction/test_execution_lineage_reconstruction.py -p no:xdist -q
```

**Pass 1:** 32 passed · **Pass 2:** 49 passed · **Pass 3:** 31 passed (recorded on accepted SHA).

---

## 9. Findings

| ID | Classification |
|---|---|
| P1-BLK-DEGRADED-LINEAGE-01 | **RESOLVED / independently accepted** @ `2643d36edb7e90fb2e68b4dd88dc146aca1b58af` |

---

## 10. Canonical roadmap state (post P1 closure bookkeeping)

```text
TRACE-X = CURRENT / MANDATORY
TRACE-X-P0 = CLOSED / independently accepted
TRACE-X-P1 = CLOSED / independently accepted
TRACE-X-P1-R1 = CLOSED / independently accepted
TRACE-X-P2 = NEXT / NOT ENTERED
TRACE-X-P3..P6 = NOT ENTERED
TRACE-X-CERT = NOT ENTERED
FRZ-TRC-02 = PASS
FRZ-TRC-12 = PASS
FRZ-TRC-01, FRZ-TRC-03..11 = OPEN
```
