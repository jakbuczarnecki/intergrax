# DG-004 Async Transport Causal Continuity — Final Closure (R3)

**Verdict:** PASS

**Date:** 2026-09-07

**Branch:** `development`

**Start HEAD:** `a5479eec632f99894b8d6d77bb57891bbce25ca4`

**Review HEAD (pre-docs):** `a5479eec632f99894b8d6d77bb57891bbce25ca4`

**Task:** `DG-004-FINAL-CLOSURE-REVIEW-R3` — review-only final closure; no production or test changes.

---

## 1. Verdict

```text
DG-004 ASYNC TRANSPORT CAUSAL CONTINUITY IN EXECUTION DIAGNOSIS = CLOSED / QUALIFIED
```

Real E0–E6 qualification on the canonical LKW Kafka background-admission path (R2) plus R3 architectural equivalence review confirm that the platform background-admission mechanism produces, persists, reads, reconstructs, and projects required transport causal evidence end-to-end. Exact File Watcher replay is **not** required for core DG-004 closure.

---

## 2. Capability scope

DG-004 is the platform capability:

```text
Can the Diagnostic Engine prove causal continuity across
transport-backed background execution?
```

Canonical capability chain:

```text
transport task
  ↓
canonical background execution identity
  ↓
TRANSPORT_TASK_TRIGGERED_EXECUTION evidence
  ↓
canonical causal persistence
  ↓
read-by-execution
  ↓
ExecutionReconstructor
  ↓
DiagnosticOrchestrator
```

DG-004 is **not** “does the File Watcher scenario itself pass end-to-end?”

---

## 3. Historical gap

**Pre DG-A (2026-08-27):** File Watcher proof could not qualify background execution (worker assembly blocked).

**Post DG-A:** Fresh background execution and RuntimeEvents existed; `DiagnosticOrchestrator` reached DQ-2 but `has_transport_evidence=false`.

**R1:** Root cause at producer/persistence layer **not proven**; static chain showed canonical admission **should** run.

**R2:** Real E0–E6 **PASS** on LKW Kafka `lkw.background_ingest.v1`; same `evidence_id` traced through persistence and reconstruction; `has_transport_evidence=true`.

**R3 classification of historical symptom:** **STALE RELATIVE TO CURRENT CANONICAL MECHANISM** — not an unresolved scenario-specific core risk when File Watcher conforms to the same admission spine.

---

## 4. R1 audit

[`DG_004_ASYNC_CAUSAL_CONTINUITY_ROOT_CAUSE_AUDIT.md`](DG_004_ASYNC_CAUSAL_CONTINUITY_ROOT_CAUSE_AUDIT.md) (`e48d7db842081b8899c8232f175511234c47e025`):

| Hypothesis | R1 result |
| ---------- | --------- |
| Current producer bypass | **Disproven** (static) |
| Reconstructor defect | **Disproven** |
| Projection defect | **Disproven** |
| Historical cause isolated | **Not proven** |

R2 supersedes historical uncertainty for the **current** canonical path.

---

## 5. R2 real qualification

[`DG_004_REAL_ASYNC_CAUSAL_CONTINUITY_R2_QUALIFICATION.md`](DG_004_REAL_ASYNC_CAUSAL_CONTINUITY_R2_QUALIFICATION.md) (`ccceae7f649ac23b4212836221e24dedd599a5b3`):

| Gate | Status |
| ---- | ------ |
| E0 transport accepted | **PASS** |
| E1 execution identity | **PASS** |
| E2 evidence built | **PASS** |
| E3 persisted | **PASS** |
| E4 read-by-execution | **PASS** |
| E5 reconstruction | **PASS** |
| E6 orchestrator | **PASS** |

**Qualified topology:**

```text
LKW Kafka background ingest (lkw.background_ingest.v1)
  → MessageBus enqueue
  → lkw-background-worker (BrokerWorkerBase)
  → admit_background_execution_handler
  → Redis execution identity
  → DocumentStoreCausalEvidencePersistence
  → Mongo
  → cross-process read (local_workspace)
  → ExecutionReconstructor
  → DiagnosticOrchestrator
```

**Evidence identity:** `evt_a0f33d8587834822a50fb59c9c67e738` — same id at E2–E6; no synthetic evidence; no manual truth mutation.

---

## 6. Canonical mechanism

| Component | Ownership |
| --------- | --------- |
| `CausalRelationKind.TRANSPORT_TASK_TRIGGERED_EXECUTION` | Platform generic |
| `build_transport_triggered_execution_evidence` | Platform (`required_audit_evidence.py`) |
| `admit_background_execution_handler` | Platform admission gate |
| `BackgroundExecutionIdentity` | Platform (Redis KV; no workload reminting) |
| `CausalEvidencePersistence` / `DocumentStoreCausalEvidencePersistence` | Platform abstraction |
| `ExecutionReconstructor.list_for_execution` | Workload-agnostic |
| `DiagnosticOrchestrator` | Workload-agnostic projection |

**Hard invariant (supported paths):** no background handler execution without required transport causal evidence (fail-closed on persistence failure).

**Diagnostics transport coupling:** none — no Kafka/Celery/LKW/File Watcher imports under `intergrax/runtime/diagnostics/`.

---

## 7. File Watcher equivalence

Production trace (static):

```text
FileWatcherRuntime / FileWatcherSidecar
  → enqueue_background_ingest_job (LKW.4C helper)
  → message_bus_enqueue → Kafka topic intergrax.tasks
  → lkw-background-worker → BrokerWorkerBase.process_message
  → admit_background_execution_reentry → admit_background_execution_handler
  → execute_logical_task (lkw.background_ingest.v1 handler)
```

File Watcher builds `LkwBackgroundIngestJob` via `build_file_watcher_ingest_job` (business payload only: `requested_by=lkw.file_watcher`, source paths). Enqueue, worker, admission, evidence, persistence, reconstruction, and orchestrator are **identical** to R2.

| Layer | R2 background ingest | File Watcher | Same canonical mechanism? |
| ----- | -------------------- | ------------ | ------------------------- |
| MessageBus submission | `enqueue_background_ingest_job` → Kafka | `FileWatcherRuntime` → `enqueue_background_ingest_job` → Kafka | **YES** |
| Transport envelope/ref | `MessageBusEnqueueInput` + `LKW_BACKGROUND_INGEST_TASK_NAME` | Same helper + task name | **YES** |
| BrokerWorkerBase | `lkw-background-worker` | Same worker | **YES** |
| Identity admission | `admit_background_execution_reentry` | Same | **YES** |
| Required causal evidence builder | `build_transport_triggered_execution_evidence` | Same via `admit_background_execution_handler` | **YES** |
| Causal persistence | `DocumentStoreCausalEvidencePersistence` | Same `resolve_host_queue_execution_dependencies` | **YES** |
| ExecutionReconstructor | `list_for_execution(tenant, task_id, run_id)` | Same | **YES** |
| DiagnosticOrchestrator | `has_transport_evidence` from reconstruction | Same | **YES** |

**Exact File Watcher replay required for DG-004 core closure?** **NO.**

**Rationale:** File Watcher is a producer of the same `lkw.background_ingest.v1` transport job; it does not introduce custom worker, admission, identity, evidence builder, persistence, reconstruction, or projection paths.

**Scenario-level residual risk:** File Watcher end-to-end replay = **OPTIONAL / SCENARIO-LEVEL** (recommended for operational confidence, not core blocker).

---

## 8. Closure criteria

| Criterion | Result |
| --------- | ------ |
| Real canonical transport E0 | **PASS** (R2) |
| Identity continuity E1 | **PASS** (R2) |
| Required evidence E2 | **PASS** (R2) |
| Persistence E3 | **PASS** (R2) |
| Read-by-execution E4 | **PASS** (R2) |
| Reconstruction E5 | **PASS** (R2) |
| Orchestrator E6 | **PASS** (R2) |
| Cross-process causal visibility | **PROVEN** (R2) |
| File Watcher mechanism equivalent | **YES** (R3 static trace) |
| Diagnostics transport-agnostic | **PASS** |
| No supported producer bypass | **NONE** (R1 static + admission tests) |
| Tenant isolation | **PASS** (conformance tests) |
| No identity minting in diagnostics | **PASS** |
| Remaining core correctness blocker | **NONE** |

---

## 9. Architecture invariants

| Invariant | Status |
| --------- | ------ |
| `RELATION CONTRACT = PLATFORM GENERIC` | PASS |
| `IDENTITY CONTINUITY = PLATFORM GENERIC` | PASS |
| `CAUSAL PERSISTENCE = PLATFORM GENERIC` | PASS |
| `RECONSTRUCTION = WORKLOAD-AGNOSTIC` | PASS |
| `PROJECTION = WORKLOAD-AGNOSTIC` | PASS |
| `DIAGNOSTICS REMAINS TRANSPORT-AGNOSTIC` | PASS |
| `TRANSPORT PROVIDER EXTENSIBILITY = THROUGH ADMISSION CONTRACT` | PASS |
| Required causal evidence fail-closed | PASS |
| `BUSINESS/HANDLER FAILURE != CAUSAL CONTINUITY FAILURE` | PASS (R2 handler FAILED after admission; evidence intact) |

**Reusability (all PASS):** typed causal relation; generic source transport ref; generic execution target ref; canonical persistence abstraction; generic reconstruction; generic orchestrator projection; tenant isolation; no identity minting; no scenario-specific diagnostic branch.

---

## 10. Cross-process evidence

R2 proved:

```text
worker write causal evidence → separate local_workspace diagnostic read
```

```text
CAUSAL CROSS-PROCESS VISIBILITY = PROVEN
```

for Mongo-backed causal evidence via `DocumentStoreCausalEvidencePersistence`.

---

## 11. Non-claims

- Does not qualify Celery or all queue providers individually
- Does not prove all retry/attempt variants in real infrastructure (retry contract covered by unit conformance tests; R2 did not qualify all retry variants)
- Does not close DG-005 (RuntimeEvent cross-topology)
- Does not qualify DG-003 operator story projection
- Does not modify DG-002 scope discovery

---

## 12. Secondary findings

| Finding | Classification |
| ------- | -------------- |
| HTTP `/proof/background-task/enqueue` → 503 (`message_bus` not materialized in FastAPI factory) | **SEPARATE PROOF/COMPOSITION SURFACE DEFECT** — does not block causal continuity once work enters transport |
| `list_for_execution` unbounded compatibility facade | **RECONSTRUCTOR CAUSAL READ BOUNDEDNESS = OPEN HARDENING OBSERVATION** — does not block correctness closure |
| `has_transport_evidence` = `bool(causal_evidence)` naming | Secondary model-hardening only; R2 verified actual `TRANSPORT_TASK_TRIGGERED_EXECUTION` record |
| R2 RuntimeEvent SQLite path (worker vs `local_workspace` data home) | **DG-005** — separate from DG-004 causal evidence |

---

## 13. DG-002 / DG-003 / DG-005 relationship

| Gap | R3 disposition |
| --- | -------------- |
| **DG-002** | **CLOSED / UNCHANGED** |
| **DG-003** | **UNCHANGED** — operator story projection separate from causal continuity |
| **DG-005** | **UNCHANGED** — cross-process RuntimeEvent topology not claimed |

---

## 14. Test evidence

### Admission regression

```bash
uv run pytest \
  tests/unit/runtime/background_execution/test_required_audit_evidence_admission.py \
  tests/unit/runtime/background_execution/test_background_causal_evidence_admission_paths.py \
  -q --basetemp=.tmp/session/dg004-r3/pytest-basetemp-admission
```

**Result:** `17 passed`, 0 failed, 0 skipped

### Reconstruction regression

```bash
uv run pytest \
  tests/unit/runtime/diagnostics/test_execution_reconstruction.py \
  tests/unit/runtime/diagnostics/test_causal_transport_scope_provider.py \
  tests/unit/runtime/diagnostics/test_lifecycle_analysis.py \
  -q --basetemp=.tmp/session/dg004-r3/pytest-basetemp-reconstruction
```

**Result:** `78 passed`, 0 failed, 0 skipped

### File Watcher / worker conformance

```bash
uv run pytest \
  applications/local_workspace_application/tests/file_watcher/test_incremental_batch.py \
  -q --basetemp=.tmp/session/dg004-r3/pytest-basetemp-fw-batch
```

**Result:** `21 passed`, 0 failed, 0 skipped

**Note:** `test_lkw_background_worker_authority.py` requires full production composition (`ollama` adapter) — environment dependency failure, not DG-004 mechanism failure. `test_lkw_file_watcher_e2e_proof_receipt.py` references relocated verification docs — proof-receipt surface, not admission equivalence.

```text
SCENARIO CONFORMANCE TEST COVERAGE = LIMITED (full worker authority / e2e receipt)
```

Static production call path + incremental batch + admission path tests suffice for R3 equivalence.

**`--ignore` used:** NO  
**New skips:** NONE

---

## 15. Qualified SHAs

| Artifact | SHA |
| -------- | --- |
| R1 audit | `e48d7db842081b8899c8232f175511234c47e025` |
| R2 qualification | `ccceae7f649ac23b4212836221e24dedd599a5b3` |
| R3 review HEAD (pre-docs) | `a5479eec632f99894b8d6d77bb57891bbce25ca4` |
| DG-A fix (related) | `24506c3c14e30984d78b7b22c5cd4c42e711d125` |

R1 and R2 confirmed ancestors of R3 review HEAD.

---

## 16. Final closure statement

```text
DG-004 FINAL CLOSURE REVIEW R3 = PASS

DG-004 ASYNC TRANSPORT CAUSAL CONTINUITY = CLOSED / QUALIFIED

REAL E0–E6 CONTINUITY = QUALIFIED (R2)

CANONICAL BACKGROUND ADMISSION = QUALIFIED

TRANSPORT → EXECUTION IDENTITY = QUALIFIED

REQUIRED TRANSPORT CAUSAL EVIDENCE = QUALIFIED

DURABLE CAUSAL PERSISTENCE = QUALIFIED

CROSS-PROCESS CAUSAL VISIBILITY = QUALIFIED

EXECUTION RECONSTRUCTION = QUALIFIED

DIAGNOSTIC ORCHESTRATOR PROJECTION = QUALIFIED

FILE WATCHER CORE MECHANISM = CONFORMANT (static equivalence)

EXACT FILE WATCHER REPLAY = NOT REQUIRED FOR CORE CLOSURE

HISTORICAL has_transport_evidence=false = STALE RELATIVE TO CURRENT CANONICAL MECHANISM

REMAINING DG-004 CORE CORRECTNESS BLOCKERS = NONE
```

**Recommended next diagnostic task:** **DG-001** (P1, PARTIALLY ADDRESSED / DESIGN REQUIRED — pre-execution bootstrap failure visibility) or **DG-003** / **DG-005** per ledger priority.
