# DG-004 — Real Async Causal Continuity Qualification (R2)

**Verdict:** PASS

**Date:** 2026-09-07

**Branch:** `development`

**START_HEAD:** `e48d7db842081b8899c8232f175511234c47e025`

**Qualification HEAD (pre-docs):** `38cde642e25593d9252b348d4e67456267ec5d41`

---

## 1. Verdict

```text
DG-004 R2 = PASS
DG-004 CURRENT CANONICAL PATH = REVALIDATED
Historical has_transport_evidence=false = NOT REPRODUCED on qualified path
Closure review = PENDING (R3)
```

---

## 2. Qualification topology

**Path (equivalence note):** LKW **Kafka background ingest** (`lkw.background_ingest.v1`) — same canonical MessageBus → `lkw-background-worker` → `BrokerWorkerBase` → `admit_background_execution_handler` spine as LKW.4E background-task proof. **Not** the File Watcher overlay path; mechanism and persistence topology are equivalent for DG-004 continuity.

| Layer | Component |
|-------|-----------|
| Transport submit | `create_local_workspace_kafka_message_bus()` → `enqueue_background_ingest_job` → topic `intergrax.tasks` |
| Worker | `lkw-core-platform-proof-lkw-background-worker-1` → `local_workspace_application.host.background_worker_main` |
| Admission | `admit_background_execution_handler` + `build_transport_triggered_execution_evidence` |
| Identity | Redis KV (`INTERGRAX_REDIS_URL`) via `resolve_host_queue_execution_dependencies` |
| Causal persistence | `DocumentStoreCausalEvidencePersistence` via `wire_causal_evidence_persistence(document_store=...)` |
| Diagnostic read | `resolve_host_diagnostic_read_dependencies` → `ExecutionReconstructor` → `DiagnosticOrchestrator` |
| E0 transport mode | `canonical_kafka_enqueue` (HTTP `/proof/background-task/enqueue` returned 503 — see §13) |

**Docker compose project:** `lkw-core-platform-proof` (core platform proof overlays: kafka, mongodb, sentry, elasticsearch).

---

## 3. Real services

| Service | Status |
|---------|--------|
| `lkw-kafka` | Up (host bootstrap `127.0.0.1:9094`) |
| `lkw-redis` | Up |
| `lkw-mongodb` | Up (host port `27018`) |
| `lkw-background-worker` | Up |
| `local_workspace` | Up (health OK) |
| `qdrant`, `ollama` | Up |

---

## 4. Real execution identity

**Marker:** `dg004-r2-20260907060109`

| Field | Value |
|-------|-------|
| E0 provider | `kafka` |
| E0 transport_task_id | `dg004-r2-6c58ba421445` |
| E1 tenant_id | `lkw-background-proof` |
| E1 TaskId | `task_4271c562c2de42fa9241f60653276054` |
| E1 RunId | `run_dec440ac39e843b18fed7497601174c6` |
| E1 AttemptId | `attempt_a241e48c02e04224b352b4f92a06985b` |
| transport_task_id ≠ execution TaskId | YES |

---

## 5. E0–E6 table

| Gate | Status | Evidence |
|------|--------|----------|
| E0 transport accepted | **PASS** | Kafka enqueue; provider=`kafka`; transport_task_id=`dg004-r2-6c58ba421445` |
| E1 execution identity | **PASS** | Canonical target from `TRANSPORT_TASK_TRIGGERED_EXECUTION`; RuntimeEvent identity consistent |
| E2 evidence built | **PASS** | `evt_a0f33d8587834822a50fb59c9c67e738`; `transport_task.triggered_execution` |
| E3 persisted | **PASS** | Same evidence_id readable after worker completion (no manual append) |
| E4 read-by-execution | **PASS** | `list_for_execution`; exact evidence_id returned |
| E5 reconstruction | **PASS** | `has_runtime_events=true`; transport relation + attempt projection |
| E6 orchestrator | **PASS** | `has_transport_evidence=true`; matches E5 projection |

**Last proven PASS boundary:** E6

**First broken boundary:** NONE

**Root cause category:** H — historical gap no longer reproduces on current canonical path

---

## 6. Transport evidence identity

| Checkpoint | Canonical identifier |
|------------|----------------------|
| E0 transport | `kafka` + `dg004-r2-6c58ba421445` |
| E1 execution | `lkw-background-proof` + `task_4271c562c2de42fa9241f60653276054` + `run_dec440ac39e843b18fed7497601174c6` + `attempt_a241e48c02e04224b352b4f92a06985b` |
| E2 relation | `evt_a0f33d8587834822a50fb59c9c67e738` |
| E3 persisted | `evt_a0f33d8587834822a50fb59c9c67e738` |
| E4 direct lookup | `evt_a0f33d8587834822a50fb59c9c67e738` |
| E5 reconstruction | `evt_a0f33d8587834822a50fb59c9c67e738` |
| E6 diagnosis | Same TaskId/RunId continuity |

---

## 7. Persistence proof

- **Worker abstraction:** `DocumentStoreCausalEvidencePersistence`
- **Diagnostic abstraction:** `DocumentStoreCausalEvidencePersistence`
- **Backend:** `_MongoDBDocumentStore` (database/collection logical names not exposed by adapter surface; partition `intergrax.causal_evidence.v1:{tenant_id}`)
- **E3 proof method:** transport page read after worker admission (no synthetic append)
- **Fail-closed:** not exercised (happy path)

---

## 8. Cross-process topology proof

| Check | Result |
|-------|--------|
| Worker vs diagnostic causal abstraction | Same class |
| Same MongoDB backing (composition) | **YES** |
| Cross-process E4 (worker write → `local_workspace` read) | **PROVEN** — `list_for_execution` returned `evt_a0f33d8587834822a50fb59c9c67e738` from separate container |
| RuntimeEvent SQLite path (worker vs `local_workspace` data home) | **DIFFER** — E5/E6 primary run used worker container for runtime-event co-location; causal cross-process read proven separately |

---

## 9. Reconstruction proof

- `ExecutionReconstructor.reconstruct_execution(tenant, task_id, run_id)`
- `has_runtime_events=true`
- `has_transport_evidence=true`
- Exact `TRANSPORT_TASK_TRIGGERED_EXECUTION` relation with E2 evidence_id
- Attempt `attempt_a241e48c02e04224b352b4f92a06985b` contains E2 record

---

## 10. Orchestrator proof

- `DiagnosticOrchestrator.run(DiagnosticOrchestrationRequest(...))`
- `has_runtime_events=true`, `has_transport_evidence=true`
- `ExecutionReconstruction.has_transport_evidence == DiagnosticExecutionAnalysis.has_transport_evidence`

---

## 11. Historical comparison

| Observation | 2026-08-27 (post DG-A) | R2 run |
|-------------|------------------------|--------|
| Runtime history | Present (DQ-2) | Present |
| `has_transport_evidence` | `false` (ledger) | `true` |

**Conclusion:** Historical symptom **does not reproduce** on current LKW Kafka background-ingest canonical path. Post–Aug-27 fix commit not isolated in this qualification.

---

## 12. Root-cause conclusion

R1 left producer/persistence cause **NOT PROVEN**. R2 real execution proves **current** path produces, persists, reads, reconstructs, and projects transport causal evidence end-to-end. Category **H** — gap remediated or stale relative to qualified topology; not a universal all-topologies closure.

---

## 13. Secondary findings

1. **HTTP background-task proof route 503** — `local_workspace` FastAPI `ToolWiringContext.message_bus` not materialized; E0 used canonical `create_local_workspace_kafka_message_bus()` enqueue (real transport).
2. **Background ingest handler FAILED** after admission — causal continuity still proven (evidence precedes handler).
3. **`list_for_execution` boundedness** — `UNBOUNDED COMPATIBILITY FACADE` in `ExecutionReconstructor`; not root cause (secondary hardening gap).
4. **Operator config** — `INTERGRAX_DIAGNOSTIC_PROBLEM_LIST_CURSOR_SECRET` required in LKW `.env` for stack boot (session-local append only).

---

## 14. Non-claims

- Does not qualify File Watcher overlay, Celery, or all queue providers
- Does not close DG-004 universally (one canonical path only)
- Does not prove all retry/attempt variants
- Does not modify DG-005

---

## 15. Next task

**DG-004-FINAL-CLOSURE-REVIEW-R3**

---

## Precondition tests

```text
uv run pytest \
  tests/unit/runtime/background_execution/test_required_audit_evidence_admission.py \
  tests/unit/runtime/background_execution/test_background_causal_evidence_admission_paths.py \
  tests/unit/runtime/diagnostics/test_execution_reconstruction.py \
  -q --basetemp=.tmp/session/dg004-r2/pytest-basetemp
→ 29 passed

uv run pytest \
  tests/unit/runtime/observability/test_durable_causal_evidence_persistence.py \
  tests/unit/runtime/observability/test_causal_evidence_paging.py \
  -q --basetemp=.tmp/session/dg004-r2/pytest-basetemp-persistence
→ 22 passed
```

**Synthetic evidence injection:** NONE  
**Manual canonical truth mutation:** NONE  
**`--ignore` used:** NO

**Qualification harness:** `.tmp/session/dg004-r2/dg004_r2_qualification.py` (not committed)
