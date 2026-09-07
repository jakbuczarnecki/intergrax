# DG-004 — Async Transport Causal Continuity Root-Cause Audit (R1)

**Audit ID:** DG-004-ASYNC-CAUSAL-CONTINUITY-ROOT-CAUSE-AUDIT-R1  
**Date:** 2026-09-07  
**Mode:** Review-only / evidence-driven root-cause localization  
**START_HEAD:** `3b9b96ace67b885093e778dc7b440d7221a920e7`  
**Auditor constraint:** No production or test changes in this task.

---

## 1. Verdict

| Field | Value |
|-------|-------|
| **Audit result** | **PASS** (localization complete within R1 static + existing-test evidence) |
| **Root cause proven** | **NO** |
| **Root cause category** | **I — NOT YET PROVEN** |
| **First broken boundary** | **NOT YET PROVEN** (symptom localizes to empty causal read; upstream producer/persistence/topology not statically proven fail) |
| **DG-004 status after R1** | **OPEN / QUALIFICATION REQUIRED** |

---

## 2. Historical symptom

**Proven terminal symptom (post DG-A, ledger `5f1fff2e9`, 2026-08-27):**

```text
fresh background execution exists
RuntimeEvents exist
DiagnosticOrchestrator reaches DQ-2
has_transport_evidence=false
```

**Historical observation layer (exact):**

```text
DiagnosticOrchestrator._analyze_execution_scope
  → ExecutionReconstructor.reconstruct_execution(...)
  → ExecutionReconstruction.has_transport_evidence  (= bool(causal_evidence))
  → DiagnosticExecutionAnalysis.has_transport_evidence
```

No automated proof artifact in-repo captures the literal `has_transport_evidence=false` field value. The observation is ledger-documented from LKW File Watcher revalidation after DG-A (`24506c3c`). **HISTORICAL OBSERVATION DOES NOT PROVE CAUSE** at the persistence/producer layer without R2 real qualification.

**DQ-2 interpretation:** `has_runtime_events=true` and timeline reconstructable; **DQ-3 blocked** because transport causal continuity absent from reconstruction.

---

## 3. Scope / non-scope

**In scope:** transport → required causal evidence → execution identity → `CausalEvidencePersistence` → `ExecutionReconstructor` → `DiagnosticOrchestrator` projection; LKW Kafka worker composition; host diagnostic wiring.

**Out of scope:** TaskQueue/Celery/Kafka architecture redesign, worker leasing, DG-005 runtime-event topology (except noting runtime history was present historically), reconstructor paging redesign as root cause.

---

## 4. Canonical causal path

**Builder:** `build_transport_triggered_execution_evidence` in `intergrax/runtime/background_execution/required_audit_evidence.py`

| Field | Source |
|-------|--------|
| `evidence_id` | `mint_event_id()` once per attempt (or reused on persistence retry) |
| `relation_kind` | `CausalRelationKind.TRANSPORT_TASK_TRIGGERED_EXECUTION` |
| `tenant_id` | `execution_identity.tenant_id` |
| `source.provider` | `transport_ref.provider` |
| `source.task_id` | `transport_ref.transport_task_id` |
| `source.tenant_id` | `execution_identity.tenant_id` |
| `target.task_id` | `execution_identity.task_id` |
| `target.run_id` | `execution_identity.run_id` |
| `target.attempt_id` | `execution_identity.attempt_id` |
| `target.tenant_id` | `execution_identity.tenant_id` |

**Admission gate:** `admit_background_execution_handler` — persist required evidence, then invoke handler. Persistence failure raises `RequiredAuditEvidencePersistenceError` (fail-closed).

**Identity authority:** `BackgroundExecutionIdentity` is established in `admit_background_execution_reentry` → `resolve_background_execution` **before** evidence build. Evidence builder does not mint TaskId/RunId. Broker message `run_id` is transport correlation only (`bootstrap.py` contract).

---

## 5. Production call chain

**Expected ordering (platform invariant):**

```text
transport task received
  → identity bootstrap / re-entry admission
  → admit_background_execution_handler (required TRANSPORT_TASK_TRIGGERED_EXECUTION)
  → execute_logical_task
```

### Admission paths table

| Path | Uses canonical admission | Evidence required | Persistence supplied |
|------|--------------------------|-------------------|----------------------|
| **broker** (`BrokerWorkerBase.process_message`) | **YES** | **YES** | Constructor-injected `CausalEvidencePersistence` |
| **WorkerRuntime** (`process_request`) | **YES** | **YES** | Constructor-injected `CausalEvidencePersistence` |
| **DocumentStore** (`DocumentStoreTaskWorker`) | **YES** | **YES** | Constructor-injected `CausalEvidencePersistence` |
| **Celery** (`register_dispatcher_task` → `intergrax.execute`) | **YES** | **YES** | `causal_evidence_persistence` parameter |
| **LKW Kafka worker** (`KafkaWorker` → `BrokerWorkerBase`) | **YES** | **YES** | `resolve_host_queue_execution_dependencies(runtime).causal_evidence_persistence` → `wire_causal_evidence_persistence(document_store=...)` |

### LKW real-path trace

| Step | Production location | Identity available | Causal persistence | Canonical admission |
|------|---------------------|--------------------|--------------------|---------------------|
| File Watcher enqueue | LKW file watcher / message bus producer | transport `task_id`, `tenant_id` | N/A (producer) | N/A |
| Kafka message | `BrokerWorkerBase.process_message` | message `task_id`, `tenant_id`, queue `run_id` (correlation) | `_causal_evidence_persistence` | **YES** via `admit_background_execution_handler` |
| Identity resolution | `admit_background_execution_reentry` | `BackgroundExecutionIdentity` (task/run/attempt) | KV identity store (Redis) | precedes evidence |
| Handler | `execute_logical_task` | `execution_identity` | — | after evidence persist |

**Does the real LKW worker path reach required causal admission?** **YES** (static production call chain).

**Required causal evidence persistence fail-closed:** **PASS** (`persist_required_audit_evidence` propagates failures; handler not invoked on failure).

---

## 6. Persistence composition

| Consumer | Abstraction | Concrete composition | Same canonical store? |
|----------|-------------|----------------------|------------------------|
| **Worker admission** | `CausalEvidencePersistence` | `wire_causal_evidence_persistence(document_store=wiring_context.document_store)` from `resolve_host_queue_execution_dependencies` | **By design YES** when shared `DocumentStore` backend |
| **Diagnostic reconstructor** | `CausalEvidencePersistence` | `wire_causal_evidence_persistence(document_store=...)` from `resolve_host_diagnostic_read_dependencies` / `resolve_host_diagnostic_runtime_dependencies` | **By design YES** when shared `DocumentStore` backend |

**LKW worker:** `build_harness_host_runtime(..., document_store=resolve_lkw_runtime_document_store(settings))` then queue deps from same runtime.

**LKW Docker (kafka overlay):** `local_workspace` and `lkw-background-worker` both set `INTERGRAX_MONGODB_URI`; LKW managed workspace collection via `LKW_MANAGED_WORKSPACE_COLLECTION` (default `lkw_managed_workspaces`). Causal partition: `intergrax.causal_evidence.v1:{tenant_id}` within that store.

**Conclusion:** **SAME CANONICAL STORE by composition intent** in queue-enabled LKW Docker topology. **NOT PROVEN** at runtime without cross-process read-after-write qualification (separate Python wrapper objects are expected).

---

## 7. Identity continuity

| Identity component | Producer value source | Persisted evidence field | Reconstructor query |
|--------------------|----------------------|--------------------------|---------------------|
| tenant | `execution_identity.tenant_id` | `evidence.tenant_id`, `target.tenant_id` | `tenant_id` arg |
| TaskId | identity persistence bootstrap | `target.task_id` | `task_id` arg |
| RunId | identity persistence bootstrap | `target.run_id` | `run_id` arg |
| AttemptId | re-entry / attempt lifecycle | `target.attempt_id` | grouping only |
| transport provider | message / worker provider | `source.provider` | N/A |
| transport task | message `task_id` | `source.task_id` | N/A |

**Handler/run_id alignment:** `execute_logical_task` uses `str(execution_identity.run_id)`; `assert_handler_run_id_matches_identity` enforces match. **No reminting** between evidence target and runtime event scope when diagnostic uses canonical execution TaskId/RunId.

**Execution identity continuity (architecture):** **PASS**. **Runtime proof for historical LKW run:** **NOT PROVEN**.

---

## 8. Reconstruction consumption

**API:** `ExecutionReconstructor.reconstruct_execution(tenant_id, task_id, run_id)`

**Causal lookup:**

```python
causal = self._causal_evidence.list_for_execution(
    tenant_id=tenant_id,
    task_id=task_id,
    run_id=run_id,
)
```

- No transport-provider filter.
- No attempt-only filter on top-level causal tuple.
- Out-of-scope evidence raises `ExecutionReconstructionIntegrityError` (not silent empty).

**`has_transport_evidence` semantics:** `bool(self.causal_evidence)` — **broader than** `TRANSPORT_TASK_TRIGGERED_EXECUTION` name; **not root cause** unless non-transport causal evidence populates without transport (currently harmless if only transport relation is written).

**Reconstructor preserves evidence:** **PASS** (tuple copied into `ExecutionReconstruction`).

**Boundedness:** `list_for_execution()` compatibility facade — **UNBOUNDED COMPATIBILITY FACADE**. **Not root cause** for `has_transport_evidence=false` (empty vs truncated distinction).

---

## 9. Projection path

```text
DiagnosticOrchestrator._analyze_execution_scope
  → reconstruction.has_transport_evidence
  → DiagnosticExecutionAnalysis.has_transport_evidence
```

**Projection preserves evidence:** **PASS** (direct copy; no recompute/filter).

Historical `false` measured at **`DiagnosticExecutionAnalysis.has_transport_evidence`**, which reflects **empty `ExecutionReconstruction.causal_evidence`**, not an outer projection defect.

---

## 10. Historical vs current code

**Historical symptom commit:** `5f1fff2e9` (2026-08-27)  
**DG-A fix:** `24506c3c` (2026-08-26) — wires `resolve_host_queue_execution_dependencies` into LKW worker; **enables** causal persistence injection; **does not** alone prove cross-process diagnostic continuity.

| Change (after `5f1fff2e9`) | Could affect DG-004? |
|----------------------------|---------------------|
| `587f3ef75` route LKW background worker through canonical execution | **YES** — identity/admission path |
| `75c1cf2a3` preserve background attempt identity on redelivery | **YES** — attempt/evidence per retry |
| `caf57aff9` converge background worker re-entry | **YES** |
| `7a6ae831f` / `18274baf1` durable re-entry conformance | **YES** — admission durability |
| `5272829dc` / `0fb9027ca` terminal/attempt provider selection | **Possible** — admission prerequisites |
| DG-002 transport bounded read (`f2198be56`… closure) | **NO** for DG-004 continuity |

**Static assessment:** Historical gap **may be stale** relative to current code; **REAL REVALIDATION REQUIRED** before any remediation authorization.

---

## 11. Evidence ladder

| Boundary | Status | Evidence |
|----------|--------|----------|
| transport submission created | **PROVEN PASS** | Ledger post DG-A: fresh background execution |
| background execution identity resolved | **PROVEN PASS** | RuntimeEvents exist for diagnosed scope |
| required causal evidence built | **NOT PROVEN** | No runtime capture from historical run |
| evidence append invoked | **NOT PROVEN** | No runtime capture |
| evidence persisted | **NOT PROVEN** | No direct store read in historical artifact |
| persisted evidence queryable by execution | **NOT PROVEN** | Symptom implies empty read; cause not isolated |
| worker/diagnostics share canonical causal store | **NOT PROVEN** | Composition intent SAME; runtime topology unqualified |
| reconstructor queries correct identity | **PROVEN PASS** | Code + unit tests |
| reconstructor receives evidence | **PROVEN FAIL** (historical symptom) | `has_transport_evidence=false` ⇒ empty `causal_evidence` |
| reconstructor preserves evidence | **PROVEN PASS** | Code path |
| operator/proof projection preserves evidence | **PROVEN PASS** | Orchestrator direct copy |

**First-failure rule:** First boundary **not PROVEN PASS** upstream of the proven empty read is **`required causal evidence built`** (or earlier persistence boundaries). None are **PROVEN FAIL** statically — only **NOT PROVEN**.

---

## 12. First broken boundary

```text
NOT YET PROVEN
```

**Localized symptom boundary (not root cause):**

```text
ExecutionReconstructor causal_evidence lookup returns empty for diagnosed execution scope
```

---

## 13. Root-cause category

**Primary: I — NOT YET PROVEN**

**Hypothesis tree (R1 disposition):**

| Hypothesis | R1 disposition |
|------------|----------------|
| H1 producer missing | **DISPROVEN for current LKW code** (canonical admission on Kafka/broker path) |
| H2 built but not persisted | **NOT PROVEN** |
| H3 persistence topology mismatch | **NOT PROVEN** (design: same DocumentStore; runtime unverified) |
| H4 execution identity mismatch | **NOT PROVEN** (architecture PASS; historical run unverified) |
| H5 lookup defect | **NOT PROVEN** (conformance tests PASS for InMemory/DocumentStore) |
| H6 reconstructor consumption defect | **DISPROVEN** (empty input, not rejection) |
| H7 projection defect | **DISPROVEN** |
| H8 historical gap remediated | **PLAUSIBLE** — material post-symptom commits; **revalidation required** |

---

## 14. Secondary observations

1. **SECONDARY HARDENING GAP: RECONSTRUCTOR CAUSAL READ BOUNDEDNESS** — `list_for_execution()` unbounded compatibility facade; unrelated to missing continuity.
2. **Property naming:** `has_transport_evidence` = any causal evidence, not relation-kind-specific; model ambiguity only.
3. **DG-A scope:** fixed worker **composition** (`causal_evidence_persistence` injection); did **not** qualify end-to-end diagnostic continuity.
4. **Wiring test suite:** 16 failed / 4 passed in audit env due to legal capability / ollama dependency resolution — **environment blocker**, not DG-004 defect (see §15).

---

## 15. Required next remediation or qualification

**Recommended single next task:** **DG-004-R2 REAL QUALIFICATION (one execution)**

Minimal experiment checkpoints:

```text
E0 transport accepted (Kafka message consumed)
E1 execution identity captured (task_id, run_id, attempt_id, tenant_id)
E2 evidence object built (relation_kind=TRANSPORT_TASK_TRIGGERED_EXECUTION)
E3 append success (worker process)
E4 list_for_execution(tenant, task_id, run_id) returns evidence (direct store query, same Mongo as worker)
E5 ExecutionReconstructor returns same evidence
E6 DiagnosticOrchestrator has_transport_evidence=true
```

**Forbidden:** manual post-hoc evidence injection, monkeypatching flags, mock-only final qualification.

---

## 16. Non-claims

- This audit does **not** claim DG-004 is fixed in current code.
- This audit does **not** claim persistence topology mismatch or identity mismatch as proven root cause.
- This audit does **not** authorize speculative production changes.
- This audit does **not** reopen DG-002 or DG-005.

---

## 17. DG-002 / DG-005 relationship

| Entry | R1 disposition |
|-------|----------------|
| **DG-002** | **CLOSED / UNCHANGED** — transport ref → execution scope discovery; separate from reconstruction continuity |
| **DG-005** | **UNCHANGED** — runtime event topology; historical run had `has_runtime_events=true` |

---

## 18. Architecture quality assessment (DG-004 path)

| Invariant | Rating |
|-----------|--------|
| Single canonical causal evidence contract | **PASS** |
| Typed transport/source and execution/target refs | **PASS** |
| Mandatory tenant scope | **PASS** |
| Read-only reconstruction | **PASS** |
| No identity minting in evidence builder | **PASS** |
| Provider/backend abstraction | **PASS** |
| No queue vendor dependency in diagnostics | **PASS** |
| Fail-closed required evidence persistence | **PASS** |

---

## 19. Enterprise anti-pattern audit (material DG-004 path only)

No material `Any` / reflection / hidden fallback on the causal continuity chain. `WorkerRuntime._emit` uses broad `except Exception` for optional correlation_id parsing — **not material** to causal continuity.

---

## 20. Admission / reconstruction / persistence tests (R1)

| Suite | Command | Result |
|-------|---------|--------|
| Admission | `uv run pytest tests/unit/runtime/background_execution/test_required_audit_evidence_admission.py tests/unit/runtime/background_execution/test_background_causal_evidence_admission_paths.py -q --basetemp=.tmp/session/dg004-audit-r1/pytest-basetemp-admission` | **17 passed** |
| Reconstruction | `uv run pytest tests/unit/runtime/diagnostics/test_execution_reconstruction.py tests/unit/runtime/diagnostics/test_diagnostic_orchestrator.py -q --basetemp=.tmp/session/dg004-audit-r1/pytest-basetemp-reconstruction` | **32 passed** |
| Persistence | `uv run pytest tests/unit/runtime/observability/test_durable_causal_evidence_persistence.py tests/unit/runtime/observability/test_causal_evidence_paging.py -q --basetemp=.tmp/session/dg004-audit-r1/pytest-basetemp-persistence` | **22 passed** |
| Wiring | `uv run pytest tests/unit/applications/local_workspace_application/test_lkw_background_worker_queue_dependencies.py tests/unit/applications/test_host_queue_execution_wiring.py applications/local_workspace_application/tests/host/test_lkw_background_canonical_execution.py -q --basetemp=.tmp/session/dg004-audit-r1/pytest-basetemp-wiring` | **16 failed, 4 passed** (legal capability / ollama env) |

`--ignore` used: **NO**  
New skips: **NONE**

---

## 21. Audit sign-off statement

```text
DG-004 ROOT-CAUSE AUDIT R1 = PASS

ASYNC CAUSAL CONTINUITY GAP = HISTORICALLY PROVEN

STATIC ROOT CAUSE = NOT YET PROVEN

EVIDENCE CHAIN = INCOMPLETE

NO SPECULATIVE FIX AUTHORIZED

DG-002 = CLOSED / UNCHANGED

DG-005 = UNCHANGED

NEXT = DG-004-R2 real qualification (E0–E6 single execution)
```
