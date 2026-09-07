# DG-001D Supervisor Pre-Engine Failure — HOST-DIAG-3 Conformance (R3)

**Verdict:** PASS

**Date:** 2026-09-07

**Branch:** `development`

**Start HEAD:** `73d023d0a34102ec35e7f01ceae56129ea72ff5e`

**Task:** `DG-001D-SUPERVISOR-HOST-DIAG-3-CONFORMANCE-R3` — conformance qualification only; no production or diagnostics-core changes.

---

## 1. Verdict

```text
DG-001D SUPERVISOR HOST-DIAG-3 CONFORMANCE R3 = PASS
```

R3 proves that the R2 supervisor pre-engine `APPLICATION_FAILED` producer is semantically compatible with the existing Central Diagnostics spine without diagnostics-core changes, manual signal construction, or direct Problem insertion.

**DG-001D status after R3:**

```text
PRODUCER + HOST-DIAG-3 CONFORMANCE QUALIFIED / REAL INTEGRATION QUALIFICATION PENDING
```

**DG-001 overall:** `PARTIALLY ADDRESSED` (unchanged).

**Next task:** `DG-001D-REAL-SUPERVISOR-PRE-ENGINE-FAILURE-QUALIFICATION-R4`

---

## 2. Tested topology

```text
HostedApplicationSupervisor.run()
  → HostedApplicationEvent(APPLICATION_FAILED)
  → HostedApplicationDiagnosticEventPublisher
      → ObservabilityHostedApplicationEventPublisher (first)
      → hosted_application_failure_to_problem_signal
      → PlatformProblemSignal
      → DiagnosticOrchestrator
      → ProblemLifecycleEngine
      → InMemoryProblemPersistence (deterministic test stack)
      → DiagnosticReadService
```

**Composition:** real production supervisor + real HOST-DIAG-3 publisher + existing typed diagnostic test stack (`_DiagnosticTestStack` pattern from HOST-DIAG-3 integration harness).

**Tenant binding:** explicit test-owned `HostedDiagnosticTenantBinding(tenant_id="dg001d-r3-test")`. Supervisor remains tenant-agnostic.

**Canonical test module:** `tests/unit/hosting/supervisor/test_supervisor_host_diag_3_conformance.py`

---

## 3. Exact scenarios (16 semantic tests)

| # | Scenario | Result |
|---|----------|--------|
| 1 | `engine_factory` raises → Problem via `supervisor.run()` | PASS |
| 2 | `engine_instance_id_mismatch` contract validation → Problem | PASS |
| 3 | `APPLICATION_INSTANCE` subject | PASS |
| 4 | No execution identity (`TaskId`/`RunId`/execution ref absent) | PASS |
| 5 | Identity fidelity event ↔ occurrence ↔ tenant = 100% | PASS |
| 6 | Raw sentinel `DG001D-R3-SECRET-SENTINEL` not persisted | PASS |
| 7 | Exactly one `APPLICATION_FAILED` and one Problem/occurrence per attempt | PASS |
| 8 | Observability export before diagnostics projection | PASS |
| 9 | Diagnostic projection failure isolated (orchestrator throws) | PASS |
| 10 | Supervisor truth preserved when diagnostics fail | PASS |
| 11 | Success path → 0 DG-001D Problems | PASS |
| 12 | Runtime engine failure → 0 supervisor pre-engine Problems | PASS |
| 13 | Recurrence: two `instance_id`s → one Problem / two occurrences | PASS |
| 14 | Grouping does not depend on `instance_id` alone | PASS |
| 15 | Composed publisher outer failure isolated | PASS |
| 16 | `DiagnosticReadService.get_problem` visibility | PASS |

Additional harness scenarios covered inline: `stop_before_launch` → 0 events / 0 Problems.

**Main proof:** every conformance scenario drives `HostedApplicationSupervisor.run()` — no bypass via direct `supervisor_pre_engine_failure_to_hosted_event()` as primary proof.

---

## 4. Identity model

| Field | Source | Problem occurrence |
|-------|--------|-------------------|
| `application_id` | `HostedApplicationDefinition` | `APPLICATION_INSTANCE.application_id` |
| `instance_id` | supervisor minted before `engine_factory` | `APPLICATION_INSTANCE.instance_id` |
| `tenant_id` | `HostedDiagnosticTenantBinding` | Problem tenant |
| Execution identity | **NONE** | `subject_ref.execution()` is `None` |

**Identity fidelity:** 100% across all failure scenarios.

---

## 5. APPLICATION_INSTANCE proof

Problems and occurrences are associated with `DiagnosticSubjectKind.APPLICATION_INSTANCE` using supervisor-minted `instance_id` and definition `application_id`. No `TaskId`, `RunId`, `AttemptId`, or `ExecutionId` fabricated.

---

## 6. Problem projection

### Factory failure (primary)

```text
phase = engine_construction
reason_code = engine_factory_failed
exception_type = HostedApplicationSupervisorError
process_role = hosted_application_supervisor
```

### Contract validation (secondary)

```text
phase = engine_contract_validation
reason_code = engine_instance_id_mismatch
```

Both project through existing `hosted_application_failure_to_problem_signal` without diagnostics-core changes.

---

## 7. Observability ordering

Recording orchestrator wrapper confirms observability envelope count ≥ 1 before `DiagnosticOrchestrator.run()` executes for the same `APPLICATION_FAILED` event. Platform signal export present.

---

## 8. Isolation

| Failure surface | Supervisor truth | Diagnostics side-effect |
|-----------------|------------------|-------------------------|
| `DiagnosticOrchestrator.run()` throws | `SUPERVISOR_ERROR` preserved | 0 Problems; observability export still present |
| Composed publisher raises on `APPLICATION_FAILED` | Same exit classification as baseline | 0 Problems |
| R2 publisher isolation (separate module) | Unchanged | Composes with R3 composed publisher isolation |

No projection exception escapes supervisor control flow.

---

## 9. Security

Factory throws `DG001D-R3-SECRET-SENTINEL`. Verified absent from:

- `HostedApplicationEvent` payload JSON
- `PlatformProblemSignal` serialization
- `Problem` read-model repr
- `ProblemOccurrence` repr
- `DiagnosticReadService` output

---

## 10. Recurrence / grouping

Two separate failed supervisor runs with the same application, phase, reason, and exception type but different `instance_id`s group under **one Problem** with **two occurrences**. Structural grouping does not key on `instance_id` or supervisor attempt number.

---

## 11. Non-claims

R3 does **not** prove:

- real Mongo / Elasticsearch durability
- cross-process persistence
- default runner HOST-DIAG-3 wiring (DG-001A)
- public launcher bootstrap (DG-001C)
- LKW product wiring
- real child-process supervisor qualification

R3 proves **semantic compatibility only** between R2 supervisor producer and existing Central Diagnostics spine.

---

## 12. Production diff gate

```text
intergrax/     = NONE
applications/  = NONE
```

Changes limited to tests + qualification documentation + ledger update.

---

## 13. Focused regression

| Suite | Count |
|-------|-------|
| Supervisor tests | included |
| DG-001D R3 conformance | 16 |
| HOST-DIAG-3 integration | included |
| Process bootstrap / guarded primitive | included |
| HOST-DIAG-3 composition gate | included |
| **Total focused** | **91 passed** |

`--ignore`: NO. New skips: NONE.

**Ruff:** PASS. **Pyright:** 0 errors on changed test file.

---

## 14. Next real qualification step

**DG-001D-REAL-SUPERVISOR-PRE-ENGINE-FAILURE-QUALIFICATION-R4**

Target:

```text
real supervisor process
→ controlled engine_factory failure
→ durable Mongo Problem persistence
→ separate-process operator read (DiagnosticReadService)
→ restart recurrence under real deployment composition
```
