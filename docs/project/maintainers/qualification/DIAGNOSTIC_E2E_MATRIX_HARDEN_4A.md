# Diagnostic E2E Matrix — HARDEN-4A

**Scope:** central diagnostic spine qualification inventory and gap classification  
**Owner:** Observability / DIAG maintainers  
**Architecture:** [`docs/project/architecture/DIAGNOSTICS.md`](../../architecture/DIAGNOSTICS.md) · [`OBSERVABILITY.md`](../../architecture/OBSERVABILITY.md)

**Related ledger:** [`DIAGNOSTIC_GAP_LEDGER.md`](DIAGNOSTIC_GAP_LEDGER.md)
**Status:** HARDEN-4 complete · HARDEN-5 documentation closeout complete

**Git baseline (4A audit):** `403dea4523a872866062d01e3469a602804720af`
**Documentation closeout:** [`DIAGNOSTIC_HARDENING_CLOSEOUT.md`](DIAGNOSTIC_HARDENING_CLOSEOUT.md) · canonical architecture [`DIAGNOSTICS.md`](../../architecture/DIAGNOSTICS.md)

---

## Proof strength legend

| Level | Meaning |
|-------|---------|
| **P1** | Unit semantics — single mechanism |
| **P2** | Integration — multiple real components, often in-process |
| **P3** | Product-host E2E — real host/API → runtime → diagnostics → read model |
| **P4** | External infrastructure E2E — real process/network/Docker/durable backend |

## Failure injection legend

| Class | Meaning |
|-------|---------|
| **FI-A** | Real external failure (e.g. Docker stop) |
| **FI-B** | Real provider capability failure through production abstraction |
| **FI-C** | Controlled deterministic injection at architectural seam |
| **FI-D** | Mock of internal/private function |

## Matrix status legend

| Status | Meaning |
|--------|---------|
| **PROVEN** | Meets minimum required level with skeptical-architect confidence |
| **PARTIALLY_PROVEN** | Compositional or lower-level coverage; production path gap remains |
| **MISSING** | No adequate proof |
| **NOT_APPLICABLE** | Invariant not in current production scope |
| **DEFERRED_WITH_REASON** | Intentionally out of 4A scope |

---

## Existing proof inventory

| File | Test | Level | Infrastructure | Invariant |
|------|------|-------|----------------|-----------|
| `tests/integration/runtime/test_harden_4b_tenant_diagnostic_isolation_e2e.py` | `test_harden_4b_same_violation_isolated_between_tenants`; `test_harden_4b_cross_tenant_problem_id_read_returns_none` | P3 | Governed-contractor HTTP host (shared), SQLite RuntimeEvents, InMemory DocumentStore, observability disabled | M17/M18 same violation class → separate Problems; tenant-scoped lists; direct-ID read isolation |
| `tests/integration/runtime/test_harden_4c_clean_diagnostic_host_e2e.py` | `test_harden_4c_clean_product_host_execution_creates_no_problem` | P3 | Governed-contractor HTTP host (shared), SQLite RuntimeEvents, InMemory DocumentStore, observability disabled, `inject_violation=False` | M1 clean HTTP execution → TASK_COMPLETED → diagnostics active → zero Problems |
| `tests/integration/runtime/test_harden_4d_problem_lifecycle_host_e2e.py` | `test_harden_4d_resolve_then_same_violation_reopens_same_problem` | P3 | Governed-contractor HTTP host (shared), SQLite RuntimeEvents, InMemory DocumentStore, observability disabled, deterministic violation injector | M5/M6 HTTP run → Problem OPEN → explicit resolve → RESOLVED read → second run reopens same `problem_id` |
| `tests/integration/runtime/test_harden_4e_diagnostic_read_truth_e2e.py` | `test_harden_4e_reconstructs_problem_from_canonical_runtime_evidence`; `test_harden_4e_missing_execution_evidence_returns_unavailable_not_fabricated_diagnosis` | P3 | Governed-contractor HTTP host (shared), SQLite RuntimeEvents, InMemory DocumentStore, observability disabled; M20 uses shared Problem persistence with isolated RuntimeEvent store | M19/M20 host read truth: reconstruction from canonical events; unavailable read without fabrication |
| `tests/integration/runtime/test_diag_final_external_otel_e2e.py` | `test_diag_final_external_otel_spine_proof` | P4 | Governed-contractor HTTP host, SQLite RuntimeEvents, InMemory DocumentStore, Docker OTLP Collector | execution → RuntimeEvent → terminal diagnostics → Problem → DiagnosticReadService; vendor DOWN/UP; restart persistence |
| `tests/integration/runtime/test_terminal_diagnostic_production_e2e.py` | `test_clean_execution_does_not_create_problem` | P3 | NexusLoop + UnifiedTaskRunner, in-memory stores | clean success → no Problem |
| same | `test_real_nexus_execution_triggers_diagnostics_without_manual_orchestrator` | P3 | NexusLoop path | violation → orchestrator findings |
| same | `test_separate_terminal_executions_reconcile_same_problem` | P3 | in-memory + DiagnosticReadService | occurrence aggregation + read model |
| same | `test_different_terminal_signatures_create_distinct_problems` | P3 | shared in-memory persistence | problem separation |
| same | `test_diagnostic_failure_does_not_change_business_outcome` | P3 | FI-C trigger monkeypatch | M7 subsystem failure evidence + business survives |
| same | `test_tenant_isolation_for_terminal_diagnostics` | P3 | orchestrator capture only | tenant passed to orchestrator (not persistence isolation) |
| `tests/integration/applications/architecture/test_harden_1c_durable_problem_restart_proof.py` | `test_harden_1c_durable_problem_survives_real_process_restart` | P4 | Mongo via IntegrationProfile → DocumentStore → ProblemPersistence; subprocess phases | cross-process durability + DiagnosticReadService after restart |
| `tests/integration/applications/architecture/test_harden_1d_durable_problem_store_failure_semantics.py` | `test_harden_1d_terminal_create_failure_preserves_business_result` (+ update/recovery) | P3 | FI-B DelegatingFailingConditionalDocumentStore | M7/M8 business survives; degradation visible |
| `tests/integration/applications/architecture/test_harden_4f_mongo_problem_store_failure_e2e.py` | `test_harden_4f_mongo_problem_store_failure_and_recovery_e2e` | P4 | Governed-contractor HTTP host, SQLite RuntimeEvents, Mongo DocumentStore via IntegrationProfile, FI-A Docker Mongo stop/start | M8 real Mongo outage + same-process recovery |
| `tests/integration/applications/architecture/test_harden_2a_durable_problem_concurrency_proof.py` | `test_harden_2a_cross_process_concurrent_update_on_mongodb` | P4 | Mongo + 2 subprocess workers | M10 concurrent occurrences preserved |
| `tests/integration/applications/architecture/test_harden_2c_durable_problem_lifecycle_concurrency_proof.py` | `test_harden_2c_cross_process_lifecycle_reconcile_on_mongodb` | P4 | Mongo + subprocess | M10/M11 lifecycle reconcile race |
| same | `test_harden_2c_cross_process_lifecycle_resolve_on_mongodb` | P4 | Mongo + subprocess + read-final worker | M5/M11 resolve OCC cross-process |
| same | `test_harden_2c_cross_process_lifecycle_create_race_on_mongodb` | P4 | Mongo + subprocess | M10 create race convergence |
| `tests/unit/runtime/diagnostics/test_harden_2a_problem_persistence_concurrency.py` | multiple `test_harden_2a_*` | P2 | in-memory / document store | OCC, tenant isolation at persistence, orphan index |
| `tests/unit/runtime/diagnostics/test_harden_2b_problem_lifecycle_occ_retry.py` | multiple `test_harden_2b_*` / `test_harden_2d_*` | P2 | in-memory persistence | M11 resolve/occurrence OCC retry semantics |
| `tests/unit/runtime/diagnostics/test_durable_problem_persistence.py` | `test_document_store_orphan_reconciliation_index_fails_closed` (+ stale/wrong-scope) | P2 | DocumentStoreProblemPersistence | M12 reconciliation index integrity |
| `tests/unit/runtime/diagnostics/test_problem_lifecycle.py` | `test_explicit_resolve_and_recurrence_reopens` | P1 | InMemoryProblemPersistence | M6 contract: **reopen** same Problem on recurrence after resolve |
| `tests/unit/runtime/diagnostics/test_diagnostic_read_service.py` | `test_get_problem_other_tenant_returns_none` | P2 | in-memory | M18 read isolation (unit) |
| same | `test_get_problem_reconstructs_through_diag_pipeline` | P2 | in-memory | M19 reconstruction |
| same | `test_get_problem_unavailable_when_execution_evidence_missing` | P2 | in-memory | M20 degraded/unavailable read |
| `tests/unit/runtime/diagnostics/test_terminal_execution_diagnostic_trigger.py` | `test_clean_execution_sequence_produces_no_problem` | P1 | seeded events + trigger | M1 no false positive at trigger layer |
| `tests/unit/runtime/diagnostics/test_diagnostic_subsystem_failure_evidence.py` | multiple | P2 | in-memory RuntimeEvent store | M7 failure evidence identity |
| `tests/unit/runtime/observability/test_harden_3c_export_failure_semantics.py` | (module) | P2 | operator wiring | M14 export failure isolation |
| `tests/unit/runtime/observability/test_harden_3d_exporter_health.py` | (module) | P2 | health registry | M15 health degraded/recovered |
| `tests/unit/runtime/architecture/test_harden_3f_qualification_matrix.py` | `test_harden_3_invariant_matrix_has_required_rows` | P1 | doc gate | maps HARDEN-3F observability matrix |

---

## E2E matrix M1–M24

| ID | Scenario | User-visible guarantee | Existing proof | Level | Deterministic | External | Status | Gap |
|----|----------|------------------------|----------------|-------|---------------|----------|--------|-----|
| M1 | Clean success / no false positive | Successful execution leaves canonical evidence and **no** Problem | `test_harden_4c_clean_diagnostic_host_e2e.py::test_harden_4c_clean_product_host_execution_creates_no_problem`; `test_terminal_diagnostic_production_e2e.py::test_clean_execution_does_not_create_problem`; `test_terminal_execution_diagnostic_trigger.py::test_clean_execution_sequence_produces_no_problem` | P3 / P1 | yes | no | **PROVEN** | — |
| M2 | Deterministic violation → Problem | execution → evidence → detection → central Problem → read API | `test_diag_final_external_otel_e2e.py::test_diag_final_external_otel_spine_proof` (`assert_problem_truth`); `test_terminal_diagnostic_production_e2e.py::test_separate_terminal_executions_reconcile_same_problem` (read path) | P4 + P3 | yes | Docker (OTel slice) | **PROVEN** | — |
| M3 | Occurrence aggregation | Same logical problem → one Problem, occurrences increment | `test_terminal_diagnostic_production_e2e.py::test_separate_terminal_executions_reconcile_same_problem`; `test_harden_2a_durable_problem_concurrency_proof.py::test_harden_2a_cross_process_concurrent_update_on_mongodb` | P3 + P4 | yes | Mongo (concurrency) | **PROVEN** | — |
| M4 | Different problem separation | Distinct signatures → distinct Problems | `test_terminal_diagnostic_production_e2e.py::test_different_terminal_signatures_create_distinct_problems` | P3 | yes | no | **PROVEN** | — |
| M5 | Resolve lifecycle | OPEN → resolved persisted; read model shows resolved; identity preserved | `test_harden_4d_problem_lifecycle_host_e2e.py::test_harden_4d_resolve_then_same_violation_reopens_same_problem`; `test_harden_2c_durable_problem_lifecycle_concurrency_proof.py::test_harden_2c_cross_process_lifecycle_resolve_on_mongodb`; `test_problem_lifecycle.py::test_explicit_resolve_and_recurrence_reopens` (resolve only); `test_diagnostic_read_service.py::test_list_status_filter` | P3 / P4 / P1 / P2 | yes | Mongo | **PROVEN** | — |
| M6 | Recurrence after resolve | Canonical contract: **reopen existing Problem** (status OPEN, occurrences continue, first_seen preserved) | `test_harden_4d_problem_lifecycle_host_e2e.py::test_harden_4d_resolve_then_same_violation_reopens_same_problem`; `test_problem_lifecycle.py::test_explicit_resolve_and_recurrence_reopens` | P3 / P1 | yes | no | **PROVEN** | — |
| M7 | Diagnostic subsystem failure | Business survives; durable failure evidence; ExecutionId preserved | `test_terminal_diagnostic_production_e2e.py::test_diagnostic_failure_does_not_change_business_outcome`; `test_harden_1d_durable_problem_store_failure_semantics.py::test_harden_1d_terminal_create_failure_preserves_business_result`; `test_diagnostic_subsystem_failure_evidence.py::test_failure_event_preserves_execution_identity` | P3 + P2 | yes | no | **PROVEN** | — |
| M8 | Problem Store outage | Persistence fails → business unaffected → diagnostic degradation visible | `test_harden_4f_mongo_problem_store_failure_e2e.py::test_harden_4f_mongo_problem_store_failure_and_recovery_e2e`; `test_harden_1d_durable_problem_store_failure_semantics.py` (create/update/recovery trio); `test_harden_1d_problem_persistence_write_failure.py` | P4 + P3 / P2 | yes | FI-A + FI-B | **PROVEN** | — |
| M9 | Cross-process durability | Process A writes Problem; Process B reads same Problem | `test_harden_1c_durable_problem_restart_proof.py::test_harden_1c_durable_problem_survives_real_process_restart` | P4 | yes | Mongo | **PROVEN** | — |
| M10 | Concurrent occurrence race | One logical Problem; both occurrences preserved; no lost update | `test_harden_2a_durable_problem_concurrency_proof.py::test_harden_2a_cross_process_concurrent_update_on_mongodb`; `test_harden_2c_*_create_race_on_mongodb` | P4 | yes | Mongo | **PROVEN** | — |
| M11 | Concurrent resolve/update OCC | CAS conflict → bounded convergence → deterministic final state | `test_harden_2b_problem_lifecycle_occ_retry.py` (in-process); `test_harden_2c_durable_problem_lifecycle_concurrency_proof.py` (cross-process) | P2 + P4 | yes | Mongo | **PROVEN** | — |
| M12 | Stale/corrupt reconciliation index | Orphan/wrong-scope index fails closed; CREATE-B retry behavior | `test_durable_problem_persistence.py::test_document_store_orphan_reconciliation_index_fails_closed` (+ related index tests); `test_harden_2a_problem_persistence_concurrency.py::test_harden_2a_orphan_reconciliation_index_raises_typed_canonical_pending_reason` | P2 | yes | no | **PROVEN** | Production-relevant; persistence-layer qualification sufficient |
| M13 | Host restart | RuntimeEvents + Problems preserved; diagnostic read works | `test_harden_1c_*` (Problem Mongo); `test_diag_final_external_otel_e2e.py` (SQLite RuntimeEvents + read after host rebuild) | P4 + P4 | yes | Mongo/SQLite | **PROVEN** | Compositional |
| M14 | Observability vendor outage | Collector DOWN → business + canonical + diagnostic truth survive | `test_diag_final_external_otel_e2e.py::test_diag_final_external_otel_spine_proof` (outage section) | P4 | yes | FI-A Docker | **PROVEN** | Maps HARDEN-3F |
| M15 | Observability recovery | DOWN → UP → new telemetry; health recovers; no replay of missed outage events | same test (recovery section) | P4 | yes | FI-A Docker | **PROVEN** | Maps HARDEN-3F-R1 |
| M16 | Identity correlation | Problem/occurrence/diagnostic output correlates to tenant/task/run/attempt/execution | `test_diag_final_external_otel_e2e.py` (collector identity attrs); `test_diagnostic_subsystem_failure_evidence.py::test_failure_event_preserves_execution_identity` | P4 + P2 | yes | Docker | **PROVEN** | — |
| M17 | Multi-tenant isolation | Same-looking issue across tenants → separate Problems; no cross-tenant occurrence merge | `test_harden_4b_tenant_diagnostic_isolation_e2e.py::test_harden_4b_same_violation_isolated_between_tenants`; `test_problem_lifecycle.py::test_same_recurrence_key_in_another_tenant_is_isolated`; `test_harden_2a_problem_persistence_concurrency.py::test_harden_2a_concurrent_create_tenant_isolation` | P3 + P1/P2 | yes | no | **PROVEN** | — |
| M18 | Diagnostic read isolation | Tenant A cannot read tenant B `problem_id` | `test_harden_4b_tenant_diagnostic_isolation_e2e.py::test_harden_4b_cross_tenant_problem_id_read_returns_none`; `test_diagnostic_read_service.py::test_get_problem_other_tenant_returns_none` | P3 + P2 | yes | no | **PROVEN** | — |
| M19 | Evidence reconstruction | Canonical RuntimeEvents → reconstruction → same diagnostic meaning | `test_harden_4e_diagnostic_read_truth_e2e.py::test_harden_4e_reconstructs_problem_from_canonical_runtime_evidence`; `test_diagnostic_read_service.py::test_get_problem_reconstructs_through_diag_pipeline` | P3 / P2 | yes | no | **PROVEN** | — |
| M20 | Missing/incomplete evidence | No fabricated diagnosis; typed unavailable/degraded | `test_harden_4e_diagnostic_read_truth_e2e.py::test_harden_4e_missing_execution_evidence_returns_unavailable_not_fabricated_diagnosis`; `test_diagnostic_read_service.py::test_get_problem_unavailable_when_execution_evidence_missing` | P3 / P2 | yes | no | **PROVEN** | — |
| M21 | AI not authority | AI output cannot create/override canonical diagnostic facts | — | — | — | — | **NOT_APPLICABLE** | Central engine is deterministic; `InvestigationConclusion` is separate non-canonical layer (`test_investigation_contracts.py`) |
| M22 | Unsupported scenario behavior | Valid-but-unsupported case must not create fake Problem or silent scenario fallback | `test_diagnostic_orchestrator.py::test_empty_subject_inputs_rejected` (invalid empty request only); M1 clean-path proofs (`test_harden_4c_*`, `test_terminal_execution_diagnostic_trigger.py::test_clean_execution_sequence_produces_no_problem`) cover supported-no-findings, not unsupported scope | P1 / P3 | yes | no | **NOT_APPLICABLE** | HARDEN-4E-R1 audit: central `DiagnosticOrchestrator` has no typed unsupported-subject/scope outcome; valid execution with `has_findings=False` is supported clean semantics (M1), not unsupported rejection |
| M23 | Side-effect safety | Diagnostic detection/read must not repeat business side effects | `test_terminal_diagnostic_production_e2e.py::test_replay_terminal_trigger_does_not_duplicate_occurrence`; `test_terminal_execution_diagnostic_trigger.py::test_trigger_replay_is_idempotent_for_same_execution` | P3 / P1 | yes | no | **PROVEN** | Terminal replay idempotency covers primary production path |
| M24 | Restart + vendor outage combined | Composition of durability + vendor failure without new mega-test | M13 + M14 proofs in `test_diag_final_external_otel_e2e.py` (restart block after outage/recovery) | P4 | yes | Docker | **PROVEN** | Justified composition |

### M25+ (additional production invariants found)

| ID | Scenario | User-visible guarantee | Existing proof | Level | Status | Gap |
|----|----------|------------------------|----------------|-------|--------|-----|
| M25 | Terminal trigger wiring on real harness host | Production host exposes terminal diagnostic trigger | `test_terminal_diagnostic_production_e2e.py::test_harness_host_runtime_wires_terminal_diagnostic_trigger` | P3 | **PROVEN** | — |
| M26 | Dashboard/read pane sees shared persistence | Operator UI read model reflects runtime-written Problems | `test_terminal_diagnostic_production_e2e.py::test_dashboard_sees_problem_on_shared_persistence_after_runtime_trigger` | P3 | **PROVEN** | — |
| M27 | One-spine architecture gates | No side Problem stores; orchestrator entrypoint discipline | `test_one_spine_problem_store_gate.py`; `test_one_spine_diagnostic_orchestrator_gate.py` | P1 | **PROVEN** | Architecture enforcement, not runtime E2E |

---

## Matrix completeness gate

| Metric | Count |
|--------|------:|
| Total required (M1–M24) | 24 |
| **PROVEN** | 22 |
| **PARTIALLY_PROVEN** | 0 |
| **MISSING** | 0 |
| **NOT_APPLICABLE** | 2 |
| **DEFERRED** | 0 |

### Critical vs non-critical gaps

| Severity | IDs | Rationale |
|----------|-----|-----------|
| **P0** | — | M17/M18 closed by HARDEN-4B product-host E2E |
| **P1** | — | M5/M6 closed by HARDEN-4D product-host lifecycle E2E |
| **P2** | — | M8 closed by HARDEN-4F real Mongo FI-A E2E |

---

## HARDEN-4A decision

**Classification: `4A-B` — Several targeted proof gaps**

Matrix is substantially built (HARDEN-1/2/3 slices cover durability, OCC, vendor OTel). Remaining gaps are **narrow, host-facing E2E slices** — not a full re-architecture program.

---

## Proposed HARDEN-4 slices (implementation — not started in 4A)

### HARDEN-4B — Product-host multi-tenant diagnostic isolation E2E ✅

| Field | Value |
|-------|-------|
| **Goal** | M17 + M18: tenant A and B same violation class → two Problems; cross-tenant `get_problem` → none |
| **Minimum level** | P3 |
| **Scope** | `build_diag_final_product_host` dual-tenant runs on shared host persistence |
| **CI class** | PR deterministic gate (`Durable diagnostics deterministic gate`), no Docker |

### HARDEN-4C — Product-host clean path + violation Problem E2E ✅

| Field | Value |
|-------|-------|
| **Goal** | M1 at P3 on governed-contractor HTTP host; explicit `inject_violation=False` run asserts zero Problems + canonical TASK_COMPLETED |
| **Minimum level** | P3 |
| **Scope** | `test_harden_4c_clean_diagnostic_host_e2e.py` on shared `build_diag_final_product_host` harness |
| **CI class** | PR deterministic gate (`Durable diagnostics deterministic gate`), no Docker required |

### HARDEN-4D — Lifecycle resolve + recurrence reopen host E2E ✅

| Field | Value |
|-------|-------|
| **Goal** | M5 + M6: host execution → Problem → explicit resolve API/path → read shows RESOLVED → second violation reopens same `problem_id` with incremented occurrences |
| **Minimum level** | P3 |
| **Scope** | `test_harden_4d_problem_lifecycle_host_e2e.py` on shared `build_diag_final_product_host` harness; `ProblemLifecycleEngine.resolve` via shared host persistence |
| **CI class** | PR deterministic gate (`Durable diagnostics deterministic gate`), no Docker |

### HARDEN-4E — Evidence reconstruction / unavailable read host E2E ✅

| Field | Value |
|-------|-------|
| **Goal** | M19 + M20 at host boundary (M22 requalified NOT_APPLICABLE in HARDEN-4E-R1 — no central unsupported-scope contract) |
| **Minimum level** | P3 |
| **Scope** | `test_harden_4e_diagnostic_read_truth_e2e.py` on shared `build_diag_final_product_host` harness; M20 uses shared Problem persistence with isolated RuntimeEvent SQLite |
| **CI class** | PR deterministic gate (`Durable diagnostics deterministic gate`), no Docker |

### HARDEN-4F — Durable store write failure E2E ✅

| Field | Value |
|-------|-------|
| **Goal** | M8 at Mongo/DocumentStore with FI-A real Docker Mongo stop/start on governed-contractor HTTP host |
| **Minimum level** | P4 |
| **Scope** | `test_harden_4f_mongo_problem_store_failure_e2e.py` — baseline / outage / recovery phases |
| **CI class** | `external_proof` / nightly / manual Mongo (`durable-diagnostics-external-proof`) |

---

## HARDEN roadmap

```text
HARDEN-4A — matrix inventory ✅
HARDEN-4B — product-host multi-tenant diagnostic isolation E2E ✅
HARDEN-4C — product-host clean / no-false-positive E2E ✅
HARDEN-4D — resolve + recurrence reopen host E2E ✅
HARDEN-4E — reconstruction / unavailable read / fail-closed host E2E ✅
HARDEN-4E-R1 ✅
HARDEN-4F — durable Mongo store failure + recovery E2E ✅
HARDEN-4 COMPLETE
HARDEN-5 — documentation review and closeout ✅
DIAGNOSTIC HARDENING COMPLETE
```

---

## Skeptic notes (PROVEN items challenged)

| Item | Skeptic concern | Verdict |
|------|-----------------|---------|
| M2 PROVEN | `diag_final` uses in-memory DocumentStore, not Mongo | Acceptable at P3 host boundary; durability is M9/M13 |
| M13 PROVEN | Split across Mongo Problems + SQLite events | Compositional proof is intentional |
| M23 PROVEN | Does not cover all execution engines | Sufficient for terminal spine production path per ONE-SPINE-3 |

---

## Production changes

**NONE** (4A audit artifact only)
