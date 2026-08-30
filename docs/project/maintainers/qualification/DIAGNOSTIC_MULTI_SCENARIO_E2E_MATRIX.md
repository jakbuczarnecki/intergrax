# Diagnostic Multi-Scenario E2E Matrix — DIAG-PLATFORM-C

**Program:** DIAG-PLATFORM-QUALIFICATION  
**Proof levels:** P1–P4 (aligned with [`DIAGNOSTIC_E2E_MATRIX_HARDEN_4A.md`](DIAGNOSTIC_E2E_MATRIX_HARDEN_4A.md))  
**Adoption inventory:** [`DIAGNOSTIC_PLATFORM_ADOPTION_MATRIX.md`](DIAGNOSTIC_PLATFORM_ADOPTION_MATRIX.md)

---

## E2E category coverage (S1–S5)

| Category | Description | Representative proof | Level | PASS |
| -------- | ----------- | ---------------------- | ----- | ---- |
| **S1** Clean application success | Real entry → success → RuntimeEvent → diagnostics → zero false Problem | `test_harden_4c_clean_diagnostic_host_e2e` (governed-contractor HTTP) | P3 | ✅ |
| **S2** Deterministic violation | Real entry → violation → central Problem → read path | `test_harden_4d_problem_lifecycle_host_e2e`; `test_harden_4e_diagnostic_read_truth_e2e` | P3 | ✅ |
| **S3** Real external integration failure | Mongo / OTLP outage through production abstractions | `test_harden_4f_mongo_problem_store_failure_e2e` (FI-A Docker Mongo); `test_diag_final_external_otel_e2e` (Docker OTLP) | P4 | ✅ |
| **S4** Async / background execution | Background child inherits terminal diagnostic spine | `test_background_execution_inherits_terminal_diagnostic_trigger` (`test_terminal_diagnostic_production_e2e`) | P3 | ✅ |
| **S5** Policy / guardrail failure | Platform denial → canonical evidence → diagnostic behavior | `test_df4_background_task_uses_shared_terminal_diagnostic_path`; harness task-control block paths (DF-4 table) | P3 | ✅ |

**Optional categories (natural existing proofs):**

| Category | Proof | Level | PASS |
| -------- | ----- | ----- | ---- |
| **S6** Cross-process execution | `test_harden_1c_durable_problem_restart_proof` | P4 | ✅ |
| **S7** Host restart durability | Same + Mongo subprocess phases | P4 | ✅ |
| **S8** External vendor outage | OTLP collector stop/start (`test_diag_final_external_otel_e2e`) | P4 | ✅ |
| **S10** Tool integration boundary | Scenario reasoning consumes `DiagnosticReadService` projections only (`test_diagnostic_platform_integration`) | P2/P3 | ✅ |

**S9 HITL/long-running:** partial via checkpoint/host wiring tests — not a dedicated diagnostic E2E category.

---

## Scenario / application selection rationale

| Scenario / application | Why selected | Different mechanism | Real integration |
| ---------------------- | ------------ | ------------------- | ---------------- |
| `governed_contractor_application` | Canonical PRODUCT HTTP host; HARDEN P3/P4 anchor | HTTP FastAPI → Nexus → terminal trigger → dashboard read | Optional Mongo (P4), Docker OTLP (P4) |
| `ai_incident_investigation` | Sole initialized scenario; `ScenarioRuntimeBaseline` | Scenario task API → baseline Nexus (no HarnessHostRuntime HTTP) | In-memory / lab DocumentStore; reasoning reads central Problems |
| `legal_application` | Second PRODUCT factory; queue worker wiring | `resolve_host_queue_execution_dependencies` on harness runtime | IntegrationProfile legal product bindings |
| Nexus + `UnifiedTaskRunner` (integration) | Spine proof without HTTP | Direct Nexus terminal path | In-memory stores |
| LKW background worker | Async worker composition root | `background_worker_factory` → shared harness | Queue deps when profile wired (unit); Kafka transport separate integration suite |

---

## Full E2E matrix

| Scenario / application | Entry | Failure / success class | Real integration | RuntimeEvent | Problem | Read | Level | PASS |
| ---------------------- | ----- | ----------------------- | ---------------- | ------------ | ------- | ---- | ----- | ---- |
| Governed contractor | HTTP POST (shared harness fixture) | S1 clean success | In-process SQLite RuntimeEvents + InMemory DocumentStore | ✅ | ✅ zero | ✅ dashboard | P3 | ✅ |
| Governed contractor | HTTP POST + violation injector | S2 lifecycle OPEN→RESOLVED→reopen | Same | ✅ | ✅ | ✅ | P3 | ✅ |
| Governed contractor | HTTP POST | S2 read truth / unavailable | Same | ✅ | ✅ | ✅ no fabrication | P3 | ✅ |
| Governed contractor | HTTP POST | S3 Mongo FI-A outage/recovery | Docker Mongo DocumentStore | ✅ | ✅ degradation | ✅ | P4 | ✅ |
| Governed contractor | HTTP POST + OTLP export | S3/S8 OTLP outage; observability intersection | Docker OTLP Collector | ✅ canonical | ✅ central | ✅ + derived spans | P4 | ✅ |
| Nexus integration suite | `nexus_loop.handle_task` | S1 clean / S2 violation | In-memory | ✅ | ✅ | ✅ DiagnosticReadService | P3 | ✅ |
| Nexus integration suite | Background child execution | S4 async inheritance | In-memory | ✅ | ✅ | ✅ | P3 | ✅ |
| `ai_incident_investigation` | `execute_scenario_task` / skeleton | S2 platform Problem → reasoning input | Lab baseline composition | ✅ | ✅ | ✅ composition read | P3 | ✅ |
| Mongo durability worker | Subprocess restart | S6/S7 cross-process | Mongo via IntegrationProfile | ✅ | ✅ durable | ✅ after restart | P4 | ✅ |
| Kafka transport | `create_kafka_integration` | Transport only (not diagnostic spine) | Docker Kafka | — | — | — | P4 transport | ⚠️ N/A diagnostic |
| Legal application harness | `build_harness_host_runtime` + queue deps | Queue wiring on PRODUCT runtime | KV cache provider | ✅ when configured | ✅ | Write path | P2 | ✅ wiring |

---

## E2E metrics

```text
P3 application/scenario flows (distinct entry classes): 5
  — HTTP product host (governed contractor)
  — Nexus direct / UnifiedTaskRunner
  — Background child execution
  — Scenario baseline task (ai_incident)
  — Legal harness + queue deps

P4 real external infrastructure proofs: 3
  — Mongo FI-A (4f)
  — OTLP collector (diag_final)
  — Cross-process Mongo restart (1c)

Distinct runtime/integration classes: 6
  — HTTP host, scenario baseline, Nexus spine, background async, Mongo durable, OTLP derived observability
```

---

## Central spine assertion

Every P3/P4 proof above confirms:

```text
application/scenario/worker entry
  → shared runtime (HarnessHostRuntime or ScenarioRuntimeBaseline)
  → RuntimeEvent persistence
  → wire_terminal_execution_diagnostics / shared terminal bridge
  → intergrax.runtime.diagnostics (DiagnosticOrchestrator spine)
  → ProblemPersistence + DiagnosticReadService
```

Static gates: `test_one_spine_diagnostic_orchestrator_gate`, `test_one_spine_problem_store_gate`, scenario architecture conformance.

**No private attribute assertions** in new platform qualification tests — public `DiagnosticWiring.readiness`, `has_terminal_diagnostic_trigger`.

---

## Observability intersection (canonical vs derived)

Qualified in `test_diag_final_external_otel_e2e`:

```text
RuntimeEvent = canonical execution evidence
Problem = central diagnostics durable state
OTel/vendor spans = derived observability (export failure does not alter canonical truth)
```

---

## Async / queue proof status

| Path | Diagnostic spine? | Status |
| ---- | ----------------- | ------ |
| Background child via Nexus (`S4`) | Yes — shared terminal trigger | **PROVEN P3** |
| LKW `background_worker_factory` → harness | Yes — same `build_harness_host_runtime` | **NATIVE adoption** (unit wiring) |
| Kafka producer → worker → Nexus → diagnostics | Not composed in one external E2E | **LIMITATION** — transport qualified separately (`test_kafka_worker_integration`) |

---

## Proof map visual

[`diagnostics-proof-map-light.svg`](../../architecture/assets/diagnostics-proof-map-light.svg)
