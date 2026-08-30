# Diagnostic Multi-Scenario E2E Matrix — DIAG-PLATFORM-C / R1

**Program:** DIAG-PLATFORM-QUALIFICATION  
**Proof levels:** P1–P4 (aligned with [`DIAGNOSTIC_E2E_MATRIX_HARDEN_4A.md`](DIAGNOSTIC_E2E_MATRIX_HARDEN_4A.md))  
**Adoption inventory:** [`DIAGNOSTIC_PLATFORM_ADOPTION_MATRIX.md`](DIAGNOSTIC_PLATFORM_ADOPTION_MATRIX.md)
**R1 audit:** Execution System owns root execution authority; Nexus participates in orchestration/planning/execution coordination — not as canonical root execution authority.

---

## Execution authority (R1 baseline)

```text
public application/scenario/worker entry
  → UnifiedTaskRunner.run_task (when task execution is involved)
  → resolve_root_task_identity
  → execute_root_task (Execution System / ExecutionRuntime / ExecutionBoundary)
  → OrchestrationExecutor → NexusLoop.handle_task (orchestration participant)
  → GraphExecutor / lower execution mechanisms
  → terminal RuntimeEvent
  → wire_terminal_execution_diagnostics / terminal bridge
  → intergrax.runtime.diagnostics
  → ProblemPersistence + DiagnosticReadService
```

**Semantic rule:** do not describe qualified proofs as “Nexus direct execution”. Nexus is reached through the Execution System orchestration backend after canonical root execution is established.

---

## E2E category coverage (S1–S5)

| Category | Description | Representative proof | Root execution authority | Level | PASS |
| -------- | ----------- | ---------------------- | ------------------------ | ----- | ---- |
| **S1** Clean application success | Real entry → success → RuntimeEvent → diagnostics → zero false Problem | `test_harden_4c_clean_diagnostic_host_e2e` (governed-contractor HTTP) | Execution System via HTTP → `UnifiedTaskRunner` | P3 | ✅ |
| **S2** Deterministic violation | Real entry → violation → central Problem → read path | `test_harden_4d_problem_lifecycle_host_e2e`; `test_harden_4e_diagnostic_read_truth_e2e` | Execution System via HTTP → `UnifiedTaskRunner` | P3 | ✅ |
| **S3** Real external integration failure | Mongo / OTLP outage through production abstractions | `test_harden_4f_mongo_problem_store_failure_e2e` (FI-A Docker Mongo); `test_diag_final_external_otel_e2e` (Docker OTLP) | Execution System via HTTP → `UnifiedTaskRunner` | P4 platform E2E | ✅ |
| **S4** Async / background execution | Background child inherits terminal diagnostic spine | `test_background_execution_inherits_terminal_diagnostic_trigger` | Execution System via child handler → `UnifiedTaskRunner` | P3 | ✅ |
| **S5** Policy / guardrail failure | Platform denial → canonical evidence → diagnostic behavior | `test_df4_background_task_uses_shared_terminal_diagnostic_path`; harness task-control block paths (DF-4 table) | Execution System via `UnifiedTaskRunner` | P3 | ✅ |

**Optional categories (natural existing proofs):**

| Category | Proof | Root execution authority | Level | PASS |
| -------- | ----- | ------------------------ | ----- | ---- |
| **S6** Cross-process execution | `test_harden_1c_durable_problem_restart_proof` | Persistence-only (no task execution entry) | P4 persistence | ✅ |
| **S7** Host restart durability | Same + Mongo subprocess phases | Persistence-only | P4 persistence | ✅ |
| **S8** External vendor outage | OTLP collector stop/start (`test_diag_final_external_otel_e2e`) | Execution System via HTTP → `UnifiedTaskRunner` | P4 platform E2E | ✅ |
| **S10** Tool integration boundary | Scenario reasoning consumes `DiagnosticReadService` projections only (`test_diagnostic_platform_integration`) | Scenario composition read path | P2/P3 | ✅ |

**S9 HITL/long-running:** partial via checkpoint/host wiring tests — not a dedicated diagnostic E2E category.

---

## Scenario / application selection rationale

| Scenario / application | Why selected | Different mechanism | Real integration |
| ---------------------- | ------------ | ------------------- | ---------------- |
| `governed_contractor_application` | Canonical PRODUCT HTTP host; HARDEN P3/P4 anchor | HTTP FastAPI → `UnifiedTaskRunner` → `execute_root_task` → Nexus orchestration → terminal trigger → dashboard read | Optional Mongo (P4 platform), Docker OTLP (P4 platform) |
| `ai_incident_investigation` | Sole initialized scenario; `ScenarioRuntimeBaseline` | `execute_scenario_task` → `UnifiedTaskRunner` → `execute_root_task` → Nexus | In-memory / lab DocumentStore; reasoning reads central Problems |
| `legal_application` | Second PRODUCT factory; queue worker wiring | `resolve_host_queue_execution_dependencies` on harness runtime (composition adoption) | IntegrationProfile legal product bindings |
| `UnifiedTaskRunner` integration suite | Execution System spine proof without HTTP | `UnifiedTaskRunner.run_task` → `execute_root_task` → Nexus orchestration | In-memory stores |
| LKW background worker | Async worker composition root | `background_worker_factory` → shared harness; child tasks via `UnifiedTaskRunner` | Queue deps when profile wired (unit); Kafka transport separate integration suite |

---

## Full E2E matrix

| Scenario / application | Entry | Root execution authority | Failure / success class | Real integration | RuntimeEvent | Problem | Read | Level | PASS |
| ---------------------- | ----- | ------------------------ | ----------------------- | ---------------- | ------------ | ------- | ---- | ----- | ---- |
| Governed contractor | HTTP POST (shared harness fixture) | Execution System / `execute_root_task` | S1 clean success | In-process SQLite RuntimeEvents + InMemory DocumentStore | ✅ | ✅ zero | ✅ dashboard | P3 | ✅ |
| Governed contractor | HTTP POST + violation injector | Execution System / `execute_root_task` | S2 lifecycle OPEN→RESOLVED→reopen | Same | ✅ | ✅ | ✅ | P3 | ✅ |
| Governed contractor | HTTP POST | Execution System / `execute_root_task` | S2 read truth / unavailable | Same | ✅ | ✅ | ✅ no fabrication | P3 | ✅ |
| Governed contractor | HTTP POST | Execution System / `execute_root_task` | S3 Mongo FI-A outage/recovery | Docker Mongo DocumentStore | ✅ | ✅ degradation | ✅ | P4 platform | ✅ |
| Governed contractor | HTTP POST + OTLP export | Execution System / `execute_root_task` | S3/S8 OTLP outage; observability intersection | Docker OTLP Collector | ✅ canonical | ✅ central | ✅ + derived spans | P4 platform | ✅ |
| Terminal diagnostic integration suite | `UnifiedTaskRunner.run_task` | Execution System / `execute_root_task` | S1 clean / S2 violation | In-memory | ✅ | ✅ | ✅ DiagnosticReadService | P3 | ✅ |
| Terminal diagnostic integration suite | Background child via `UnifiedTaskRunner` | Execution System / `execute_root_task` | S4 async inheritance | In-memory | ✅ | ✅ | ✅ | P3 | ✅ |
| `ai_incident_investigation` | `execute_scenario_task` / skeleton | Execution System / `execute_root_task` | S2 platform Problem → reasoning input | Lab baseline composition | ✅ | ✅ | ✅ composition read | P3 | ✅ |
| Mongo durability worker | Subprocess restart (direct Problem write) | Persistence-only proof | S6/S7 cross-process | Mongo via IntegrationProfile | — | ✅ durable | ✅ after restart | P4 persistence | ✅ |
| Kafka transport | `create_kafka_integration` | Transport-only (not diagnostic spine) | Transport only | Docker Kafka | — | — | — | P4 transport | ⚠️ N/A diagnostic |
| Legal application harness | `build_harness_host_runtime` + queue deps | Composition adoption (no execution-path proof) | Queue wiring on PRODUCT runtime | KV cache provider | ✅ when configured | ✅ | Write path | P2 | ✅ wiring |

---

## Proof requalification (R1)

| Proof | Previous level/name | Actual entry | Root authority | Correct level | Action |
| ----- | ------------------- | ------------ | -------------- | ------------- | ------ |
| `test_harden_4c_clean_diagnostic_host_e2e` | P3 HTTP host | HTTP POST → `GovernedContractorRunService` → `UnifiedTaskRunner` | `execute_root_task` | P3 | Keep; authority documented |
| `test_harden_4d_*` / `test_harden_4e_*` | P3 HTTP host | Same HTTP chain | `execute_root_task` | P3 | Keep |
| `test_harden_4f_mongo_problem_store_failure_e2e` | P4 | HTTP POST → `UnifiedTaskRunner` | `execute_root_task` + Mongo | P4 platform | Keep; not persistence-only |
| `test_diag_final_external_otel_e2e` | P4 | HTTP POST → `UnifiedTaskRunner` | `execute_root_task` + OTLP | P4 platform | Keep; chain documented |
| `test_harden_1c_durable_problem_restart_proof` | P4 platform (stale) | Direct `DocumentStoreProblemPersistence.create` in subprocess | Persistence-only | P4 persistence | Downgrade classification |
| `test_real_nexus_execution_triggers_diagnostics_without_manual_orchestrator` | P3 “Nexus direct” (stale) | `UnifiedTaskRunner.run_task` | `execute_root_task` | P3 | Fix description |
| `test_clean_execution_does_not_create_problem` | P3 | `UnifiedTaskRunner.run_task` | `execute_root_task` | P3 | Keep |
| `test_diagnostic_failure_does_not_change_business_outcome` | P3 | `UnifiedTaskRunner.run_task` | `execute_root_task` | P3 | Keep |
| `test_separate_terminal_executions_reconcile_same_problem` | P3 | `UnifiedTaskRunner.run_task` | `execute_root_task` | P3 | Keep |
| `test_different_terminal_signatures_create_distinct_problems` | P3 | `UnifiedTaskRunner.run_task` | `execute_root_task` | P3 | Keep |
| `test_background_execution_inherits_terminal_diagnostic_trigger` | P3 | Background handler → `UnifiedTaskRunner.run_task` | `execute_root_task` | P3 | Keep |
| `test_df4_background_task_uses_shared_terminal_diagnostic_path` | P3 | Background handler → `UnifiedTaskRunner.run_task` | `execute_root_task` | P3 | Keep |
| `execute_scenario_task` / `ai_incident_investigation` | P3 “Scenario → Nexus direct” (stale) | `execute_scenario_task` → `UnifiedTaskRunner` | `execute_root_task` | P3 | Fix description |
| `legal_application` queue wiring | Counted in P3 flows (stale) | `build_harness_host_runtime` composition only | Composition adoption | P2 | Remove from P3 count |

---

## E2E metrics

```text
true P3 application/scenario flows (distinct entry classes): 4
  — HTTP product host (governed contractor) → Execution System
  — Scenario baseline task API (ai_incident) → Execution System
  — UnifiedTaskRunner integration suite (terminal diagnostic production E2E)
  — Background child execution → UnifiedTaskRunner → Execution System

true P4 platform E2E (canonical execution entry + external infra): 2
  — Mongo FI-A (4f): HTTP → Execution System → diagnostics → Mongo
  — OTLP collector (diag_final): HTTP → Execution System → diagnostics → OTLP

P4 persistence proofs (external infra, no execution entry): 1
  — Cross-process Mongo restart (1c)

Distinct runtime/integration classes: 6
  — HTTP host, scenario baseline, Execution System spine, background async, Mongo durable, OTLP derived observability
```

---

## Central spine assertion

Every **true P3 / P4 platform** proof above confirms:

```text
application/scenario/worker entry
  → shared runtime (HarnessHostRuntime or ScenarioRuntimeBaseline)
  → UnifiedTaskRunner (when executing a task)
  → execute_root_task (Execution System root authority)
  → Nexus orchestration participant
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
HTTP → UnifiedTaskRunner → execute_root_task → Nexus orchestration → RuntimeEvent (canonical execution evidence)
  → central diagnostics → Problem (durable state)
  → OTLP/vendor spans (derived observability; export failure does not alter canonical truth)
```

---

## Async / queue proof status

| Path | Diagnostic spine? | Root authority | Status |
| ---- | ----------------- | -------------- | ------ |
| Background child via `UnifiedTaskRunner` (`S4`) | Yes — shared terminal trigger | `execute_root_task` | **PROVEN P3** |
| LKW `background_worker_factory` → harness | Yes — same `build_harness_host_runtime` | Composition + worker path | **NATIVE adoption** (unit wiring) |
| Kafka producer → worker → Nexus → diagnostics | Not composed in one external E2E | — | **LIMITATION** — transport qualified separately (`test_kafka_worker_integration`) |

---

## Governed HTTP path (R1)

```text
HTTP POST /v1/governed_contractor/run
  → GovernedContractorRunService.run_task
  → UnifiedTaskRunner.run_task
  → execute_root_task
  → Nexus orchestration
  → terminal RuntimeEvent
  → central diagnostics
```

**Status: PASS** — no HTTP bypass of canonical root execution.

## Scenario baseline path (R1)

```text
execute_scenario_task
  → UnifiedTaskRunner.run_task
  → execute_root_task
  → Nexus orchestration
  → terminal RuntimeEvent
  → central diagnostics
```

**Status: PASS** — no scenario bypass of canonical root execution.

## Background path (R1)

```text
background worker handler
  → UnifiedTaskRunner.run_task (with inherited execution identity)
  → execute_root_task
  → Nexus orchestration
  → terminal RuntimeEvent
  → central diagnostics
```

**Classification: P3** — canonical runner entry; not a separate public HTTP/scenario surface.

---

## Proof map visual

[`diagnostics-proof-map-light.svg`](../../architecture/assets/diagnostics-proof-map-light.svg)
