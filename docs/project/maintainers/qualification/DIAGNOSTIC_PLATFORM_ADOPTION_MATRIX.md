# Diagnostic Platform Adoption Matrix — DIAG-PLATFORM-A

**Program:** DIAG-PLATFORM-QUALIFICATION  
**Branch baseline:** `development` @ `1657d0010b4f6e51e765843c1f5c3101146e5585`  
**Engine qualification:** [`DIAGNOSTIC_HARDENING_CLOSEOUT.md`](DIAGNOSTIC_HARDENING_CLOSEOUT.md) (HARDEN complete)  
**Architecture:** [`docs/project/architecture/DIAGNOSTICS.md`](../../architecture/DIAGNOSTICS.md)

---

## Adoption status legend

| Status | Meaning |
| ------ | ------- |
| **NATIVE** | Shared runtime composition; central diagnostics spine wired per profile contract |
| **CONDITIONAL** | Shared spine; diagnostics or read path depends on runtime prerequisites or profile |
| **LEGACY** | Manual / local diagnostic composition — migration required |
| **BYPASS** | Production-capable surface bypasses shared harness or baseline |
| **NOT_APPLICABLE** | Lab / synthetic profile with explicit diagnostics-not-required contract |
| **UNKNOWN** | Not audited |

**Target invariant (production-capable):**

```text
production application → HarnessHostRuntime → RuntimeEvent persistence
  → wire_terminal_execution_diagnostics → central ProblemPersistence
```

**Target invariant (initialized scenario):**

```text
ScenarioRuntimeBaseline (or canonical facade) → shared Nexus → terminal diagnostics
```

---

## Adoption metrics (production surfaces)

| Classification | Count |
| -------------- | ----: |
| **NATIVE** | 5 |
| **CONDITIONAL** | 1 |
| **LEGACY** | 0 |
| **BYPASS** | 0 |
| **NOT_APPLICABLE** | 3 |

```text
Production-capable application surfaces (PRODUCT): NATIVE = 4, BYPASS = 0
Initialized scenario surfaces: NATIVE = 1, LEGACY = 0
```

---

## Tier-3 application composition roots

| Surface | Type | Runtime entry | RuntimeEvent persistence | Terminal diagnostic trigger | ProblemPersistence | Central read path | Default? | Status |
| ------- | ---- | ------------- | ------------------------ | --------------------------- | ------------------ | ----------------- | -------- | ------ |
| `governed_contractor_application` | PRODUCT host | `build_harness_host_runtime` | Yes (`runtime_events_db_path`) | Yes (`DiagnosticWiring` via harness) | Yes (`wire_problem_persistence`) | Yes (`wire_harness_product_observability_dashboard` → `DiagnosticReadService`) | **REQUIRED** | **NATIVE** |
| `legal_application` | PRODUCT host | `build_harness_host_runtime` | Yes | Yes | Yes | Write only at factory (no HTTP read routes) | **REQUIRED** | **NATIVE** |
| `dispute_sim_application` | PRODUCT host | `build_harness_host_runtime` | Yes | Yes | Yes | Write only at factory | **REQUIRED** | **NATIVE** |
| `local_workspace_application` | PRODUCT host (reference) | `build_harness_host_runtime` | Yes | Yes | Yes | Write at factory; product observability optional | **REQUIRED** | **NATIVE** |
| `research_application` | PRODUCT prototype host | `build_harness_host_runtime` | Yes | Yes | Yes | Write only; `ApiEnvironment.DEV` | **REQUIRED** | **CONDITIONAL** |
| `lab_application` | LAB host | `build_harness_host_runtime` | Yes (when configured) | Yes when prerequisites present | Yes when store present | Debug API only | NOT_REQUIRED unless `DiagnosticPosture.REQUIRED` | **CONDITIONAL** |
| `poc_template_application` | LAB scaffold | `build_harness_host_runtime` | Conditional (`use_in_memory_trace` when `db_path` None) | Optional | Optional | Debug only | **NOT_REQUIRED_UNAVAILABLE** (explicit lab) | **NOT_APPLICABLE** |
| `attestation_demo` | LAB / partner PoC | `build_harness_host_runtime` | Conditional | Optional | Optional | Debug only | **NOT_REQUIRED_UNAVAILABLE** | **NOT_APPLICABLE** |
| `intergrax_assistant_application` | LAB scaffold | `build_harness_host_runtime` | Conditional | Optional | Optional | Debug only | **NOT_REQUIRED_UNAVAILABLE** | **NOT_APPLICABLE** |

**Factory audit:** all 9 `applications/*/host/factory.py` call `build_harness_host_runtime`. No direct `NexusLoop()` in factories (`check_application_production_gates` APP-PROD scope: `*_application`).

---

## Application workers and secondary composition roots

| Surface | Type | Runtime entry | RuntimeEvent | Terminal trigger | ProblemPersistence | Read path | Default? | Status |
| ------- | ---- | ------------- | ------------ | ---------------- | ------------------ | --------- | -------- | ------ |
| `local_workspace_application` background worker | PRODUCT worker | `build_harness_host_runtime` (`background_worker_factory`) | Yes (production profile) | Yes (shared harness wiring) | Yes | Worker uses host queue deps; no separate read API | **REQUIRED** | **NATIVE** |
| `local_workspace_application` model_runtime_proof | Internal proof runtime | `build_harness_host_runtime` | Yes | Yes | Yes | Proof-local | Lab contract | **CONDITIONAL** |
| `intergrax/harness/app.py` | Harness CLI entry | `build_harness_host_runtime` | Config-dependent | Harness-internal | Config-dependent | N/A | Profile-driven | **CONDITIONAL** |

---

## Platform proof scenarios (initialized)

| Scenario | Runtime entry | Baseline? | RuntimeEvent | Terminal trigger | ProblemPersistence | Read path | Mode | Status |
| -------- | ------------- | --------- | ------------ | ---------------- | ------------------ | --------- | ---- | ------ |
| `ai_incident_investigation` | `build_scenario_runtime_from_environment` via `build_scenario_runtime_composition` | **Yes** (`ScenarioRuntimeBaseline`) | Yes | Yes (`composition.diagnostic_wiring`) | Yes (shared `DocumentStore`) | `scenario_composition` → `DiagnosticReadService` (composition layer) | `ScenarioRuntimeMode.LAB` | **NATIVE** |

**Design-only scenarios (12):** no `application/` package — adoption **NOT_APPLICABLE** until `IMPLEMENTATION_INITIALIZED`.

**Forbidden-pattern scan (`ai_incident_investigation/application/`):** no local `GraphExecutor`, `DiagnosticOrchestrator`, `ProblemLifecycleEngine`, or manual terminal publishers.

---

## Shared runtime entry points

| Entry point | Module | Adoption role |
| ----------- | ------ | --------------- |
| `build_harness_host_runtime` | `harness_host_runtime.py` | Canonical Tier-3 host composition |
| `build_scenario_runtime_from_environment` | `scenario_runtime_baseline.py` | Canonical scenario Nexus runtime |
| `wire_terminal_execution_diagnostics` | `diagnostic_runtime_wiring.py` | Policy-aware terminal trigger attach |
| `build_nexus_loop_from_environment` | `nexus_factory.py` | Shared Nexus assembly |
| `UnifiedTaskRunner` | Nexus execution path | Terminal diagnostics via shared bridge |

---

## Default enforcement summary

| Question | Answer |
| -------- | ------ |
| Can a **production-capable application** start without diagnostics when RuntimeEvent + Problem persistence are part of the production profile? | **NO** — `DiagnosticAssemblyError` fail-closed (`wire_terminal_execution_diagnostics` + `assert_diagnostic_assembly_valid`) |
| Can a **production-attached scenario** start without diagnostics? | **NO** — `ScenarioRuntimeBuildError` when prerequisites missing |
| Can a **lab scenario/application** start without diagnostics? | **YES** — explicit `NOT_REQUIRED_UNAVAILABLE` when prerequisites absent and posture not `REQUIRED` |
| Silent accidental disable in production? | **Blocked** — missing DocumentStore or RuntimeEvents on PRODUCT / `PRODUCTION_ATTACHED` raises |

**Semantic split:** production requirement violated → **FAIL CLOSED**; lab capability not requested/unavailable → **allowed** with typed readiness.

---

## Conformance gates (automated)

| Gate | Location | CI |
| ---- | -------- | -- |
| Initialized scenario architecture | `assert_all_initialized_scenario_architectures` | `test_all_initialized_scenario_architecture.py` (`ci_smoke`) |
| Scenario scaffold destructive | `test_scenario_scaffold_conformance_proof.py` | unit gate |
| Application factory harness spine | `check_no_ad_hoc_nexus_in_factories` | `check_application_production_gates.py` |
| Platform adoption gate | `test_diag_platform_adoption_gate.py` | `ci_smoke` |
| Hosting diagnostics bridge | `test_one_spine_hosting_diagnostics_gate.py` | unit gate |
| DF-5 destructive proofs | `test_diag_foundation_5_destructive_proof.py` | unit gate |
| One-spine orchestrator / problem store | `test_one_spine_diagnostic_orchestrator_gate.py`, `test_one_spine_problem_store_gate.py` | unit gate |

---

## Known gaps (non-BYPASS)

| Gap | Impact |
| --- | ------ |
| HTTP `DiagnosticReadService` only on `governed_contractor_application` factory | Other PRODUCT hosts: write path NATIVE; operator read via shared wiring elsewhere or future dashboard adoption |
| `check_application_production_gates` scans `*_application` only | `attestation_demo`, `poc_template`, `intergrax_assistant` outside APP-PROD factory scan (lab scaffolds) |
| Only **1** initialized scenario | Second scenario proof blocked until next `IMPLEMENTATION_INITIALIZED` scenario ships |
| Kafka queue → worker → Nexus → diagnostics | Queue transport qualified separately; full P4 async diagnostic spine not yet composed in one external proof |

---

## Visual reference

Flagship adoption diagram: [`diagnostics-platform-adoption-light.svg`](../../architecture/assets/diagnostics-platform-adoption-light.svg) · [`diagnostics-platform-backbone-light.svg`](../../architecture/assets/diagnostics-platform-backbone-light.svg)
