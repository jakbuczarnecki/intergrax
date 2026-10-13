# PROD-Q-R1-R4 — Canonical Production Startup Path

**Disposition:** **READY FOR AUDIT** (does **not** close **PROD-Q-R1** or **FRZ-PRD-04**)

**START_HEAD:** `904d05a64c0fd3c1ee8f627bf5aa30ba35abfef1` · **Branch:** `development` · **Parent:** **PROD-Q-R1** — **BLOCKED**

## Mission outcome

| Scenario | Result |
| --- | --- |
| **A — Sanctioned production entry** | `create_governed_contractor_process_app` verified: registry bootstrap → factory strict gates → **fail-closed** before coherent ready state (no harness mock). |
| **B — Negative startup** | Reuses **PROD-Q-R1-R1** admission matrix; adds provider-invocation + inactive lifecycle negatives on current path. |
| **C — Environment isolation** | Unchanged **PROD-Q-R1-R3** evidence (`INTERGRAX_RELATIONAL_DB`, no `build/intergrax.db` repair). |
| **D — Full strict ready state** | **NOT RUN — TEST BUDGET / EVIDENCE REQUIRED** for end-to-end ready without explicit continuation binding on sanctioned entry API. |

## Sanctioned startup chain (GCA)

```text
create_governed_contractor_process_app(process_composition)
  → bootstrap_production_registry_projection(AP-9/AP-10 stores)
  → compose_governed_contractor_host_runtime(settings) [default host ports]
  → create_governed_contractor_backend_app(...)
       → resolve_reference_production_strict_host_environment (when process_composition + strict)
       → resolve_harness_host_profile_persistence_kwargs_from_composition
       → validate_execution_continuation_for_composition (explicit store kwarg only)
       → build_harness_host_runtime (real assembly; admission via env reliability profile)
       → FastAPI + apply_factory_lifespans + attach_plugin_shutdown
```

## Dependency matrix (startup slice)

| Dependency | Canonical owner | Required in strict? | Actual provider (GCA process path) | Evidence |
| --- | --- | --- | --- | --- |
| Continuation store | `validate_execution_continuation_for_composition` · durability policy | Yes — durable `ExecutionContinuationStateStore` | Factory kwarg only; **not** on `create_governed_contractor_process_app`; **not** from `ProductionProcessComposition` | `test_sanctioned_process_app_fails_closed_without_durable_continuation_store` |
| Tool admission | `materialize_tool_dependency_attempt_boundary` · declarative tool wiring | Yes — `reliability_profile.dependency_concurrency_admission` | Environment profile via harness / declarative invoker (not synthesized) | **PROD-Q-R1-R1** tests (unchanged) |
| Provider invocation store | `ProductionProcessComposition.provider_invocation_store` | Yes when `process_composition` + strict | Composition field; factory fail-closed if missing | `test_strict_process_composition_requires_provider_invocation_store_at_factory` |
| Persistent state | `ProductionAgentPlatformRuntime.platform_persistence` | Yes for strict profile disambiguation | `resolve_harness_host_profile_persistence_kwargs_from_composition` | GR6-style integration tests (not re-run here); composition-backed KV/document |
| Lifecycle/shutdown | `apply_factory_lifespans` · `attach_plugin_shutdown` | Yes for long-running host | Wired after successful `build_harness_host_runtime` | **Not reached** on sanctioned entry until continuation (+ admission) satisfied |

## First failure on supported composition factory (GR6-style wiring)

When lifecycle stores are seeded and durable continuation is injected via **backend** factory kwargs (integration test pattern), real `build_harness_host_runtime` can complete. The **sanctioned** `create_governed_contractor_process_app` surface does **not** expose continuation injection — **architecture gap** for FRZ-PRD-04, not a test-harness limitation.

## FRZ / blockers

| ID | Status |
| --- | --- |
| **FRZ-PRD-04** | **OPEN** — no end-to-end production startup PASS |
| **PROD-Q-R1** | **BLOCKED** — continuation ownership on sanctioned entry + admission materialization on full harness |
| **R1-SQLITE-ENV-01** | **CLASSIFIED** per **PROD-Q-R1-R3** |

## Tests (executed on qualification HEAD)

```text
uv run pytest tests/unit/applications/governed_contractor/test_prod_q_r1_r4_production_startup_path.py -p no:xdist -q
uv run pytest tests/unit/applications/shared/test_strict_host_dependency_concurrency_admission.py::test_strict_missing_configuration_fails_closed_at_materialization -p no:xdist -q
```

**Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z commitu znajdującego się na GitHubie. Raport Cursor AI nie jest podstawą do finalnego zamknięcia zadania.**
