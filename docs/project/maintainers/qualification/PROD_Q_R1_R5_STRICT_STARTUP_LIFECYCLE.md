# PROD-Q-R1-R5 — Canonical Strict Startup, Admission & Lifecycle Qualification

**Disposition:** **READY FOR AUDIT** (does **not** close **PROD-Q-R1** or **FRZ-PRD-04**)

**START_HEAD:** `2a0336010704c3b33f297e62b86156c099bf80fb` · **Branch:** `development` · **Parent:** **PROD-Q-R1** — **BLOCKED**

## Mission outcome

| Scenario | Result |
| --- | --- |
| **A — Real sanctioned in-process startup** | `create_governed_contractor_process_app` → real `build_harness_host_runtime` (no mock); composition-bound `SerializedSealExecutionContinuationStateStore`; AP-9/AP-10 bootstrap; `provider_invocation_store`; platform persistence from composition; strict `dependency_concurrency_admission` on resolved host environment. |
| **B — Fail-closed materialization** | Second writer on same continuation state file; non-durable lab store rejected by `validate_execution_continuation_for_composition` (R1-R1 admission matrix unchanged — cite existing tests). |
| **C — Writer lease / shutdown** | DUR-02 store lifecycle tests (close, in-flight, OS subprocess second writer); R5 test closes composition-owned store after `TestClient` lifespan; **gap:** factory / `attach_plugin_shutdown` does **not** invoke `execution_continuation_state_store.close()` — owner remains deploy-activate / composition launcher per ADR-01/02. |
| **D — GR-10 / continuation smoke** | DUR-02 composition binding tests (kwargs capture with harness mock); not a substitute for A. |
| **E — Process restart** | **NOT RUN** — covered by DUR-02 `test_os_process_restart_reads_*` where applicable. |

## Sanctioned startup chain (verified A)

```text
ProductionProcessComposition (+ open_production_execution_continuation_state_store)
  → create_governed_contractor_process_app
  → bootstrap_production_registry_projection (activated serving revision)
  → create_governed_contractor_backend_app
       → resolve_reference_production_strict_host_environment
       → resolve_harness_host_profile_persistence_kwargs_from_composition
       → validate_execution_continuation_for_composition (composition store)
       → build_harness_host_runtime (real)
       → apply_factory_lifespans / attach_plugin_shutdown
```

## Dependency matrix (startup slice)

| Dependency | Owner | Evidence |
| --- | --- | --- |
| Continuation | `ProductionProcessComposition.execution_continuation_state_store` + factory resolution | `test_prod_q_r1_dur_02_gca_composition_binding.py`, R5 A |
| Provider invocation | `ProductionProcessComposition.provider_invocation_store` | R4 negative + R5 A |
| Platform persistence | `agent_platform_runtime.platform_persistence` | R5 A (`build_reference_production_platform_persistence`) |
| Tool admission | `ReliabilityProfile.dependency_concurrency_admission` (reference host defaults) | R5 A post-wiring assert; **PROD-Q-R1-R1** `test_strict_host_dependency_concurrency_admission.py` |
| Registry | Lifecycle stores + `bootstrap_production_registry_projection` | R5 A |

## Lifecycle gap (documented — no factory bypass)

| Edge | Status |
| --- | --- |
| Open once per process | Launcher/test opens `open_production_execution_continuation_state_store` before `create_governed_contractor_process_app` |
| Writer lease through serve lifetime | Held while process runs; not released by FastAPI shutdown hooks today |
| Normal shutdown `close()` once | **Manual / launcher responsibility** — not wired in GCA factory @ audited implementation |
| Startup failure lease release | DUR-02 `test_startup_failure_after_open_does_not_pin_writer_lock` |
| Cross-app close | N/A in single-process qualification; store instance scoped to composition |

## Tests (executed on qualification HEAD)

```text
uv run pytest tests/unit/applications/governed_contractor/test_prod_q_r1_r5_strict_gca_startup.py tests/unit/applications/governed_contractor/test_prod_q_r1_dur_02_gca_composition_binding.py tests/unit/applications/governed_contractor/test_prod_q_r1_r4_production_startup_path.py tests/unit/runtime/execution/continuation/test_prod_q_r1_dur_02_store_lifecycle.py -p no:xdist -q
```

**Aggregate:** ~14 s (session capture under `.tmp/session/prod-q-r1-r5/`).

## FRZ / blockers

| ID | Status |
| --- | --- |
| **FRZ-PRD-04** | **OPEN** — independent GitHub audit + launcher-owned continuation shutdown not proven on deploy path |
| **PROD-Q-R1** | **BLOCKED** — composition shutdown edge + full admission on all production materialization paths |
| **DUR-02** | **ACCEPTED** (composition binding + store lifecycle) — parent scope only |

**Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z commitu znajdującego się na GitHubie. Raport Cursor AI nie jest podstawą do finalnego zamknięcia zadania.**
