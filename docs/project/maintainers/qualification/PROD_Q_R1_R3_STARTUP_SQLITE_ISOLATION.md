# PROD-Q-R1-R3 — Startup Evidence & SQLite Isolation

**Disposition:** **READY FOR AUDIT** (does not close **PROD-Q-R1** or **FRZ-PRD-04**)

**START_HEAD:** `9c0fbc5702c56cd28c81ee8af687be2c518f38be` · **Branch:** `development` · **Parent:** **PROD-Q-R1** — **BLOCKED**

## Mission outcome

| Scenario | Result |
| --- | --- |
| **A — SQLite isolation** | Default `build/intergrax.db` remains **non-SQLite** (`%3\|…\|FAIL\|…`). `INTERGRAX_RELATIONAL_DB` routes relational paths to a fresh file; `sqlite3` can create a valid **SQLite format 3** file at that path without touching `build/intergrax.db`. |
| **B — strict missing dependency** | **PASS** — `tests/unit/applications/shared/test_strict_host_dependency_concurrency_admission.py` (PROD-Q-R1-R1): `ToolDependencyAttemptBoundaryMaterializationError` when `dependency_concurrency_admission` absent in strict mode. |
| **C — supported lightweight strict wiring** | **PARTIAL** — `create_governed_contractor_backend_app(document_store=…)` requires explicit durable `ExecutionContinuationStateStore` before `build_harness_host_runtime` (PROD-Q-R1-R2). Full HTTP harness still stops at the same boundary until admission + remaining strict deps are wired; **not** a production PASS. |
| **D — P0 reproduction reconciliation** | GCA HTTP reproduction @ isolated `INTERGRAX_RELATIONAL_DB` now fails first on **`ExecutionContinuationError`** (`NON_DURABLE_CONTINUATION_STORE`), not corrupted default SQLite. With durable continuation injected, failure advances to **`ToolDependencyAttemptBoundaryMaterializationError`** (strict admission) — aligns with historical P0 classification as **DEPENDENCY_MATERIALIZATION**, not SQLite defect. |

## Supported vs unsupported strict GCA configuration

| Configuration | Supported strict startup? | Notes |
| --- | --- | --- |
| `create_governed_contractor_process_app(process_composition=…)` | **Sanctioned production entry** | Requires activated `ProductionProcessComposition` + lifecycle deploy/activate (`host/main.py`). |
| `create_governed_contractor_backend_app` + `document_store` only | **Harness / test slice only** | Strict mode: durable continuation mandatory; does **not** imply full production qualification. |
| Default `build/intergrax.db` without `INTERGRAX_RELATIONAL_DB` | **Unsupported for isolated qualification** | Polluted artifact; do not repair in qualification runs. |

## Ownership graph (startup slice)

| Concern | Owner |
| --- | --- |
| Strict host environment | `resolve_reference_production_strict_host_environment` · `intergrax/applications/_shared/production_platform_persistence.py` |
| Continuation durability gate | `validate_execution_continuation_for_composition` · `intergrax/runtime/execution/continuation/durability_policy.py` |
| Tool dependency admission materialization | `materialize_tool_dependency_attempt_boundary` · `intergrax/runtime/resilience/dependency_attempt_boundary_composition.py` |
| Harness runtime assembly | `build_harness_host_runtime` · `intergrax/applications/_shared/harness_host_runtime.py` |
| GCA factory composition | `create_governed_contractor_backend_app` · `applications/governed_contractor_application/host/factory.py` |
| SQLite path authority | `resolve_sqlite_store_paths` / `INTERGRAX_RELATIONAL_DB` · `intergrax/integrations/providers/relational_store/sqlite/paths.py` |

## Tests (executed on qualification HEAD)

```text
uv run pytest tests/unit/applications/governed_contractor/test_prod_q_r1_r3_startup_sqlite_isolation.py -p no:xdist -q
uv run pytest tests/unit/applications/shared/test_strict_host_dependency_concurrency_admission.py tests/unit/applications/governed_contractor/test_prod_q_r1_r2_gca_startup_continuation.py -p no:xdist -q
```

GCA HTTP reproduction (P0 subprocess): `applications/governed_contractor_application/tests/host/test_governed_contractor_canonical_execution.py::test_governed_contractor_http_root_uses_canonical_execution_facade` with `INTERGRAX_RELATIONAL_DB` → **ExecutionContinuationError** @ current HEAD.

## FRZ / blockers

| ID | Status |
| --- | --- |
| **FRZ-PRD-04** | **OPEN** — startup lifecycle not qualified end-to-end |
| **R1-SQLITE-ENV-01** | **CLASSIFIED** — environment/test isolation; not a production SQLite defect |
| **PROD-Q-R1** | **BLOCKED** — dependency materialization + full strict host composition remain |

**R1-R1 / R1-R2:** unchanged corrective intent; R1-R3 adds current-HEAD reproduction reconciliation only.

**Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z commitu znajdującego się na GitHubie. Raport Cursor AI nie jest podstawą do finalnego zamknięcia zadania.**
