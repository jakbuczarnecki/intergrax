# PROD-Q-R1-R6 — Canonical Production Lifecycle Ownership & Shutdown

**Disposition:** **READY FOR AUDIT** (does **not** close **PROD-Q-R1** or **FRZ-PRD-04**)

**START_HEAD:** `9a5dfc6bc16ef109b4b4c92964d22f753d749bd3` · **Branch:** `development` · **Parent:** **PROD-Q-R1** — **BLOCKED**

## Lifecycle owner (verified)

| Responsibility | Owner |
| --- | --- |
| Process composition + AP runtime | `open_reference_production_process_lifecycle` → `create_reference_production_process_composition` / `composition_builder` |
| Continuation state file path | Caller supplies `continuation_state_file` to lifecycle opener |
| Open durable store once | `open_production_execution_continuation_state_store` inside lifecycle opener |
| Writer lease during serve | `ProductionProcessComposition.execution_continuation_state_store` (same instance as lifecycle-owned store) |
| Deploy / activate registry | `ReferenceProductionLifecycleLauncher` on the same composition |
| Startup exception cleanup | Lifecycle opener rolls back `store.close()` on construction failure; context `__exit__` on partial serve |
| Normal shutdown `close()` once | `ReferenceProductionProcessLifecycle.close_process_resources()` / context manager |
| GCA factory / FastAPI shutdown | **Consumer only** — does **not** close composition continuation store |

## Scenarios

| ID | Result |
| --- | --- |
| **A — Normal lifecycle** | `test_launcher_owned_normal_startup_serve_and_shutdown_releases_writer_lease` — real `create_governed_contractor_process_app`, `TestClient` serve, mutation while serving; writer lease released only after lifecycle context exit (no manual `store.close()` in test body) |
| **B — Failed startup** | `test_launcher_owned_startup_failure_releases_writer_lease` — injected `build_harness_host_runtime` failure; lifecycle exit allows reopen |
| **C — Ownership isolation** | `test_factory_shutdown_does_not_close_composition_owned_store` — `TestClient` teardown leaves lease held; launcher `close_process_resources` releases |
| **D — Strict continuation** | `test_strict_continuation_fail_closed_non_durable_unchanged` |

## Tests (executed)

```text
python -c "import subprocess,sys; sys.exit(subprocess.run(['uv','run','pytest','tests/unit/applications/governed_contractor/test_prod_q_r1_r6_strict_gca_lifecycle.py','-p','no:xdist','-q'], timeout=30).returncode)"
```

**Evidence:** `.tmp/session/prod-q-r1-r6/pytest.log` (4 passed, ~8 s pytest body).

## Remaining gaps

| ID | Status |
| --- | --- |
| **FRZ-PRD-04** | **OPEN** — no production deploy path (e.g. GCA `run()`) wired to lifecycle opener on GitHub |
| **PROD-Q-R1** | **BLOCKED** — full admission on all materialization paths; independent audit |
| **R5 manual close in legacy test** | `test_prod_q_r1_r5_strict_gca_startup.py` still documents pre-R6 harness proof |

**Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z commitu znajdującego się na GitHubie. Raport Cursor AI nie jest podstawą do finalnego zamknięcia zadania.**
