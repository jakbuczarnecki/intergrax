# OBS-DIAG-X3A — Universal Surface Discovery & Anti-Drift Closure

> **Maintainer audit / qualification evidence — not architecture SSOT.**
> **CURRENT AUTHORITY:** [`OBSERVABILITY.md`](../../architecture/OBSERVABILITY.md) · [`DIAGNOSTICS.md`](../../architecture/DIAGNOSTICS.md)

| Field | Value |
| ----- | ----- |
| **Program** | OBS-DIAG-X3A |
| **Predecessor** | [`OBS_DIAG_UNIVERSAL_SPINE_ADOPTION_X3.md`](OBS_DIAG_UNIVERSAL_SPINE_ADOPTION_X3.md) |
| **Verdict** | `PASS — UNIVERSAL OBS/DIAG SURFACE ANTI-DRIFT CLOSED` (static gates) |

## Gaps closed (audit findings)

| Finding | Closure |
| ------- | ------- |
| Worker discovery hard-coded to `local_workspace_application` / `background_worker_factory.py` | Generic `execution_surface_discovery.py` — `BootstrapSurfaceKind.WORKER_BACKGROUND` marker across all Tier-3 `host/` trees |
| AST worker scan independent of qualification discovery | `obs_diag_x3_ast_gates._iter_worker_entry_python` delegates to `iter_worker_execution_surface_python_paths` |
| PRODUCT/LAB reclassification not validated vs manifest | `assert_obs_diag_x3_surface_coverage_complete` runs registry integrity + manifest/profile classification parity |
| Harness entry hard-coded path | Harness modules under `intergrax/harness/` discovered via `build_harness_host_runtime` composition marker |

## Discovery architecture

```text
applications / scenarios / workers / harness CLI
        ↓
execution_surface_discovery + existing list_application_projects / scenario slugs
        ↓
OBS_DIAG_X3_QUALIFIED_SURFACES (anti-drift registry)
        ↓
registry integrity + classification validation + coverage
        ↓
AST zero-bypass gates (shared worker path set)
```

## Primary gates

| Gate | Module |
| ---- | ------ |
| X3A adoption + anti-drift proofs | `tests/unit/applications/_shared/test_obs_diag_x3_universal_spine_adoption.py` |
| Execution surface discovery | `intergrax/applications/_shared/execution_surface_discovery.py` |
| Qualification + classification | `intergrax/applications/_shared/obs_diag_surface_qualification.py` |
| AST production layer | `intergrax/runtime/architecture/obs_diag_x3_ast_gates.py` |

## Successor

- OBS-DIAG-X4 — external Kafka async + cross-process HITL (pending)

## Explicit non-claims

- Real Kafka cross-process spine (X4)
- Live persistence/transport provider qualification matrix (X5)
- Final exact-SHA OBS/DIAG backbone gate (X6)
