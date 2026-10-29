# DUP-X — Responsibility Delta: observability export routing

| Field | Value |
| --- | --- |
| **Symbols** | `IntegrationCategory.OBSERVABILITY_BACKEND` · `ObservabilityExportBackendRegistry` · `operator_wiring.build_observability_export_*` |
| **Candidate** | DUPX-CAND-09 |
| **Disposition** | **DISTINCT** (integration catalog vs export operator plane) |

## 1. Exact responsibility

**Integration category `observability_backend`:** catalog-backed `PlatformIntegrationContract` materialization via sanctioned `factory.resolve*` (same Class D spine as other categories).

**Observability export routing:** explicit operator wiring — backend registry, OTLP/Elasticsearch/Sentry builder registration, runtime export plugins — owned by `intergrax/runtime/observability/operator_wiring.py`.

## 2. Authority / truth owned

| Plane | Owner | Config surface |
| --- | --- | --- |
| Integration materialization | `integrations/registry/factory.py` | `INTERGRAX_INTEGRATION_OBSERVABILITY_BACKEND` → slug |
| Export delivery | `runtime/observability/operator_wiring.py` | App settings → `ObservabilityExportOperatorConfig` (e.g. LKW `observability_export_*` env fields) |

No second slug registry: LKW `observability_wiring.py` calls `build_observability_export_integration(config, registry=DEFAULT_OBSERVABILITY_EXPORT_BACKEND_REGISTRY)`.

## 3. I/O contracts + lifecycle

Export plugins attach at host composition (`build_observability_export_runtime_plugin`); diagnostics health registries remain evidence ports (CAND-11 N/A).

## 4. Caller graph

```text
LKW host/factory.py
  → settings.build_observability_export_config()
  → build_local_workspace_observability_plugins(observability_export)
       → operator_wiring.build_observability_export_runtime_plugin

GCA host/factory.py (optional)
  → ObservabilityExportOperatorConfig passed in
  → build_observability_export_runtime_plugin(observability_export)

Integration profile (environment_profile)
  → observability_backend IntegrationBinding (catalog) — separate from export operator config
```

## 5. Why canonical owner cannot absorb

Catalog contracts describe vendor capability instances; export operator wiring describes telemetry egress policy and backend builders — different semantic truths (DUP-10 vs DUP-12).

## 6. Convergence point

Keep factory-first for integration slugs; keep `ObservabilityExportBackendRegistry` as export builder SSOT.

## 7. Negative bypass proof

LKW does not re-implement env slug reads for export backends outside `operator_wiring` + typed settings; integration OBSERVABILITY_BACKEND env remains factory-owned (Class D matrix).
