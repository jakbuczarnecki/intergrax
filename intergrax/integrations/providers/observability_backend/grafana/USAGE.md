# Grafana (grafana)

Category: `observability_backend`

## Single public entrypoint

- **`GrafanaObservabilityIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `GrafanaObservabilityIntegration`.
- Contract factory: `create_grafana_observability_backend_integration()`.
