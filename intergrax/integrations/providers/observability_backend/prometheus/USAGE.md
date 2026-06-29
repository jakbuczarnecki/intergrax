# Prometheus (prometheus)

Category: `observability_backend`

## Single public entrypoint

- **`PrometheusObservabilityIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `PrometheusObservabilityIntegration`.
- Contract factory: `create_prometheus_observability_backend_integration()`.
