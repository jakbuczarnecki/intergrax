# Datadog (datadog)

Category: `observability_backend`

## Single public entrypoint

- **`DatadogObservabilityIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `DatadogObservabilityIntegration`.
- Contract factory: `create_datadog_observability_backend_integration()`.
