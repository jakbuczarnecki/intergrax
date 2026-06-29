# Otel (otel)

Category: `observability_backend`

## Single public entrypoint

- **`OtelObservabilityIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `OtelObservabilityIntegration`.
- Contract factory: `create_otel_observability_backend_integration()`.
