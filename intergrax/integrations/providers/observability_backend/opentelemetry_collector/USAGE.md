# Opentelemetry Collector (opentelemetry_collector)

Category: `observability_backend`

## Single public entrypoint

- **`OpenTelemetryCollectorObservabilityIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `OpenTelemetryCollectorObservabilityIntegration`.
- Contract factory: `create_opentelemetry_collector_observability_backend_integration()`.
