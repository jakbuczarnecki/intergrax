# Loki (loki)

Category: `observability_backend`

## Single public entrypoint

- **`LokiObservabilityIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `LokiObservabilityIntegration`.
- Contract factory: `create_loki_observability_backend_integration()`.
