# Phoenix (phoenix)

Category: `observability_backend`

## Single public entrypoint

- **`PhoenixObservabilityIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `PhoenixObservabilityIntegration`.
- Contract factory: `create_phoenix_observability_backend_integration()`.
