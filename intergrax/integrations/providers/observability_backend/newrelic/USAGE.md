# Newrelic (newrelic)

Category: `observability_backend`

## Single public entrypoint

- **`NewRelicObservabilityIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `NewRelicObservabilityIntegration`.
- Contract factory: `create_newrelic_observability_backend_integration()`.
