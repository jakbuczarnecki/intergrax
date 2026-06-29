# Tempo (tempo)

Category: `observability_backend`

## Single public entrypoint

- **`TempoObservabilityIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `TempoObservabilityIntegration`.
- Contract factory: `create_tempo_observability_backend_integration()`.
