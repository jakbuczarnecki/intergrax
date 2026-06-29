# Arize (arize)

Category: `observability_backend`

## Single public entrypoint

- **`ArizeObservabilityIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `ArizeObservabilityIntegration`.
- Contract factory: `create_arize_observability_backend_integration()`.
