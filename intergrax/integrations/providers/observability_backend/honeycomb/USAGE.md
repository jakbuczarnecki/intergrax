# Honeycomb (honeycomb)

Category: `observability_backend`

## Single public entrypoint

- **`HoneycombObservabilityIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `HoneycombObservabilityIntegration`.
- Contract factory: `create_honeycomb_observability_backend_integration()`.
