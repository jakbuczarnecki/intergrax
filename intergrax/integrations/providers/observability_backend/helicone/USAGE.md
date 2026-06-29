# Helicone (helicone)

Category: `observability_backend`

## Single public entrypoint

- **`HeliconeObservabilityIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `HeliconeObservabilityIntegration`.
- Contract factory: `create_helicone_observability_backend_integration()`.
