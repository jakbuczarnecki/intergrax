# Signoz (signoz)

Category: `observability_backend`

## Single public entrypoint

- **`SignozObservabilityIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `SignozObservabilityIntegration`.
- Contract factory: `create_signoz_observability_backend_integration()`.
