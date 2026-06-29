# Braintrust (braintrust)

Category: `observability_backend`

## Single public entrypoint

- **`BraintrustObservabilityIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `BraintrustObservabilityIntegration`.
- Contract factory: `create_braintrust_observability_backend_integration()`.
