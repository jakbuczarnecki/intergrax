# Opensearch (opensearch)

Category: `observability_backend`

## Single public entrypoint

- **`OpensearchObservabilityIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `OpensearchObservabilityIntegration`.
- Contract factory: `create_opensearch_observability_backend_integration()`.
