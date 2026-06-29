# Mlflow (mlflow)

Category: `observability_backend`

## Single public entrypoint

- **`MlflowObservabilityIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `MlflowObservabilityIntegration`.
- Contract factory: `create_mlflow_observability_backend_integration()`.
