# Wandb (wandb)

Category: `observability_backend`

## Single public entrypoint

- **`WandbObservabilityIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `WandbObservabilityIntegration`.
- Contract factory: `create_wandb_observability_backend_integration()`.
