# Replicate (replicate)

Category: `ml_inference_host`

## Single public entrypoint

- **`ReplicateMlInferenceHostIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `ReplicateMlInferenceHostIntegration`.
- Contract factory: `create_replicate_ml_inference_host_integration()`.
