# Triton (triton)

Category: `vision_serving`

## Single public entrypoint

- **`TritonVisionServingIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `TritonVisionServingIntegration`.
- Contract factory: `create_triton_vision_serving_integration()`.
