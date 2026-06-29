# Langfuse (langfuse)

Category: `observability_backend`

## Single public entrypoint

- **`LangfuseObservabilityIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `LangfuseObservabilityIntegration`.
- Contract factory: `create_langfuse_observability_backend_integration()`.
