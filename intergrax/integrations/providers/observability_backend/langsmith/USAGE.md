# Langsmith (langsmith)

Category: `observability_backend`

## Single public entrypoint

- **`LangsmithObservabilityIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `LangsmithObservabilityIntegration`.
- Contract factory: `create_langsmith_observability_backend_integration()`.
