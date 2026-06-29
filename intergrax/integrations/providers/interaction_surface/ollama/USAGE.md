# Ollama (ollama)

Category: `interaction_surface`

## Single public entrypoint

- **`OllamaInteractionSurfaceIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `OllamaInteractionSurfaceIntegration`.
- Contract factory: `create_ollama_interaction_surface_integration()`.
