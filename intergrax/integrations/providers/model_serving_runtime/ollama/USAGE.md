# Ollama (ollama)

Category: `model_serving_runtime`

- **`OllamaModelServingRuntimeIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `OllamaModelServingRuntimeIntegration`.
- Contract factory: `create_ollama_model_serving_runtime_integration()`.
- Host ops: ``list_models``, ``health``. LLM chat/embeddings stay in ``llm_adapters`` / RAG.
