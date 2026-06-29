# Cohere Rerank (cohere_rerank)

Category: `rerank_provider`

## Single public entrypoint

- **`CohereRerankRerankProviderIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `CohereRerankRerankProviderIntegration`.
- Contract factory: `create_cohere_rerank_rerank_provider_integration()`.
