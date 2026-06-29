# Jina Rerank (jina_rerank)

Category: `rerank_provider`

## Single public entrypoint

- **`JinaRerankRerankProviderIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `JinaRerankRerankProviderIntegration`.
- Contract factory: `create_jina_rerank_rerank_provider_integration()`.
