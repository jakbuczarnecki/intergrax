# Semantic Scholar (semantic_scholar)

Category: `search_provider`

## Single public entrypoint

- **`SemanticScholarSearchProviderIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `SemanticScholarSearchProviderIntegration`.
- Contract factory: `create_semantic_scholar_search_provider_integration()`.
