# Arxiv (arxiv)

Category: `search_provider`

## Single public entrypoint

- **`ArxivSearchProviderIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `ArxivSearchProviderIntegration`.
- Contract factory: `create_arxiv_search_provider_integration()`.
