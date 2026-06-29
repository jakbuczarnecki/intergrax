# Perplexity (perplexity)

Category: `search_provider`

## Single public entrypoint

- **`PerplexitySearchProviderIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `PerplexitySearchProviderIntegration`.
- Contract factory: `create_perplexity_search_provider_integration()`.
