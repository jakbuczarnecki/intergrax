# Algolia (algolia)

Category: `search_provider`

## Single public entrypoint

- **`AlgoliaSearchProviderIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `AlgoliaSearchProviderIntegration`.
- Contract factory: `create_algolia_search_provider_integration()`.
