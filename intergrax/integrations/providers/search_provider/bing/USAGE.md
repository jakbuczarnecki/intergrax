# Bing (bing)

Category: `search_provider`

## Single public entrypoint

- **`BingSearchProviderIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `BingSearchProviderIntegration`.
- Contract factory: `create_bing_search_provider_integration()`.
