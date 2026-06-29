# Serpapi (serpapi)

Category: `search_provider`

## Single public entrypoint

- **`SerpapiSearchProviderIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `SerpapiSearchProviderIntegration`.
- Contract factory: `create_serpapi_search_provider_integration()`.
