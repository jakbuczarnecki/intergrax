# Brave (brave)

Category: `search_provider`

## Single public entrypoint

- **`BraveSearchProviderIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `BraveSearchProviderIntegration`.
- Contract factory: `create_brave_search_provider_integration()`.
