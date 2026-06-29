# Exa (exa)

Category: `search_provider`

## Single public entrypoint

- **`ExaSearchProviderIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `ExaSearchProviderIntegration`.
- Contract factory: `create_exa_search_provider_integration()`.
