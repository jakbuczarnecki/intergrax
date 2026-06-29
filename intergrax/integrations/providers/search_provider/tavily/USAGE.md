# Tavily (tavily)

Category: `search_provider`

## Single public entrypoint

- **`TavilySearchProviderIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `TavilySearchProviderIntegration`.
- Contract factory: `create_tavily_search_provider_integration()`.
