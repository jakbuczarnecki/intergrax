# Google Cse (google_cse)

Category: `search_provider`

## Single public entrypoint

- **`GoogleCseSearchProviderIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `GoogleCseSearchProviderIntegration`.
- Contract factory: `create_google_cse_search_provider_integration()`.
