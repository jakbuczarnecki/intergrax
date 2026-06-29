# Google Places (google_places)

Category: `search_provider`

## Single public entrypoint

- **`GooglePlacesSearchProviderIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `GooglePlacesSearchProviderIntegration`.
- Contract factory: `create_google_places_search_provider_integration()`.
