# Reddit (reddit)

Category: `search_provider`

## Single public entrypoint

- **`RedditSearchProviderIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `RedditSearchProviderIntegration`.
- Contract factory: `create_reddit_search_provider_integration()`.
