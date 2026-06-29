# Launchdarkly (launchdarkly)

Category: `feature_flag`

## Single public entrypoint

- **`LaunchdarklyFeatureFlagIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `LaunchdarklyFeatureFlagIntegration`.
- Contract factory: `create_launchdarkly_feature_flag_integration()`.
