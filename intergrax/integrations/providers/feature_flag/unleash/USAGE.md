# Unleash (unleash)

Category: `feature_flag`

## Single public entrypoint

- **`UnleashFeatureFlagIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `UnleashFeatureFlagIntegration`.
- Contract factory: `create_unleash_feature_flag_integration()`.
