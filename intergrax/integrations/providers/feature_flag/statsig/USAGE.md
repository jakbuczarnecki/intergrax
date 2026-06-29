# Statsig (statsig)

Category: `feature_flag`

## Single public entrypoint

- **`StatsigFeatureFlagIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `StatsigFeatureFlagIntegration`.
- Contract factory: `create_statsig_feature_flag_integration()`.
