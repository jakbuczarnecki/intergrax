# Launchdarkly (launchdarkly)

Category: `feature_flag`

## Legacy facade

- `create_launchdarkly_feature_flag()` remains backward-compatible.

## Contract-based integration

- `LaunchdarklyFeatureFlagIntegration` derives from the category-specific contract.
- Factory: `create_launchdarkly_feature_flag_integration()`.
- Disabled by default (`enabled=False`).
- No vendor SDK or network I/O in the contract adapter.
- Injectable `{prefix}Client` required when `enabled=True`.

## Registry

- `register.py` remains legacy-compatible.
- Registry v2 / contract registry wiring deferred.
