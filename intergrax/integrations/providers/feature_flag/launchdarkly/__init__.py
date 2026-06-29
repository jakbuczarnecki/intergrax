# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "LAUNCHDARKLY_FEATURE_FLAG_PROVIDER_ID",
    "LaunchdarklyFeatureFlagIntegration",
    "LaunchdarklyFeatureFlagIntegrationConfig",
    "LaunchdarklyFeatureFlagClient",
    "create_launchdarkly_feature_flag",
    "create_launchdarkly_feature_flag_integration",
    "register_launchdarkly_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_launchdarkly_feature_flag",
        "create_launchdarkly_feature_flag_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "LAUNCHDARKLY_FEATURE_FLAG_PROVIDER_ID",
        "LaunchdarklyFeatureFlagIntegration",
        "LaunchdarklyFeatureFlagIntegrationConfig",
        "LaunchdarklyFeatureFlagClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "LAUNCHDARKLY_FEATURE_FLAG_PROVIDER_ID",
        "LaunchdarklyFeatureFlagIntegration",
        "LaunchdarklyFeatureFlagIntegrationConfig",
        "LaunchdarklyFeatureFlagClient",
    }
)

def __getattr__(name: str):
    if name == "register_launchdarkly_integration":
        from intergrax.integrations.providers.feature_flag.launchdarkly.register import register_launchdarkly_integration

        return register_launchdarkly_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.feature_flag.launchdarkly import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.feature_flag.launchdarkly import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.feature_flag.launchdarkly import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
