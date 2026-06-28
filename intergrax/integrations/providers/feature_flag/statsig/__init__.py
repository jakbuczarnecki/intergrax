# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "STATSIG_FEATURE_FLAG_PROVIDER_ID",
    "StatsigFeatureFlagIntegration",
    "StatsigFeatureFlagIntegrationConfig",
    "StatsigFeatureFlagClient",
    "create_statsig_feature_flag",
    "create_statsig_feature_flag_integration",
    "register_statsig_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_statsig_feature_flag",
        "create_statsig_feature_flag_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "STATSIG_FEATURE_FLAG_PROVIDER_ID",
        "StatsigFeatureFlagIntegration",
        "StatsigFeatureFlagIntegrationConfig",
        "StatsigFeatureFlagClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "STATSIG_FEATURE_FLAG_PROVIDER_ID",
        "StatsigFeatureFlagIntegration",
        "StatsigFeatureFlagIntegrationConfig",
        "StatsigFeatureFlagClient",
    }
)

def __getattr__(name: str):
    if name == "register_statsig_integration":
        from intergrax.integrations.providers.feature_flag.statsig.register import register_statsig_integration

        return register_statsig_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.feature_flag.statsig import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.feature_flag.statsig import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.feature_flag.statsig import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
