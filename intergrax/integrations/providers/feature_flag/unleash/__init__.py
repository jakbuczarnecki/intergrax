# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "UNLEASH_FEATURE_FLAG_PROVIDER_ID",
    "UnleashFeatureFlagIntegration",
    "UnleashFeatureFlagIntegrationConfig",
    "UnleashFeatureFlagClient",
    "create_unleash_feature_flag",
    "create_unleash_feature_flag_integration",
    "register_unleash_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_unleash_feature_flag",
        "create_unleash_feature_flag_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "UNLEASH_FEATURE_FLAG_PROVIDER_ID",
        "UnleashFeatureFlagIntegration",
        "UnleashFeatureFlagIntegrationConfig",
        "UnleashFeatureFlagClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "UNLEASH_FEATURE_FLAG_PROVIDER_ID",
        "UnleashFeatureFlagIntegration",
        "UnleashFeatureFlagIntegrationConfig",
        "UnleashFeatureFlagClient",
    }
)

def __getattr__(name: str):
    if name == "register_unleash_integration":
        from intergrax.integrations.providers.feature_flag.unleash.register import register_unleash_integration

        return register_unleash_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.feature_flag.unleash import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.feature_flag.unleash import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.feature_flag.unleash import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
