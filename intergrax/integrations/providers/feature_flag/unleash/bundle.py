# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p5.factories import create_unleash_feature_flag as _legacy_create_unleash_feature_flag

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.feature_flag.unleash.integration import (
    UNLEASH_FEATURE_FLAG_PROVIDER_ID,
    UnleashFeatureFlagIntegration,
    UnleashFeatureFlagIntegrationConfig,
    UnleashFeatureFlagClient,
)

__all__ = [
    "create_unleash_feature_flag",
    "create_unleash_feature_flag_integration",
]


def create_unleash_feature_flag_integration(
    *,
    client: UnleashFeatureFlagClient | None = None,
    enabled: bool = False,
) -> UnleashFeatureFlagIntegration:
    """
    Build a contract-based Unleash feature flag integration.

    The legacy facade (create_unleash_feature_flag) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Unleash feature flag integration requires an injected client when enabled=True",
        )
    if client is not None:
        return UnleashFeatureFlagIntegration.from_client(client, enabled=enabled)
    return UnleashFeatureFlagIntegration.for_provider(
        provider_id=UNLEASH_FEATURE_FLAG_PROVIDER_ID,
        display_name="Unleash",
        config=UnleashFeatureFlagIntegrationConfig(enabled=enabled),
    )


def create_unleash_feature_flag(**kwargs: object) -> UnleashFeatureFlagIntegration:
    """Compatibility shim — constructs UnleashFeatureFlagIntegration from legacy runtime."""
    runtime = _legacy_create_unleash_feature_flag(**kwargs)
    if isinstance(runtime, UnleashFeatureFlagIntegration):
        return runtime
    return UnleashFeatureFlagIntegration.from_runtime(runtime)
