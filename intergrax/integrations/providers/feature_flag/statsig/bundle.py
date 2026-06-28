# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p7.factories import create_statsig_feature_flag

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.feature_flag.statsig.integration import (
    STATSIG_FEATURE_FLAG_PROVIDER_ID,
    StatsigFeatureFlagIntegration,
    StatsigFeatureFlagIntegrationConfig,
    StatsigFeatureFlagClient,
)

__all__ = [
    "create_statsig_feature_flag",
    "create_statsig_feature_flag_integration",
]


def create_statsig_feature_flag_integration(
    *,
    client: StatsigFeatureFlagClient | None = None,
    enabled: bool = False,
) -> StatsigFeatureFlagIntegration:
    """
    Build a contract-based Statsig feature flag integration.

    The legacy facade (create_statsig_feature_flag) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Statsig feature flag integration requires an injected client when enabled=True",
        )
    if client is not None:
        return StatsigFeatureFlagIntegration.from_client(client, enabled=enabled)
    return StatsigFeatureFlagIntegration.for_provider(
        provider_id=STATSIG_FEATURE_FLAG_PROVIDER_ID,
        display_name="Statsig",
        config=StatsigFeatureFlagIntegrationConfig(enabled=enabled),
    )
