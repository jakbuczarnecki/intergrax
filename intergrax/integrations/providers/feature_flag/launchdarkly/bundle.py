# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p5.factories import create_launchdarkly_feature_flag

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.feature_flag.launchdarkly.integration import (
    LAUNCHDARKLY_FEATURE_FLAG_PROVIDER_ID,
    LaunchdarklyFeatureFlagIntegration,
    LaunchdarklyFeatureFlagIntegrationConfig,
    LaunchdarklyFeatureFlagClient,
)

__all__ = [
    "create_launchdarkly_feature_flag",
    "create_launchdarkly_feature_flag_integration",
]


def create_launchdarkly_feature_flag_integration(
    *,
    client: LaunchdarklyFeatureFlagClient | None = None,
    enabled: bool = False,
) -> LaunchdarklyFeatureFlagIntegration:
    """
    Build a contract-based Launchdarkly feature flag integration.

    The legacy facade (create_launchdarkly_feature_flag) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Launchdarkly feature flag integration requires an injected client when enabled=True",
        )
    if client is not None:
        return LaunchdarklyFeatureFlagIntegration.from_client(client, enabled=enabled)
    return LaunchdarklyFeatureFlagIntegration.for_provider(
        provider_id=LAUNCHDARKLY_FEATURE_FLAG_PROVIDER_ID,
        display_name="Launchdarkly",
        config=LaunchdarklyFeatureFlagIntegrationConfig(enabled=enabled),
    )
