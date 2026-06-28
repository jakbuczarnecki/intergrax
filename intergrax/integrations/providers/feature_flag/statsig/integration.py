# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Statsig feature flag integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.security import FeatureFlagIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

STATSIG_FEATURE_FLAG_PROVIDER_ID = "statsig"


class StatsigFeatureFlagIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Statsig feature flag integration."""

    pass


@runtime_checkable
class StatsigFeatureFlagClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class StatsigFeatureFlagIntegration(FeatureFlagIntegrationContract):
    """
    Statsig feature flag integration.

    The legacy facade (create_statsig_feature_flag) remains separate and backward-compatible.
    """

    config: StatsigFeatureFlagIntegrationConfig = StatsigFeatureFlagIntegrationConfig()
    _client: StatsigFeatureFlagClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: StatsigFeatureFlagClient,
        *,
        enabled: bool = False,
    ) -> StatsigFeatureFlagIntegration:
        integration = cls.for_provider(
            provider_id=STATSIG_FEATURE_FLAG_PROVIDER_ID,
            display_name="Statsig",
            config=StatsigFeatureFlagIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> StatsigFeatureFlagClient | None:
        return self._client
