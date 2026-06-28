# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unleash feature flag integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.security import FeatureFlagIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

UNLEASH_FEATURE_FLAG_PROVIDER_ID = "unleash"


class UnleashFeatureFlagIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Unleash feature flag integration."""

    pass


@runtime_checkable
class UnleashFeatureFlagClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class UnleashFeatureFlagIntegration(FeatureFlagIntegrationContract):
    """
    Unleash feature flag integration.

    The legacy facade (create_unleash_feature_flag) remains separate and backward-compatible.
    """

    config: UnleashFeatureFlagIntegrationConfig = UnleashFeatureFlagIntegrationConfig()
    _client: UnleashFeatureFlagClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: UnleashFeatureFlagClient,
        *,
        enabled: bool = False,
    ) -> UnleashFeatureFlagIntegration:
        integration = cls.for_provider(
            provider_id=UNLEASH_FEATURE_FLAG_PROVIDER_ID,
            display_name="Unleash",
            config=UnleashFeatureFlagIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> UnleashFeatureFlagClient | None:
        return self._client
