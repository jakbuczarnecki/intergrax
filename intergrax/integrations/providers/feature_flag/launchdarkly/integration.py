# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Launchdarkly feature flag integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.security import FeatureFlagIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

LAUNCHDARKLY_FEATURE_FLAG_PROVIDER_ID = "launchdarkly"


class LaunchdarklyFeatureFlagIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Launchdarkly feature flag integration."""

    pass


@runtime_checkable
class LaunchdarklyFeatureFlagClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class LaunchdarklyFeatureFlagIntegration(FeatureFlagIntegrationContract):
    """
    Launchdarkly feature flag integration.

    The legacy facade (create_launchdarkly_feature_flag) remains separate and backward-compatible.
    """

    config: LaunchdarklyFeatureFlagIntegrationConfig = LaunchdarklyFeatureFlagIntegrationConfig()
    _client: LaunchdarklyFeatureFlagClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: LaunchdarklyFeatureFlagClient,
        *,
        enabled: bool = False,
    ) -> LaunchdarklyFeatureFlagIntegration:
        integration = cls.for_provider(
            provider_id=LAUNCHDARKLY_FEATURE_FLAG_PROVIDER_ID,
            display_name="Launchdarkly",
            config=LaunchdarklyFeatureFlagIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> LaunchdarklyFeatureFlagClient | None:
        return self._client
