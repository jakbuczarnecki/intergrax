# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Launchdarkly feature flag integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.feature_flag import FeatureFlagBackend
from intergrax.runtime.integrations.categories.security import FeatureFlagIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

LAUNCHDARKLY_FEATURE_FLAG_PROVIDER_ID = "launchdarkly"


class LaunchdarklyFeatureFlagIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Launchdarkly feature flag integration."""

    pass


LaunchdarklyFeatureFlagClient = FeatureFlagBackend

class LaunchdarklyFeatureFlagIntegration(FeatureFlagIntegrationContract):
    """
    Single public Launchdarkly feature flag entrypoint.

    Legacy catalog factory (create_launchdarkly_feature_flag) owns catalog behavior; legacy factories use from_client().
    """

    config: LaunchdarklyFeatureFlagIntegrationConfig = LaunchdarklyFeatureFlagIntegrationConfig()
    _client: LaunchdarklyFeatureFlagClient | None = PrivateAttr(default=None)
    

    def is_enabled(self, flag_key: str, *, tenant_id: str, user_id: str = "") -> bool:
        return self._require_client().is_enabled(flag_key, tenant_id=tenant_id, user_id=user_id)

    def evaluate(self, flag_key: str, *, tenant_id: str, user_id: str = ""):
        return self._require_client().evaluate(flag_key, tenant_id=tenant_id, user_id=user_id)

    def _require_client(self) -> FeatureFlagBackend:
        if self._client is None:
            raise IntegrationConfigurationError(
                f"{type(self).__name__} requires a catalog client for operations",
            )
        return self._client


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

FeatureFlagBackend.register(LaunchdarklyFeatureFlagIntegration)
