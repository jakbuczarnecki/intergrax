# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Statsig feature flag integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.feature_flag import FeatureFlagBackend
from intergrax.runtime.integrations.categories.security import FeatureFlagIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

STATSIG_FEATURE_FLAG_PROVIDER_ID = "statsig"


class StatsigFeatureFlagIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Statsig feature flag integration."""

    pass


StatsigFeatureFlagClient = FeatureFlagBackend

class StatsigFeatureFlagIntegration(FeatureFlagIntegrationContract):
    """
    Single public Statsig feature flag entrypoint.

    Legacy catalog factory (create_statsig_feature_flag) owns catalog behavior; legacy factories use from_client().
    """

    config: StatsigFeatureFlagIntegrationConfig = StatsigFeatureFlagIntegrationConfig()
    _client: StatsigFeatureFlagClient | None = PrivateAttr(default=None)
    

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

FeatureFlagBackend.register(StatsigFeatureFlagIntegration)
