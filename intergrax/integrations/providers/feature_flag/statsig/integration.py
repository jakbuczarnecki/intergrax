# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Statsig feature flag integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.feature_flag import FeatureFlagBackend
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
    Single public Statsig feature flag entrypoint.

    Legacy catalog factory (create_statsig_feature_flag) delegates to this class.
    """

    config: StatsigFeatureFlagIntegrationConfig = StatsigFeatureFlagIntegrationConfig()
    _client: StatsigFeatureFlagClient | None = PrivateAttr(default=None)
    _runtime: Any | None = PrivateAttr(default=None)

    @classmethod
    def from_runtime(cls, runtime: Any, *, enabled: bool = True) -> StatsigFeatureFlagIntegration:
        integration = cls.for_provider(
            provider_id=STATSIG_FEATURE_FLAG_PROVIDER_ID,
            display_name="Statsig",
            config=StatsigFeatureFlagIntegrationConfig(enabled=enabled),
        )
        integration._runtime = runtime
        return integration

    def _require_runtime(self) -> Any:
        if self._runtime is None:
            raise IntegrationConfigurationError("Statsig integration requires a runtime delegate")
        return self._runtime



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
    def __getattr__(self, name: str) -> object:
        if name.startswith("_"):
            private = object.__getattribute__(self, "__pydantic_private__")
            if name in private:
                return private[name]
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        return getattr(self._require_runtime(), name)

FeatureFlagBackend.register(StatsigFeatureFlagIntegration)
