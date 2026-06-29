# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Launchdarkly feature flag integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.feature_flag import FeatureFlagBackend
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
    Single public Launchdarkly feature flag entrypoint.

    Legacy catalog factory (create_launchdarkly_feature_flag) delegates to this class.
    """

    config: LaunchdarklyFeatureFlagIntegrationConfig = LaunchdarklyFeatureFlagIntegrationConfig()
    _client: LaunchdarklyFeatureFlagClient | None = PrivateAttr(default=None)
    _runtime: Any | None = PrivateAttr(default=None)

    @classmethod
    def from_runtime(cls, runtime: Any, *, enabled: bool = True) -> LaunchdarklyFeatureFlagIntegration:
        integration = cls.for_provider(
            provider_id=LAUNCHDARKLY_FEATURE_FLAG_PROVIDER_ID,
            display_name="Launchdarkly",
            config=LaunchdarklyFeatureFlagIntegrationConfig(enabled=enabled),
        )
        integration._runtime = runtime
        return integration

    def _require_runtime(self) -> Any:
        if self._runtime is None:
            raise IntegrationConfigurationError("Launchdarkly integration requires a runtime delegate")
        return self._runtime



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
    def __getattr__(self, name: str) -> object:
        if name.startswith("_"):
            private = object.__getattribute__(self, "__pydantic_private__")
            if name in private:
                return private[name]
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        return getattr(self._require_runtime(), name)

FeatureFlagBackend.register(LaunchdarklyFeatureFlagIntegration)
