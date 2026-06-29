# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Gcp cloud platform integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.cloud_platform import CloudPlatform
from intergrax.runtime.integrations.categories.devops import CloudPlatformIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

GCP_CLOUD_PLATFORM_PROVIDER_ID = "gcp"


class GcpCloudPlatformIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Gcp cloud platform integration."""

    pass


@runtime_checkable
class GcpCloudPlatformClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class GcpCloudPlatformIntegration(CloudPlatformIntegrationContract):
    """
    Single public Gcp cloud platform entrypoint.

    Legacy catalog factory (create_gcp_integration) delegates to this class.
    """

    config: GcpCloudPlatformIntegrationConfig = GcpCloudPlatformIntegrationConfig()
    _client: _GcpCloudPlatformClient | None = PrivateAttr(default=None)
    _runtime: Any | None = PrivateAttr(default=None)

    @classmethod
    def from_runtime(cls, runtime: Any, *, enabled: bool = True) -> GcpCloudPlatformIntegration:
        integration = cls.for_provider(
            provider_id=GCP_CLOUD_PLATFORM_PROVIDER_ID,
            display_name="Gcp",
            config=GcpCloudPlatformIntegrationConfig(enabled=enabled),
        )
        integration._runtime = runtime
        return integration

    def _require_runtime(self) -> Any:
        if self._runtime is None:
            raise IntegrationConfigurationError("Gcp integration requires a runtime delegate")
        return self._runtime



    @classmethod
    def from_client(
        cls,
        client: _GcpCloudPlatformClient,
        *,
        enabled: bool = False,
    ) -> GcpCloudPlatformIntegration:
        integration = cls.for_provider(
            provider_id=GCP_CLOUD_PLATFORM_PROVIDER_ID,
            display_name="Gcp",
            config=GcpCloudPlatformIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> GcpCloudPlatformClient | None:
        return self._client
    def __getattr__(self, name: str) -> object:
        if name.startswith("_"):
            private = object.__getattribute__(self, "__pydantic_private__")
            if name in private:
                return private[name]
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        return getattr(self._require_runtime(), name)

CloudPlatform.register(GcpCloudPlatformIntegration)
