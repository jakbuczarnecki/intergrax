# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Azure cloud platform integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.cloud_platform import CloudPlatform
from intergrax.runtime.integrations.categories.devops import CloudPlatformIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

AZURE_CLOUD_PLATFORM_PROVIDER_ID = "azure"


class AzureCloudPlatformIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Azure cloud platform integration."""

    pass


@runtime_checkable
class AzureCloudPlatformClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class AzureCloudPlatformIntegration(CloudPlatformIntegrationContract):
    """
    Single public Azure cloud platform entrypoint.

    Legacy catalog factory (create_azure_integration) delegates to this class.
    """

    config: AzureCloudPlatformIntegrationConfig = AzureCloudPlatformIntegrationConfig()
    _client: _AzureCloudPlatformClient | None = PrivateAttr(default=None)
    _runtime: Any | None = PrivateAttr(default=None)

    @classmethod
    def from_runtime(cls, runtime: Any, *, enabled: bool = True) -> AzureCloudPlatformIntegration:
        integration = cls.for_provider(
            provider_id=AZURE_CLOUD_PLATFORM_PROVIDER_ID,
            display_name="Azure",
            config=AzureCloudPlatformIntegrationConfig(enabled=enabled),
        )
        integration._runtime = runtime
        return integration

    def _require_runtime(self) -> Any:
        if self._runtime is None:
            raise IntegrationConfigurationError("Azure integration requires a runtime delegate")
        return self._runtime



    @classmethod
    def from_client(
        cls,
        client: _AzureCloudPlatformClient,
        *,
        enabled: bool = False,
    ) -> AzureCloudPlatformIntegration:
        integration = cls.for_provider(
            provider_id=AZURE_CLOUD_PLATFORM_PROVIDER_ID,
            display_name="Azure",
            config=AzureCloudPlatformIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> AzureCloudPlatformClient | None:
        return self._client
    def __getattr__(self, name: str) -> object:
        if name.startswith("_"):
            private = object.__getattribute__(self, "__pydantic_private__")
            if name in private:
                return private[name]
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        return getattr(self._require_runtime(), name)

CloudPlatform.register(AzureCloudPlatformIntegration)
