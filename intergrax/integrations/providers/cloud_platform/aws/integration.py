# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Aws cloud platform integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.cloud_platform import CloudPlatform
from intergrax.runtime.integrations.categories.devops import CloudPlatformIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

AWS_CLOUD_PLATFORM_PROVIDER_ID = "aws"


class AwsCloudPlatformIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Aws cloud platform integration."""

    pass


@runtime_checkable
class AwsCloudPlatformClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class AwsCloudPlatformIntegration(CloudPlatformIntegrationContract):
    """
    Single public Aws cloud platform entrypoint.

    Legacy catalog factory (create_aws_integration) delegates to this class.
    """

    config: AwsCloudPlatformIntegrationConfig = AwsCloudPlatformIntegrationConfig()
    _client: _AwsCloudPlatformClient | None = PrivateAttr(default=None)
    _runtime: Any | None = PrivateAttr(default=None)

    @classmethod
    def from_runtime(cls, runtime: Any, *, enabled: bool = True) -> AwsCloudPlatformIntegration:
        integration = cls.for_provider(
            provider_id=AWS_CLOUD_PLATFORM_PROVIDER_ID,
            display_name="Aws",
            config=AwsCloudPlatformIntegrationConfig(enabled=enabled),
        )
        integration._runtime = runtime
        return integration

    def _require_runtime(self) -> Any:
        if self._runtime is None:
            raise IntegrationConfigurationError("Aws integration requires a runtime delegate")
        return self._runtime



    @classmethod
    def from_client(
        cls,
        client: _AwsCloudPlatformClient,
        *,
        enabled: bool = False,
    ) -> AwsCloudPlatformIntegration:
        integration = cls.for_provider(
            provider_id=AWS_CLOUD_PLATFORM_PROVIDER_ID,
            display_name="Aws",
            config=AwsCloudPlatformIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> AwsCloudPlatformClient | None:
        return self._client
    def __getattr__(self, name: str) -> object:
        if name.startswith("_"):
            private = object.__getattribute__(self, "__pydantic_private__")
            if name in private:
                return private[name]
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        return getattr(self._require_runtime(), name)

CloudPlatform.register(AwsCloudPlatformIntegration)
