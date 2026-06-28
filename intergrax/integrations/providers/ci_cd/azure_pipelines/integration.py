# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Azure Pipelines ci cd integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.devops import CiCdIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

AZURE_PIPELINES_CI_CD_PROVIDER_ID = "azure_pipelines"


class AzurePipelinesCiCdIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Azure Pipelines ci cd integration."""

    pass


@runtime_checkable
class AzurePipelinesCiCdClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class AzurePipelinesCiCdIntegration(CiCdIntegrationContract):
    """
    Azure Pipelines ci cd integration.

    The legacy facade (create_azure_pipelines_ci_cd) remains separate and backward-compatible.
    """

    config: AzurePipelinesCiCdIntegrationConfig = AzurePipelinesCiCdIntegrationConfig()
    _client: AzurePipelinesCiCdClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: AzurePipelinesCiCdClient,
        *,
        enabled: bool = False,
    ) -> AzurePipelinesCiCdIntegration:
        integration = cls.for_provider(
            provider_id=AZURE_PIPELINES_CI_CD_PROVIDER_ID,
            display_name="Azure Pipelines",
            config=AzurePipelinesCiCdIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> AzurePipelinesCiCdClient | None:
        return self._client
