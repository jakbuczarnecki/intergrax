# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.project_status.config import ProjectStatusIntegrationConfig
from intergrax.integrations.providers.project_status.knowledge_read import (
    PROJECT_STATUS_PROVIDER_ID,
    PROJECT_STATUS_SOURCE_KIND,
    ProjectStatusReadClient,
    ProjectStatusSnapshotV1,
)
from intergrax.runtime.integrations.categories.collaboration import IssueTrackerIntegrationContract

__all__ = [
    "PROJECT_STATUS_PROVIDER_ID",
    "PROJECT_STATUS_SOURCE_KIND",
    "ProjectStatusIntegration",
    "ProjectStatusIntegrationConfig",
]


class ProjectStatusIntegration(IssueTrackerIntegrationContract):
    """Single public Project Status entrypoint for Vendor Knowledge live reads."""

    config: ProjectStatusIntegrationConfig = ProjectStatusIntegrationConfig(
        base_url="http://127.0.0.1:8765",
    )
    _client: ProjectStatusReadClient | None = PrivateAttr(default=None)

    async def read_project_status(self, *, project_id: str) -> ProjectStatusSnapshotV1:
        return await self._require_client().read_project_status(project_id=project_id)

    def _require_client(self) -> ProjectStatusReadClient:
        if self._client is None:
            raise IntegrationConfigurationError(
                f"{type(self).__name__} requires a configured read client",
            )
        return self._client

    @classmethod
    def from_client(
        cls,
        client: ProjectStatusReadClient,
        *,
        config: ProjectStatusIntegrationConfig | None = None,
    ) -> ProjectStatusIntegration:
        integration = cls.for_provider(
            provider_id=PROJECT_STATUS_PROVIDER_ID,
            display_name="Project Status",
            config=config
            or ProjectStatusIntegrationConfig(
                base_url="http://127.0.0.1:8765",
            ),
        )
        integration._client = client
        return integration
