# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Azure Devops issue tracker integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.collaboration import IssueTrackerIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

AZURE_DEVOPS_ISSUE_TRACKER_PROVIDER_ID = "azure_devops"


class AzureDevopsIssueTrackerIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Azure Devops issue tracker integration."""

    pass


@runtime_checkable
class AzureDevopsIssueTrackerClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class AzureDevopsIssueTrackerIntegration(IssueTrackerIntegrationContract):
    """
    Azure Devops issue tracker integration.

    The legacy facade (create_azure_devops_issue_tracker) remains separate and backward-compatible.
    """

    config: AzureDevopsIssueTrackerIntegrationConfig = AzureDevopsIssueTrackerIntegrationConfig()
    _client: AzureDevopsIssueTrackerClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: AzureDevopsIssueTrackerClient,
        *,
        enabled: bool = False,
    ) -> AzureDevopsIssueTrackerIntegration:
        integration = cls.for_provider(
            provider_id=AZURE_DEVOPS_ISSUE_TRACKER_PROVIDER_ID,
            display_name="Azure Devops",
            config=AzureDevopsIssueTrackerIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> AzureDevopsIssueTrackerClient | None:
        return self._client
