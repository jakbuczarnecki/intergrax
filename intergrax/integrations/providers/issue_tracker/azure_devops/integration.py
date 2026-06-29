# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Azure Devops issue tracker integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.issue_tracker import IssueTracker
from intergrax.integrations.contracts.issue_tracker import IssueRecord, IssueSearchResult, IssueTracker
from intergrax.runtime.integrations.categories.collaboration import IssueTrackerIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

AZURE_DEVOPS_ISSUE_TRACKER_PROVIDER_ID = "azure_devops"


class AzureDevopsIssueTrackerIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Azure Devops issue tracker integration."""

    pass


AzureDevopsIssueTrackerClient = IssueTracker

class AzureDevopsIssueTrackerIntegration(IssueTrackerIntegrationContract):
    """
    Single public Azure Devops issue tracker entrypoint.

    Legacy catalog factory (create_azure_devops_issue_tracker) owns catalog behavior; legacy factories use from_client().
    """

    config: AzureDevopsIssueTrackerIntegrationConfig = AzureDevopsIssueTrackerIntegrationConfig()
    _client: AzureDevopsIssueTrackerClient | None = PrivateAttr(default=None)
    


    def search_issues(self, query: str, *, limit: int = 20) -> IssueSearchResult:
        return self._require_client().search_issues(query, limit=limit)

    def get_issue(self, issue_id: str) -> IssueRecord | None:
        return self._require_client().get_issue(issue_id)

    def create_issue(self, *, title: str, body: str = "", labels: Sequence[str] = ()) -> IssueRecord:
        return self._require_client().create_issue(title=title, body=body, labels=labels)


    def add_comment(self, issue_key, body):
        return self._require_client().add_comment(issue_key, body)

    def _require_client(self) -> IssueTracker:
        if self._client is None:
            raise IntegrationConfigurationError(
                f"{type(self).__name__} requires a catalog client for operations",
            )
        return self._client


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

IssueTracker.register(AzureDevopsIssueTrackerIntegration)
