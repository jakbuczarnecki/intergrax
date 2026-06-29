# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Zendesk issue tracker integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.issue_tracker import IssueTracker
from intergrax.integrations.contracts.issue_tracker import IssueRecord, IssueSearchResult, IssueTracker
from intergrax.runtime.integrations.categories.collaboration import IssueTrackerIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

ZENDESK_ISSUE_TRACKER_PROVIDER_ID = "zendesk"


class ZendeskIssueTrackerIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Zendesk issue tracker integration."""

    pass


ZendeskIssueTrackerClient = IssueTracker

class ZendeskIssueTrackerIntegration(IssueTrackerIntegrationContract):
    """
    Single public Zendesk issue tracker entrypoint.

    Legacy catalog factory (create_zendesk_issue_tracker) owns catalog behavior; legacy factories use from_client().
    """

    config: ZendeskIssueTrackerIntegrationConfig = ZendeskIssueTrackerIntegrationConfig()
    _client: ZendeskIssueTrackerClient | None = PrivateAttr(default=None)
    


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
        client: ZendeskIssueTrackerClient,
        *,
        enabled: bool = False,
    ) -> ZendeskIssueTrackerIntegration:
        integration = cls.for_provider(
            provider_id=ZENDESK_ISSUE_TRACKER_PROVIDER_ID,
            display_name="Zendesk",
            config=ZendeskIssueTrackerIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> ZendeskIssueTrackerClient | None:
        return self._client

IssueTracker.register(ZendeskIssueTrackerIntegration)
