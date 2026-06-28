# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Jira issue tracker integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.collaboration import IssueTrackerIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

JIRA_ISSUE_TRACKER_PROVIDER_ID = "jira"


class JiraIssueTrackerIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Jira issue tracker integration."""

    pass


@runtime_checkable
class JiraIssueTrackerClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class JiraIssueTrackerIntegration(IssueTrackerIntegrationContract):
    """
    Jira issue tracker integration.

    The legacy facade (create_jira_integration) remains separate and backward-compatible.
    """

    config: JiraIssueTrackerIntegrationConfig = JiraIssueTrackerIntegrationConfig()
    _client: JiraIssueTrackerClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: JiraIssueTrackerClient,
        *,
        enabled: bool = False,
    ) -> JiraIssueTrackerIntegration:
        integration = cls.for_provider(
            provider_id=JIRA_ISSUE_TRACKER_PROVIDER_ID,
            display_name="Jira",
            config=JiraIssueTrackerIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> JiraIssueTrackerClient | None:
        return self._client
