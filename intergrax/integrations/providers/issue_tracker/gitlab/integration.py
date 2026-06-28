# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Gitlab issue tracker integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.collaboration import IssueTrackerIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

GITLAB_ISSUE_TRACKER_PROVIDER_ID = "gitlab"


class GitlabIssueTrackerIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Gitlab issue tracker integration."""

    pass


@runtime_checkable
class GitlabIssueTrackerClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class GitlabIssueTrackerIntegration(IssueTrackerIntegrationContract):
    """
    Gitlab issue tracker integration.

    The legacy facade (create_gitlab_integration) remains separate and backward-compatible.
    """

    config: GitlabIssueTrackerIntegrationConfig = GitlabIssueTrackerIntegrationConfig()
    _client: GitlabIssueTrackerClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: GitlabIssueTrackerClient,
        *,
        enabled: bool = False,
    ) -> GitlabIssueTrackerIntegration:
        integration = cls.for_provider(
            provider_id=GITLAB_ISSUE_TRACKER_PROVIDER_ID,
            display_name="Gitlab",
            config=GitlabIssueTrackerIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> GitlabIssueTrackerClient | None:
        return self._client
