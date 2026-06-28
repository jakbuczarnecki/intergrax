# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Github issue tracker integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.collaboration import IssueTrackerIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

GITHUB_ISSUE_TRACKER_PROVIDER_ID = "github"


class GithubIssueTrackerIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Github issue tracker integration."""

    pass


@runtime_checkable
class GithubIssueTrackerClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class GithubIssueTrackerIntegration(IssueTrackerIntegrationContract):
    """
    Github issue tracker integration.

    The legacy facade (create_github_issue_tracker) remains separate and backward-compatible.
    """

    config: GithubIssueTrackerIntegrationConfig = GithubIssueTrackerIntegrationConfig()
    _client: GithubIssueTrackerClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: GithubIssueTrackerClient,
        *,
        enabled: bool = False,
    ) -> GithubIssueTrackerIntegration:
        integration = cls.for_provider(
            provider_id=GITHUB_ISSUE_TRACKER_PROVIDER_ID,
            display_name="Github",
            config=GithubIssueTrackerIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> GithubIssueTrackerClient | None:
        return self._client
