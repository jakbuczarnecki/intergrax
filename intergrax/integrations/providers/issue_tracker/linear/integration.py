# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Linear issue tracker integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.collaboration import IssueTrackerIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

LINEAR_ISSUE_TRACKER_PROVIDER_ID = "linear"


class LinearIssueTrackerIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Linear issue tracker integration."""

    pass


@runtime_checkable
class LinearIssueTrackerClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class LinearIssueTrackerIntegration(IssueTrackerIntegrationContract):
    """
    Linear issue tracker integration.

    The legacy facade (create_linear_issue_tracker) remains separate and backward-compatible.
    """

    config: LinearIssueTrackerIntegrationConfig = LinearIssueTrackerIntegrationConfig()
    _client: LinearIssueTrackerClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: LinearIssueTrackerClient,
        *,
        enabled: bool = False,
    ) -> LinearIssueTrackerIntegration:
        integration = cls.for_provider(
            provider_id=LINEAR_ISSUE_TRACKER_PROVIDER_ID,
            display_name="Linear",
            config=LinearIssueTrackerIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> LinearIssueTrackerClient | None:
        return self._client
