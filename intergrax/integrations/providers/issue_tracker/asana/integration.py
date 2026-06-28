# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Asana issue tracker integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.collaboration import IssueTrackerIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

ASANA_ISSUE_TRACKER_PROVIDER_ID = "asana"


class AsanaIssueTrackerIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Asana issue tracker integration."""

    pass


@runtime_checkable
class AsanaIssueTrackerClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class AsanaIssueTrackerIntegration(IssueTrackerIntegrationContract):
    """
    Asana issue tracker integration.

    The legacy facade (create_asana_issue_tracker) remains separate and backward-compatible.
    """

    config: AsanaIssueTrackerIntegrationConfig = AsanaIssueTrackerIntegrationConfig()
    _client: AsanaIssueTrackerClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: AsanaIssueTrackerClient,
        *,
        enabled: bool = False,
    ) -> AsanaIssueTrackerIntegration:
        integration = cls.for_provider(
            provider_id=ASANA_ISSUE_TRACKER_PROVIDER_ID,
            display_name="Asana",
            config=AsanaIssueTrackerIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> AsanaIssueTrackerClient | None:
        return self._client
