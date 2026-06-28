# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Zendesk issue tracker integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.collaboration import IssueTrackerIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

ZENDESK_ISSUE_TRACKER_PROVIDER_ID = "zendesk"


class ZendeskIssueTrackerIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Zendesk issue tracker integration."""

    pass


@runtime_checkable
class ZendeskIssueTrackerClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class ZendeskIssueTrackerIntegration(IssueTrackerIntegrationContract):
    """
    Zendesk issue tracker integration.

    The legacy facade (create_zendesk_issue_tracker) remains separate and backward-compatible.
    """

    config: ZendeskIssueTrackerIntegrationConfig = ZendeskIssueTrackerIntegrationConfig()
    _client: ZendeskIssueTrackerClient | None = PrivateAttr(default=None)

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
