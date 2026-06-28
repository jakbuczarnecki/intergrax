# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Bitbucket issue tracker integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.collaboration import IssueTrackerIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

BITBUCKET_ISSUE_TRACKER_PROVIDER_ID = "bitbucket"


class BitbucketIssueTrackerIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Bitbucket issue tracker integration."""

    pass


@runtime_checkable
class BitbucketIssueTrackerClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class BitbucketIssueTrackerIntegration(IssueTrackerIntegrationContract):
    """
    Bitbucket issue tracker integration.

    The legacy facade (create_bitbucket_issue_tracker) remains separate and backward-compatible.
    """

    config: BitbucketIssueTrackerIntegrationConfig = BitbucketIssueTrackerIntegrationConfig()
    _client: BitbucketIssueTrackerClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: BitbucketIssueTrackerClient,
        *,
        enabled: bool = False,
    ) -> BitbucketIssueTrackerIntegration:
        integration = cls.for_provider(
            provider_id=BITBUCKET_ISSUE_TRACKER_PROVIDER_ID,
            display_name="Bitbucket",
            config=BitbucketIssueTrackerIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> BitbucketIssueTrackerClient | None:
        return self._client
