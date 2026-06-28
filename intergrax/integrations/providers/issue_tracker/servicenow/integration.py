# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Servicenow issue tracker integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.collaboration import IssueTrackerIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

SERVICENOW_ISSUE_TRACKER_PROVIDER_ID = "servicenow"


class ServicenowIssueTrackerIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Servicenow issue tracker integration."""

    pass


@runtime_checkable
class ServicenowIssueTrackerClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class ServicenowIssueTrackerIntegration(IssueTrackerIntegrationContract):
    """
    Servicenow issue tracker integration.

    The legacy facade (create_servicenow_issue_tracker) remains separate and backward-compatible.
    """

    config: ServicenowIssueTrackerIntegrationConfig = ServicenowIssueTrackerIntegrationConfig()
    _client: ServicenowIssueTrackerClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: ServicenowIssueTrackerClient,
        *,
        enabled: bool = False,
    ) -> ServicenowIssueTrackerIntegration:
        integration = cls.for_provider(
            provider_id=SERVICENOW_ISSUE_TRACKER_PROVIDER_ID,
            display_name="Servicenow",
            config=ServicenowIssueTrackerIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> ServicenowIssueTrackerClient | None:
        return self._client
