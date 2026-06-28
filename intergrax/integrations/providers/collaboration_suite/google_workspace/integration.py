# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Google Workspace collaboration suite integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.collaboration import CollaborationSuiteIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID = "google_workspace"


class GoogleWorkspaceCollaborationSuiteIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Google Workspace collaboration suite integration."""

    pass


@runtime_checkable
class GoogleWorkspaceCollaborationSuiteClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class GoogleWorkspaceCollaborationSuiteIntegration(CollaborationSuiteIntegrationContract):
    """
    Google Workspace collaboration suite integration.

    The legacy facade (create_google_workspace_collaboration_suite) remains separate and backward-compatible.
    """

    config: GoogleWorkspaceCollaborationSuiteIntegrationConfig = GoogleWorkspaceCollaborationSuiteIntegrationConfig()
    _client: GoogleWorkspaceCollaborationSuiteClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: GoogleWorkspaceCollaborationSuiteClient,
        *,
        enabled: bool = False,
    ) -> GoogleWorkspaceCollaborationSuiteIntegration:
        integration = cls.for_provider(
            provider_id=GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
            display_name="Google Workspace",
            config=GoogleWorkspaceCollaborationSuiteIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> GoogleWorkspaceCollaborationSuiteClient | None:
        return self._client
