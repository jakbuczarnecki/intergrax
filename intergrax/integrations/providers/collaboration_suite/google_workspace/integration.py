# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Google Workspace collaboration suite integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.collaboration_suite import CollaborationSuite
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
    Single public Google Workspace collaboration suite entrypoint.

    Legacy catalog factory (create_google_workspace_collaboration_suite) delegates to this class.
    """

    config: GoogleWorkspaceCollaborationSuiteIntegrationConfig = GoogleWorkspaceCollaborationSuiteIntegrationConfig()
    _client: GoogleWorkspaceCollaborationSuiteClient | None = PrivateAttr(default=None)
    _runtime: Any | None = PrivateAttr(default=None)

    @classmethod
    def from_runtime(cls, runtime: Any, *, enabled: bool = True) -> GoogleWorkspaceCollaborationSuiteIntegration:
        integration = cls.for_provider(
            provider_id=GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
            display_name="Google Workspace",
            config=GoogleWorkspaceCollaborationSuiteIntegrationConfig(enabled=enabled),
        )
        integration._runtime = runtime
        return integration

    def _require_runtime(self) -> Any:
        if self._runtime is None:
            raise IntegrationConfigurationError("Google Workspace integration requires a runtime delegate")
        return self._runtime



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
    def __getattr__(self, name: str) -> object:
        if name.startswith("_"):
            private = object.__getattribute__(self, "__pydantic_private__")
            if name in private:
                return private[name]
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        return getattr(self._require_runtime(), name)

CollaborationSuite.register(GoogleWorkspaceCollaborationSuiteIntegration)
