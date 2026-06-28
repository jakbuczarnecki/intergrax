# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Ms365 Graph collaboration suite integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.collaboration import CollaborationSuiteIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID = "ms365_graph"


class Ms365GraphCollaborationSuiteIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Ms365 Graph collaboration suite integration."""

    pass


@runtime_checkable
class Ms365GraphCollaborationSuiteClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class Ms365GraphCollaborationSuiteIntegration(CollaborationSuiteIntegrationContract):
    """
    Ms365 Graph collaboration suite integration.

    The legacy facade (create_ms365_graph_integration) remains separate and backward-compatible.
    """

    config: Ms365GraphCollaborationSuiteIntegrationConfig = Ms365GraphCollaborationSuiteIntegrationConfig()
    _client: Ms365GraphCollaborationSuiteClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: Ms365GraphCollaborationSuiteClient,
        *,
        enabled: bool = False,
    ) -> Ms365GraphCollaborationSuiteIntegration:
        integration = cls.for_provider(
            provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
            display_name="Ms365 Graph",
            config=Ms365GraphCollaborationSuiteIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> Ms365GraphCollaborationSuiteClient | None:
        return self._client
