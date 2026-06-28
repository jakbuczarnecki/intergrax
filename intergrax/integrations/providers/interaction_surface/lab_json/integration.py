# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Lab Json interaction surface integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.collaboration import InteractionSurfaceIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

LAB_JSON_INTERACTION_SURFACE_PROVIDER_ID = "lab_json"


class LabJsonInteractionSurfaceIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Lab Json interaction surface integration."""

    pass


@runtime_checkable
class LabJsonInteractionSurfaceClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class LabJsonInteractionSurfaceIntegration(InteractionSurfaceIntegrationContract):
    """
    Lab Json interaction surface integration.

    The legacy facade (create_lab_json_integration) remains separate and backward-compatible.
    """

    config: LabJsonInteractionSurfaceIntegrationConfig = LabJsonInteractionSurfaceIntegrationConfig()
    _client: LabJsonInteractionSurfaceClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: LabJsonInteractionSurfaceClient,
        *,
        enabled: bool = False,
    ) -> LabJsonInteractionSurfaceIntegration:
        integration = cls.for_provider(
            provider_id=LAB_JSON_INTERACTION_SURFACE_PROVIDER_ID,
            display_name="Lab Json",
            config=LabJsonInteractionSurfaceIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> LabJsonInteractionSurfaceClient | None:
        return self._client
