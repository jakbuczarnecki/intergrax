# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Lab Json interaction surface integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Optional, Sequence

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.interaction_surface import InteractionSurface
from intergrax.runtime.integrations.categories.collaboration import InteractionSurfaceIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig
from intergrax.utils import attribute_access


LAB_JSON_INTERACTION_SURFACE_PROVIDER_ID = "lab_json"


class LabJsonInteractionSurfaceIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Lab Json interaction surface integration."""

    pass


LabJsonInteractionSurfaceClient = InteractionSurface

class LabJsonInteractionSurfaceIntegration(InteractionSurfaceIntegrationContract):
    """
    Single public Lab Json interaction surface entrypoint.

    Legacy catalog factory (create_lab_json_integration) owns catalog behavior; legacy factories use from_client().
    """

    config: LabJsonInteractionSurfaceIntegrationConfig = LabJsonInteractionSurfaceIntegrationConfig()
    _client: LabJsonInteractionSurfaceClient | None = PrivateAttr(default=None)
    

    def can_handle(self, payload):
        return self._require_client().can_handle(payload)

    @property
    def channel(self):
        return attribute_access.optional_str(self._require_client(), 'channel')

    def to_inbound(self, payload, *, tenant_id: str, user_id: str):
        return self._require_client().to_inbound(payload, tenant_id=tenant_id, user_id=user_id)

    def to_task(self, payload, *, tenant_id: str, user_id: Optional[str] = None):
        return self._require_client().to_task(payload, tenant_id=tenant_id, user_id=user_id)

    def _require_client(self) -> InteractionSurface:
        if self._client is None:
            raise IntegrationConfigurationError(
                f"{type(self).__name__} requires a catalog client for operations",
            )
        return self._client


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

InteractionSurface.register(LabJsonInteractionSurfaceIntegration)
