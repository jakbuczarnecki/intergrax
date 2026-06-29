# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Lab Json interaction surface integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.interaction_surface import InteractionSurface
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
    Single public Lab Json interaction surface entrypoint.

    Legacy catalog factory (create_lab_json_integration) delegates to this class.
    """

    config: LabJsonInteractionSurfaceIntegrationConfig = LabJsonInteractionSurfaceIntegrationConfig()
    _client: LabJsonInteractionSurfaceClient | None = PrivateAttr(default=None)
    _runtime: Any | None = PrivateAttr(default=None)

    @classmethod
    def from_runtime(cls, runtime: Any, *, enabled: bool = True) -> LabJsonInteractionSurfaceIntegration:
        integration = cls.for_provider(
            provider_id=LAB_JSON_INTERACTION_SURFACE_PROVIDER_ID,
            display_name="Lab Json",
            config=LabJsonInteractionSurfaceIntegrationConfig(enabled=enabled),
        )
        integration._runtime = runtime
        return integration

    def _require_runtime(self) -> Any:
        if self._runtime is None:
            raise IntegrationConfigurationError("Lab Json integration requires a runtime delegate")
        return self._runtime



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
    def __getattr__(self, name: str) -> object:
        if name.startswith("_"):
            private = object.__getattribute__(self, "__pydantic_private__")
            if name in private:
                return private[name]
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        return getattr(self._require_runtime(), name)

InteractionSurface.register(LabJsonInteractionSurfaceIntegration)
