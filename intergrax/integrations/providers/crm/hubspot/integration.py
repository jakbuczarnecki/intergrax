# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Hubspot crm integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.crm import CrmBackend
from intergrax.runtime.integrations.categories.automation import CrmIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

HUBSPOT_CRM_PROVIDER_ID = "hubspot"


class HubspotCrmIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Hubspot crm integration."""

    pass


@runtime_checkable
class HubspotCrmClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class HubspotCrmIntegration(CrmIntegrationContract):
    """
    Single public Hubspot crm entrypoint.

    Legacy catalog factory (create_hubspot_crm) delegates to this class.
    """

    config: HubspotCrmIntegrationConfig = HubspotCrmIntegrationConfig()
    _client: HubspotCrmClient | None = PrivateAttr(default=None)
    _runtime: Any | None = PrivateAttr(default=None)

    @classmethod
    def from_runtime(cls, runtime: Any, *, enabled: bool = True) -> HubspotCrmIntegration:
        integration = cls.for_provider(
            provider_id=HUBSPOT_CRM_PROVIDER_ID,
            display_name="Hubspot",
            config=HubspotCrmIntegrationConfig(enabled=enabled),
        )
        integration._runtime = runtime
        return integration

    def _require_runtime(self) -> Any:
        if self._runtime is None:
            raise IntegrationConfigurationError("Hubspot integration requires a runtime delegate")
        return self._runtime



    @classmethod
    def from_client(
        cls,
        client: HubspotCrmClient,
        *,
        enabled: bool = False,
    ) -> HubspotCrmIntegration:
        integration = cls.for_provider(
            provider_id=HUBSPOT_CRM_PROVIDER_ID,
            display_name="Hubspot",
            config=HubspotCrmIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> HubspotCrmClient | None:
        return self._client
    def __getattr__(self, name: str) -> object:
        if name.startswith("_"):
            private = object.__getattribute__(self, "__pydantic_private__")
            if name in private:
                return private[name]
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        return getattr(self._require_runtime(), name)

CrmBackend.register(HubspotCrmIntegration)
