# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Hubspot crm integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

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
    Hubspot crm integration.

    The legacy facade (create_hubspot_crm) remains separate and backward-compatible.
    """

    config: HubspotCrmIntegrationConfig = HubspotCrmIntegrationConfig()
    _client: HubspotCrmClient | None = PrivateAttr(default=None)

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
