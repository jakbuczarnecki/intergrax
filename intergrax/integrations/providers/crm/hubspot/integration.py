# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Hubspot crm integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.crm import CrmBackend
from intergrax.runtime.integrations.categories.automation import CrmIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

HUBSPOT_CRM_PROVIDER_ID = "hubspot"


class HubspotCrmIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Hubspot crm integration."""

    pass


HubspotCrmClient = CrmBackend

class HubspotCrmIntegration(CrmIntegrationContract):
    """
    Single public Hubspot crm entrypoint.

    Legacy catalog factory (create_hubspot_crm) owns catalog behavior; legacy factories use from_client().
    """

    config: HubspotCrmIntegrationConfig = HubspotCrmIntegrationConfig()
    _client: HubspotCrmClient | None = PrivateAttr(default=None)
    

    def get_account(self, account_id):
        return self._require_client().get_account(account_id)

    def list_contacts(self, account_id, limit: int = 50):
        return self._require_client().list_contacts(account_id, limit=limit)

    def list_tickets(self, account_id, limit: int = 50):
        return self._require_client().list_tickets(account_id, limit=limit)

    def _require_client(self) -> CrmBackend:
        if self._client is None:
            raise IntegrationConfigurationError(
                f"{type(self).__name__} requires a catalog client for operations",
            )
        return self._client


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

CrmBackend.register(HubspotCrmIntegration)
