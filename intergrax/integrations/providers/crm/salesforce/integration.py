# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Salesforce crm integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.crm import CrmBackend
from intergrax.runtime.integrations.categories.automation import CrmIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

SALESFORCE_CRM_PROVIDER_ID = "salesforce"


class SalesforceCrmIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Salesforce crm integration."""

    pass


SalesforceCrmClient = CrmBackend

class SalesforceCrmIntegration(CrmIntegrationContract):
    """
    Single public Salesforce crm entrypoint.

    Legacy catalog factory (create_salesforce_crm) owns catalog behavior; legacy factories use from_client().
    """

    config: SalesforceCrmIntegrationConfig = SalesforceCrmIntegrationConfig()
    _client: SalesforceCrmClient | None = PrivateAttr(default=None)
    

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
        client: SalesforceCrmClient,
        *,
        enabled: bool = False,
    ) -> SalesforceCrmIntegration:
        integration = cls.for_provider(
            provider_id=SALESFORCE_CRM_PROVIDER_ID,
            display_name="Salesforce",
            config=SalesforceCrmIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> SalesforceCrmClient | None:
        return self._client

CrmBackend.register(SalesforceCrmIntegration)
