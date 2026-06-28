# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Salesforce crm integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.automation import CrmIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

SALESFORCE_CRM_PROVIDER_ID = "salesforce"


class SalesforceCrmIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Salesforce crm integration."""

    pass


@runtime_checkable
class SalesforceCrmClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class SalesforceCrmIntegration(CrmIntegrationContract):
    """
    Salesforce crm integration.

    The legacy facade (create_salesforce_crm) remains separate and backward-compatible.
    """

    config: SalesforceCrmIntegrationConfig = SalesforceCrmIntegrationConfig()
    _client: SalesforceCrmClient | None = PrivateAttr(default=None)

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
