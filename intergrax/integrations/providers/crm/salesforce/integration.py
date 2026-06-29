# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Salesforce crm integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.crm import CrmBackend
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
    Single public Salesforce crm entrypoint.

    Legacy catalog factory (create_salesforce_crm) delegates to this class.
    """

    config: SalesforceCrmIntegrationConfig = SalesforceCrmIntegrationConfig()
    _client: SalesforceCrmClient | None = PrivateAttr(default=None)
    _runtime: Any | None = PrivateAttr(default=None)

    @classmethod
    def from_runtime(cls, runtime: Any, *, enabled: bool = True) -> SalesforceCrmIntegration:
        integration = cls.for_provider(
            provider_id=SALESFORCE_CRM_PROVIDER_ID,
            display_name="Salesforce",
            config=SalesforceCrmIntegrationConfig(enabled=enabled),
        )
        integration._runtime = runtime
        return integration

    def _require_runtime(self) -> Any:
        if self._runtime is None:
            raise IntegrationConfigurationError("Salesforce integration requires a runtime delegate")
        return self._runtime



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
    def __getattr__(self, name: str) -> object:
        if name.startswith("_"):
            private = object.__getattribute__(self, "__pydantic_private__")
            if name in private:
                return private[name]
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        return getattr(self._require_runtime(), name)

CrmBackend.register(SalesforceCrmIntegration)
