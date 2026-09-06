# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Provider-neutral registry for tenant connection integration factories."""

from __future__ import annotations

from collections.abc import Iterable

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.credential import supports_late_credential_resolution
from intergrax.runtime.vendor_knowledge.models import JsonValue
from intergrax.runtime.vendor_knowledge.tenant_connection_rehydration import (
    TenantConnectionIntegrationFactory,
)

type TenantConnectionFactoryKey = tuple[str, IntegrationCategory]


class TenantConnectionIntegrationFactoryRegistry:
    """Route safe durable connection records to provider-owned factories."""

    def __init__(
        self,
        factories: Iterable[
            tuple[str, IntegrationCategory, TenantConnectionIntegrationFactory]
        ] = (),
    ) -> None:
        self._factories: dict[
            TenantConnectionFactoryKey,
            TenantConnectionIntegrationFactory,
        ] = {}
        for provider_id, integration_kind, factory in factories:
            self.register(
                provider_id=provider_id,
                integration_kind=integration_kind,
                factory=factory,
            )

    def register(
        self,
        *,
        provider_id: str,
        integration_kind: IntegrationCategory,
        factory: TenantConnectionIntegrationFactory,
    ) -> None:
        cleaned_provider = provider_id.strip()
        if not cleaned_provider:
            raise ValueError("provider_id must be a non-empty string")
        if not isinstance(integration_kind, IntegrationCategory):
            raise ValueError("integration_kind must be an IntegrationCategory")
        if not isinstance(factory, TenantConnectionIntegrationFactory):
            raise TypeError("factory must implement TenantConnectionIntegrationFactory")
        key = (cleaned_provider, integration_kind)
        if key in self._factories:
            raise ValueError("tenant connection integration factory is already registered")
        self._factories[key] = factory

    def factory_for(
        self,
        *,
        provider_id: str,
        integration_kind: IntegrationCategory,
    ) -> TenantConnectionIntegrationFactory | None:
        return self._factories.get((provider_id.strip(), integration_kind))

    def supports_late_credential_resolution(
        self,
        *,
        provider_id: str,
        integration_kind: IntegrationCategory,
    ) -> bool:
        factory = self.factory_for(
            provider_id=provider_id,
            integration_kind=integration_kind,
        )
        if factory is None:
            return False
        return supports_late_credential_resolution(factory)

    def create_integration(
        self,
        *,
        tenant_id: str,
        connection_ref: str,
        provider_id: str,
        integration_kind: IntegrationCategory,
        credential_ref: str,
        credential: str,
        secret_free_config: dict[str, JsonValue],
    ) -> object:
        factory = self._factories.get((provider_id.strip(), integration_kind))
        if factory is None:
            raise ValueError("tenant connection integration factory is unavailable")
        return factory.create_integration(
            tenant_id=tenant_id,
            connection_ref=connection_ref,
            provider_id=provider_id,
            integration_kind=integration_kind,
            credential_ref=credential_ref,
            credential=credential,
            secret_free_config=secret_free_config,
        )


__all__ = [
    "TenantConnectionFactoryKey",
    "TenantConnectionIntegrationFactoryRegistry",
]
