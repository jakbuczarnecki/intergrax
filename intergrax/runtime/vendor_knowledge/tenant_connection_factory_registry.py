# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Provider-neutral registry for tenant connection integration factories."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.credential import CredentialResolutionMode
from intergrax.runtime.vendor_knowledge.models import JsonValue
from intergrax.runtime.vendor_knowledge.tenant_connection_factory_contract import (
    require_valid_credential_resolution_mode,
)
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
        require_valid_credential_resolution_mode(factory.credential_resolution_mode)
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

    def credential_resolution_mode_for(
        self,
        *,
        provider_id: str,
        integration_kind: IntegrationCategory,
    ) -> CredentialResolutionMode:
        factory = self.factory_for(
            provider_id=provider_id,
            integration_kind=integration_kind,
        )
        if factory is None:
            raise ValueError("tenant connection integration factory is unavailable")
        return require_valid_credential_resolution_mode(
            factory.credential_resolution_mode,
        )

    def create_integration_with_resolved_credential(
        self,
        *,
        tenant_id: str,
        connection_ref: str,
        provider_id: str,
        integration_kind: IntegrationCategory,
        credential_ref: str,
        resolved_credential: str,
        secret_free_config: Mapping[str, JsonValue],
    ) -> object:
        factory = self._require_factory(
            provider_id=provider_id,
            integration_kind=integration_kind,
        )
        mode = require_valid_credential_resolution_mode(
            factory.credential_resolution_mode,
        )
        if mode is not CredentialResolutionMode.RESOLVED_MATERIAL:
            raise ValueError(
                "factory does not support resolved-material credential resolution",
            )
        return factory.create_integration_with_resolved_credential(
            tenant_id=tenant_id,
            connection_ref=connection_ref,
            provider_id=provider_id,
            integration_kind=integration_kind,
            credential_ref=credential_ref,
            resolved_credential=resolved_credential,
            secret_free_config=secret_free_config,
        )

    def create_late_bound_integration(
        self,
        *,
        tenant_id: str,
        connection_ref: str,
        provider_id: str,
        integration_kind: IntegrationCategory,
        credential_ref: str,
        secret_free_config: Mapping[str, JsonValue],
    ) -> object:
        factory = self._require_factory(
            provider_id=provider_id,
            integration_kind=integration_kind,
        )
        mode = require_valid_credential_resolution_mode(
            factory.credential_resolution_mode,
        )
        if mode is not CredentialResolutionMode.LATE_BOUND:
            raise ValueError(
                "factory does not support late-bound credential resolution",
            )
        return factory.create_late_bound_integration(
            tenant_id=tenant_id,
            connection_ref=connection_ref,
            provider_id=provider_id,
            integration_kind=integration_kind,
            credential_ref=credential_ref,
            secret_free_config=secret_free_config,
        )

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
        return self.create_integration_with_resolved_credential(
            tenant_id=tenant_id,
            connection_ref=connection_ref,
            provider_id=provider_id,
            integration_kind=integration_kind,
            credential_ref=credential_ref,
            resolved_credential=credential,
            secret_free_config=secret_free_config,
        )

    def _require_factory(
        self,
        *,
        provider_id: str,
        integration_kind: IntegrationCategory,
    ) -> TenantConnectionIntegrationFactory:
        factory = self.factory_for(
            provider_id=provider_id,
            integration_kind=integration_kind,
        )
        if factory is None:
            raise ValueError("tenant connection integration factory is unavailable")
        return factory


__all__ = [
    "TenantConnectionFactoryKey",
    "TenantConnectionIntegrationFactoryRegistry",
]
