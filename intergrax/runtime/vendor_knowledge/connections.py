# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Instance-local connection registry and connection-aware integration resolver."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.runtime.vendor_knowledge.contracts import VendorIntegrationResolver
from intergrax.runtime.vendor_knowledge.errors import (
    VendorKnowledgeError,
    VendorKnowledgeErrorCode,
)
from intergrax.runtime.vendor_knowledge.models import KnowledgeSourceRef

type ConnectionRegistryKey = tuple[str, str]


@dataclass(frozen=True, slots=True)
class _ConnectionEntry:
    tenant_id: str
    connection_ref: str
    provider_id: str
    integration_kind: IntegrationCategory
    integration: object


class KnowledgeConnectionRegistry:
    """Instance-local registry of already constructed integration instances.

    Keyed by ``(tenant_id, connection_ref)``. Does not look up secrets, create
    clients, or replace ``IntegrationProfile``.
    """

    def __init__(self) -> None:
        self._entries: dict[ConnectionRegistryKey, _ConnectionEntry] = {}

    def register(
        self,
        *,
        tenant_id: str,
        connection_ref: str,
        provider_id: str,
        integration_kind: IntegrationCategory,
        integration: object,
    ) -> None:
        cleaned_tenant = tenant_id.strip()
        cleaned_ref = connection_ref.strip()
        cleaned_provider = provider_id.strip()
        if not cleaned_tenant:
            raise ValueError("tenant_id must be a non-empty string")
        if not cleaned_ref:
            raise ValueError("connection_ref must be a non-empty string")
        if not cleaned_provider:
            raise ValueError("provider_id must be a non-empty string")
        if not isinstance(integration_kind, IntegrationCategory):
            raise ValueError("integration_kind must be an IntegrationCategory")

        key: ConnectionRegistryKey = (cleaned_tenant, cleaned_ref)
        if key in self._entries:
            raise ValueError("connection is already registered for this tenant")

        self._entries[key] = _ConnectionEntry(
            tenant_id=cleaned_tenant,
            connection_ref=cleaned_ref,
            provider_id=cleaned_provider,
            integration_kind=integration_kind,
            integration=integration,
        )

    def resolve(
        self,
        *,
        tenant_id: str,
        connection_ref: str,
        provider_id: str,
        integration_kind: IntegrationCategory,
    ) -> object:
        cleaned_tenant = tenant_id.strip()
        cleaned_ref = connection_ref.strip()
        cleaned_provider = provider_id.strip()
        if not cleaned_tenant:
            raise ValueError("tenant_id must be a non-empty string")
        if not cleaned_ref:
            raise ValueError("connection_ref must be a non-empty string")
        if not cleaned_provider:
            raise ValueError("provider_id must be a non-empty string")

        key: ConnectionRegistryKey = (cleaned_tenant, cleaned_ref)
        entry = self._entries.get(key)
        if entry is None:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INTEGRATION_NOT_FOUND,
                safe_message="Requested connection is not registered for this tenant",
                provider_id=cleaned_provider,
                retryable=False,
            )

        if entry.provider_id != cleaned_provider:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INTEGRATION_NOT_FOUND,
                safe_message="Registered connection provider does not match the request",
                provider_id=cleaned_provider,
                retryable=False,
            )

        if entry.integration_kind != integration_kind:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INTEGRATION_CATEGORY_MISMATCH,
                safe_message="Registered connection category does not match the request",
                provider_id=cleaned_provider,
                retryable=False,
            )

        return entry.integration


class ConnectionAwareVendorResolver:
    """Resolve integrations via connection registry, with profile fallback."""

    def __init__(
        self,
        *,
        tenant_id: str,
        connection_registry: KnowledgeConnectionRegistry,
        fallback_resolver: VendorIntegrationResolver,
    ) -> None:
        cleaned_tenant = str(tenant_id).strip()
        if not cleaned_tenant:
            raise ValueError("tenant_id must be a non-empty string")
        self._tenant_id = cleaned_tenant
        self._connection_registry = connection_registry
        self._fallback_resolver = fallback_resolver

    def resolve(self, *, source: KnowledgeSourceRef) -> object:
        if source.tenant_id != self._tenant_id:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.TENANT_MISMATCH,
                safe_message=(
                    "Knowledge source tenant does not match the configured resolver tenant"
                ),
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )

        if source.connection_ref is None:
            return self._fallback_resolver.resolve(source=source)

        return self._connection_registry.resolve(
            tenant_id=source.tenant_id,
            connection_ref=source.connection_ref,
            provider_id=source.provider_id,
            integration_kind=source.integration_kind,
        )
