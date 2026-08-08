# © Artur Czarnecki. All rights reserved.

"""Tenant-safe source-kind and mode catalog for Vendor Knowledge."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.runtime.vendor_knowledge.plugin import (
    VendorKnowledgeMode,
    VendorKnowledgeSourceIdentity,
    VendorKnowledgeSourcePluginRegistry,
)
from intergrax.runtime.vendor_knowledge.tenant_connection_capabilities import (
    TenantConnectionPort,
)
from intergrax.runtime.vendor_knowledge.tenant_connections import (
    TenantConnectionAdministrativeStatus,
    TenantConnectionInvalidState,
    TenantConnectionNotFound,
)


@dataclass(frozen=True, slots=True)
class TenantSourceKindCapabilitiesV1:
    """Provider-neutral capabilities for one active tenant connection source kind."""

    identity: VendorKnowledgeSourceIdentity
    modes: tuple[VendorKnowledgeMode, ...]


class TenantVendorKnowledgeSourceCatalog:
    """Match the authoritative plugin registry to tenant-owned connections."""

    def __init__(
        self,
        *,
        connection_port: TenantConnectionPort,
        plugin_registry: VendorKnowledgeSourcePluginRegistry,
    ) -> None:
        self._connection_port = connection_port
        self._plugin_registry = plugin_registry

    def list_source_kind_capabilities(
        self,
        *,
        tenant_id: str,
        connection_ref: str,
    ) -> tuple[TenantSourceKindCapabilitiesV1, ...]:
        connection = self._connection_port.get_connection(
            tenant_id=tenant_id,
            connection_ref=connection_ref,
        )
        if connection is None:
            raise TenantConnectionNotFound("tenant connection was not found")
        if connection.tenant_id != tenant_id:
            raise TenantConnectionNotFound("tenant connection was not found")
        if connection.administrative_status is not TenantConnectionAdministrativeStatus.ACTIVE:
            raise TenantConnectionInvalidState(
                "tenant connection does not allow source-kind listing",
            )

        matched: list[TenantSourceKindCapabilitiesV1] = []
        for plugin in self._plugin_registry.list_plugins():
            if (
                plugin.identity.provider_id != connection.provider_id
                or plugin.identity.integration_category != connection.integration_kind
            ):
                continue
            modes = tuple(capability.mode for capability in plugin.capabilities)
            matched.append(
                TenantSourceKindCapabilitiesV1(
                    identity=plugin.identity,
                    modes=modes,
                )
            )
        matched.sort(
            key=lambda item: (
                item.identity.source_kind,
                item.identity.provider_id,
                item.identity.integration_category.value,
            )
        )
        return tuple(matched)

    def list_source_kinds(
        self,
        *,
        tenant_id: str,
        connection_ref: str,
    ) -> tuple[VendorKnowledgeSourceIdentity, ...]:
        return tuple(
            item.identity
            for item in self.list_source_kind_capabilities(
                tenant_id=tenant_id,
                connection_ref=connection_ref,
            )
        )

    def list_modes(
        self,
        *,
        tenant_id: str,
        connection_ref: str,
        source_kind: str,
    ) -> tuple[VendorKnowledgeMode, ...]:
        normalized_kind = source_kind.strip()
        for item in self.list_source_kind_capabilities(
            tenant_id=tenant_id,
            connection_ref=connection_ref,
        ):
            if item.identity.source_kind == normalized_kind:
                return item.modes
        return ()


__all__ = [
    "TenantSourceKindCapabilitiesV1",
    "TenantVendorKnowledgeSourceCatalog",
]
