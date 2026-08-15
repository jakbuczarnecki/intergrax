# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tenant-safe connection reads and typed live capability catalog."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.runtime.vendor_knowledge.live.contracts import (
    HARD_MAX_CONTENT_BYTES_PER_ITEM,
    HARD_MAX_PROVIDER_PAGE_SIZE,
    HARD_MAX_PROVIDER_PAGES,
    HARD_MAX_PROVIDER_REQUESTS,
    HARD_MAX_RESULT_BYTES,
    HARD_MAX_RESULT_ITEMS,
    HARD_MAX_UPSTREAM_ITEMS,
)
from intergrax.runtime.vendor_knowledge.live.identity import (
    LIVE_CONTRACT_VERSION,
    validate_capability_identity,
)
from intergrax.runtime.vendor_knowledge.tenant_connections import (
    SafeTenantConnectionV1,
    TenantConnectionAdministrativeStatus,
    TenantConnectionInvalidState,
    TenantConnectionNotFound,
    TenantConnectionRepository,
    to_safe_tenant_connection,
)

_FORBIDDEN_CAPABILITY_SUFFIXES: tuple[str, ...] = (
    ".write",
    ".create",
    ".delete",
    ".update",
)


class CapabilityEffectV1(StrEnum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    ADMIN = "admin"


def _require_non_empty(value: str, *, field_name: str, max_length: int | None = None) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be a non-empty string")
    if max_length is not None and len(cleaned) > max_length:
        raise ValueError(f"{field_name} exceeds maximum length {max_length}")
    return cleaned


class LiveCapabilityDescriptorV1(BaseModel):
    """Typed descriptor for a provider-declared live capability."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    capability_id: str
    provider_id: str
    integration_kind: IntegrationCategory
    source_kind: str
    contract_version: str

    effect: CapabilityEffectV1
    read_only: bool

    resource_scope_required: bool
    supported_resource_types: tuple[str, ...] = ()

    request_schema_ref: str
    result_schema_ref: str

    max_result_items: int = HARD_MAX_RESULT_ITEMS
    max_result_bytes: int = HARD_MAX_RESULT_BYTES
    max_provider_pages: int = HARD_MAX_PROVIDER_PAGES
    max_provider_requests: int = HARD_MAX_PROVIDER_REQUESTS
    max_upstream_items: int = HARD_MAX_UPSTREAM_ITEMS
    max_provider_page_size: int = HARD_MAX_PROVIDER_PAGE_SIZE
    max_content_bytes_per_item: int = HARD_MAX_CONTENT_BYTES_PER_ITEM
    available: bool = True

    @field_validator("capability_id")
    @classmethod
    def _valid_capability_id(cls, value: str) -> str:
        return _require_non_empty(value, field_name="capability_id", max_length=128)

    @field_validator("provider_id")
    @classmethod
    def _valid_provider_id(cls, value: str) -> str:
        return _require_non_empty(value, field_name="provider_id", max_length=64)

    @field_validator("source_kind")
    @classmethod
    def _valid_source_kind(cls, value: str) -> str:
        return _require_non_empty(value, field_name="source_kind", max_length=64)

    @field_validator("contract_version")
    @classmethod
    def _valid_contract_version(cls, value: str) -> str:
        if value != LIVE_CONTRACT_VERSION:
            raise ValueError("live_contract_version_unsupported")
        return value

    @field_validator("request_schema_ref", "result_schema_ref")
    @classmethod
    def _valid_schema_ref(cls, value: str, info) -> str:
        field_name = info.field_name or "schema_ref"
        return _require_non_empty(value, field_name=field_name, max_length=256)

    @field_validator("supported_resource_types")
    @classmethod
    def _valid_resource_types(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized: list[str] = []
        seen: set[str] = set()
        for item in value:
            cleaned = _require_non_empty(item, field_name="supported_resource_types", max_length=64)
            if cleaned not in seen:
                seen.add(cleaned)
                normalized.append(cleaned)
        return tuple(sorted(normalized))

    @field_validator(
        "max_result_items",
        "max_result_bytes",
        "max_provider_pages",
        "max_provider_requests",
        "max_upstream_items",
        "max_provider_page_size",
        "max_content_bytes_per_item",
    )
    @classmethod
    def _valid_positive_limit(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError("limit must be greater than or equal to 1")
        return value

    @model_validator(mode="after")
    def _validate_canonical_identity(self) -> LiveCapabilityDescriptorV1:
        validate_capability_identity(
            capability_id=self.capability_id,
            provider_id=self.provider_id,
            integration_kind=self.integration_kind,
            source_kind=self.source_kind,
            contract_version=self.contract_version,
        )
        return self


def is_bindable_read_only_capability(descriptor: LiveCapabilityDescriptorV1) -> bool:
    if descriptor.effect is not CapabilityEffectV1.READ:
        return False
    if not descriptor.read_only:
        return False
    if not descriptor.available:
        return False
    try:
        validate_capability_identity(
            capability_id=descriptor.capability_id,
            provider_id=descriptor.provider_id,
            integration_kind=descriptor.integration_kind,
            source_kind=descriptor.source_kind,
            contract_version=descriptor.contract_version,
        )
    except ValueError:
        return False
    normalized_id = descriptor.capability_id.strip()
    return (
        descriptor.effect is CapabilityEffectV1.READ
        and descriptor.read_only
        and descriptor.available
        and not any(
        normalized_id.endswith(suffix) for suffix in _FORBIDDEN_CAPABILITY_SUFFIXES
        )
    )


@runtime_checkable
class TenantConnectionPort(Protocol):
    def get_connection(
        self,
        *,
        tenant_id: str,
        connection_ref: str,
    ) -> SafeTenantConnectionV1 | None:
        ...

    def list_connections(
        self,
        *,
        tenant_id: str,
        limit: int = 100,
        administrative_status: TenantConnectionAdministrativeStatus | None = None,
    ) -> tuple[SafeTenantConnectionV1, ...]:
        ...


class RepositoryTenantConnectionPort:
    """Safe tenant connection reads backed by a durable repository."""

    def __init__(self, repository: TenantConnectionRepository) -> None:
        self._repository = repository

    def get_connection(
        self,
        *,
        tenant_id: str,
        connection_ref: str,
    ) -> SafeTenantConnectionV1 | None:
        cleaned_tenant = tenant_id.strip()
        cleaned_ref = connection_ref.strip()
        if not cleaned_tenant or not cleaned_ref:
            return None
        connection = self._repository.get(
            tenant_id=cleaned_tenant,
            connection_ref=cleaned_ref,
        )
        if connection is None or connection.tenant_id != cleaned_tenant:
            return None
        return to_safe_tenant_connection(connection)

    def list_connections(
        self,
        *,
        tenant_id: str,
        limit: int = 100,
        administrative_status: TenantConnectionAdministrativeStatus | None = None,
    ) -> tuple[SafeTenantConnectionV1, ...]:
        connections = self._repository.list(
            tenant_id=tenant_id,
            limit=limit,
            administrative_status=administrative_status,
        )
        safe_connections = [
            to_safe_tenant_connection(connection)
            for connection in connections
            if connection.tenant_id == tenant_id.strip()
        ]
        safe_connections.sort(key=lambda item: item.connection_ref)
        return tuple(safe_connections)


@runtime_checkable
class TenantLiveCapabilityCatalogPort(Protocol):
    def list_capabilities(
        self,
        *,
        tenant_id: str,
        connection_ref: str,
        remote_resource_id: str | None,
    ) -> tuple[LiveCapabilityDescriptorV1, ...]:
        ...


def _validate_remote_resource_id(remote_resource_id: str | None) -> None:
    if remote_resource_id is None:
        return
    cleaned = remote_resource_id.strip()
    if not cleaned:
        raise ValueError("remote_resource_id must be a non-empty string when provided")
    if len(cleaned) > 256:
        raise ValueError("remote_resource_id exceeds maximum length 256")


def _require_active_connection(
    connection_port: TenantConnectionPort,
    *,
    tenant_id: str,
    connection_ref: str,
) -> SafeTenantConnectionV1:
    connection = connection_port.get_connection(
        tenant_id=tenant_id,
        connection_ref=connection_ref,
    )
    if connection is None:
        raise TenantConnectionNotFound("tenant connection was not found")
    if connection.administrative_status is TenantConnectionAdministrativeStatus.DISABLED:
        raise TenantConnectionInvalidState(
            "tenant connection administrative status does not allow capability listing"
        )
    if connection.administrative_status is TenantConnectionAdministrativeStatus.REVOKED:
        raise TenantConnectionInvalidState(
            "tenant connection administrative status does not allow capability listing"
        )
    if connection.administrative_status is not TenantConnectionAdministrativeStatus.ACTIVE:
        raise TenantConnectionInvalidState(
            "tenant connection administrative status does not allow capability listing"
        )
    return connection


class TenantLiveCapabilityCatalog:
    """Runtime registry of typed live capabilities matched to tenant connections."""

    def __init__(self, *, connection_port: TenantConnectionPort) -> None:
        self._connection_port = connection_port
        self._registry: dict[
            tuple[str, IntegrationCategory, str, str],
            LiveCapabilityDescriptorV1,
        ] = {}

    def register(self, descriptor: LiveCapabilityDescriptorV1) -> None:
        key = (
            descriptor.provider_id,
            descriptor.integration_kind,
            descriptor.capability_id,
            descriptor.contract_version,
        )
        if key in self._registry:
            raise ValueError("capability descriptor is already registered")
        self._registry[key] = descriptor

    def list_capabilities(
        self,
        *,
        tenant_id: str,
        connection_ref: str,
        remote_resource_id: str | None,
    ) -> tuple[LiveCapabilityDescriptorV1, ...]:
        _validate_remote_resource_id(remote_resource_id)
        connection = _require_active_connection(
            self._connection_port,
            tenant_id=tenant_id,
            connection_ref=connection_ref,
        )
        matched = [
            descriptor
            for descriptor in self._registry.values()
            if descriptor.provider_id == connection.provider_id
            and descriptor.integration_kind == connection.integration_kind
        ]
        matched.sort(key=lambda item: item.capability_id)
        return tuple(matched)


class TenantConnectionCapabilityReadService:
    """Public tenant-scoped read boundary for connections and read-only capabilities."""

    def __init__(
        self,
        *,
        tenant_id: str,
        connection_port: TenantConnectionPort,
        capability_catalog: TenantLiveCapabilityCatalogPort,
    ) -> None:
        cleaned_tenant = str(tenant_id).strip()
        if not cleaned_tenant:
            raise ValueError("tenant_id must be a non-empty string")
        self._tenant_id = cleaned_tenant
        self._connection_port = connection_port
        self._capability_catalog = capability_catalog

    def get_connection(self, connection_ref: str) -> SafeTenantConnectionV1:
        connection = self._connection_port.get_connection(
            tenant_id=self._tenant_id,
            connection_ref=connection_ref,
        )
        if connection is None:
            raise TenantConnectionNotFound("tenant connection was not found")
        return connection

    def list_connections(
        self,
        *,
        limit: int = 100,
        administrative_status: TenantConnectionAdministrativeStatus | None = None,
    ) -> tuple[SafeTenantConnectionV1, ...]:
        return self._connection_port.list_connections(
            tenant_id=self._tenant_id,
            limit=limit,
            administrative_status=administrative_status,
        )

    def list_read_only_capabilities(
        self,
        *,
        connection_ref: str,
        remote_resource_id: str | None = None,
    ) -> tuple[LiveCapabilityDescriptorV1, ...]:
        capabilities = self._capability_catalog.list_capabilities(
            tenant_id=self._tenant_id,
            connection_ref=connection_ref,
            remote_resource_id=remote_resource_id,
        )
        bindable = [
            descriptor
            for descriptor in capabilities
            if is_bindable_read_only_capability(descriptor)
        ]
        bindable.sort(key=lambda item: item.capability_id)
        return tuple(bindable)
