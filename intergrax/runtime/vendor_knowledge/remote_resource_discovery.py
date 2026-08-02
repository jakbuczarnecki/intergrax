# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Provider-neutral remote resource discovery boundary for tenant connections."""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Protocol, runtime_checkable
from urllib.parse import parse_qsl, urlparse

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.runtime.vendor_knowledge.connections import KnowledgeConnectionRegistry
from intergrax.runtime.vendor_knowledge.errors import (
    VendorKnowledgeError,
    VendorKnowledgeErrorCode,
)
from intergrax.runtime.vendor_knowledge.tenant_connection_capabilities import (
    LiveCapabilityDescriptorV1,
    TenantConnectionPort,
    TenantLiveCapabilityCatalogPort,
    is_bindable_read_only_capability,
)
from intergrax.runtime.vendor_knowledge.tenant_connections import (
    SafeTenantConnectionV1,
    TenantConnectionAdministrativeStatus,
    TenantConnectionInvalidState,
    TenantConnectionNotFound,
)

_URL_IN_TEXT_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://\S+")
_AUTHORIZATION_ASSIGN_RE = re.compile(r"\bauthorization\s*[:=]", re.IGNORECASE)
_BEARER_TOKEN_RE = re.compile(r"\bbearer\s+[A-Za-z0-9._\-+=/]+", re.IGNORECASE)
_API_KEY_ASSIGN_RE = re.compile(r"\bapi[_-]?key\s*[:=]", re.IGNORECASE)
_SECRET_QUERY_NAMES = frozenset(
    {"token", "access_token", "refresh_token", "password", "secret", "api_key", "authorization", "credential", "bearer"}
)
_MAX_PAGE_TOKEN_LENGTH = 4096


class RemoteResourceAvailabilityV1(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    PERMISSION_DENIED = "permission_denied"
    NOT_FOUND = "not_found"


def _require_non_empty(value: str, *, field_name: str, max_length: int | None = None) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be a non-empty string")
    if max_length is not None and len(cleaned) > max_length:
        raise ValueError(f"{field_name} exceeds maximum length {max_length}")
    return cleaned


def _optional_non_empty(value: str | None, *, field_name: str, max_length: int) -> str | None:
    return None if value is None else _require_non_empty(value, field_name=field_name, max_length=max_length)


def _url_embeds_secrets(url: str) -> bool:
    parsed = urlparse(url.strip())
    if parsed.username is not None or parsed.password is not None:
        return True
    return any(
        normalized in _SECRET_QUERY_NAMES
        or any(normalized.endswith(s) for s in ("_token", "_password", "_secret", "_api_key", "_authorization", "_credential", "_bearer"))
        for raw_key, _ in parse_qsl(parsed.query, keep_blank_values=True)
        for normalized in (raw_key.strip().lower().replace("-", "_"),)
        if normalized
    )


def _assert_safe_text(value: str, *, field_name: str, allow_empty: bool = False) -> str:
    cleaned = value.strip()
    if not cleaned:
        if allow_empty:
            return ""
        raise ValueError(f"{field_name} must be a non-empty string")
    if _AUTHORIZATION_ASSIGN_RE.search(cleaned) or _BEARER_TOKEN_RE.search(cleaned) or _API_KEY_ASSIGN_RE.search(cleaned):
        raise ValueError(f"{field_name} must not contain credential material")
    for match in _URL_IN_TEXT_RE.finditer(cleaned):
        if _url_embeds_secrets(match.group(0).rstrip(".,);]'\"")):
            raise ValueError(f"{field_name} must not contain credential-bearing or secret-bearing URLs")
    return cleaned


def _normalize_capability_ids(value: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    normalized = [
        cleaned
        for item in value
        for cleaned in (_require_non_empty(item, field_name="supported_capability_ids", max_length=128),)
        if cleaned not in seen and not seen.add(cleaned)
    ]
    return tuple(sorted(normalized))


def _assert_utc_aware(value: datetime, *, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must be a timezone-aware UTC datetime")
    return value


def _validate_page_token(token: str | None) -> str | None:
    if token is None:
        return None
    if not token.strip():
        raise ValueError("page_token must not be blank when provided")
    if len(token) > _MAX_PAGE_TOKEN_LENGTH:
        raise ValueError("page_token exceeds maximum length 4096")
    return token


class _RemoteResourceCoreV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    remote_resource_id: str
    resource_type: str
    safe_display_label: str
    safe_description: str = ""
    availability: RemoteResourceAvailabilityV1
    supported_capability_ids: tuple[str, ...] = ()
    parent_scope_ref: str | None = None

    @field_validator("remote_resource_id")
    @classmethod
    def _valid_remote_resource_id(cls, value: str) -> str:
        return _require_non_empty(value, field_name="remote_resource_id", max_length=256)

    @field_validator("resource_type")
    @classmethod
    def _valid_resource_type(cls, value: str) -> str:
        return _require_non_empty(value, field_name="resource_type", max_length=64)

    @field_validator("safe_display_label")
    @classmethod
    def _valid_display_label(cls, value: str) -> str:
        cleaned = _assert_safe_text(value, field_name="safe_display_label")
        if len(cleaned) > 256:
            raise ValueError("safe_display_label exceeds maximum length 256")
        return cleaned

    @field_validator("safe_description")
    @classmethod
    def _valid_description(cls, value: str) -> str:
        cleaned = _assert_safe_text(value, field_name="safe_description", allow_empty=True)
        if len(cleaned) > 1024:
            raise ValueError("safe_description exceeds maximum length 1024")
        return cleaned

    @field_validator("supported_capability_ids")
    @classmethod
    def _valid_capability_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _normalize_capability_ids(value)

    @field_validator("parent_scope_ref")
    @classmethod
    def _valid_parent_scope_ref(cls, value: str | None) -> str | None:
        return _optional_non_empty(value, field_name="parent_scope_ref", max_length=256)


class RemoteResourceCandidateV1(_RemoteResourceCoreV1):
    ...


class _PagedSnapshotV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    next_page_token: str | None = None
    snapshot_version: str

    @field_validator("next_page_token")
    @classmethod
    def _valid_next_page_token(cls, value: str | None) -> str | None:
        return _validate_page_token(value)

    @field_validator("snapshot_version")
    @classmethod
    def _valid_snapshot_version(cls, value: str) -> str:
        return _require_non_empty(value, field_name="snapshot_version", max_length=64)


class RemoteResourceCandidatePageV1(_PagedSnapshotV1):
    resources: tuple[RemoteResourceCandidateV1, ...] = ()


class RemoteResourceDescriptorV1(_RemoteResourceCoreV1):
    connection_ref: str
    provider_id: str
    integration_kind: IntegrationCategory
    source_kind: str
    discovered_at: datetime
    snapshot_version: str

    @field_validator("connection_ref", "provider_id", "source_kind", "snapshot_version")
    @classmethod
    def _valid_descriptor_strings(cls, value: str, info) -> str:
        limits = {"connection_ref": 128, "provider_id": 64, "source_kind": 64, "snapshot_version": 64}
        field_name = info.field_name or "field"
        return _require_non_empty(value, field_name=field_name, max_length=limits[field_name])

    @field_validator("discovered_at")
    @classmethod
    def _valid_discovered_at(cls, value: datetime) -> datetime:
        return _assert_utc_aware(value, field_name="discovered_at")


class RemoteResourceDiscoveryPageV1(_PagedSnapshotV1):
    resources: tuple[RemoteResourceDescriptorV1, ...] = ()

    @model_validator(mode="after")
    def _resource_snapshot_versions_match(self) -> RemoteResourceDiscoveryPageV1:
        if any(resource.snapshot_version != self.snapshot_version for resource in self.resources):
            raise ValueError("resource snapshot_version must match page snapshot_version")
        return self


@runtime_checkable
class RemoteResourceDiscoveryProvider(Protocol):
    @property
    def provider_id(self) -> str: ...
    @property
    def integration_kind(self) -> IntegrationCategory: ...
    @property
    def source_kind(self) -> str: ...
    async def list_remote_resources(
        self,
        *,
        integration: object,
        connection: SafeTenantConnectionV1,
        page_token: str | None,
        limit: int,
    ) -> RemoteResourceCandidatePageV1: ...


type _DiscoveryRegistryKey = tuple[str, IntegrationCategory, str]


def _discovery_invalid(
    *,
    provider_id: str,
    source_kind: str | None = None,
    message: str = "Remote resource discovery failed",
) -> VendorKnowledgeError:
    return VendorKnowledgeError(
        code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
        safe_message=message,
        provider_id=provider_id,
        source_kind=source_kind,
        retryable=False,
    )


def _reraise_discovery_boundary(exc: BaseException, *, provider_id: str, source_kind: str) -> None:
    if isinstance(exc, VendorKnowledgeError):
        raise exc
    raise _discovery_invalid(provider_id=provider_id, source_kind=source_kind) from None


class RemoteResourceDiscoveryRegistry:
    def __init__(self) -> None:
        self._providers: dict[_DiscoveryRegistryKey, RemoteResourceDiscoveryProvider] = {}

    def register(self, provider: RemoteResourceDiscoveryProvider) -> None:
        provider_id = _require_non_empty(provider.provider_id, field_name="provider_id", max_length=64)
        integration_kind = provider.integration_kind
        if not isinstance(integration_kind, IntegrationCategory):
            raise ValueError("integration_kind must be an IntegrationCategory")
        source_kind = _require_non_empty(provider.source_kind, field_name="source_kind", max_length=64)
        key = (provider_id, integration_kind, source_kind)
        if key in self._providers:
            raise ValueError("discovery provider is already registered")
        self._providers[key] = provider

    def resolve(
        self,
        *,
        provider_id: str,
        integration_kind: IntegrationCategory,
        source_kind: str,
    ) -> RemoteResourceDiscoveryProvider:
        cleaned_provider = _require_non_empty(provider_id, field_name="provider_id", max_length=64)
        cleaned_source = _require_non_empty(source_kind, field_name="source_kind", max_length=64)
        if not isinstance(integration_kind, IntegrationCategory):
            raise ValueError("integration_kind must be an IntegrationCategory")
        provider = self._providers.get((cleaned_provider, integration_kind, cleaned_source))
        if provider is None:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.ADAPTER_NOT_FOUND,
                safe_message="Remote resource discovery provider was not found",
                provider_id=cleaned_provider,
                source_kind=cleaned_source,
                retryable=False,
            )
        return provider

    def registered_source_kinds(
        self,
        *,
        provider_id: str,
        integration_kind: IntegrationCategory,
    ) -> tuple[str, ...]:
        cleaned_provider = _require_non_empty(provider_id, field_name="provider_id", max_length=64)
        if not isinstance(integration_kind, IntegrationCategory):
            raise ValueError("integration_kind must be an IntegrationCategory")
        return tuple(sorted(s for p, k, s in self._providers if p == cleaned_provider and k is integration_kind))


def _require_active_connection(
    connection_port: TenantConnectionPort,
    *,
    tenant_id: str,
    connection_ref: str,
) -> SafeTenantConnectionV1:
    connection = connection_port.get_connection(tenant_id=tenant_id, connection_ref=connection_ref)
    if connection is None:
        raise TenantConnectionNotFound("tenant connection was not found")
    if connection.administrative_status is not TenantConnectionAdministrativeStatus.ACTIVE:
        raise TenantConnectionInvalidState(
            "tenant connection administrative status does not allow resource discovery"
        )
    return connection


def _candidate_fingerprint(candidate: RemoteResourceCandidateV1) -> tuple[object, ...]:
    return (candidate.resource_type, candidate.safe_display_label, candidate.safe_description, candidate.availability, candidate.supported_capability_ids, candidate.parent_scope_ref)


def _deduplicate_candidates_ordered(
    candidates: tuple[RemoteResourceCandidateV1, ...],
) -> tuple[RemoteResourceCandidateV1, ...]:
    by_id: dict[str, RemoteResourceCandidateV1] = {}
    order: list[str] = []
    for candidate in candidates:
        existing = by_id.get(candidate.remote_resource_id)
        if existing is None:
            by_id[candidate.remote_resource_id] = candidate
            order.append(candidate.remote_resource_id)
        elif _candidate_fingerprint(existing) != _candidate_fingerprint(candidate):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Remote resource discovery returned conflicting duplicates",
                retryable=False,
            )
    return tuple(by_id[item] for item in order)


def _validate_catalog_capabilities(
    descriptors: object,
    *,
    expected_provider_id: str,
    expected_integration_kind: IntegrationCategory,
) -> dict[str, LiveCapabilityDescriptorV1]:
    if not isinstance(descriptors, tuple):
        raise TypeError("catalog must return a tuple")
    catalog: dict[str, LiveCapabilityDescriptorV1] = {}
    for raw_descriptor in descriptors:
        if not isinstance(raw_descriptor, BaseModel):
            raise TypeError("catalog descriptor must be a Pydantic model")
        descriptor = LiveCapabilityDescriptorV1.model_validate(raw_descriptor.model_dump())
        if descriptor.provider_id != expected_provider_id:
            raise _discovery_invalid(provider_id=expected_provider_id, message="Capability catalog provider does not match the connection")
        if descriptor.integration_kind != expected_integration_kind:
            raise _discovery_invalid(provider_id=expected_provider_id, message="Capability catalog integration kind does not match the connection")
        if descriptor.capability_id in catalog:
            raise _discovery_invalid(provider_id=expected_provider_id, message="Capability catalog returned duplicate capability identifiers")
        catalog[descriptor.capability_id] = descriptor
    return catalog


def _intersect_capabilities(*, candidate: RemoteResourceCandidateV1, catalog: dict[str, LiveCapabilityDescriptorV1]) -> tuple[str, ...]:
    return tuple(sorted(capability_id for capability_id in candidate.supported_capability_ids if (descriptor := catalog.get(capability_id)) is not None and is_bindable_read_only_capability(descriptor)))


class TenantRemoteResourceDiscoveryService:
    def __init__(
        self,
        *,
        tenant_id: str,
        connection_port: TenantConnectionPort,
        capability_catalog: TenantLiveCapabilityCatalogPort,
        connection_registry: KnowledgeConnectionRegistry,
        discovery_registry: RemoteResourceDiscoveryRegistry,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        cleaned_tenant = str(tenant_id).strip()
        if not cleaned_tenant:
            raise ValueError("tenant_id must be a non-empty string")
        self._tenant_id = cleaned_tenant
        self._connection_port = connection_port
        self._capability_catalog = capability_catalog
        self._connection_registry = connection_registry
        self._discovery_registry = discovery_registry
        self._clock = clock or (lambda: datetime.now(UTC))

    def list_source_kinds(self, *, connection_ref: str) -> tuple[str, ...]:
        connection = _require_active_connection(
            self._connection_port, tenant_id=self._tenant_id, connection_ref=connection_ref
        )
        return self._discovery_registry.registered_source_kinds(
            provider_id=connection.provider_id,
            integration_kind=connection.integration_kind,
        )

    async def list_remote_resources(
        self,
        *,
        connection_ref: str,
        source_kind: str,
        page_token: str | None = None,
        limit: int = 100,
    ) -> RemoteResourceDiscoveryPageV1:
        if not isinstance(limit, int) or limit < 1 or limit > 100:
            raise ValueError("limit must be an integer between 1 and 100")
        cleaned_source_kind = _require_non_empty(source_kind, field_name="source_kind", max_length=64)
        validated_page_token = _validate_page_token(page_token)
        connection = _require_active_connection(
            self._connection_port, tenant_id=self._tenant_id, connection_ref=connection_ref
        )
        provider = self._discovery_registry.resolve(
            provider_id=connection.provider_id,
            integration_kind=connection.integration_kind,
            source_kind=cleaned_source_kind,
        )
        integration = self._connection_registry.resolve(
            tenant_id=self._tenant_id,
            connection_ref=connection.connection_ref,
            provider_id=connection.provider_id,
            integration_kind=connection.integration_kind,
        )
        try:
            raw_page = await provider.list_remote_resources(
                integration=integration,
                connection=connection,
                page_token=validated_page_token,
                limit=limit,
            )
        except Exception as exc:
            _reraise_discovery_boundary(exc, provider_id=connection.provider_id, source_kind=cleaned_source_kind)
        try:
            if not isinstance(raw_page, BaseModel):
                raise TypeError("provider result must be a Pydantic model")
            candidate_page = RemoteResourceCandidatePageV1.model_validate(raw_page.model_dump())
            if len(candidate_page.resources) > limit:
                raise ValueError("provider returned more resources than requested")
            deduplicated = _deduplicate_candidates_ordered(candidate_page.resources)
            clock_value = self._clock()
            if not isinstance(clock_value, datetime):
                raise TypeError("clock must return datetime")
            discovered_at = _assert_utc_aware(clock_value, field_name="discovered_at")
        except Exception as exc:
            _reraise_discovery_boundary(exc, provider_id=connection.provider_id, source_kind=cleaned_source_kind)
        descriptors: list[RemoteResourceDescriptorV1] = []
        for candidate in deduplicated:
            try:
                catalog = _validate_catalog_capabilities(
                    self._capability_catalog.list_capabilities(
                        tenant_id=self._tenant_id,
                        connection_ref=connection.connection_ref,
                        remote_resource_id=candidate.remote_resource_id,
                    ),
                    expected_provider_id=connection.provider_id,
                    expected_integration_kind=connection.integration_kind,
                )
                descriptors.append(
                    RemoteResourceDescriptorV1(
                        connection_ref=connection.connection_ref,
                        remote_resource_id=candidate.remote_resource_id,
                        provider_id=connection.provider_id,
                        integration_kind=connection.integration_kind,
                        source_kind=cleaned_source_kind,
                        resource_type=candidate.resource_type,
                        safe_display_label=candidate.safe_display_label,
                        safe_description=candidate.safe_description,
                        availability=candidate.availability,
                        supported_capability_ids=_intersect_capabilities(candidate=candidate, catalog=catalog),
                        parent_scope_ref=candidate.parent_scope_ref,
                        discovered_at=discovered_at,
                        snapshot_version=candidate_page.snapshot_version,
                    )
                )
            except Exception as exc:
                _reraise_discovery_boundary(exc, provider_id=connection.provider_id, source_kind=cleaned_source_kind)
        descriptors.sort(key=lambda item: (item.connection_ref, item.remote_resource_id, item.source_kind))
        return RemoteResourceDiscoveryPageV1(resources=tuple(descriptors), next_page_token=candidate_page.next_page_token, snapshot_version=candidate_page.snapshot_version)
