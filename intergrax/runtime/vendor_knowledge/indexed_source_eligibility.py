# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Provider-neutral qualification for durable Indexed Sources."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Protocol, runtime_checkable
from urllib.parse import parse_qsl, urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.runtime.vendor_knowledge.errors import (
    VendorKnowledgeError,
    VendorKnowledgeErrorCode,
)
from intergrax.runtime.vendor_knowledge.remote_resource_discovery import (
    RemoteResourceAvailabilityV1,
    RemoteResourceDescriptorV1,
    RemoteResourceDiscoveryPageV1,
)
from intergrax.runtime.vendor_knowledge.tenant_connection_capabilities import (
    TenantConnectionPort,
)
from intergrax.runtime.vendor_knowledge.tenant_connections import (
    SafeTenantConnectionV1,
    TenantConnectionAdministrativeStatus,
)

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_CODE_RE = re.compile(r"^[a-z][a-z0-9_]{0,127}$")
_MAX_PROOF_TTL_SECONDS = 300
_MAX_DISCOVERY_PAGES = 20
_MATERIALIZATION_CONTRACT_VERSION = "vendor_knowledge.indexed_materialization.v1"
_SECRET_QUERY_NAMES = frozenset(
    {
        "access_token",
        "api_key",
        "authorization",
        "bearer",
        "credential",
        "password",
        "refresh_token",
        "secret",
        "token",
    }
)


class IndexedSourceEligibilityStatusV1(StrEnum):
    ELIGIBLE = "eligible"
    NOT_SUPPORTED = "not_supported"
    CONNECTION_INACTIVE = "connection_inactive"
    RESOURCE_UNAVAILABLE = "resource_unavailable"
    SNAPSHOT_STALE = "snapshot_stale"
    HANDLER_UNAVAILABLE = "handler_unavailable"


class IndexedSourceEligibilityRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str = Field(..., min_length=1, max_length=128)
    connection_ref: str = Field(..., min_length=1, max_length=128)
    remote_resource_id: str = Field(..., min_length=1, max_length=256)
    source_kind: str = Field(..., min_length=1, max_length=64)
    discovery_snapshot_version: str = Field(..., min_length=1, max_length=64)

    @field_validator(
        "tenant_id",
        "connection_ref",
        "remote_resource_id",
        "source_kind",
        "discovery_snapshot_version",
    )
    @classmethod
    def _normalize_identifiers(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("indexed_source_eligibility_identifier_blank")
        return cleaned


def _safe_text(value: str, *, field_name: str, allow_empty: bool = False) -> str:
    cleaned = value.strip()
    if not cleaned:
        if allow_empty:
            return ""
        raise ValueError(f"{field_name}_blank")
    lowered = cleaned.lower()
    if any(
        marker in lowered
        for marker in ("authorization:", "authorization=", "bearer ", "api_key:", "api_key=")
    ):
        raise ValueError(f"{field_name}_contains_credentials")
    for raw_url in re.findall(r"[A-Za-z][A-Za-z0-9+.-]*://\S+", cleaned):
        parsed = urlparse(raw_url.rstrip(".,);]'\""))
        if parsed.username or parsed.password:
            raise ValueError(f"{field_name}_contains_credentials")
        if any(
            key.strip().lower().replace("-", "_") in _SECRET_QUERY_NAMES
            for key, _ in parse_qsl(parsed.query, keep_blank_values=True)
        ):
            raise ValueError(f"{field_name}_contains_credentials")
    return cleaned


def _utc(value: datetime, *, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name}_must_be_utc")
    return value


def _sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_indexed_source_ref(
    *,
    tenant_id: str,
    connection_ref: str,
    provider_id: str,
    integration_kind: IntegrationCategory,
    remote_resource_id: str,
    source_kind: str,
) -> str:
    """Return a stable, opaque source identity without using a display label."""
    return "vksrc:" + _sha256(
        {
            "tenant_id": tenant_id,
            "connection_ref": connection_ref,
            "provider_id": provider_id,
            "integration_kind": integration_kind.value,
            "remote_resource_id": remote_resource_id,
            "source_kind": source_kind,
        }
    )


class IndexedSourceDescriptorV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_id: str = Field(..., min_length=1, max_length=64)
    integration_kind: IntegrationCategory
    connection_ref: str = Field(..., min_length=1, max_length=128)
    remote_resource_id: str = Field(..., min_length=1, max_length=256)
    source_kind: str = Field(..., min_length=1, max_length=64)
    canonical_source_ref: str = Field(..., min_length=70, max_length=70)
    safe_display_label: str = Field(..., min_length=1, max_length=256)
    safe_description: str = Field(default="", max_length=1024)
    resource_type: str = Field(..., min_length=1, max_length=64)
    discovery_snapshot_version: str = Field(..., min_length=1, max_length=64)
    materialization_contract_version: str = Field(..., min_length=1, max_length=64)

    @field_validator(
        "provider_id",
        "connection_ref",
        "remote_resource_id",
        "source_kind",
        "resource_type",
        "discovery_snapshot_version",
        "materialization_contract_version",
    )
    @classmethod
    def _non_empty_strings(cls, value: str, info) -> str:
        return _safe_text(value, field_name=info.field_name or "field")

    @field_validator("safe_display_label")
    @classmethod
    def _safe_label(cls, value: str) -> str:
        return _safe_text(value, field_name="safe_display_label")

    @field_validator("safe_description")
    @classmethod
    def _safe_description(cls, value: str) -> str:
        return _safe_text(value, field_name="safe_description", allow_empty=True)

    @field_validator("canonical_source_ref")
    @classmethod
    def _canonical_ref(cls, value: str) -> str:
        cleaned = _safe_text(value, field_name="canonical_source_ref")
        if not cleaned.startswith("vksrc:") or _SHA256_HEX_RE.fullmatch(cleaned[6:]) is None:
            raise ValueError("canonical_source_ref_invalid")
        return cleaned


class IndexedSourceBindingPlanV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str = Field(..., min_length=1, max_length=128)
    connection_ref: str = Field(..., min_length=1, max_length=128)
    provider_id: str = Field(..., min_length=1, max_length=64)
    integration_kind: IntegrationCategory
    remote_resource_id: str = Field(..., min_length=1, max_length=256)
    source_kind: str = Field(..., min_length=1, max_length=64)
    canonical_source_ref: str = Field(..., min_length=70, max_length=70)
    source_descriptor: IndexedSourceDescriptorV1
    sync_handler_ref: str = Field(..., min_length=1, max_length=256)
    materialization_contract_version: str = Field(..., min_length=1, max_length=64)
    proof_revision: str = Field(..., min_length=64, max_length=64)

    @field_validator(
        "tenant_id",
        "connection_ref",
        "provider_id",
        "remote_resource_id",
        "source_kind",
        "sync_handler_ref",
        "materialization_contract_version",
    )
    @classmethod
    def _safe_plan_strings(cls, value: str, info) -> str:
        return _safe_text(value, field_name=info.field_name or "field")

    @field_validator("canonical_source_ref", "proof_revision")
    @classmethod
    def _hash_fields(cls, value: str, info) -> str:
        cleaned = _safe_text(value, field_name=info.field_name or "field")
        if info.field_name == "canonical_source_ref":
            if not cleaned.startswith("vksrc:") or _SHA256_HEX_RE.fullmatch(cleaned[6:]) is None:
                raise ValueError("canonical_source_ref_invalid")
        elif _SHA256_HEX_RE.fullmatch(cleaned) is None:
            raise ValueError("proof_revision_invalid")
        return cleaned

    @model_validator(mode="after")
    def _descriptor_matches_plan(self) -> IndexedSourceBindingPlanV1:
        descriptor = self.source_descriptor
        if (
            self.connection_ref != descriptor.connection_ref
            or self.provider_id != descriptor.provider_id
            or self.integration_kind is not descriptor.integration_kind
            or self.remote_resource_id != descriptor.remote_resource_id
            or self.source_kind != descriptor.source_kind
            or self.canonical_source_ref != descriptor.canonical_source_ref
            or self.materialization_contract_version
            != descriptor.materialization_contract_version
        ):
            raise ValueError("indexed_source_binding_plan_descriptor_mismatch")
        return self


class IndexedSourceEligibilityProofV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: IndexedSourceEligibilityStatusV1
    eligible: bool
    binding_plan: IndexedSourceBindingPlanV1 | None = None
    safe_reason_code: str | None = None
    evaluated_at: datetime
    expires_at: datetime
    proof_revision: str = Field(..., min_length=64, max_length=64)

    @field_validator("safe_reason_code")
    @classmethod
    def _safe_reason(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if _SAFE_CODE_RE.fullmatch(cleaned) is None:
            raise ValueError("safe_reason_code_invalid")
        return cleaned

    @field_validator("evaluated_at", "expires_at")
    @classmethod
    def _utc_timestamps(cls, value: datetime, info) -> datetime:
        return _utc(value, field_name=info.field_name or "timestamp")

    @field_validator("proof_revision")
    @classmethod
    def _proof_hash(cls, value: str) -> str:
        if _SHA256_HEX_RE.fullmatch(value) is None:
            raise ValueError("proof_revision_invalid")
        return value

    @model_validator(mode="after")
    def _proof_invariants(self) -> IndexedSourceEligibilityProofV1:
        if self.expires_at <= self.evaluated_at:
            raise ValueError("eligibility_proof_expiration_invalid")
        if self.expires_at - self.evaluated_at > timedelta(seconds=_MAX_PROOF_TTL_SECONDS):
            raise ValueError("eligibility_proof_ttl_exceeded")
        if self.status is IndexedSourceEligibilityStatusV1.ELIGIBLE:
            if not self.eligible or self.binding_plan is None or self.safe_reason_code is not None:
                raise ValueError("eligible_proof_invariant_violation")
        elif self.eligible or self.binding_plan is not None or self.safe_reason_code is None:
            raise ValueError("ineligible_proof_invariant_violation")
        return self

    def is_current(self, now: datetime) -> bool:
        """Return whether the bounded proof can still be consumed as current."""
        current = _utc(now, field_name="now")
        return self.evaluated_at <= current < self.expires_at


@runtime_checkable
class IndexedSourceMaterializationProvider(Protocol):
    @property
    def provider_id(self) -> str: ...

    @property
    def integration_kind(self) -> IntegrationCategory: ...

    @property
    def source_kind(self) -> str: ...

    @property
    def materialization_contract_version(self) -> str: ...

    def qualify(
        self,
        *,
        connection: SafeTenantConnectionV1,
        resource: RemoteResourceDescriptorV1,
    ) -> IndexedSourceDescriptorV1 | None: ...

    def sync_handler_ref(self) -> str: ...

    def sync_handler_available(self) -> bool: ...


type IndexedSourceMaterializationRegistryKey = tuple[str, IntegrationCategory, str]


class IndexedSourceMaterializationRegistry:
    """Atomic instance-local registry of complete qualification bundles."""

    def __init__(self) -> None:
        self._providers: dict[
            IndexedSourceMaterializationRegistryKey,
            IndexedSourceMaterializationProvider,
        ] = {}

    def register(self, provider: IndexedSourceMaterializationProvider) -> None:
        provider_id = _safe_text(str(provider.provider_id), field_name="provider_id")
        source_kind = _safe_text(str(provider.source_kind), field_name="source_kind")
        integration_kind = provider.integration_kind
        contract_version = getattr(provider, "materialization_contract_version", None)
        if not isinstance(integration_kind, IntegrationCategory):
            raise ValueError("integration_kind_invalid")
        if not isinstance(contract_version, str) or not contract_version.strip():
            raise ValueError("materialization_contract_version_required")
        if not callable(getattr(provider, "qualify", None)):
            raise ValueError("materialization_qualifier_required")
        if not hasattr(provider, "sync_handler_ref") or not hasattr(
            provider, "sync_handler_available"
        ):
            raise ValueError("materialization_handler_registration_incomplete")
        key = (provider_id, integration_kind, source_kind)
        if key in self._providers:
            raise ValueError("materialization_provider_already_registered")
        self._providers[key] = provider

    def resolve(
        self,
        *,
        provider_id: str,
        integration_kind: IntegrationCategory,
        source_kind: str,
    ) -> IndexedSourceMaterializationProvider | None:
        return self._providers.get((provider_id, integration_kind, source_kind))

    def registered_keys(self) -> tuple[IndexedSourceMaterializationRegistryKey, ...]:
        return tuple(sorted(self._providers, key=lambda item: (item[0], item[1].value, item[2])))

    def unregister(
        self,
        *,
        provider_id: str,
        integration_kind: IntegrationCategory,
        source_kind: str,
    ) -> bool:
        return self._providers.pop((provider_id, integration_kind, source_kind), None) is not None


class IndexedSourceDiscoveryPort(Protocol):
    async def list_remote_resources(
        self,
        *,
        connection_ref: str,
        source_kind: str,
        page_token: str | None = None,
        limit: int = 100,
    ) -> RemoteResourceDiscoveryPageV1:
        ...


class IndexedSourceEligibilityPort(Protocol):
    async def resolve(
        self,
        request: IndexedSourceEligibilityRequestV1,
    ) -> IndexedSourceEligibilityProofV1:
        ...


class _SnapshotStale(Exception):
    pass


class _DiscoveryUnavailable(Exception):
    pass


class IndexedSourceEligibilityResolverV1:
    """Authoritative, read-only qualification boundary for applications."""

    def __init__(
        self,
        *,
        connection_port: TenantConnectionPort,
        discovery_service_factory: Callable[[str], IndexedSourceDiscoveryPort],
        materialization_registry: IndexedSourceMaterializationRegistry,
        clock: Callable[[], datetime] | None = None,
        proof_ttl_seconds: int = 300,
    ) -> None:
        if proof_ttl_seconds < 1 or proof_ttl_seconds > _MAX_PROOF_TTL_SECONDS:
            raise ValueError("proof_ttl_seconds must be in range 1..300")
        self._connection_port = connection_port
        self._discovery_service_factory = discovery_service_factory
        self._materialization_registry = materialization_registry
        self._clock = clock or (lambda: datetime.now(UTC))
        self._proof_ttl_seconds = proof_ttl_seconds

    async def resolve(
        self,
        request: IndexedSourceEligibilityRequestV1,
    ) -> IndexedSourceEligibilityProofV1:
        evaluated_at = _utc(self._clock(), field_name="evaluated_at")
        connection = self._connection_port.get_connection(
            tenant_id=request.tenant_id,
            connection_ref=request.connection_ref,
        )
        if connection is None or connection.tenant_id != request.tenant_id:
            return self._negative(
                request=request,
                status=IndexedSourceEligibilityStatusV1.CONNECTION_INACTIVE,
                reason="indexed_source_eligibility_connection_not_found",
                evaluated_at=evaluated_at,
            )
        if connection.administrative_status is not TenantConnectionAdministrativeStatus.ACTIVE:
            return self._negative(
                request=request,
                status=IndexedSourceEligibilityStatusV1.CONNECTION_INACTIVE,
                reason="indexed_source_eligibility_connection_not_active",
                evaluated_at=evaluated_at,
            )

        try:
            resource = await self._find_resource(request=request)
        except _SnapshotStale:
            return self._negative(
                request=request,
                status=IndexedSourceEligibilityStatusV1.SNAPSHOT_STALE,
                reason="indexed_source_eligibility_snapshot_stale",
                evaluated_at=evaluated_at,
            )
        except _DiscoveryUnavailable:
            return self._negative(
                request=request,
                status=IndexedSourceEligibilityStatusV1.NOT_SUPPORTED,
                reason="indexed_source_eligibility_unavailable",
                evaluated_at=evaluated_at,
            )
        if resource is None:
            return self._negative(
                request=request,
                status=IndexedSourceEligibilityStatusV1.RESOURCE_UNAVAILABLE,
                reason="indexed_source_eligibility_resource_not_found",
                evaluated_at=evaluated_at,
            )
        if resource.availability is not RemoteResourceAvailabilityV1.AVAILABLE:
            return self._negative(
                request=request,
                status=IndexedSourceEligibilityStatusV1.RESOURCE_UNAVAILABLE,
                reason="indexed_source_eligibility_resource_unavailable",
                evaluated_at=evaluated_at,
            )
        if (
            resource.connection_ref != connection.connection_ref
            or resource.remote_resource_id != request.remote_resource_id
            or resource.source_kind != request.source_kind
            or resource.provider_id != connection.provider_id
            or resource.integration_kind is not connection.integration_kind
        ):
            return self._negative(
                request=request,
                status=IndexedSourceEligibilityStatusV1.NOT_SUPPORTED,
                reason="indexed_source_eligibility_invalid_provider_response",
                evaluated_at=evaluated_at,
            )

        provider = self._materialization_registry.resolve(
            provider_id=connection.provider_id,
            integration_kind=connection.integration_kind,
            source_kind=request.source_kind,
        )
        if provider is None:
            return self._negative(
                request=request,
                status=IndexedSourceEligibilityStatusV1.NOT_SUPPORTED,
                reason="indexed_source_eligibility_materialization_not_supported",
                evaluated_at=evaluated_at,
            )

        try:
            handler_ref = provider.sync_handler_ref()
            handler_available = provider.sync_handler_available()
        except Exception:
            handler_ref = ""
            handler_available = False
        if (
            not isinstance(handler_ref, str)
            or not handler_ref.strip()
            or not isinstance(handler_available, bool)
            or not handler_available
        ):
            return self._negative(
                request=request,
                status=IndexedSourceEligibilityStatusV1.HANDLER_UNAVAILABLE,
                reason="indexed_source_eligibility_handler_unavailable",
                evaluated_at=evaluated_at,
            )

        try:
            descriptor = provider.qualify(connection=connection, resource=resource)
        except Exception:
            descriptor = None
        if descriptor is None or not self._descriptor_is_authoritative(
            descriptor=descriptor,
            connection=connection,
            request=request,
            resource=resource,
            contract_version=provider.materialization_contract_version,
        ):
            return self._negative(
                request=request,
                status=IndexedSourceEligibilityStatusV1.NOT_SUPPORTED,
                reason="indexed_source_eligibility_invalid_provider_response",
                evaluated_at=evaluated_at,
            )

        plan_without_revision = {
            "tenant_id": request.tenant_id,
            "connection_ref": request.connection_ref,
            "provider_id": connection.provider_id,
            "integration_kind": connection.integration_kind.value,
            "remote_resource_id": request.remote_resource_id,
            "source_kind": request.source_kind,
            "canonical_source_ref": descriptor.canonical_source_ref,
            "source_descriptor": descriptor.model_dump(mode="json"),
            "sync_handler_ref": handler_ref.strip(),
            "materialization_contract_version": provider.materialization_contract_version,
        }
        proof_revision = _sha256(
            {
                **plan_without_revision,
                "discovery_snapshot_version": request.discovery_snapshot_version,
            }
        )
        plan = IndexedSourceBindingPlanV1(
            **plan_without_revision,
            proof_revision=proof_revision,
        )
        return self._positive(
            plan=plan,
            evaluated_at=evaluated_at,
            proof_revision=proof_revision,
        )

    async def _find_resource(
        self,
        *,
        request: IndexedSourceEligibilityRequestV1,
    ) -> RemoteResourceDescriptorV1 | None:
        service = self._discovery_service_factory(request.tenant_id)
        page_token: str | None = None
        snapshot_version: str | None = None
        for page_number in range(_MAX_DISCOVERY_PAGES):
            try:
                page = await service.list_remote_resources(
                    connection_ref=request.connection_ref,
                    source_kind=request.source_kind,
                    page_token=page_token,
                    limit=100,
                )
            except VendorKnowledgeError as exc:
                if exc.code is VendorKnowledgeErrorCode.ADAPTER_NOT_FOUND:
                    return None
                raise _DiscoveryUnavailable from None
            except Exception:
                raise _DiscoveryUnavailable from None
            if not isinstance(page, RemoteResourceDiscoveryPageV1):
                raise _DiscoveryUnavailable
            try:
                page_snapshot = str(page.snapshot_version).strip()
            except Exception:
                raise _DiscoveryUnavailable from None
            if snapshot_version is None:
                snapshot_version = page_snapshot
                if page_snapshot != request.discovery_snapshot_version:
                    raise _SnapshotStale
            elif page_snapshot != snapshot_version:
                raise _SnapshotStale
            for resource in page.resources:
                if resource.remote_resource_id == request.remote_resource_id:
                    return resource
            page_token = page.next_page_token
            if page_token is None:
                return None
        raise _DiscoveryUnavailable

    @staticmethod
    def _descriptor_is_authoritative(
        *,
        descriptor: object,
        connection: SafeTenantConnectionV1,
        request: IndexedSourceEligibilityRequestV1,
        resource: RemoteResourceDescriptorV1,
        contract_version: object,
    ) -> bool:
        if not isinstance(descriptor, IndexedSourceDescriptorV1):
            return False
        if not isinstance(contract_version, str) or not contract_version.strip():
            return False
        expected_ref = canonical_indexed_source_ref(
            tenant_id=request.tenant_id,
            connection_ref=connection.connection_ref,
            provider_id=connection.provider_id,
            integration_kind=connection.integration_kind,
            remote_resource_id=request.remote_resource_id,
            source_kind=request.source_kind,
        )
        return (
            descriptor.provider_id == connection.provider_id
            and descriptor.integration_kind is connection.integration_kind
            and descriptor.connection_ref == connection.connection_ref
            and descriptor.remote_resource_id == request.remote_resource_id
            and descriptor.source_kind == request.source_kind
            and descriptor.canonical_source_ref == expected_ref
            and descriptor.resource_type == resource.resource_type
            and descriptor.discovery_snapshot_version == request.discovery_snapshot_version
            and descriptor.materialization_contract_version == contract_version
        )

    def _positive(
        self,
        *,
        plan: IndexedSourceBindingPlanV1,
        evaluated_at: datetime,
        proof_revision: str,
    ) -> IndexedSourceEligibilityProofV1:
        return IndexedSourceEligibilityProofV1(
            status=IndexedSourceEligibilityStatusV1.ELIGIBLE,
            eligible=True,
            binding_plan=plan,
            evaluated_at=evaluated_at,
            expires_at=evaluated_at + timedelta(seconds=self._proof_ttl_seconds),
            proof_revision=proof_revision,
        )

    def _negative(
        self,
        *,
        request: IndexedSourceEligibilityRequestV1,
        status: IndexedSourceEligibilityStatusV1,
        reason: str,
        evaluated_at: datetime,
    ) -> IndexedSourceEligibilityProofV1:
        proof_revision = _sha256(
            {
                "tenant_id": request.tenant_id,
                "connection_ref": request.connection_ref,
                "remote_resource_id": request.remote_resource_id,
                "source_kind": request.source_kind,
                "discovery_snapshot_version": request.discovery_snapshot_version,
                "status": status.value,
                "safe_reason_code": reason,
            }
        )
        return IndexedSourceEligibilityProofV1(
            status=status,
            eligible=False,
            safe_reason_code=reason,
            evaluated_at=evaluated_at,
            expires_at=evaluated_at + timedelta(seconds=self._proof_ttl_seconds),
            proof_revision=proof_revision,
        )
