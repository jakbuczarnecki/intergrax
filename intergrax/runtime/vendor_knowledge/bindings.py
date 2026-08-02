# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tenant-scoped knowledge source bindings for the Vendor Knowledge Facade."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.runtime.vendor_knowledge.contracts import VendorIntegrationResolver
from intergrax.runtime.vendor_knowledge.errors import (
    VendorKnowledgeError,
    VendorKnowledgeErrorCode,
)
from intergrax.runtime.vendor_knowledge.models import (
    KnowledgeSourceRef,
    KnowledgeSourceScope,
)
from intergrax.runtime.vendor_knowledge.registry import KnowledgeAdapterRegistry


class KnowledgeSourceBindingStatus(StrEnum):
    ACTIVE = "active"
    DISABLED = "disabled"
    REVOKED = "revoked"
    EXPIRED = "expired"


class KnowledgeSourceBindingAlreadyExists(Exception):
    """Binding already exists for the requested identity."""


class KnowledgeSourceBindingNotFound(Exception):
    """Binding was not found for the requested tenant-scoped identity."""


class KnowledgeSourceBindingVersionConflict(Exception):
    """Optimistic configuration version conflict."""


class KnowledgeSourceBindingCorruptRecord(Exception):
    """Durable binding record is corrupt or inconsistent."""


def _require_non_empty(value: str, *, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be a non-empty string")
    return cleaned


class KnowledgeSourceBinding(BaseModel):
    """Durable tenant-scoped binding from a logical source to an opaque connection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    binding_id: str
    tenant_id: str
    provider_id: str
    integration_kind: IntegrationCategory
    source_kind: str
    connection_ref: str
    credential_ref: str | None = None
    safe_display_name: str
    scope: KnowledgeSourceScope
    status: KnowledgeSourceBindingStatus
    configuration_version: int = Field(ge=1)
    broad_scope: bool = False
    scope_approval_ref: str | None = None

    @field_validator(
        "binding_id",
        "tenant_id",
        "provider_id",
        "source_kind",
        "connection_ref",
        "safe_display_name",
    )
    @classmethod
    def _non_empty_required(cls, value: str, info: ValidationInfo) -> str:
        field_name = info.field_name or "field"
        return _require_non_empty(value, field_name=field_name)

    @field_validator("credential_ref", "scope_approval_ref")
    @classmethod
    def _optional_opaque_ref(cls, value: str | None, info: ValidationInfo) -> str | None:
        if value is None:
            return None
        field_name = info.field_name or "field"
        return _require_non_empty(value, field_name=field_name)

    @model_validator(mode="after")
    def _broad_scope_requires_approval(self) -> KnowledgeSourceBinding:
        if self.broad_scope and self.scope_approval_ref is None:
            raise ValueError("broad_scope=True requires a non-empty scope_approval_ref")
        return self


def to_source_ref(binding: KnowledgeSourceBinding) -> KnowledgeSourceRef:
    """Project a binding into a knowledge source reference (preserves connection_ref)."""
    return KnowledgeSourceRef(
        tenant_id=binding.tenant_id,
        provider_id=binding.provider_id,
        integration_kind=binding.integration_kind,
        source_kind=binding.source_kind,
        connection_ref=binding.connection_ref,
        scope=binding.scope,
    )


@runtime_checkable
class KnowledgeSourceBindingRepository(Protocol):
    """Durable repository port for tenant-scoped knowledge source bindings."""

    def create(self, binding: KnowledgeSourceBinding) -> None:
        ...

    def get(
        self,
        *,
        tenant_id: str,
        binding_id: str,
    ) -> KnowledgeSourceBinding | None:
        ...

    def update(
        self,
        binding: KnowledgeSourceBinding,
        *,
        expected_configuration_version: int,
    ) -> None:
        ...

    def list(
        self,
        *,
        tenant_id: str,
        limit: int = 100,
        status: KnowledgeSourceBindingStatus | None = None,
    ) -> tuple[KnowledgeSourceBinding, ...]:
        ...


class KnowledgeSourceBindingService:
    """Create, update and resolve tenant-scoped knowledge source bindings."""

    def __init__(
        self,
        *,
        tenant_id: str,
        repository: KnowledgeSourceBindingRepository,
        integration_resolver: VendorIntegrationResolver,
        adapter_registry: KnowledgeAdapterRegistry,
    ) -> None:
        cleaned_tenant = str(tenant_id).strip()
        if not cleaned_tenant:
            raise ValueError("tenant_id must be a non-empty string")
        self._tenant_id = cleaned_tenant
        self._repository = repository
        self._integration_resolver = integration_resolver
        self._adapter_registry = adapter_registry

    def create(self, binding: KnowledgeSourceBinding) -> KnowledgeSourceBinding:
        self._assert_service_tenant(binding.tenant_id, provider_id=binding.provider_id)
        self._assert_broad_scope(binding)
        self._assert_resolvable(binding)
        try:
            self._repository.create(binding)
        except KnowledgeSourceBindingAlreadyExists:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.CONFIGURATION_ERROR,
                safe_message="Knowledge source binding already exists",
                provider_id=binding.provider_id,
                source_kind=binding.source_kind,
                retryable=False,
            ) from None
        except KnowledgeSourceBindingCorruptRecord:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Knowledge source binding record is invalid",
                provider_id=binding.provider_id,
                source_kind=binding.source_kind,
                retryable=False,
            ) from None
        except VendorKnowledgeError:
            raise
        except Exception:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Knowledge source binding persistence failed",
                provider_id=binding.provider_id,
                source_kind=binding.source_kind,
                retryable=False,
            ) from None
        return binding

    def update(
        self,
        binding: KnowledgeSourceBinding,
        *,
        expected_configuration_version: int,
    ) -> KnowledgeSourceBinding:
        self._assert_service_tenant(binding.tenant_id, provider_id=binding.provider_id)
        self._assert_broad_scope(binding)
        self._assert_resolvable(binding)
        try:
            self._repository.update(
                binding,
                expected_configuration_version=expected_configuration_version,
            )
        except KnowledgeSourceBindingNotFound:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INTEGRATION_NOT_FOUND,
                safe_message="Knowledge source binding was not found",
                provider_id=binding.provider_id,
                source_kind=binding.source_kind,
                retryable=False,
            ) from None
        except KnowledgeSourceBindingVersionConflict:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.CONFIGURATION_ERROR,
                safe_message="Knowledge source binding configuration version conflict",
                provider_id=binding.provider_id,
                source_kind=binding.source_kind,
                retryable=False,
            ) from None
        except KnowledgeSourceBindingCorruptRecord:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Knowledge source binding record is invalid",
                provider_id=binding.provider_id,
                source_kind=binding.source_kind,
                retryable=False,
            ) from None
        except VendorKnowledgeError:
            raise
        except Exception:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Knowledge source binding persistence failed",
                provider_id=binding.provider_id,
                source_kind=binding.source_kind,
                retryable=False,
            ) from None
        return binding

    def get(self, binding_id: str) -> KnowledgeSourceBinding:
        cleaned_id = _require_non_empty(binding_id, field_name="binding_id")
        try:
            binding = self._repository.get(
                tenant_id=self._tenant_id,
                binding_id=cleaned_id,
            )
        except KnowledgeSourceBindingCorruptRecord:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Knowledge source binding record is invalid",
                retryable=False,
            ) from None
        except VendorKnowledgeError:
            raise
        except Exception:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Knowledge source binding persistence failed",
                retryable=False,
            ) from None
        if binding is None:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INTEGRATION_NOT_FOUND,
                safe_message="Knowledge source binding was not found",
                retryable=False,
            )
        self._assert_service_tenant(binding.tenant_id, provider_id=binding.provider_id)
        return binding

    def list(
        self,
        *,
        limit: int = 100,
        status: KnowledgeSourceBindingStatus | None = None,
    ) -> tuple[KnowledgeSourceBinding, ...]:
        try:
            bindings = self._repository.list(
                tenant_id=self._tenant_id,
                limit=limit,
                status=status,
            )
        except KnowledgeSourceBindingCorruptRecord:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Knowledge source binding record is invalid",
                retryable=False,
            ) from None
        except VendorKnowledgeError:
            raise
        except Exception:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Knowledge source binding persistence failed",
                retryable=False,
            ) from None

        for binding in bindings:
            self._assert_service_tenant(binding.tenant_id, provider_id=binding.provider_id)
            if status is not None and binding.status is not status:
                raise VendorKnowledgeError(
                    code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                    safe_message="Knowledge source binding list status filter mismatch",
                    provider_id=binding.provider_id,
                    source_kind=binding.source_kind,
                    retryable=False,
                )
        return bindings

    def resolve_source(self, binding_id: str) -> KnowledgeSourceRef:
        binding = self.get(binding_id)
        if binding.status is KnowledgeSourceBindingStatus.DISABLED:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.CONFIGURATION_ERROR,
                safe_message="Knowledge source binding is disabled",
                provider_id=binding.provider_id,
                source_kind=binding.source_kind,
                retryable=False,
            )
        if binding.status is KnowledgeSourceBindingStatus.REVOKED:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.AUTHORIZATION_DENIED,
                safe_message="Knowledge source binding is revoked",
                provider_id=binding.provider_id,
                source_kind=binding.source_kind,
                retryable=False,
            )
        if binding.status is KnowledgeSourceBindingStatus.EXPIRED:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.AUTHENTICATION_FAILED,
                safe_message="Knowledge source binding is expired",
                provider_id=binding.provider_id,
                source_kind=binding.source_kind,
                retryable=False,
            )
        if binding.status is not KnowledgeSourceBindingStatus.ACTIVE:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.CONFIGURATION_ERROR,
                safe_message="Knowledge source binding is not active",
                provider_id=binding.provider_id,
                source_kind=binding.source_kind,
                retryable=False,
            )

        self._assert_service_tenant(binding.tenant_id, provider_id=binding.provider_id)
        self._assert_broad_scope(binding)
        self._assert_resolvable(binding)
        return to_source_ref(binding)

    def create_or_get_equivalent(self, binding: KnowledgeSourceBinding) -> KnowledgeSourceBinding:
        """Create a binding or return an exactly equivalent existing binding."""
        self._assert_service_tenant(binding.tenant_id, provider_id=binding.provider_id)
        self._assert_broad_scope(binding)
        self._assert_resolvable(binding)

        existing = self._repository.get(
            tenant_id=self._tenant_id,
            binding_id=binding.binding_id,
        )
        if existing is not None:
            if not self._bindings_equivalent(existing, binding):
                raise VendorKnowledgeError(
                    code=VendorKnowledgeErrorCode.CONFIGURATION_ERROR,
                    safe_message="Knowledge source binding identity conflict",
                    provider_id=binding.provider_id,
                    source_kind=binding.source_kind,
                    retryable=False,
                )
            return existing

        try:
            self._repository.create(binding)
        except KnowledgeSourceBindingAlreadyExists:
            reloaded = self._repository.get(
                tenant_id=self._tenant_id,
                binding_id=binding.binding_id,
            )
            if reloaded is None:
                raise VendorKnowledgeError(
                    code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                    safe_message="Knowledge source binding persistence failed",
                    provider_id=binding.provider_id,
                    source_kind=binding.source_kind,
                    retryable=False,
                ) from None
            if not self._bindings_equivalent(reloaded, binding):
                raise VendorKnowledgeError(
                    code=VendorKnowledgeErrorCode.CONFIGURATION_ERROR,
                    safe_message="Knowledge source binding identity conflict",
                    provider_id=binding.provider_id,
                    source_kind=binding.source_kind,
                    retryable=False,
                )
            return reloaded
        except KnowledgeSourceBindingCorruptRecord:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Knowledge source binding record is invalid",
                provider_id=binding.provider_id,
                source_kind=binding.source_kind,
                retryable=False,
            ) from None
        except VendorKnowledgeError:
            raise
        except Exception:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Knowledge source binding persistence failed",
                provider_id=binding.provider_id,
                source_kind=binding.source_kind,
                retryable=False,
            ) from None
        return binding

    def _bindings_equivalent(
        self,
        left: KnowledgeSourceBinding,
        right: KnowledgeSourceBinding,
    ) -> bool:
        return (
            left.binding_id == right.binding_id
            and left.tenant_id == right.tenant_id
            and left.provider_id == right.provider_id
            and left.integration_kind == right.integration_kind
            and left.source_kind == right.source_kind
            and left.connection_ref == right.connection_ref
            and left.credential_ref == right.credential_ref
            and left.safe_display_name == right.safe_display_name
            and left.scope == right.scope
            and left.status == right.status
            and left.configuration_version == right.configuration_version
            and left.broad_scope == right.broad_scope
            and left.scope_approval_ref == right.scope_approval_ref
        )

    def _assert_service_tenant(self, tenant_id: str, *, provider_id: str | None) -> None:
        if tenant_id != self._tenant_id:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.TENANT_MISMATCH,
                safe_message="Knowledge source binding tenant does not match the service tenant",
                provider_id=provider_id,
                retryable=False,
            )

    def _assert_broad_scope(self, binding: KnowledgeSourceBinding) -> None:
        if binding.broad_scope and binding.scope_approval_ref is None:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.AUTHORIZATION_DENIED,
                safe_message="Broad-scope binding requires an approval reference",
                provider_id=binding.provider_id,
                source_kind=binding.source_kind,
                retryable=False,
            )

    def _assert_resolvable(self, binding: KnowledgeSourceBinding) -> None:
        source = to_source_ref(binding)
        self._adapter_registry.resolve(source=source)
        self._integration_resolver.resolve(source=source)
