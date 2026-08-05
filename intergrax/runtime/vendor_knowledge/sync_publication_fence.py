# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Provider-neutral lifecycle publication fence for Vendor Knowledge sync."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    ValidationInfo,
    field_validator,
    model_validator,
)

from intergrax.integrations.contracts.document_store import (
    ConditionalDocumentStore,
    DocumentRecord,
    DocumentStore,
)

_FENCE_SCHEMA = "vendor_knowledge.sync_publication_fence.v1"
_FENCE_PARTITION_PREFIX = _FENCE_SCHEMA
_MAX_TOKEN_LENGTH = 256


def _require_non_empty(value: str, *, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be a non-empty string")
    return cleaned


class KnowledgeSyncPublicationFenceV1(BaseModel):
    """Immutable application-owned lifecycle authority snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str
    binding_id: str
    lifecycle_revision: StrictInt = Field(ge=1)
    lifecycle_token: str = Field(repr=False, min_length=1, max_length=_MAX_TOKEN_LENGTH)
    enabled: StrictBool
    detached: StrictBool

    @field_validator("tenant_id", "binding_id", "lifecycle_token")
    @classmethod
    def _identifiers(cls, value: str, info: ValidationInfo) -> str:
        field_name = info.field_name or "field"
        return _require_non_empty(value, field_name=field_name)

    @model_validator(mode="after")
    def _lifecycle_state(self) -> KnowledgeSyncPublicationFenceV1:
        if self.detached and self.enabled:
            raise ValueError("detached publication fence cannot be enabled")
        return self


@runtime_checkable
class KnowledgeSyncPublicationFencePort(Protocol):
    """Read-only application authority consumed by Vendor Knowledge."""

    def read_fence(
        self,
        *,
        tenant_id: str,
        binding_id: str,
    ) -> KnowledgeSyncPublicationFenceV1 | None:
        ...


class KnowledgeSyncPublicationFenceConflict(Exception):
    """Optimistic lifecycle fence write conflict."""


class DocumentStoreKnowledgeSyncPublicationFenceRepository:
    """Durable CAS-backed fence authority for application lifecycle writers."""

    def __init__(self, document_store: DocumentStore) -> None:
        if not isinstance(document_store, ConditionalDocumentStore):
            raise TypeError(
                "publication fence repository requires ConditionalDocumentStore"
            )
        self._store = document_store

    def read_fence(
        self,
        *,
        tenant_id: str,
        binding_id: str,
    ) -> KnowledgeSyncPublicationFenceV1 | None:
        tenant = _require_non_empty(tenant_id, field_name="tenant_id")
        binding = _require_non_empty(binding_id, field_name="binding_id")
        document = self._store.get(
            self._partition_key(tenant),
            self._row_key(binding),
        )
        if document is None:
            return None
        return self._parse(document, tenant_id=tenant, binding_id=binding)

    def write_fence(
        self,
        fence: KnowledgeSyncPublicationFenceV1,
        *,
        expected_revision: int | None,
    ) -> None:
        self._validate_identity(fence)
        partition = self._partition_key(fence.tenant_id)
        row = self._row_key(fence.binding_id)
        current = self._store.get(partition, row)
        if expected_revision is None:
            if current is not None:
                raise KnowledgeSyncPublicationFenceConflict(
                    "publication fence already exists"
                )
            if not self._store.put_if_absent(self._to_document(fence)):
                raise KnowledgeSyncPublicationFenceConflict(
                    "publication fence create conflict"
                )
            return
        if current is None:
            raise KnowledgeSyncPublicationFenceConflict("publication fence is missing")
        parsed = self._parse(
            current,
            tenant_id=fence.tenant_id,
            binding_id=fence.binding_id,
        )
        if parsed.lifecycle_revision != expected_revision:
            raise KnowledgeSyncPublicationFenceConflict("publication fence revision conflict")
        if fence.lifecycle_revision <= expected_revision:
            raise KnowledgeSyncPublicationFenceConflict(
                "publication fence revision must advance"
            )
        if not self._store.replace_if_match(
            expected=current,
            replacement=self._to_document(fence),
        ):
            raise KnowledgeSyncPublicationFenceConflict("publication fence write conflict")

    def enable(
        self,
        *,
        tenant_id: str,
        binding_id: str,
        lifecycle_revision: int,
        lifecycle_token: str,
        expected_revision: int | None,
    ) -> KnowledgeSyncPublicationFenceV1:
        fence = KnowledgeSyncPublicationFenceV1(
            tenant_id=tenant_id,
            binding_id=binding_id,
            lifecycle_revision=lifecycle_revision,
            lifecycle_token=lifecycle_token,
            enabled=True,
            detached=False,
        )
        self.write_fence(fence, expected_revision=expected_revision)
        return fence

    def disable(
        self,
        *,
        tenant_id: str,
        binding_id: str,
        lifecycle_revision: int,
        lifecycle_token: str,
        expected_revision: int,
    ) -> KnowledgeSyncPublicationFenceV1:
        fence = KnowledgeSyncPublicationFenceV1(
            tenant_id=tenant_id,
            binding_id=binding_id,
            lifecycle_revision=lifecycle_revision,
            lifecycle_token=lifecycle_token,
            enabled=False,
            detached=False,
        )
        self.write_fence(fence, expected_revision=expected_revision)
        return fence

    def detach(
        self,
        *,
        tenant_id: str,
        binding_id: str,
        lifecycle_revision: int,
        lifecycle_token: str,
        expected_revision: int,
    ) -> KnowledgeSyncPublicationFenceV1:
        fence = KnowledgeSyncPublicationFenceV1(
            tenant_id=tenant_id,
            binding_id=binding_id,
            lifecycle_revision=lifecycle_revision,
            lifecycle_token=lifecycle_token,
            enabled=False,
            detached=True,
        )
        self.write_fence(fence, expected_revision=expected_revision)
        return fence

    @staticmethod
    def _partition_key(tenant_id: str) -> str:
        return f"{_FENCE_PARTITION_PREFIX}:{_require_non_empty(tenant_id, field_name='tenant_id')}"

    @staticmethod
    def _row_key(binding_id: str) -> str:
        return f"binding:{_require_non_empty(binding_id, field_name='binding_id')}"

    def _to_document(self, fence: KnowledgeSyncPublicationFenceV1) -> DocumentRecord:
        return DocumentRecord(
            partition_key=self._partition_key(fence.tenant_id),
            row_key=self._row_key(fence.binding_id),
            data={
                "schema_version": _FENCE_SCHEMA,
                "tenant_id": fence.tenant_id,
                "binding_id": fence.binding_id,
                "fence": fence.model_dump(mode="json"),
            },
        )

    def _parse(
        self,
        document: DocumentRecord,
        *,
        tenant_id: str,
        binding_id: str,
    ) -> KnowledgeSyncPublicationFenceV1:
        data: dict[str, Any] = dict(document.data)
        if (
            data.get("schema_version") != _FENCE_SCHEMA
            or document.partition_key != self._partition_key(tenant_id)
            or document.row_key != self._row_key(binding_id)
            or data.get("tenant_id") != tenant_id
            or data.get("binding_id") != binding_id
            or not isinstance(data.get("fence"), dict)
        ):
            raise ValueError("publication fence record is corrupt")
        try:
            fence = KnowledgeSyncPublicationFenceV1.model_validate(data["fence"])
        except (TypeError, ValueError):
            raise ValueError("publication fence record is corrupt") from None
        self._validate_identity(fence)
        return fence

    @staticmethod
    def _validate_identity(fence: KnowledgeSyncPublicationFenceV1) -> None:
        if not fence.tenant_id.strip() or not fence.binding_id.strip():
            raise ValueError("publication fence identity is invalid")
