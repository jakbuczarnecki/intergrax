# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Provider-neutral lifecycle publication fence for Vendor Knowledge sync."""

from __future__ import annotations

import hashlib
import json
import secrets
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
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
_PERMIT_SCHEMA = "vendor_knowledge.sync_publication_permit.v1"
_PUBLICATION_COMMIT_NODE_SCHEMA = "vendor_knowledge.sync_publication_commit_node.v1"
_PUBLICATION_COMMIT_PARTITION_PREFIX = f"{_FENCE_PARTITION_PREFIX}:commit"
_MAX_PERMIT_TTL_SECONDS = 3600
_SHA256_LENGTH = 64
_MAX_COMMITTED_CHAIN_NODES = 1_000_000


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


class KnowledgeSyncPublicationPermitV1(BaseModel):
    """Durable, bounded authority to publish one binding page."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str
    binding_id: str
    lifecycle_revision: StrictInt = Field(ge=1)
    lifecycle_token: str = Field(repr=False, min_length=1, max_length=_MAX_TOKEN_LENGTH)
    permit_id: str = Field(repr=False, min_length=1, max_length=_MAX_TOKEN_LENGTH)
    owner_id: str
    acquired_at: datetime
    expires_at: datetime

    @field_validator(
        "tenant_id",
        "binding_id",
        "lifecycle_token",
        "permit_id",
        "owner_id",
    )
    @classmethod
    def _identifiers(cls, value: str, info: ValidationInfo) -> str:
        field_name = info.field_name or "field"
        return _require_non_empty(value, field_name=field_name)

    @field_validator("acquired_at", "expires_at")
    @classmethod
    def _utc_datetimes(cls, value: datetime, info: ValidationInfo) -> datetime:
        field_name = info.field_name or "datetime"
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{field_name} must be a timezone-aware UTC datetime")
        if value.utcoffset() != timedelta(0):
            raise ValueError(f"{field_name} must be a timezone-aware UTC datetime")
        return value

    @model_validator(mode="after")
    def _valid_interval(self) -> KnowledgeSyncPublicationPermitV1:
        if self.expires_at <= self.acquired_at:
            raise ValueError("publication permit must have a positive TTL")
        return self


class KnowledgeSyncCommittedPublicationV1(BaseModel):
    """Bounded descriptor authorized by the publication fence CAS."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str
    binding_id: str
    workspace_id: str
    source_id: str
    indexed_source_binding_id: str
    delivery_id: str
    materialization_sequence: StrictInt = Field(gt=0)
    manifest_id: str = Field(min_length=1, max_length=512)
    manifest_fingerprint: str = Field(
        min_length=_SHA256_LENGTH, max_length=_SHA256_LENGTH
    )
    committed_at: datetime

    @field_validator(
        "tenant_id",
        "binding_id",
        "workspace_id",
        "source_id",
        "indexed_source_binding_id",
        "delivery_id",
    )
    @classmethod
    def _identifiers(cls, value: str, info: ValidationInfo) -> str:
        field_name = info.field_name or "field"
        return _require_non_empty(value, field_name=field_name)

    @field_validator("manifest_id")
    @classmethod
    def _manifest_id(cls, value: str, info: ValidationInfo) -> str:
        field_name = info.field_name or "fingerprint"
        return _require_non_empty(value, field_name=field_name)

    @field_validator("manifest_fingerprint")
    @classmethod
    def _fingerprint(cls, value: str, info: ValidationInfo) -> str:
        field_name = info.field_name or "fingerprint"
        cleaned = _require_non_empty(value, field_name=field_name)
        if len(cleaned) != _SHA256_LENGTH or any(
            character not in "0123456789abcdef" for character in cleaned
        ):
            raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
        return cleaned

    @field_validator("committed_at")
    @classmethod
    def _utc_datetime(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError("committed_at must be a timezone-aware UTC datetime")
        return value


class KnowledgeSyncCommittedPublicationV2(KnowledgeSyncCommittedPublicationV1):
    """Bounded publication descriptor linked into immutable history."""

    publication_commit_id: str = Field(
        min_length=_SHA256_LENGTH,
        max_length=_SHA256_LENGTH,
    )
    previous_publication_commit_id: str | None = Field(
        default=None,
        min_length=_SHA256_LENGTH,
        max_length=_SHA256_LENGTH,
    )

    @field_validator("publication_commit_id", "previous_publication_commit_id")
    @classmethod
    def _commit_ids(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if any(character not in "0123456789abcdef" for character in value):
            raise ValueError("publication commit id must be a lowercase SHA-256 digest")
        return value


class KnowledgeSyncPublicationCommitNodeV1(BaseModel):
    """Immutable, non-visible publication history node."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    commit_id: str = Field(min_length=_SHA256_LENGTH, max_length=_SHA256_LENGTH)
    descriptor: KnowledgeSyncCommittedPublicationV2
    previous_commit_id: str | None = Field(
        default=None,
        min_length=_SHA256_LENGTH,
        max_length=_SHA256_LENGTH,
    )

    @model_validator(mode="after")
    def _identity(self) -> KnowledgeSyncPublicationCommitNodeV1:
        if self.commit_id != self.descriptor.publication_commit_id:
            raise ValueError("publication commit node id mismatch")
        if self.previous_commit_id != self.descriptor.previous_publication_commit_id:
            raise ValueError("publication commit node predecessor mismatch")
        return self


def _descriptor_identity_payload(
    descriptor: KnowledgeSyncCommittedPublicationV1,
    *,
    previous_commit_id: str | None,
) -> dict[str, object]:
    return {
        "tenant_id": descriptor.tenant_id,
        "binding_id": descriptor.binding_id,
        "workspace_id": descriptor.workspace_id,
        "source_id": descriptor.source_id,
        "indexed_source_binding_id": descriptor.indexed_source_binding_id,
        "delivery_id": descriptor.delivery_id,
        "materialization_sequence": descriptor.materialization_sequence,
        "manifest_id": descriptor.manifest_id,
        "manifest_fingerprint": descriptor.manifest_fingerprint,
        "previous_commit_id": previous_commit_id,
    }


def publication_commit_id(
    descriptor: KnowledgeSyncCommittedPublicationV1,
    *,
    previous_commit_id: str | None,
) -> str:
    """Return the deterministic identity of one immutable publication node."""
    payload = json.dumps(
        _descriptor_identity_payload(
            descriptor,
            previous_commit_id=previous_commit_id,
        ),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _descriptor_identity(
    descriptor: KnowledgeSyncCommittedPublicationV1,
) -> tuple[object, ...]:
    return tuple(
        _descriptor_identity_payload(
            descriptor,
            previous_commit_id=None,
        ).values()
    )


class KnowledgeSyncPublicationCommitStatus(StrEnum):
    COMMITTED = "committed"
    REPLAYED = "replayed"
    STALE = "stale"


@dataclass(frozen=True, slots=True)
class KnowledgeSyncPublicationCommitResult:
    status: KnowledgeSyncPublicationCommitStatus
    descriptor: KnowledgeSyncCommittedPublicationV2


@runtime_checkable
class KnowledgeSyncPublicationFencePort(Protocol):
    """Application lifecycle and publication authority consumed by Vendor Knowledge."""

    def read_fence(
        self,
        *,
        tenant_id: str,
        binding_id: str,
    ) -> KnowledgeSyncPublicationFenceV1 | None:
        ...

    def acquire_publication_permit(
        self,
        *,
        tenant_id: str,
        binding_id: str,
        expected_revision: int,
        expected_token: str,
        owner_id: str,
        ttl_seconds: int,
    ) -> KnowledgeSyncPublicationPermitV1 | None:
        ...

    def release_publication_permit(
        self,
        permit: KnowledgeSyncPublicationPermitV1,
    ) -> bool:
        ...

    def is_current_publication_permit(
        self,
        *,
        permit: KnowledgeSyncPublicationPermitV1,
    ) -> bool:
        ...

    def commit_publication_under_permit(
        self,
        *,
        expected_fence: KnowledgeSyncPublicationFenceV1,
        publication_permit: KnowledgeSyncPublicationPermitV1,
        publication_descriptor: KnowledgeSyncCommittedPublicationV1,
    ) -> KnowledgeSyncPublicationCommitResult:
        ...

    def read_committed_publication(
        self,
        *,
        tenant_id: str,
        binding_id: str,
    ) -> KnowledgeSyncCommittedPublicationV2 | None:
        ...

    def read_publication_head(
        self,
        *,
        tenant_id: str,
        binding_id: str,
    ) -> KnowledgeSyncCommittedPublicationV2 | None:
        ...

    def list_committed_publications(
        self,
        *,
        tenant_id: str,
        binding_id: str,
    ) -> tuple[KnowledgeSyncCommittedPublicationV2, ...]:
        ...

    def read_committed_publication_for_delivery(
        self,
        *,
        tenant_id: str,
        binding_id: str,
        delivery_id: str,
    ) -> KnowledgeSyncCommittedPublicationV2 | None:
        ...

    def invalidate_for_purge(
        self,
        *,
        tenant_id: str,
        binding_id: str,
        requested_lifecycle_revision: int,
        purge_id: str,
    ) -> KnowledgeSyncPublicationFenceV1:
        ...

    def read_publication_commit_node(
        self,
        *,
        tenant_id: str,
        binding_id: str,
        commit_id: str,
    ) -> KnowledgeSyncPublicationCommitNodeV1:
        ...

    def advance_purge_publication_head(
        self,
        *,
        expected_fence: KnowledgeSyncPublicationFenceV1,
        expected_head: KnowledgeSyncCommittedPublicationV2 | None,
        replacement_head: KnowledgeSyncCommittedPublicationV2 | None,
    ) -> KnowledgeSyncPublicationFenceV1:
        ...

    def delete_publication_commit_node(
        self,
        *,
        node: KnowledgeSyncPublicationCommitNodeV1,
    ) -> bool:
        ...
class KnowledgeSyncPublicationFenceConflict(Exception):
    """Optimistic lifecycle fence write conflict."""


class KnowledgeSyncPublicationInProgress(KnowledgeSyncPublicationFenceConflict):
    """Lifecycle mutation is blocked by an unexpired publication permit."""


class DocumentStoreKnowledgeSyncPublicationFenceRepository:
    """Durable CAS-backed fence authority for application lifecycle writers."""

    def __init__(
        self,
        document_store: DocumentStore,
        *,
        clock: Callable[[], datetime] | None = None,
        permit_id_factory: Callable[[], str] | None = None,
    ) -> None:
        if not isinstance(document_store, ConditionalDocumentStore):
            raise TypeError(
                "publication fence repository requires ConditionalDocumentStore"
            )
        self._store = document_store
        self._clock = clock or (lambda: datetime.now(UTC))
        self._permit_id_factory = permit_id_factory or (
            lambda: secrets.token_urlsafe(24)
        )

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
        return self._parse(document, tenant_id=tenant, binding_id=binding)[0]

    def acquire_publication_permit(
        self,
        *,
        tenant_id: str,
        binding_id: str,
        expected_revision: int,
        expected_token: str,
        owner_id: str,
        ttl_seconds: int,
    ) -> KnowledgeSyncPublicationPermitV1 | None:
        tenant = _require_non_empty(tenant_id, field_name="tenant_id")
        binding = _require_non_empty(binding_id, field_name="binding_id")
        token = _require_non_empty(expected_token, field_name="expected_token")
        owner = _require_non_empty(owner_id, field_name="owner_id")
        if expected_revision < 1:
            raise ValueError("expected_revision must be >= 1")
        if ttl_seconds < 1 or ttl_seconds > _MAX_PERMIT_TTL_SECONDS:
            raise ValueError(
                f"ttl_seconds must be in range 1..{_MAX_PERMIT_TTL_SECONDS}"
            )
        partition = self._partition_key(tenant)
        row = self._row_key(binding)
        current = self._store.get(partition, row)
        if current is None:
            return None
        fence, existing_permit, current_descriptor = self._parse(
            current, tenant_id=tenant, binding_id=binding
        )
        now = self._utc_now()
        if (
            fence.lifecycle_revision != expected_revision
            or fence.lifecycle_token != token
            or not fence.enabled
            or fence.detached
            or (
                existing_permit is not None
                and existing_permit.expires_at > now
            )
        ):
            return None
        permit = KnowledgeSyncPublicationPermitV1(
            tenant_id=tenant,
            binding_id=binding,
            lifecycle_revision=expected_revision,
            lifecycle_token=token,
            permit_id=_require_non_empty(
                str(self._permit_id_factory()), field_name="permit_id"
            ),
            owner_id=owner,
            acquired_at=now,
            expires_at=now + timedelta(seconds=ttl_seconds),
        )
        replacement = self._to_document(
            fence,
            publication_permit=permit,
            committed_publication=current_descriptor,
        )
        if not self._store.replace_if_match(expected=current, replacement=replacement):
            return None
        return permit

    def commit_publication_under_permit(
        self,
        *,
        expected_fence: KnowledgeSyncPublicationFenceV1,
        publication_permit: KnowledgeSyncPublicationPermitV1,
        publication_descriptor: KnowledgeSyncCommittedPublicationV1,
    ) -> KnowledgeSyncPublicationCommitResult:
        self._validate_identity(expected_fence)
        self._validate_permit_identity(publication_permit)
        self._validate_descriptor_identity(publication_descriptor)
        if (
            publication_descriptor.tenant_id != expected_fence.tenant_id
            or publication_descriptor.binding_id != expected_fence.binding_id
            or publication_permit.tenant_id != expected_fence.tenant_id
            or publication_permit.binding_id != expected_fence.binding_id
            or publication_permit.lifecycle_revision != expected_fence.lifecycle_revision
            or publication_permit.lifecycle_token != expected_fence.lifecycle_token
        ):
            raise KnowledgeSyncPublicationFenceConflict(
                "publication identity does not match fence"
            )
        partition = self._partition_key(expected_fence.tenant_id)
        row = self._row_key(expected_fence.binding_id)
        for _ in range(4):
            current = self._store.get(partition, row)
            if current is None:
                raise KnowledgeSyncPublicationFenceConflict("publication fence is missing")
            fence, current_permit, current_descriptor = self._parse(
                current,
                tenant_id=expected_fence.tenant_id,
                binding_id=expected_fence.binding_id,
            )
            now = self._utc_now()
            if fence != expected_fence:
                raise KnowledgeSyncPublicationFenceConflict("publication fence changed")
            if not fence.enabled or fence.detached:
                raise KnowledgeSyncPublicationFenceConflict("publication fence is disabled")
            if current_permit != publication_permit:
                raise KnowledgeSyncPublicationFenceConflict(
                    "permit_lost: publication permit is not current"
                )
            if publication_permit.expires_at <= now:
                raise KnowledgeSyncPublicationFenceConflict("publication permit expired")

            chain = self._read_chain(
                tenant_id=fence.tenant_id,
                binding_id=fence.binding_id,
                head=current_descriptor,
            )
            existing_result = self._resolve_existing_descriptor(
                publication_descriptor,
                chain,
            )
            if existing_result is not None:
                return existing_result

            previous_commit_id = (
                None
                if current_descriptor is None
                else current_descriptor.publication_commit_id
            )
            committed_descriptor = self._committed_descriptor(
                publication_descriptor,
                previous_commit_id=previous_commit_id,
            )
            self._put_commit_node(
                KnowledgeSyncPublicationCommitNodeV1(
                    commit_id=committed_descriptor.publication_commit_id,
                    descriptor=committed_descriptor,
                    previous_commit_id=previous_commit_id,
                )
            )
            replacement = self._to_document(
                fence,
                publication_permit=publication_permit,
                committed_publication=committed_descriptor,
            )
            if self._store.replace_if_match(expected=current, replacement=replacement):
                return KnowledgeSyncPublicationCommitResult(
                    status=KnowledgeSyncPublicationCommitStatus.COMMITTED,
                    descriptor=committed_descriptor,
                )
        raise KnowledgeSyncPublicationFenceConflict("publication commit conflict")

    def release_publication_permit(
        self,
        permit: KnowledgeSyncPublicationPermitV1,
    ) -> bool:
        self._validate_permit_identity(permit)
        current = self._store.get(
            self._partition_key(permit.tenant_id),
            self._row_key(permit.binding_id),
        )
        if current is None:
            return True
        fence, current_permit, current_descriptor = self._parse(
            current,
            tenant_id=permit.tenant_id,
            binding_id=permit.binding_id,
        )
        if current_permit is None:
            return True
        if current_permit != permit:
            return False
        return self._store.replace_if_match(
            expected=current,
            replacement=self._to_document(
                fence,
                committed_publication=current_descriptor,
            ),
        )

    def is_current_publication_permit(
        self,
        *,
        permit: KnowledgeSyncPublicationPermitV1,
    ) -> bool:
        self._validate_permit_identity(permit)
        current = self._store.get(
            self._partition_key(permit.tenant_id),
            self._row_key(permit.binding_id),
        )
        if current is None:
            return False
        fence, current_permit, _current_descriptor = self._parse(
            current,
            tenant_id=permit.tenant_id,
            binding_id=permit.binding_id,
        )
        return (
            fence.enabled
            and not fence.detached
            and fence.lifecycle_revision == permit.lifecycle_revision
            and fence.lifecycle_token == permit.lifecycle_token
            and current_permit == permit
            and permit.expires_at > self._utc_now()
        )

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
        parsed, current_permit, current_descriptor = self._parse(
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
        if current_permit is not None and current_permit.expires_at > self._utc_now():
            raise KnowledgeSyncPublicationInProgress("publication_in_progress")
        if not self._store.replace_if_match(
            expected=current,
            replacement=self._to_document(
                fence,
                committed_publication=current_descriptor,
            ),
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

    def _to_document(
        self,
        fence: KnowledgeSyncPublicationFenceV1,
        *,
        publication_permit: KnowledgeSyncPublicationPermitV1 | None = None,
        committed_publication: KnowledgeSyncCommittedPublicationV2 | None = None,
    ) -> DocumentRecord:
        if publication_permit is not None:
            self._validate_permit_identity(publication_permit)
            if (
                publication_permit.tenant_id != fence.tenant_id
                or publication_permit.binding_id != fence.binding_id
                or publication_permit.lifecycle_revision != fence.lifecycle_revision
                or publication_permit.lifecycle_token != fence.lifecycle_token
            ):
                raise ValueError("publication permit does not match fence")
        if committed_publication is not None:
            self._validate_descriptor_identity(committed_publication)
            if (
                committed_publication.tenant_id != fence.tenant_id
                or committed_publication.binding_id != fence.binding_id
            ):
                raise ValueError("publication descriptor does not match fence")
        data: dict[str, Any] = {
            "schema_version": _FENCE_SCHEMA,
            "tenant_id": fence.tenant_id,
            "binding_id": fence.binding_id,
            "fence": fence.model_dump(mode="json"),
        }
        if publication_permit is not None:
            data["publication_permit"] = publication_permit.model_dump(mode="json")
        if committed_publication is not None:
            data["committed_publication"] = committed_publication.model_dump(mode="json")
        return DocumentRecord(
            partition_key=self._partition_key(fence.tenant_id),
            row_key=self._row_key(fence.binding_id),
            data=data,
        )

    def _parse(
        self,
        document: DocumentRecord,
        *,
        tenant_id: str,
        binding_id: str,
    ) -> tuple[
        KnowledgeSyncPublicationFenceV1,
        KnowledgeSyncPublicationPermitV1 | None,
        KnowledgeSyncCommittedPublicationV2 | None,
    ]:
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
        raw_permit = data.get("publication_permit")
        permit: KnowledgeSyncPublicationPermitV1 | None = None
        if raw_permit is not None:
            if not isinstance(raw_permit, dict):
                raise ValueError("publication fence record is corrupt")
            try:
                permit = KnowledgeSyncPublicationPermitV1.model_validate(raw_permit)
            except (TypeError, ValueError):
                raise ValueError("publication fence record is corrupt") from None
            if (
                permit.tenant_id != fence.tenant_id
                or permit.binding_id != fence.binding_id
                or permit.lifecycle_revision != fence.lifecycle_revision
                or permit.lifecycle_token != fence.lifecycle_token
            ):
                raise ValueError("publication fence record is corrupt")
        raw_descriptor = data.get("committed_publication")
        descriptor: KnowledgeSyncCommittedPublicationV2 | None = None
        if raw_descriptor is not None:
            if not isinstance(raw_descriptor, dict):
                raise ValueError("publication fence record is corrupt")
            try:
                descriptor = KnowledgeSyncCommittedPublicationV2.model_validate(
                    raw_descriptor
                )
            except (TypeError, ValueError):
                raise ValueError("publication fence record is corrupt") from None
            if (
                descriptor.tenant_id != fence.tenant_id
                or descriptor.binding_id != fence.binding_id
            ):
                raise ValueError("publication fence record is corrupt")
        return fence, permit, descriptor

    def read_committed_publication(
        self,
        *,
        tenant_id: str,
        binding_id: str,
    ) -> KnowledgeSyncCommittedPublicationV2 | None:
        tenant = _require_non_empty(tenant_id, field_name="tenant_id")
        binding = _require_non_empty(binding_id, field_name="binding_id")
        document = self._store.get(self._partition_key(tenant), self._row_key(binding))
        if document is None:
            return None
        descriptor = self._parse(document, tenant_id=tenant, binding_id=binding)[2]
        self._read_chain(tenant_id=tenant, binding_id=binding, head=descriptor)
        return descriptor

    def read_publication_head(
        self,
        *,
        tenant_id: str,
        binding_id: str,
    ) -> KnowledgeSyncCommittedPublicationV2 | None:
        tenant = _require_non_empty(tenant_id, field_name="tenant_id")
        binding = _require_non_empty(binding_id, field_name="binding_id")
        document = self._store.get(self._partition_key(tenant), self._row_key(binding))
        if document is None:
            return None
        return self._parse(document, tenant_id=tenant, binding_id=binding)[2]

    def list_committed_publications(
        self,
        *,
        tenant_id: str,
        binding_id: str,
    ) -> tuple[KnowledgeSyncCommittedPublicationV2, ...]:
        tenant = _require_non_empty(tenant_id, field_name="tenant_id")
        binding = _require_non_empty(binding_id, field_name="binding_id")
        document = self._store.get(self._partition_key(tenant), self._row_key(binding))
        if document is None:
            return ()
        descriptor = self._parse(document, tenant_id=tenant, binding_id=binding)[2]
        return self._read_chain(
            tenant_id=tenant,
            binding_id=binding,
            head=descriptor,
        )

    def read_committed_publication_for_delivery(
        self,
        *,
        tenant_id: str,
        binding_id: str,
        delivery_id: str,
    ) -> KnowledgeSyncCommittedPublicationV2 | None:
        tenant = _require_non_empty(tenant_id, field_name="tenant_id")
        binding = _require_non_empty(binding_id, field_name="binding_id")
        delivery = _require_non_empty(delivery_id, field_name="delivery_id")
        for descriptor in self.list_committed_publications(
            tenant_id=tenant,
            binding_id=binding,
        ):
            if descriptor.delivery_id == delivery:
                return descriptor
        return None

    def invalidate_for_purge(
        self,
        *,
        tenant_id: str,
        binding_id: str,
        requested_lifecycle_revision: int,
        purge_id: str,
    ) -> KnowledgeSyncPublicationFenceV1:
        tenant = _require_non_empty(tenant_id, field_name="tenant_id")
        binding = _require_non_empty(binding_id, field_name="binding_id")
        purge = _require_non_empty(purge_id, field_name="purge_id")
        if requested_lifecycle_revision < 1:
            raise ValueError("requested_lifecycle_revision must be >= 1")
        current = self._store.get(self._partition_key(tenant), self._row_key(binding))
        if current is None:
            raise KnowledgeSyncPublicationFenceConflict("publication fence is missing")
        fence, permit, descriptor = self._parse(
            current,
            tenant_id=tenant,
            binding_id=binding,
        )
        if permit is not None and permit.expires_at > self._utc_now():
            raise KnowledgeSyncPublicationInProgress("publication_in_progress")
        if not fence.enabled and fence.detached and fence.lifecycle_revision >= requested_lifecycle_revision:
            return fence
        revision = max(
            fence.lifecycle_revision + 1,
            requested_lifecycle_revision,
        )
        token = hashlib.sha256(
            f"purge:{tenant}:{binding}:{purge}:{revision}:{secrets.token_urlsafe(24)}".encode()
        ).hexdigest()
        replacement = KnowledgeSyncPublicationFenceV1(
            tenant_id=tenant,
            binding_id=binding,
            lifecycle_revision=revision,
            lifecycle_token=token,
            enabled=False,
            detached=True,
        )
        if not self._store.replace_if_match(
            expected=current,
            replacement=self._to_document(
                replacement,
                committed_publication=descriptor,
            ),
        ):
            raise KnowledgeSyncPublicationFenceConflict("publication purge invalidation conflict")
        return replacement

    def read_publication_commit_node(
        self,
        *,
        tenant_id: str,
        binding_id: str,
        commit_id: str,
    ) -> KnowledgeSyncPublicationCommitNodeV1:
        tenant = _require_non_empty(tenant_id, field_name="tenant_id")
        binding = _require_non_empty(binding_id, field_name="binding_id")
        commit = _require_non_empty(commit_id, field_name="commit_id")
        record = self._store.get(
            self._commit_partition(tenant, binding),
            self._commit_row_key(commit),
        )
        if record is None:
            raise ValueError("publication commit node is missing")
        node = self._parse_commit_node(record)
        if (
            node.commit_id != commit
            or node.descriptor.tenant_id != tenant
            or node.descriptor.binding_id != binding
        ):
            raise ValueError("publication commit chain identity mismatch")
        return node

    def advance_purge_publication_head(
        self,
        *,
        expected_fence: KnowledgeSyncPublicationFenceV1,
        expected_head: KnowledgeSyncCommittedPublicationV2 | None,
        replacement_head: KnowledgeSyncCommittedPublicationV2 | None,
    ) -> KnowledgeSyncPublicationFenceV1:
        self._validate_identity(expected_fence)
        if expected_fence.enabled or not expected_fence.detached:
            raise KnowledgeSyncPublicationFenceConflict("publication fence is not purge-invalidated")
        if expected_head is not None:
            self._validate_descriptor_identity(expected_head)
            if (
                expected_head.tenant_id != expected_fence.tenant_id
                or expected_head.binding_id != expected_fence.binding_id
            ):
                raise KnowledgeSyncPublicationFenceConflict("publication purge head identity mismatch")
        if replacement_head is not None:
            self._validate_descriptor_identity(replacement_head)
            if (
                replacement_head.tenant_id != expected_fence.tenant_id
                or replacement_head.binding_id != expected_fence.binding_id
            ):
                raise KnowledgeSyncPublicationFenceConflict(
                    "publication purge predecessor identity mismatch"
                )
        current = self._store.get(
            self._partition_key(expected_fence.tenant_id),
            self._row_key(expected_fence.binding_id),
        )
        if current is None:
            raise KnowledgeSyncPublicationFenceConflict("publication fence is missing")
        fence, permit, current_head = self._parse(
            current,
            tenant_id=expected_fence.tenant_id,
            binding_id=expected_fence.binding_id,
        )
        if fence != expected_fence or permit is not None:
            raise KnowledgeSyncPublicationFenceConflict("publication purge head CAS conflict")
        if current_head != expected_head:
            raise KnowledgeSyncPublicationFenceConflict("publication purge head changed")
        replacement = self._to_document(
            fence,
            committed_publication=replacement_head,
        )
        if not self._store.replace_if_match(expected=current, replacement=replacement):
            raise KnowledgeSyncPublicationFenceConflict("publication purge head CAS conflict")
        return fence

    def delete_publication_commit_node(
        self,
        *,
        node: KnowledgeSyncPublicationCommitNodeV1,
    ) -> bool:
        self._validate_descriptor_identity(node.descriptor)
        record = DocumentRecord(
            partition_key=self._commit_partition(
                node.descriptor.tenant_id,
                node.descriptor.binding_id,
            ),
            row_key=self._commit_row_key(node.commit_id),
            data={
                "schema_version": _PUBLICATION_COMMIT_NODE_SCHEMA,
                "commit_id": node.commit_id,
                "previous_commit_id": node.previous_commit_id,
                "node": node.model_dump(mode="json"),
            },
        )
        if not isinstance(self._store, ConditionalDocumentStore):
            raise TypeError("publication purge requires ConditionalDocumentStore")
        return self._store.delete_if_match(expected=record)

    @staticmethod
    def _commit_partition(tenant_id: str, binding_id: str) -> str:
        return (
            f"{_PUBLICATION_COMMIT_PARTITION_PREFIX}:"
            f"{_require_non_empty(tenant_id, field_name='tenant_id')}:"
            f"{_require_non_empty(binding_id, field_name='binding_id')}"
        )

    @staticmethod
    def _commit_row_key(commit_id: str) -> str:
        return f"commit:{_require_non_empty(commit_id, field_name='commit_id')}"

    @staticmethod
    def _committed_descriptor(
        descriptor: KnowledgeSyncCommittedPublicationV1,
        *,
        previous_commit_id: str | None,
    ) -> KnowledgeSyncCommittedPublicationV2:
        return KnowledgeSyncCommittedPublicationV2(
            **descriptor.model_dump(),
            publication_commit_id=publication_commit_id(
                descriptor,
                previous_commit_id=previous_commit_id,
            ),
            previous_publication_commit_id=previous_commit_id,
        )

    def _put_commit_node(
        self,
        node: KnowledgeSyncPublicationCommitNodeV1,
    ) -> None:
        record = DocumentRecord(
            partition_key=self._commit_partition(
                node.descriptor.tenant_id,
                node.descriptor.binding_id,
            ),
            row_key=self._commit_row_key(node.commit_id),
            data={
                "schema_version": _PUBLICATION_COMMIT_NODE_SCHEMA,
                "commit_id": node.commit_id,
                "previous_commit_id": node.previous_commit_id,
                "node": node.model_dump(mode="json"),
            },
        )
        existing = self._store.get(record.partition_key, record.row_key)
        if existing is not None:
            if self._parse_commit_node(existing) != node:
                raise KnowledgeSyncPublicationFenceConflict(
                    "publication commit node identity conflict"
                )
            return
        if not self._store.put_if_absent(record):
            retry = self._store.get(record.partition_key, record.row_key)
            if retry is None or self._parse_commit_node(retry) != node:
                raise KnowledgeSyncPublicationFenceConflict(
                    "publication commit node write conflict"
                )

    def _read_chain(
        self,
        *,
        tenant_id: str,
        binding_id: str,
        head: KnowledgeSyncCommittedPublicationV2 | None,
    ) -> tuple[KnowledgeSyncCommittedPublicationV2, ...]:
        if head is None:
            return ()
        current: KnowledgeSyncCommittedPublicationV2 | None = head
        previous_sequence: int | None = None
        visited: set[str] = set()
        descriptors: list[KnowledgeSyncCommittedPublicationV2] = []
        while current is not None:
            if len(descriptors) >= _MAX_COMMITTED_CHAIN_NODES:
                raise ValueError("publication commit chain safety bound exceeded")
            commit_id = current.publication_commit_id
            if commit_id in visited:
                raise ValueError("publication commit chain predecessor cycle")
            visited.add(commit_id)
            record = self._store.get(
                self._commit_partition(tenant_id, binding_id),
                self._commit_row_key(commit_id),
            )
            if record is None:
                raise ValueError("publication commit node is missing")
            node = self._parse_commit_node(record)
            if node.descriptor != current:
                raise ValueError("publication commit head/node mismatch")
            if (
                node.descriptor.tenant_id != tenant_id
                or node.descriptor.binding_id != binding_id
            ):
                raise ValueError("publication commit chain identity mismatch")
            if (
                previous_sequence is not None
                and current.materialization_sequence >= previous_sequence
            ):
                raise ValueError("publication commit chain sequence is non-monotonic")
            descriptors.append(current)
            previous_sequence = current.materialization_sequence
            previous_id = current.previous_publication_commit_id
            if previous_id is None:
                break
            if previous_id in visited:
                raise ValueError("publication commit chain predecessor cycle")
            previous_record = self._store.get(
                self._commit_partition(tenant_id, binding_id),
                self._commit_row_key(previous_id),
            )
            if previous_record is None:
                raise ValueError("publication commit predecessor is missing")
            previous_node = self._parse_commit_node(previous_record)
            if previous_node.commit_id != previous_id:
                raise ValueError("publication commit predecessor id mismatch")
            current = previous_node.descriptor
        return tuple(descriptors)

    @staticmethod
    def _parse_commit_node(
        record: DocumentRecord,
    ) -> KnowledgeSyncPublicationCommitNodeV1:
        if (
            record.data.get("schema_version") != _PUBLICATION_COMMIT_NODE_SCHEMA
            or not isinstance(record.data.get("node"), dict)
            or not isinstance(record.data.get("commit_id"), str)
            or record.row_key != f"commit:{record.data.get('commit_id')}"
        ):
            raise ValueError("publication commit node is corrupt")
        try:
            node = KnowledgeSyncPublicationCommitNodeV1.model_validate(
                record.data["node"]
            )
        except (TypeError, ValueError):
            raise ValueError("publication commit node is corrupt") from None
        if (
            node.commit_id != record.data.get("commit_id")
            or node.previous_commit_id != record.data.get("previous_commit_id")
        ):
            raise ValueError("publication commit node is corrupt")
        return node

    @staticmethod
    def _resolve_existing_descriptor(
        candidate: KnowledgeSyncCommittedPublicationV1,
        chain: tuple[KnowledgeSyncCommittedPublicationV2, ...],
    ) -> KnowledgeSyncPublicationCommitResult | None:
        candidate_identity = _descriptor_identity(candidate)
        for position, committed in enumerate(chain):
            if _descriptor_identity(committed) == candidate_identity:
                return KnowledgeSyncPublicationCommitResult(
                    status=(
                        KnowledgeSyncPublicationCommitStatus.REPLAYED
                        if position == 0
                        else KnowledgeSyncPublicationCommitStatus.STALE
                    ),
                    descriptor=committed,
                )
            if (
                committed.materialization_sequence
                == candidate.materialization_sequence
            ):
                raise KnowledgeSyncPublicationFenceConflict(
                    "publication sequence conflict"
                )
        if chain and (
            chain[0].materialization_sequence
            > candidate.materialization_sequence
        ):
            return KnowledgeSyncPublicationCommitResult(
                status=KnowledgeSyncPublicationCommitStatus.STALE,
                descriptor=chain[0],
            )
        return None

    def _utc_now(self) -> datetime:
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None or now.utcoffset() != timedelta(0):
            raise ValueError("publication fence clock must return timezone-aware UTC")
        return now

    @staticmethod
    def _validate_identity(fence: KnowledgeSyncPublicationFenceV1) -> None:
        if not fence.tenant_id.strip() or not fence.binding_id.strip():
            raise ValueError("publication fence identity is invalid")

    @staticmethod
    def _validate_permit_identity(
        permit: KnowledgeSyncPublicationPermitV1,
    ) -> None:
        if not permit.tenant_id.strip() or not permit.binding_id.strip():
            raise ValueError("publication permit identity is invalid")

    @staticmethod
    def _validate_descriptor_identity(
        descriptor: KnowledgeSyncCommittedPublicationV1,
    ) -> None:
        if not descriptor.tenant_id.strip() or not descriptor.binding_id.strip():
            raise ValueError("publication descriptor identity is invalid")
