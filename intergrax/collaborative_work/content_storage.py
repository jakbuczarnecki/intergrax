# © Artur Czarnecki. All rights reserved.

"""Provider-neutral artifact content storage boundary (MP-3F).

Content-addressed storage behind the frozen MP-3A ``ArtifactContentRef`` contract.
Physical object storage is adapter-private; collaborative publication remains
in ``CollaborativeWorkArtifactService``.
"""

from __future__ import annotations

import re
from typing import Final, Literal, Mapping, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.contracts.collaborative_work import ArtifactContentRef
from intergrax.contracts.validation import compute_sha256_content_digest
from intergrax.integrations.contracts.object_storage import ConditionalObjectStorage

SCHEMA_STORE_ARTIFACT_CONTENT_REQUEST_V1: Final = "store_artifact_content_request.v1"
SCHEMA_GET_ARTIFACT_CONTENT_REQUEST_V1: Final = "get_artifact_content_request.v1"
SCHEMA_STORED_ARTIFACT_CONTENT_V1: Final = "stored_artifact_content.v1"

_LOGICAL_CONTENT_REF_SCHEME: Final = "artifact-content"
_LOGICAL_CONTENT_REF_PREFIX: Final = f"{_LOGICAL_CONTENT_REF_SCHEME}://sha256/"
_LOGICAL_CONTENT_REF_RE: Final = re.compile(
    r"^artifact-content://sha256/[0-9a-f]{64}$",
)
_PHYSICAL_KEY_PREFIX: Final = "intergrax/collaborative-work/artifact-content/v1/"

_NON_EMPTY = Field(min_length=1)


class ArtifactContentStorageError(Exception):
    """Base error for artifact content storage boundary failures."""


class ArtifactContentIntegrityError(ArtifactContentStorageError):
    """Raised when stored bytes fail digest or size verification."""


class ArtifactContentReferenceUnsupported(ArtifactContentStorageError):
    """Raised when a logical content reference format is not supported."""


class ArtifactContentPersistenceError(ArtifactContentStorageError):
    """Raised when provider write/read consistency cannot be verified."""


class StoreArtifactContentRequest(BaseModel):
    """Immutable request to persist collaborative artifact content bytes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["store_artifact_content_request.v1"] = (
        SCHEMA_STORE_ARTIFACT_CONTENT_REQUEST_V1
    )
    tenant_id: str = _NON_EMPTY
    workspace_id: str = _NON_EMPTY
    body: bytes
    media_type: str = _NON_EMPTY

    @field_validator("tenant_id", "workspace_id", "media_type")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized


class GetArtifactContentRequest(BaseModel):
    """Immutable request to retrieve collaborative artifact content bytes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["get_artifact_content_request.v1"] = (
        SCHEMA_GET_ARTIFACT_CONTENT_REQUEST_V1
    )
    tenant_id: str = _NON_EMPTY
    workspace_id: str = _NON_EMPTY
    content_ref: ArtifactContentRef

    @field_validator("tenant_id", "workspace_id")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized


class StoredArtifactContent(BaseModel):
    """Verified artifact content returned by a successful ``get`` operation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["stored_artifact_content.v1"] = SCHEMA_STORED_ARTIFACT_CONTENT_V1
    content_ref: ArtifactContentRef
    body: bytes


@runtime_checkable
class ArtifactContentStore(Protocol):
    """Provider-neutral port for content-addressed artifact byte storage."""

    def put(self, request: StoreArtifactContentRequest) -> ArtifactContentRef:
        """Persist bytes and return a verified neutral content descriptor."""

    def get(self, request: GetArtifactContentRequest) -> StoredArtifactContent | None:
        """Return verified bytes or ``None`` when scoped content is missing."""


def _scope_digest(tenant_id: str, workspace_id: str) -> str:
    scope_bytes = f"{tenant_id}\0{workspace_id}".encode("utf-8")
    return compute_sha256_content_digest(scope_bytes).removeprefix("sha256:")


def _digest_hex_from_integrity_digest(integrity_digest: str) -> str:
    return integrity_digest.removeprefix("sha256:")


def _format_logical_content_ref(digest_hex: str) -> str:
    return f"{_LOGICAL_CONTENT_REF_PREFIX}{digest_hex}"


def _parse_logical_content_ref(content_ref: str) -> str:
    if "?" in content_ref or "#" in content_ref:
        raise ArtifactContentReferenceUnsupported(
            "logical content reference must not include query or fragment components",
        )
    if not _LOGICAL_CONTENT_REF_RE.fullmatch(content_ref):
        raise ArtifactContentReferenceUnsupported(
            "logical content reference must match artifact-content://sha256/<64 lowercase hex>",
        )
    return content_ref.removeprefix(_LOGICAL_CONTENT_REF_PREFIX)


def _physical_object_key(tenant_id: str, workspace_id: str, digest_hex: str) -> str:
    scope = _scope_digest(tenant_id, workspace_id)
    return f"{_PHYSICAL_KEY_PREFIX}{scope}/sha256/{digest_hex}"


def _build_artifact_content_ref(
    *,
    digest: str,
    media_type: str,
    size_bytes: int,
) -> ArtifactContentRef:
    digest_hex = _digest_hex_from_integrity_digest(digest)
    return ArtifactContentRef(
        content_ref=_format_logical_content_ref(digest_hex),
        media_type=media_type,
        integrity_digest=digest,
        size_bytes=size_bytes,
    )


def _verify_reference_digest_consistency(content_ref: ArtifactContentRef) -> str:
    logical_digest_hex = _parse_logical_content_ref(content_ref.content_ref)
    integrity_digest_hex = _digest_hex_from_integrity_digest(content_ref.integrity_digest)
    if logical_digest_hex != integrity_digest_hex:
        raise ArtifactContentIntegrityError(
            "logical content reference digest does not match integrity_digest",
        )
    return logical_digest_hex


def _verify_body_matches_descriptor(body: bytes, content_ref: ArtifactContentRef) -> None:
    actual_digest = compute_sha256_content_digest(body)
    if actual_digest != content_ref.integrity_digest:
        raise ArtifactContentIntegrityError("stored body digest does not match content descriptor")
    if content_ref.size_bytes is not None and len(body) != content_ref.size_bytes:
        raise ArtifactContentIntegrityError("stored body size does not match content descriptor")


class ObjectStorageArtifactContentStore:
    """Object-storage-backed ``ArtifactContentStore`` implementation."""

    def __init__(self, object_storage: ConditionalObjectStorage) -> None:
        self._object_storage = object_storage

    def put(self, request: StoreArtifactContentRequest) -> ArtifactContentRef:
        digest = compute_sha256_content_digest(request.body)
        digest_hex = _digest_hex_from_integrity_digest(digest)
        physical_key = _physical_object_key(
            request.tenant_id,
            request.workspace_id,
            digest_hex,
        )
        content_ref = _build_artifact_content_ref(
            digest=digest,
            media_type=request.media_type,
            size_bytes=len(request.body),
        )

        existing = self._object_storage.get(physical_key)
        if existing is not None:
            self._verify_existing_object(
                body=existing.body,
                expected_digest=digest,
                expected_size=len(request.body),
            )
            return content_ref

        metadata: Mapping[str, str] = {
            "integrity_digest": digest,
            "schema_version": SCHEMA_STORED_ARTIFACT_CONTENT_V1,
            "size_bytes": str(len(request.body)),
        }
        created = self._object_storage.put_if_absent(
            physical_key,
            request.body,
            content_type=request.media_type,
            metadata=metadata,
        )
        if created:
            persisted = self._object_storage.get(physical_key)
            if persisted is None:
                raise ArtifactContentPersistenceError(
                    "object storage reported success but content is not readable",
                )
            self._verify_existing_object(
                body=persisted.body,
                expected_digest=digest,
                expected_size=len(request.body),
            )
            return content_ref

        winner = self._object_storage.get(physical_key)
        if winner is None:
            raise ArtifactContentPersistenceError(
                "object storage reported conflict but content is not readable",
            )
        self._verify_existing_object(
            body=winner.body,
            expected_digest=digest,
            expected_size=len(request.body),
        )
        return content_ref

    def get(self, request: GetArtifactContentRequest) -> StoredArtifactContent | None:
        digest_hex = _verify_reference_digest_consistency(request.content_ref)
        physical_key = _physical_object_key(
            request.tenant_id,
            request.workspace_id,
            digest_hex,
        )
        stored = self._object_storage.get(physical_key)
        if stored is None:
            return None
        _verify_body_matches_descriptor(stored.body, request.content_ref)
        return StoredArtifactContent(
            content_ref=request.content_ref,
            body=stored.body,
        )

    @staticmethod
    def _verify_existing_object(
        *,
        body: bytes,
        expected_digest: str,
        expected_size: int,
    ) -> None:
        actual_digest = compute_sha256_content_digest(body)
        if actual_digest != expected_digest:
            raise ArtifactContentIntegrityError(
                "existing stored content digest does not match expected content identity",
            )
        if len(body) != expected_size:
            raise ArtifactContentIntegrityError(
                "existing stored content size does not match expected content identity",
            )
