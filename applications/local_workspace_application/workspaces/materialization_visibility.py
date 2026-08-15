# © Artur Czarnecki. All rights reserved.

"""Fail-closed visibility authority for materialized workspace knowledge."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

if TYPE_CHECKING:
    from local_workspace_application.workspaces.repository import (
        ManagedWorkspaceRepository,
    )

from local_workspace_application.workspaces.connected_source_manifest import (
    ConnectedSourceMaterializationManifestConflict,
    ConnectedSourceMaterializationManifestRepository,
)

from intergrax.runtime.vendor_knowledge.sync_publication_fence import (
    DocumentStoreKnowledgeSyncPublicationFenceRepository,
)

_IDENTIFIER_RE = re.compile(r"^[^\x00-\x1f\x7f]{1,512}$")


def _normalized_identifier(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name}_must_be_string")
    if value != value.strip() or not _IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"{field_name}_must_be_normalized")
    return value


class KnowledgeMaterializationOwnershipModeV1(StrEnum):
    CONNECTED_SOURCE = "connected_source"
    LEGACY = "legacy"


class KnowledgeMaterializationOwnershipV1(BaseModel):
    """Canonical, credential-free ownership of one materialized document."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    tenant_id: str = Field(..., min_length=1, max_length=512)
    workspace_id: str = Field(..., min_length=1, max_length=512)
    source_id: str = Field(..., min_length=1, max_length=512)
    indexed_source_binding_id: str | None = Field(default=None, max_length=512)
    knowledge_source_binding_ref: str | None = Field(default=None, max_length=512)
    delivery_id: str | None = Field(default=None, max_length=512)
    materialization_generation: str | None = Field(default=None, max_length=512)
    materialization_sequence: int | None = Field(default=None, gt=0)
    remote_id: str | None = Field(default=None, max_length=512)
    ownership_mode: KnowledgeMaterializationOwnershipModeV1 = (
        KnowledgeMaterializationOwnershipModeV1.CONNECTED_SOURCE
    )

    _validate_required_ids = field_validator(
        "tenant_id", "workspace_id", "source_id"
    )(
        lambda value, info: _normalized_identifier(
            value, info.field_name or "identifier"
        )
    )
    _validate_optional_ids = field_validator(
        "indexed_source_binding_id",
        "knowledge_source_binding_ref",
        "delivery_id",
        "materialization_generation",
        "remote_id",
    )(
        lambda value, info: (
            None
            if value is None
            else _normalized_identifier(value, info.field_name or "identifier")
        )
    )

    @model_validator(mode="after")
    def _validate_ownership_shape(self) -> KnowledgeMaterializationOwnershipV1:
        connected_fields = (
            self.indexed_source_binding_id,
            self.knowledge_source_binding_ref,
            self.delivery_id,
            self.remote_id,
        )
        if self.ownership_mode is KnowledgeMaterializationOwnershipModeV1.CONNECTED_SOURCE:
            if any(value is None for value in connected_fields):
                raise ValueError("connected_materialization_ownership_incomplete")
        elif any(value is not None for value in connected_fields):
            raise ValueError("legacy_materialization_ownership_has_connected_fields")
        if (
            self.ownership_mode is KnowledgeMaterializationOwnershipModeV1.LEGACY
            and self.materialization_generation is not None
        ):
            raise ValueError("legacy_materialization_generation_forbidden")
        return self

    @classmethod
    def connected(
        cls,
        *,
        tenant_id: str,
        workspace_id: str,
        source_id: str,
        indexed_source_binding_id: str,
        knowledge_source_binding_ref: str,
        delivery_id: str,
        remote_id: str,
        materialization_generation: str | None = None,
        materialization_sequence: int | None = None,
    ) -> KnowledgeMaterializationOwnershipV1:
        return cls(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            source_id=source_id,
            indexed_source_binding_id=indexed_source_binding_id,
            knowledge_source_binding_ref=knowledge_source_binding_ref,
            delivery_id=delivery_id,
            materialization_generation=materialization_generation or delivery_id,
            materialization_sequence=materialization_sequence,
            remote_id=remote_id,
        )

    @classmethod
    def legacy(
        cls,
        *,
        tenant_id: str,
        workspace_id: str,
        source_id: str,
    ) -> KnowledgeMaterializationOwnershipV1:
        return cls(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            source_id=source_id,
            ownership_mode=KnowledgeMaterializationOwnershipModeV1.LEGACY,
        )

    @property
    def identity_scope(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )


class KnowledgeMaterializationVisibilityAuthorityTypeV1(StrEnum):
    LEGACY_IMMEDIATE = "legacy_immediate"
    DELIVERY_RECEIPT = "delivery_receipt"
    MATERIALIZATION_GENERATION = "materialization_generation"
    DELIVERY_MANIFEST = "delivery_manifest"


class KnowledgeMaterializationVisibilityStatusV1(StrEnum):
    PREPARED = "prepared"
    COMMITTED = "committed"
    ABORTED = "aborted"


class KnowledgeMaterializationActivePointerV1(BaseModel):
    """Durable active version pointer; prepared writes never update it."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    tenant_id: str = Field(..., min_length=1, max_length=512)
    workspace_id: str = Field(..., min_length=1, max_length=512)
    source_id: str = Field(..., min_length=1, max_length=512)
    indexed_source_binding_id: str = Field(..., min_length=1, max_length=512)
    remote_id: str = Field(..., min_length=1, max_length=512)
    delivery_id: str = Field(..., min_length=1, max_length=512)
    materialization_generation: str = Field(..., min_length=1, max_length=512)
    document_id: str = Field(..., min_length=1, max_length=512)
    materialization_revision: int = Field(..., gt=0)
    committed_at: datetime

    _validate_ids = field_validator(
        "tenant_id",
        "workspace_id",
        "source_id",
        "indexed_source_binding_id",
        "remote_id",
        "delivery_id",
        "materialization_generation",
        "document_id",
    )(
        lambda value, info: _normalized_identifier(
            value, info.field_name or "identifier"
        )
    )

    @field_validator("committed_at")
    @classmethod
    def _validate_committed_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("materialization_committed_at_must_be_timezone_aware")
        if value.utcoffset() != timedelta(0):
            raise ValueError("materialization_committed_at_must_be_utc")
        return value.astimezone(UTC)

    @classmethod
    def for_ownership(
        cls,
        *,
        ownership: KnowledgeMaterializationOwnershipV1,
        document_id: str,
        materialization_revision: int,
        committed_at: datetime,
    ) -> KnowledgeMaterializationActivePointerV1:
        if ownership.ownership_mode is not KnowledgeMaterializationOwnershipModeV1.CONNECTED_SOURCE:
            raise ValueError("legacy_materialization_has_no_active_pointer")
        assert ownership.indexed_source_binding_id is not None
        assert ownership.remote_id is not None
        assert ownership.delivery_id is not None
        assert ownership.materialization_generation is not None
        return cls(
            tenant_id=ownership.tenant_id,
            workspace_id=ownership.workspace_id,
            source_id=ownership.source_id,
            indexed_source_binding_id=ownership.indexed_source_binding_id,
            remote_id=ownership.remote_id,
            delivery_id=ownership.delivery_id,
            materialization_generation=ownership.materialization_generation,
            document_id=document_id,
            materialization_revision=materialization_revision,
            committed_at=committed_at,
        )


class KnowledgeMaterializationVisibilityPort(Protocol):
    def is_visible(
        self,
        *,
        ownership: KnowledgeMaterializationOwnershipV1,
        document_id: str | None = None,
        content_hash: str | None = None,
    ) -> bool:
        ...


class RepositoryKnowledgeMaterializationVisibility:
    """Resolve visibility from durable receipt and active-pointer authority."""

    def __init__(self, repository: ManagedWorkspaceRepository) -> None:
        self._repository = repository

    def is_visible(
        self,
        *,
        ownership: KnowledgeMaterializationOwnershipV1,
        document_id: str | None = None,
        content_hash: str | None = None,
    ) -> bool:
        if not isinstance(ownership, KnowledgeMaterializationOwnershipV1):
            return False
        try:
            source = self._repository.get_source(
                tenant_id=ownership.tenant_id,
                workspace_id=ownership.workspace_id,
                source_id=ownership.source_id,
            )
        except (TypeError, ValueError, AttributeError):
            return False
        if source is None:
            return False
        is_connected = str(getattr(source.source_type, "value", source.source_type)) == (
            "connected_source"
        )
        if ownership.ownership_mode is KnowledgeMaterializationOwnershipModeV1.LEGACY:
            return not is_connected
        if not is_connected:
            return False
        try:
            publication_fence = DocumentStoreKnowledgeSyncPublicationFenceRepository(
                self._repository.document_store
            ).read_fence(
                tenant_id=ownership.tenant_id,
                binding_id=ownership.knowledge_source_binding_ref or "",
            )
        except (TypeError, ValueError, AttributeError):
            return False
        if publication_fence is not None and (
            not publication_fence.enabled or publication_fence.detached
        ):
            return False
        if (
            ownership.indexed_source_binding_id is None
            or ownership.knowledge_source_binding_ref is None
            or ownership.delivery_id is None
            or ownership.remote_id is None
            or ownership.materialization_generation is None
        ):
            return False
        authority = None
        if document_id is not None:
            try:
                ref = self._repository.get_document_ref(
                    tenant_id=ownership.tenant_id,
                    workspace_id=ownership.workspace_id,
                    document_id=document_id,
                )
            except (TypeError, ValueError, AttributeError):
                return False
            if ref is None or ref.materialization_ownership != ownership:
                return False
            authority = ref.visibility_authority_type
        else:
            try:
                authority = next(
                    (
                        ref.visibility_authority_type
                        for ref in self._repository.list_document_refs(
                            tenant_id=ownership.tenant_id,
                            workspace_id=ownership.workspace_id,
                        )
                        if ref.materialization_ownership == ownership
                    ),
                    None,
                )
            except (TypeError, ValueError, AttributeError):
                return False
        if authority is KnowledgeMaterializationVisibilityAuthorityTypeV1.DELIVERY_MANIFEST:
            if ownership.materialization_sequence is None:
                return False
            try:
                manifest_repository = ConnectedSourceMaterializationManifestRepository(
                    self._repository.document_store
                )
                manifest = manifest_repository.get_committed_for_delivery(
                    tenant_id=ownership.tenant_id,
                    workspace_id=ownership.workspace_id,
                    source_id=ownership.source_id,
                    indexed_source_binding_id=ownership.indexed_source_binding_id,
                    delivery_id=ownership.delivery_id,
                )
            except (
                ConnectedSourceMaterializationManifestConflict,
                TypeError,
                ValueError,
                AttributeError,
            ):
                return False
            if (
                manifest is None
                or manifest.tenant_id != ownership.tenant_id
                or manifest.workspace_id != ownership.workspace_id
                or manifest.source_id != ownership.source_id
                or manifest.indexed_source_binding_id
                != ownership.indexed_source_binding_id
                or manifest.knowledge_source_binding_ref
                != ownership.knowledge_source_binding_ref
                or manifest.delivery_id != ownership.delivery_id
                or manifest.materialization_sequence
                != ownership.materialization_sequence
            ):
                return False
            entry = next(
                (
                    item
                    for item in manifest.document_entries
                    if item.remote_id == ownership.remote_id
                ),
                None,
            )
            if entry is not None:
                if not (
                    manifest.delivery_id == ownership.delivery_id
                    and entry.materialization_generation
                    == ownership.materialization_generation
                    and (document_id is None or entry.document_id == document_id)
                    and (content_hash is None or entry.content_hash == content_hash)
                ):
                    return False
                try:
                    newest = manifest_repository.get_committed_for_remote(
                        tenant_id=ownership.tenant_id,
                        workspace_id=ownership.workspace_id,
                        source_id=ownership.source_id,
                        indexed_source_binding_id=ownership.indexed_source_binding_id,
                        knowledge_source_binding_ref=ownership.knowledge_source_binding_ref,
                        remote_id=ownership.remote_id,
                    )
                except (
                    ConnectedSourceMaterializationManifestConflict,
                    TypeError,
                    ValueError,
                    AttributeError,
                ):
                    return False
                return (
                    newest is not None
                    and newest.materialization_sequence
                    == manifest.materialization_sequence
                )
            return False
        try:
            receipt = self._repository.get_connected_source_delivery_receipt(
                tenant_id=ownership.tenant_id,
                workspace_id=ownership.workspace_id,
                source_id=ownership.source_id,
                delivery_id=ownership.delivery_id,
            )
        except (TypeError, ValueError, AttributeError):
            return False
        if (
            receipt is None
            or receipt.tenant_id != ownership.tenant_id
            or receipt.workspace_id != ownership.workspace_id
            or receipt.source_id != ownership.source_id
            or receipt.indexed_source_binding_id != ownership.indexed_source_binding_id
            or receipt.knowledge_source_binding_ref != ownership.knowledge_source_binding_ref
            or str(receipt.status.value) != "completed"
            or receipt.completed_at is None
            or receipt.items_failed != 0
            or receipt.materialization_sequence is None
        ):
            return False
        try:
            assignment = self._repository.get_connected_source_delivery_sequence_assignment(
                tenant_id=ownership.tenant_id,
                workspace_id=ownership.workspace_id,
                source_id=ownership.source_id,
                indexed_source_binding_id=ownership.indexed_source_binding_id,
                delivery_id=ownership.delivery_id,
            )
        except (TypeError, ValueError, AttributeError):
            return False
        if (
            assignment is not None
            and assignment.materialization_sequence != receipt.materialization_sequence
        ):
            return False
        try:
            pointer = self._repository.get_active_materialization_pointer(
                tenant_id=ownership.tenant_id,
                workspace_id=ownership.workspace_id,
                source_id=ownership.source_id,
                indexed_source_binding_id=ownership.indexed_source_binding_id,
                remote_id=ownership.remote_id,
            )
        except (TypeError, ValueError, AttributeError):
            return False
        return bool(
            pointer is not None
            and pointer.delivery_id == ownership.delivery_id
            and pointer.materialization_generation == ownership.materialization_generation
            and pointer.materialization_revision == receipt.materialization_sequence
        )
