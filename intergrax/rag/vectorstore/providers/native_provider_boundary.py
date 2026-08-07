# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Shared validation and transport helpers for native vector-store providers."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.knowledge.contracts.validation import (
    JsonValue,
    knowledge_metadata_to_plain,
)
from intergrax.rag.vectorstore.contracts.native_vectorstore import (
    MetadataFilter,
    VectorStoreContractError,
    VectorStoreHit,
    VectorStoreRecord,
    VectorStoreScope,
)

SYSTEM_METADATA_KEYS = frozenset(
    {
        "schema_version",
        "document_id",
        "root_document_id",
        "parent_document_id",
        "tenant_id",
        "namespace",
        "workspace_id",
        "source_kind",
        "source_id",
        "source_parent_id",
        "provider_id",
        "source_revision",
        "source_uri",
        "content_hash",
        "text",
        "logical_id",
    }
)


def validate_scope(scope: VectorStoreScope, *, tenant_id: str) -> VectorStoreScope:
    if not isinstance(scope, VectorStoreScope):
        raise TypeError("scope must be a VectorStoreScope")
    if scope.tenant_id != tenant_id:
        raise VectorStoreContractError("scope tenant_id differs from provider tenant")
    return scope


def validate_records(
    records: Sequence[VectorStoreRecord],
    *,
    scope: VectorStoreScope,
    tenant_id: str,
) -> list[VectorStoreRecord]:
    validate_scope(scope, tenant_id=tenant_id)
    materialized = list(records)
    validated: list[VectorStoreRecord] = []
    for record in materialized:
        if not isinstance(record, VectorStoreRecord):
            raise TypeError("records must contain only VectorStoreRecord values")
        checked = VectorStoreRecord(
            document=record.document,
            embedding=record.embedding,
            vector_id=record.vector_id,
        )
        if not scope.matches_document(checked.document):
            raise VectorStoreContractError(
                "record document scope does not match operation scope"
            )
        validated.append(checked)
    return validated


def validate_query(
    query_embedding: NDArray[np.float32] | Sequence[float],
    *,
    top_k: int,
) -> tuple[NDArray[np.float32], int]:
    try:
        vector = np.array(query_embedding, dtype=np.float32, copy=True)
    except (TypeError, ValueError) as exc:
        raise VectorStoreContractError("query_embedding must be numeric") from exc
    if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
        raise VectorStoreContractError(
            "query_embedding must be a finite non-empty 1D vector"
        )
    if type(top_k) is not int or top_k <= 0:
        raise VectorStoreContractError("top_k must be an exact positive int")
    vector.setflags(write=False)
    return vector, top_k


def effective_filter(
    scope: VectorStoreScope,
    metadata_filter: MetadataFilter | None,
) -> MetadataFilter:
    return MetadataFilter.for_scope(scope, metadata_filter)


def provider_metadata(
    document: KnowledgeDocument,
    *,
    scope: VectorStoreScope,
) -> dict[str, JsonValue]:
    if not scope.matches_document(document):
        raise VectorStoreContractError(
            "document scope does not match provider operation scope"
        )
    provenance = document.provenance
    system: dict[str, JsonValue] = {
        "schema_version": document.schema_version,
        "document_id": document.identity.document_id,
        "root_document_id": document.identity.root_document_id,
        "tenant_id": document.scope.tenant_id,
        "source_kind": provenance.source_kind,
        "source_id": provenance.source_id,
    }
    optional_identity = {
        "parent_document_id": document.identity.parent_document_id,
        "namespace": document.scope.namespace,
        "workspace_id": document.scope.workspace_id,
        "source_parent_id": provenance.source_parent_id,
        "provider_id": provenance.provider_id,
        "source_revision": provenance.source_revision,
        "source_uri": provenance.source_uri,
        "content_hash": provenance.content_hash,
    }
    system.update(
        {
            key: value
            for key, value in optional_identity.items()
            if value is not None
        }
    )
    user_metadata = {
        key: value
        for key, value in knowledge_metadata_to_plain(document.metadata).items()
        if key not in SYSTEM_METADATA_KEYS
    }
    return {**user_metadata, **system}


def reconstruct_document(
    content: str,
    metadata: Mapping[str, object],
    *,
    scope: VectorStoreScope,
) -> KnowledgeDocument:
    if not isinstance(metadata, Mapping):
        raise VectorStoreContractError("provider metadata must be a mapping")
    provider_tenant = metadata.get("tenant_id")
    provider_namespace = metadata.get("namespace")
    provider_workspace = metadata.get("workspace_id")
    if provider_tenant != scope.tenant_id:
        raise VectorStoreContractError("provider result belongs to another tenant")
    if provider_namespace != scope.namespace:
        raise VectorStoreContractError("provider result belongs to another namespace")
    if provider_workspace != scope.workspace_id:
        raise VectorStoreContractError("provider result belongs to another workspace")

    def required(name: str) -> object:
        value = metadata.get(name)
        if value is None:
            raise VectorStoreContractError(f"provider metadata lacks '{name}'")
        return value

    identity = {
        "document_id": required("document_id"),
        "root_document_id": required("root_document_id"),
    }
    if metadata.get("parent_document_id") is not None:
        identity["parent_document_id"] = metadata["parent_document_id"]

    provenance = {
        "source_kind": required("source_kind"),
        "source_id": required("source_id"),
    }
    for key in (
        "source_parent_id",
        "provider_id",
        "source_revision",
        "source_uri",
        "content_hash",
    ):
        if metadata.get(key) is not None:
            provenance[key] = metadata[key]

    user_metadata = {
        key: value
        for key, value in metadata.items()
        if key not in SYSTEM_METADATA_KEYS
    }
    try:
        return KnowledgeDocument.model_validate(
            {
                "schema_version": required("schema_version"),
                "identity": identity,
                "scope": {
                    "tenant_id": provider_tenant,
                    "namespace": provider_namespace,
                    "workspace_id": provider_workspace,
                },
                "content": content,
                "metadata": user_metadata,
                "provenance": provenance,
            }
        )
    except Exception as exc:
        raise VectorStoreContractError(
            "provider result cannot reconstruct KnowledgeDocument"
        ) from exc


def native_hit(
    *,
    vector_id: object,
    content: object,
    metadata: Mapping[str, object],
    similarity_score: object,
    rank: int,
    scope: VectorStoreScope,
    embedding: NDArray[np.float32] | Sequence[float] | None = None,
) -> VectorStoreHit:
    if not isinstance(vector_id, str) or not vector_id:
        raise VectorStoreContractError("provider result lacks a valid vector_id")
    if not isinstance(content, str):
        content = str(content)
    if isinstance(similarity_score, bool) or not isinstance(
        similarity_score, (int, float)
    ):
        raise VectorStoreContractError("provider result has an invalid score")
    score = float(similarity_score)
    if not math.isfinite(score):
        raise VectorStoreContractError("provider result score must be finite")
    score = max(0.0, min(1.0, score))
    document = reconstruct_document(content, metadata, scope=scope)
    return VectorStoreHit(
        vector_id=vector_id,
        document=document,
        similarity_score=score,
        rank=rank,
        embedding=embedding,
    )
