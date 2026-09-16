# © Artur Czarnecki. All rights reserved.

"""Memory session turn index ports backed by RAG embedding/vectorstore managers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.memory.contracts.session_turn_index import (
    SessionTurnIndexEmbeddingPort,
    SessionTurnIndexMetadataFilter,
    SessionTurnIndexStoreCreationContext,
    SessionTurnIndexVectorQueryHit,
    SessionTurnIndexVectorScope,
    SessionTurnIndexVectorUpsertRecord,
    SessionTurnIndexVectorstorePort,
)
from intergrax.rag.embedding.contracts.base_embedding_manager import BaseEmbeddingManager
from intergrax.rag.vectorstore.contracts.base_vectorstore_manager import BaseVectorstoreManager
from intergrax.rag.vectorstore.contracts.native_vectorstore import (
    MetadataFilter,
    VectorStoreRecord,
    VectorStoreScope,
)


def _embedding_rows_to_sequences(
    matrix: NDArray[np.float32],
) -> tuple[tuple[float, ...], ...]:
    if matrix.ndim == 1:
        return (tuple(float(x) for x in matrix.tolist()),)
    return tuple(
        tuple(float(x) for x in matrix[index].tolist()) for index in range(matrix.shape[0])
    )


@dataclass(frozen=True, slots=True)
class SessionTurnIndexEmbeddingAdapter:
    """Adapt RAG ``BaseEmbeddingManager`` to ``SessionTurnIndexEmbeddingPort``."""

    _delegate: BaseEmbeddingManager

    def embed_texts(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        matrix = self._delegate.embed_texts(texts)
        rows = _embedding_rows_to_sequences(matrix)
        return rows


def _to_vector_store_scope(scope: SessionTurnIndexVectorScope) -> VectorStoreScope:
    return VectorStoreScope(
        tenant_id=scope.tenant_id,
        namespace=scope.namespace,
        workspace_id=scope.workspace_id,
    )


def _to_metadata_filter(
    metadata_filter: SessionTurnIndexMetadataFilter | None,
) -> MetadataFilter | None:
    if metadata_filter is None:
        return None
    return MetadataFilter(conditions=dict(metadata_filter.conditions))


def _metadata_for_hit(
    metadata: Mapping[str, object],
) -> dict[str, str | int | float]:
    out: dict[str, str | int | float] = {}
    for key, value in metadata.items():
        if isinstance(value, bool):
            out[key] = int(value)
        elif isinstance(value, (str, int, float)):
            out[key] = value
    return out


@dataclass(frozen=True, slots=True)
class _AdaptedSessionTurnIndexVectorHit:
    similarity_score: float
    document_content: str
    document_id: str
    document_metadata: Mapping[str, str | int | float]


def _record_to_vector_store_record(
    record: SessionTurnIndexVectorUpsertRecord,
    *,
    scope: SessionTurnIndexVectorScope,
) -> VectorStoreRecord:
    rag_scope = _to_vector_store_scope(scope)
    session_id = str(record.document_metadata.get("session_id") or "")
    document = KnowledgeDocument.model_validate(
        {
            "schema_version": 1,
            "identity": {
                "document_id": record.vector_id,
                "root_document_id": record.vector_id,
            },
            "scope": {
                "tenant_id": rag_scope.tenant_id,
                "namespace": rag_scope.namespace,
                "workspace_id": rag_scope.workspace_id,
            },
            "content": record.document_content,
            "metadata": dict(record.document_metadata),
            "provenance": {
                "source_kind": "conversation_turn",
                "source_id": record.vector_id,
                "source_parent_id": session_id,
            },
        }
    )
    embedding = np.asarray(record.embedding, dtype=np.float32)
    return VectorStoreRecord(
        document=document,
        embedding=embedding,
        vector_id=record.vector_id,
    )


@dataclass(frozen=True, slots=True)
class SessionTurnIndexVectorstoreAdapter:
    """Adapt RAG ``BaseVectorstoreManager`` to ``SessionTurnIndexVectorstorePort``."""

    _delegate: BaseVectorstoreManager

    def add_records(
        self,
        records: Sequence[SessionTurnIndexVectorUpsertRecord],
        *,
        scope: SessionTurnIndexVectorScope,
    ) -> None:
        rag_scope = _to_vector_store_scope(scope)
        rag_records = [
            _record_to_vector_store_record(record, scope=scope) for record in records
        ]
        self._delegate.add_records(rag_records, scope=rag_scope)

    def query(
        self,
        embedding: Sequence[float],
        *,
        scope: SessionTurnIndexVectorScope,
        top_k: int,
        metadata_filter: SessionTurnIndexMetadataFilter | None = None,
    ) -> Sequence[SessionTurnIndexVectorQueryHit]:
        rag_scope = _to_vector_store_scope(scope)
        query_vector = np.asarray(embedding, dtype=np.float32)
        hits = self._delegate.query(
            query_vector,
            scope=rag_scope,
            top_k=top_k,
            metadata_filter=_to_metadata_filter(metadata_filter),
            include_embeddings=False,
        )
        adapted: list[_AdaptedSessionTurnIndexVectorHit] = []
        for hit in hits:
            document = hit.document
            adapted.append(
                _AdaptedSessionTurnIndexVectorHit(
                    similarity_score=float(hit.similarity_score),
                    document_content=document.content,
                    document_id=document.identity.document_id,
                    document_metadata=_metadata_for_hit(document.metadata),
                )
            )
        return adapted

    def delete(
        self,
        ids: Sequence[str],
        *,
        scope: SessionTurnIndexVectorScope,
    ) -> None:
        rag_scope = _to_vector_store_scope(scope)
        self._delegate.delete(ids, scope=rag_scope)


def adapt_rag_managers_to_session_turn_index_ports(
    *,
    embedding_manager: BaseEmbeddingManager,
    vectorstore_manager: BaseVectorstoreManager,
) -> tuple[SessionTurnIndexEmbeddingPort, SessionTurnIndexVectorstorePort]:
    return (
        SessionTurnIndexEmbeddingAdapter(_delegate=embedding_manager),
        SessionTurnIndexVectorstoreAdapter(_delegate=vectorstore_manager),
    )


def build_session_turn_index_creation_context(
    *,
    tenant_id: str,
    embedding_manager: BaseEmbeddingManager,
    vectorstore_manager: BaseVectorstoreManager,
    index_roles: tuple[str, ...] = ("user", "assistant"),
    vector_index_namespace: str | None = None,
    workspace_id: str | None = None,
) -> SessionTurnIndexStoreCreationContext:
    embedding_port, vectorstore_port = adapt_rag_managers_to_session_turn_index_ports(
        embedding_manager=embedding_manager,
        vectorstore_manager=vectorstore_manager,
    )
    return SessionTurnIndexStoreCreationContext(
        tenant_id=tenant_id,
        index_roles=index_roles,
        vector_index_namespace=vector_index_namespace,
        workspace_id=workspace_id,
        embedding_manager=embedding_port,
        vectorstore_manager=vectorstore_port,
    )
