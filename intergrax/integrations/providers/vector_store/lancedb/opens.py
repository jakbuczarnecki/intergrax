# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Low-level LanceDB openers — internal to the lancedb integration package."""

from __future__ import annotations

from typing import Callable, Optional, Sequence

from intergrax.integrations._shared.p3.configs import VectorIntegrationConfig
from intergrax.integrations.contracts.vector_store import (
    MetadataFilter,
    VectorStore,
    VectorStoreHit,
    VectorStoreRecord,
    VectorStoreScope,
)
from intergrax.integrations.providers.vector_store.lancedb.integration import LancedbVectorStoreIntegration
from intergrax.rag.vectorstore.providers.native_provider_boundary import (
    effective_filter,
    native_hit,
    provider_metadata,
    validate_query,
    validate_records,
    validate_scope,
)


def _open_rag_store(
    *,
    store_factory: Optional[Callable[[], VectorStore]] = None,
) -> VectorStore:
    if store_factory is not None:
        return store_factory()

    class _LanceClient(VectorStore):
        def __init__(self) -> None:
            self._tenant_id: str | None = None
            self._docs: dict[str, tuple[str, dict[str, object], list[float]]] = {}

        def add_records(
            self,
            records: Sequence[VectorStoreRecord],
            *,
            scope: VectorStoreScope,
        ) -> Sequence[str]:
            if self._tenant_id is None:
                self._tenant_id = scope.tenant_id
            validated = validate_records(
                records,
                scope=scope,
                tenant_id=self._tenant_id,
            )
            if not validated:
                return []
            ids: list[str] = []
            for record in validated:
                metadata = provider_metadata(record.document, scope=scope)
                self._docs[record.vector_id] = (
                    record.document.content,
                    dict(metadata),
                    record.embedding.tolist(),
                )
                ids.append(record.vector_id)
            return ids

        def query(
            self,
            query_embedding: Sequence[float],
            *,
            scope: VectorStoreScope,
            top_k: int,
            metadata_filter: Optional[MetadataFilter] = None,
            include_embeddings: bool = False,
        ) -> list[VectorStoreHit]:
            _vector, limit = validate_query(query_embedding, top_k=top_k)
            validate_scope(scope, tenant_id=self._tenant_id or scope.tenant_id)
            conditions = effective_filter(scope, metadata_filter).conditions
            hits: list[VectorStoreHit] = []
            for idx, (doc_id, (content, metadata, embedding)) in enumerate(
                self._docs.items()
            ):
                if not all(
                    metadata.get(key) == value for key, value in conditions.items()
                ):
                    continue
                if len(hits) >= limit:
                    break
                hits.append(
                    native_hit(
                        vector_id=doc_id,
                        content=content,
                        metadata=metadata,
                        similarity_score=1.0 / float(idx + 1),
                        rank=len(hits),
                        scope=scope,
                        embedding=embedding if include_embeddings else None,
                    )
                )
            return hits

        def delete(self, ids: Sequence[str], *, scope: VectorStoreScope) -> None:
            validate_scope(scope, tenant_id=self._tenant_id or scope.tenant_id)
            conditions = effective_filter(scope, None).conditions
            for doc_id in ids:
                row = self._docs.get(doc_id)
                if row is not None and all(
                    row[1].get(key) == value for key, value in conditions.items()
                ):
                    self._docs.pop(doc_id, None)

        def count(self, *, scope: VectorStoreScope) -> int:
            validate_scope(scope, tenant_id=self._tenant_id or scope.tenant_id)
            conditions = effective_filter(scope, None).conditions
            return sum(
                all(metadata.get(key) == value for key, value in conditions.items())
                for _, (_, metadata, _) in self._docs.items()
            )

    return _LanceClient()


def open_lancedb_vector_store(
    config: VectorIntegrationConfig,
    *,
    implementation: Optional[VectorStore] = None,
    store: Optional[VectorStore] = None,
    store_factory: Optional[Callable[[], VectorStore]] = None,
) -> VectorStore:
    inner = (
        implementation
        if implementation is not None
        else store
        if store is not None
        else _open_rag_store(store_factory=store_factory)
    )
    return LancedbVectorStoreIntegration.from_store(config, inner)
