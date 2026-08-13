# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.knowledge.contracts.validation import require_non_empty_str
from intergrax.distributed.source_operation import (
    RagSourceOperationKey,
    SOURCE_PUBLICATION_GENERATION_METADATA_KEY,
    SourceOperationCoordinator,
)
from intergrax.rag.vectorstore.contracts.base_vectorstore_manager import BaseVectorstoreManager
from intergrax.rag.vectorstore.contracts.native_vectorstore import (
    MetadataFilter,
    VectorStoreContractError,
    VectorStoreHit,
    VectorStoreRecord,
    VectorStoreScope,
)
from intergrax.rag.vectorstore.contracts.hybrid_search import (
    provider_supports_native_hybrid_search,
    resolve_native_hybrid_search_provider,
)
from intergrax.rag.vectorstore.contracts.vector_store import VectorStore
from intergrax.rag.vectorstore.governance.collection_access_policy import (
    CollectionAccessPolicy,
    enforce_collection_access,
)
from intergrax.rag.vectorstore.publication_visibility import vector_record_visible
from intergrax.logging import IntergraxLogging

logger = IntergraxLogging.get_logger(__name__, component="rag")


class VectorstoreManager(BaseVectorstoreManager):
    """
    Native core boundary around provider implementations.

    Providers receive validated native records and return native hits.
    """

    def __init__(
        self,
        store: VectorStore,
        *,
        access_policy: Optional[CollectionAccessPolicy] = None,
        collection_name: Optional[str] = None,
        scope: VectorStoreScope | None = None,
    ) -> None:
        self._store = store
        self._access_policy = access_policy
        self._collection_name = collection_name
        self._bound_scope = scope
        self._source_coordinator: SourceOperationCoordinator | None = None
        provider_tenant = getattr(store, "_tenant_id", None)
        if isinstance(provider_tenant, str) and provider_tenant.strip():
            provider_tenant = provider_tenant.strip()
            provider_scope = VectorStoreScope(tenant_id=provider_tenant)
            if (
                scope is not None
                and scope.tenant_id != provider_scope.tenant_id
            ):
                raise ValueError("manager scope tenant_id does not match provider tenant")
            if scope is None:
                self._bound_scope = provider_scope
        elif scope is None:
            raise ValueError(
                "VectorstoreManager requires an explicit scope or a tenant-bound provider"
            )

    @property
    def bound_scope(self) -> VectorStoreScope | None:
        return self._bound_scope

    def set_source_operation_coordinator(
        self,
        coordinator: SourceOperationCoordinator,
    ) -> None:
        self._source_coordinator = coordinator

    def _resolve_scope(self, scope: VectorStoreScope | None) -> VectorStoreScope:
        if scope is not None and not isinstance(scope, VectorStoreScope):
            raise TypeError("scope must be a VectorStoreScope")
        resolved = scope or self._bound_scope
        if resolved is None:
            raise ValueError("vector-store operation requires an explicit tenant scope")
        if self._bound_scope is not None:
            if resolved.tenant_id != self._bound_scope.tenant_id:
                raise VectorStoreContractError("operation tenant_id differs from bound scope")
            if (
                self._bound_scope.namespace is not None
                and resolved.namespace != self._bound_scope.namespace
            ):
                raise VectorStoreContractError("operation namespace differs from bound scope")
            if (
                self._bound_scope.workspace_id is not None
                and resolved.workspace_id != self._bound_scope.workspace_id
            ):
                raise VectorStoreContractError("operation workspace_id differs from bound scope")
        return resolved

    def _enforce_access(self, operation: str, scope: VectorStoreScope) -> None:
        if self._access_policy is not None and self._access_policy.tenant_id != scope.tenant_id:
            raise ValueError("collection access policy tenant_id differs from scope")
        enforce_collection_access(
            self._access_policy,
            operation,
            workspace_id=scope.workspace_id,
            collection_name=self._collection_name,
        )

    @staticmethod
    def _validate_query_vector(
        value: NDArray[np.float32] | Sequence[float],
    ) -> NDArray[np.float32]:
        try:
            vector = np.array(value, dtype=np.float32, copy=True)
        except (TypeError, ValueError) as exc:
            raise VectorStoreContractError("query_embedding must be numeric") from exc
        if vector.ndim != 1:
            raise VectorStoreContractError("query_embedding must be exactly 1D")
        if vector.size == 0:
            raise VectorStoreContractError(
                "query_embedding must have a positive dimension"
            )
        if not np.isfinite(vector).all():
            raise VectorStoreContractError(
                "query_embedding must contain only finite values"
            )
        vector.setflags(write=False)
        return vector

    @staticmethod
    def _validate_top_k(top_k: int) -> int:
        if type(top_k) is not int or top_k <= 0:
            raise VectorStoreContractError("top_k must be an exact positive int")
        return top_k

    def add_records(
        self,
        records: Sequence[VectorStoreRecord],
        *,
        scope: VectorStoreScope | None = None,
    ) -> Sequence[str] | None:
        materialized = list(records)
        if not materialized:
            return []

        validated: list[VectorStoreRecord] = []
        for record in materialized:
            if not isinstance(record, VectorStoreRecord):
                raise TypeError("records must contain only VectorStoreRecord values")
            checked = VectorStoreRecord(
                document=record.document,
                embedding=record.embedding,
                vector_id=record.vector_id,
            )
            validated.append(checked)

        first_document_scope = validated[0].document.scope
        if any(
            record.document.scope.tenant_id != first_document_scope.tenant_id
            or record.document.scope.namespace != first_document_scope.namespace
            or record.document.scope.workspace_id != first_document_scope.workspace_id
            for record in validated[1:]
        ):
            raise VectorStoreContractError(
                "records must share the same document tenant, namespace and workspace"
            )

        document_scope = VectorStoreScope(
            tenant_id=first_document_scope.tenant_id,
            namespace=first_document_scope.namespace,
            workspace_id=first_document_scope.workspace_id,
        )
        if scope is not None:
            resolved_scope = self._resolve_scope(scope)
        else:
            bound_scope = self._bound_scope
            if bound_scope is not None:
                if bound_scope.tenant_id != document_scope.tenant_id:
                    raise VectorStoreContractError(
                        "document tenant_id differs from bound scope"
                    )
                if (
                    bound_scope.namespace is not None
                    and bound_scope.namespace != document_scope.namespace
                ):
                    raise VectorStoreContractError(
                        "document namespace differs from bound scope"
                    )
            resolved_scope = self._resolve_scope(
                VectorStoreScope(
                    tenant_id=document_scope.tenant_id,
                    namespace=document_scope.namespace,
                    workspace_id=document_scope.workspace_id,
                )
            )

        if any(
            not resolved_scope.matches_document(record.document)
            for record in validated
        ):
            raise VectorStoreContractError(
                "record document scope does not match operation scope"
            )

        self._enforce_access("write", resolved_scope)
        return self._store.add_records(validated, scope=resolved_scope)

    def query(
        self,
        query_embedding: NDArray[np.float32] | Sequence[float],
        *,
        scope: VectorStoreScope | None = None,
        top_k: int,
        metadata_filter: MetadataFilter | None = None,
        include_embeddings: bool = False,
    ) -> Sequence[VectorStoreHit]:
        resolved_scope = self._resolve_scope(scope)
        self._enforce_access("read", resolved_scope)
        vector = self._validate_query_vector(query_embedding)
        limit = self._validate_top_k(top_k)
        provider_filter = MetadataFilter.for_scope(resolved_scope, metadata_filter)
        provider_hits = self._store.query(
            query_embedding=vector.tolist(),
            scope=resolved_scope,
            top_k=self._publication_query_limit(limit, resolved_scope),
            metadata_filter=provider_filter,
            include_embeddings=include_embeddings,
        )
        normalized = self._normalize_hits(
            provider_hits,
            scope=resolved_scope,
            include_embeddings=include_embeddings,
        )
        return self._filter_visible_publication_hits(normalized, resolved_scope)[:limit]

    def _normalize_hits(
        self,
        provider_hits: Sequence[object],
        *,
        scope: VectorStoreScope,
        include_embeddings: bool,
    ) -> list[VectorStoreHit]:
        normalized: list[VectorStoreHit] = []
        for provider_hit in provider_hits:
            try:
                if not isinstance(provider_hit, VectorStoreHit):
                    raise VectorStoreContractError(
                        "provider returned a non-native vector-store hit"
                    )
                document = provider_hit.document
                if not isinstance(document, KnowledgeDocument):
                    raise VectorStoreContractError(
                        "provider hit document is not a KnowledgeDocument"
                    )
                if not scope.matches_document(document):
                    raise VectorStoreContractError(
                        "provider hit document scope does not match query scope"
                    )
                embedding = (
                    provider_hit.embedding
                    if include_embeddings
                    else None
                )
                normalized.append(
                    VectorStoreHit(
                        vector_id=provider_hit.vector_id,
                        document=document,
                        similarity_score=provider_hit.similarity_score,
                        rank=provider_hit.rank,
                        embedding=embedding,
                    )
                )
            except VectorStoreContractError:
                raise
            except (AttributeError, TypeError, ValueError, KeyError) as exc:
                raise VectorStoreContractError(
                    "provider returned a malformed vector-store hit"
                ) from exc
        return normalized

    def supports_native_hybrid_search(self) -> bool:
        return provider_supports_native_hybrid_search(self._store)

    def query_hybrid(
        self,
        query_embedding: NDArray[np.float32] | Sequence[float],
        query_text: str,
        *,
        scope: VectorStoreScope | None = None,
        top_k: int,
        metadata_filter: MetadataFilter | None = None,
        include_embeddings: bool = False,
        alpha: float = 0.5,
    ) -> Sequence[VectorStoreHit]:
        resolved_scope = self._resolve_scope(scope)
        self._enforce_access("read", resolved_scope)
        vector = self._validate_query_vector(query_embedding)
        limit = self._validate_top_k(top_k)
        provider_filter = MetadataFilter.for_scope(resolved_scope, metadata_filter)
        native_provider = resolve_native_hybrid_search_provider(self._store)
        if native_provider is None:
            raise VectorStoreContractError(
                "provider does not support native hybrid search"
            )
        provider_hits = native_provider.query_hybrid(
            vector.tolist(),
            query_text,
            scope=resolved_scope,
            top_k=self._publication_query_limit(limit, resolved_scope),
            metadata_filter=provider_filter,
            include_embeddings=include_embeddings,
            alpha=alpha,
        )
        normalized = self._normalize_hits(
            provider_hits,
            scope=resolved_scope,
            include_embeddings=include_embeddings,
        )
        return self._filter_visible_publication_hits(normalized, resolved_scope)[:limit]

    def _publication_query_limit(
        self,
        limit: int,
        scope: VectorStoreScope,
    ) -> int:
        if self._source_coordinator is None:
            return limit
        try:
            return max(limit, int(self._store.count(scope=scope)))
        except (AttributeError, TypeError, ValueError):
            return limit

    def _filter_visible_publication_hits(
        self,
        hits: Sequence[VectorStoreHit],
        scope: VectorStoreScope,
    ) -> list[VectorStoreHit]:
        visible: list[VectorStoreHit] = []
        for hit in hits:
            source_id = str(hit.document.provenance.source_id)
            publication_scope_id = str(hit.document.identity.root_document_id).strip()
            key = RagSourceOperationKey(
                tenant_id=scope.tenant_id,
                namespace=scope.namespace,
                workspace_id=scope.workspace_id,
                source_id=source_id,
                publication_scope_id=publication_scope_id,
            )
            record_generation = hit.document.metadata.get(
                SOURCE_PUBLICATION_GENERATION_METADATA_KEY
            )
            generation_value = (
                str(record_generation)
                if record_generation is not None
                else None
            )
            if not vector_record_visible(
                record_generation=generation_value,
                source_key=key,
                coordinator=self._source_coordinator,
            ):
                continue
            visible.append(
                VectorStoreHit(
                    vector_id=hit.vector_id,
                    document=hit.document,
                    similarity_score=hit.similarity_score,
                    rank=len(visible),
                    embedding=hit.embedding,
                )
            )
        return visible

    def delete(
        self,
        ids: Sequence[str],
        *,
        scope: VectorStoreScope | None = None,
    ) -> None:
        resolved_scope = self._resolve_scope(scope)
        self._enforce_access("delete", resolved_scope)
        try:
            self._store.delete(ids, scope=resolved_scope)
        except TypeError as exc:
            raise VectorStoreContractError(
                "provider does not support scoped delete"
            ) from exc

    def list_source_record_ids(
        self,
        *,
        source_id: str,
        scope: VectorStoreScope | None = None,
        root_document_id: str | None = None,
    ) -> Sequence[str]:
        resolved_scope = self._resolve_scope(scope)
        self._enforce_access("read", resolved_scope)
        try:
            canonical_source_id = require_non_empty_str(
                source_id,
                field_name="source_id",
            )
            canonical_root_document_id = (
                require_non_empty_str(
                    root_document_id,
                    field_name="root_document_id",
                )
                if root_document_id is not None
                else None
            )
        except ValueError as exc:
            raise VectorStoreContractError(
                "source_id and root_document_id must be non-empty strings when provided"
            ) from exc

        provider_ids = self._store.list_source_record_ids(
            source_id=canonical_source_id,
            scope=resolved_scope,
            root_document_id=canonical_root_document_id,
        )
        return tuple(sorted(provider_ids))

    def count(self, *, scope: VectorStoreScope | None = None) -> int:
        resolved_scope = self._resolve_scope(scope)
        self._enforce_access("count", resolved_scope)
        try:
            return self._store.count(scope=resolved_scope)
        except TypeError as exc:
            raise VectorStoreContractError(
                "provider does not support scoped count"
            ) from exc

    def list_collections(self) -> list[str]:
        return list(self._store.list_collections())

    def list_document_ids(self, *, limit: int = 100, offset: int = 0) -> list[str]:
        from intergrax.tools.registry.runtime_bindings import VectorStoreDocumentListerBinding

        store = self._store
        if isinstance(store, VectorStoreDocumentListerBinding):
            return list(store.list_document_ids(limit=limit, offset=offset))
        raise RuntimeError("vectorstore_list_documents_not_supported")

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        from intergrax.tools.registry.runtime_bindings import VectorStoreDocumentListerBinding

        store = self._store
        if isinstance(store, VectorStoreDocumentListerBinding):
            return store.get_document(document_id.strip())
        raise RuntimeError("vectorstore_get_document_not_supported")

    def search_by_metadata(
        self,
        *,
        conditions: dict[str, Any],
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        from intergrax.tools.registry.runtime_bindings import VectorstoreIndexLifecycleBinding

        store = self._store
        if isinstance(store, VectorstoreIndexLifecycleBinding):
            return list(store.search_by_metadata(conditions=conditions, limit=limit))
        raise RuntimeError("vectorstore_search_by_metadata_not_supported")

    def purge_collection(self, *, dry_run: bool = True, tenant_id: str = "") -> dict[str, Any]:
        from intergrax.tools.registry.runtime_bindings import VectorstoreIndexLifecycleBinding

        store = self._store
        if isinstance(store, VectorstoreIndexLifecycleBinding):
            return dict(store.purge_collection(dry_run=dry_run, tenant_id=tenant_id))
        raise RuntimeError("vectorstore_purge_collection_not_supported")