# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from intergrax.rag.vectorstore.contracts.base_vectorstore_manager import BaseVectorstoreManager
from intergrax.rag.vectorstore.contracts.native_vectorstore import (
    MetadataFilter,
    VectorStoreContractError,
    VectorStoreHit,
    VectorStoreRecord,
    VectorStoreScope,
)
from intergrax.rag.vectorstore.contracts.vector_store import VectorStore
from intergrax.rag.document_loaders.compat.legacy_runtime_document import (
    from_legacy_rag_hit,
    to_legacy_rag_document,
)
from intergrax.rag.vectorstore.governance.collection_access_policy import (
    CollectionAccessPolicy,
    enforce_collection_access,
)
from intergrax.logging import IntergraxLogging

logger = IntergraxLogging.get_logger(__name__, component="rag")


class VectorstoreManager(BaseVectorstoreManager):
    """
    Native core boundary around a legacy provider compatibility port.

    Providers receive legacy documents only after the complete native batch
    has been validated. Provider hits are normalized before returning.
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
        self._provider_scope_bound = False
        provider_tenant = getattr(store, "_tenant_id", None)
        if provider_tenant is not None:
            provider_scope = VectorStoreScope(tenant_id=provider_tenant)
            self._provider_scope_bound = (
                scope is None
                or (scope.namespace is None and scope.workspace_id is None)
            )
            if scope is not None and scope.tenant_id != provider_scope.tenant_id:
                raise ValueError("manager scope tenant_id does not match provider tenant")
            if scope is None:
                self._bound_scope = provider_scope
        elif scope is None:
            raise ValueError(
                "VectorstoreManager requires an explicit scope or a tenant-bound provider"
            )

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
            for record in validated[1:]
        ):
            raise VectorStoreContractError(
                "records must share the same document tenant and namespace"
            )

        document_scope = VectorStoreScope(
            tenant_id=first_document_scope.tenant_id,
            namespace=first_document_scope.namespace,
        )
        if scope is None:
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
                    workspace_id=(
                        bound_scope.workspace_id if bound_scope is not None else None
                    ),
                )
            )
        else:
            resolved_scope = self._resolve_scope(scope)

        if any(
            not resolved_scope.matches_document(record.document)
            for record in validated
        ):
            raise VectorStoreContractError(
                "record document scope does not match operation scope"
            )

        self._enforce_access("write", resolved_scope)

        legacy_documents = []
        for record in validated:
            legacy = to_legacy_rag_document(record.document)
            metadata = dict(getattr(legacy, "metadata", {}) or {})
            if resolved_scope.workspace_id is None:
                metadata.pop("workspace_id", None)
            else:
                metadata["workspace_id"] = resolved_scope.workspace_id
            legacy_documents.append(
                type(legacy)(
                    id=getattr(legacy, "id", None),
                    page_content=getattr(legacy, "page_content"),
                    metadata=metadata,
                )
            )

        return self._store.add_documents(
            documents=legacy_documents,
            embeddings=[record.embedding.tolist() for record in validated],
            ids=[record.vector_id for record in validated],
        )

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
            top_k=limit,
            metadata_filter=provider_filter,
            include_embeddings=include_embeddings,
        )
        return self._normalize_hits(
            provider_hits,
            scope=resolved_scope,
            include_embeddings=include_embeddings,
        )

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
                raw_metadata = getattr(provider_hit, "metadata")
                if not isinstance(raw_metadata, Mapping):
                    raise VectorStoreContractError(
                        "provider hit metadata must be a mapping"
                    )
                if raw_metadata.get("tenant_id") != scope.tenant_id:
                    raise VectorStoreContractError(
                        "provider hit belongs to a different tenant"
                    )
                if (
                    scope.namespace is not None
                    and raw_metadata.get("namespace") != scope.namespace
                ):
                    raise VectorStoreContractError(
                        "provider hit belongs to a different namespace"
                    )
                if (
                    scope.workspace_id is not None
                    and raw_metadata.get("workspace_id") != scope.workspace_id
                ):
                    raise VectorStoreContractError(
                        "provider hit belongs to a different workspace"
                    )
                document = from_legacy_rag_hit(provider_hit)
                if not scope.matches_document(document):
                    raise VectorStoreContractError(
                        "provider hit document scope does not match query scope"
                    )
                embedding = (
                    getattr(provider_hit, "embedding", None)
                    if include_embeddings
                    else None
                )
                normalized.append(
                    VectorStoreHit(
                        vector_id=getattr(provider_hit, "id"),
                        document=document,
                        similarity_score=getattr(provider_hit, "similarity_score"),
                        rank=getattr(provider_hit, "rank"),
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
        if hasattr(self._store, "query_hybrid"):
            provider_hits = self._store.query_hybrid(
                vector.tolist(),
                query_text,
                top_k=limit,
                metadata_filter=provider_filter,
                include_embeddings=include_embeddings,
                alpha=alpha,
            )
        else:
            provider_hits = self._store.query(
                query_embedding=vector.tolist(),
                top_k=limit,
                metadata_filter=provider_filter,
                include_embeddings=include_embeddings,
            )
        return self._normalize_hits(
            provider_hits,
            scope=resolved_scope,
            include_embeddings=include_embeddings,
        )

    def delete(
        self,
        ids: Sequence[str],
        *,
        scope: VectorStoreScope | None = None,
    ) -> None:
        resolved_scope = self._resolve_scope(scope)
        self._enforce_access("delete", resolved_scope)
        self._require_scope_bound_operation(resolved_scope)
        self._store.delete(ids)

    def count(self, *, scope: VectorStoreScope | None = None) -> int:
        resolved_scope = self._resolve_scope(scope)
        self._enforce_access("count", resolved_scope)
        self._require_scope_bound_operation(resolved_scope)
        return self._store.count()

    def _require_scope_bound_operation(self, scope: VectorStoreScope) -> None:
        if not self._provider_scope_bound or self._bound_scope is None:
            raise VectorStoreContractError(
                "delete/count require a tenant-bound provider scope"
            )
        if not self._bound_scope.matches(scope):
            raise VectorStoreContractError(
                "delete/count require the provider's exact bound scope"
            )

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