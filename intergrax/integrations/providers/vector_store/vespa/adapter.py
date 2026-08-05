# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Vespa vector store adapter implementing ``VectorStore``."""

from __future__ import annotations

from typing import Optional, Sequence

from intergrax.integrations.contracts.vector_store import (
    MetadataFilter,
    VectorStore,
    VectorStoreHit,
    VectorStoreRecord,
    VectorStoreScope,
)
from intergrax.integrations.providers.vector_store.vespa.client import VespaRestClient
from intergrax.integrations.providers.vector_store.vespa.config import VespaIntegrationConfig
from intergrax.utils import attribute_access
from intergrax.rag.vectorstore.providers.native_provider_boundary import (
    effective_filter,
    native_hit,
    provider_metadata,
    validate_query,
    validate_records,
    validate_scope,
)
from intergrax.rag.vectorstore.contracts.native_vectorstore import (
    VectorStoreContractError,
)


class _VespaVectorStore(VectorStore):
    """Feed/query facade over Vespa document API."""

    def __init__(self, config: VespaIntegrationConfig, client: VespaRestClient) -> None:
        self._config = config
        self._client = client
        self._doc_ids: list[str] = []
        self._metadata: dict[str, dict[str, object]] = {}

    @property
    def rest_client(self) -> VespaRestClient:
        return self._client

    def add_records(
        self,
        records: Sequence[VectorStoreRecord],
        *,
        scope: VectorStoreScope,
    ) -> Sequence[str]:
        validated = validate_records(
            records,
            scope=scope,
            tenant_id=self._config.tenant_id,
        )
        if not validated:
            return []
        ids: list[str] = []
        for record in validated:
            doc_id = record.vector_id
            metadata = provider_metadata(record.document, scope=scope)
            self._client.feed_document(
                doc_id=doc_id,
                fields={
                    "content": record.document.content,
                    "metadata": metadata,
                    "tenant_id": scope.tenant_id,
                    **(
                        {"namespace": scope.namespace}
                        if scope.namespace is not None
                        else {}
                    ),
                    **(
                        {"workspace_id": scope.workspace_id}
                        if scope.workspace_id is not None
                        else {}
                    ),
                    "embedding": {"values": record.embedding.tolist()},
                },
            )
            if doc_id in self._doc_ids:
                self._doc_ids.remove(doc_id)
            self._doc_ids.append(doc_id)
            self._metadata[doc_id] = dict(metadata)
            ids.append(doc_id)
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
        validate_scope(scope, tenant_id=self._config.tenant_id)
        conditions = effective_filter(scope, metadata_filter).conditions
        clauses = [
            self._yql_condition(key, value)
            for key, value in conditions.items()
        ]
        yql = (
            f"select * from sources {self._config.collection} where "
            + (" and ".join(clauses) if clauses else "true")
        )
        rows = self._client.query_yql(yql, hits=limit, ranking="default")
        hits: list[VectorStoreHit] = []
        for rank, row in enumerate(rows[:top_k], start=1):
            fields = row.get("fields") if isinstance(row, dict) else {}
            if not isinstance(fields, dict):
                fields = {}
            content = str(fields.get("content") or "")
            metadata = dict(fields.get("metadata") or {})
            hits.append(
                native_hit(
                    vector_id=str(row.get("id") or fields.get("id") or rank),
                    content=content,
                    metadata=metadata,
                    similarity_score=1.0 / float(rank),
                    rank=rank,
                    scope=scope,
                    embedding=(
                        attribute_access.optional(fields.get("embedding"), "values", None)
                        if include_embeddings
                        else None
                    ),
                )
            )
        return hits

    def delete(self, ids: Sequence[str], *, scope: VectorStoreScope) -> None:
        validate_scope(scope, tenant_id=self._config.tenant_id)
        requested_ids = list(ids)
        if not requested_ids:
            return
        for doc_id in requested_ids:
            metadata = self._metadata.get(doc_id)
            if metadata is None:
                raise VectorStoreContractError(
                    "vespa scoped delete is unsupported for unknown IDs"
                )
            if any(
                metadata.get(key) != value
                for key, value in self._scope_conditions(scope).items()
            ):
                continue
            self._client.delete_document(doc_id)
            if doc_id in self._doc_ids:
                self._doc_ids.remove(doc_id)
            self._metadata.pop(doc_id, None)

    def count(self, *, scope: VectorStoreScope) -> int:
        validate_scope(scope, tenant_id=self._config.tenant_id)
        conditions = self._scope_conditions(scope)
        yql = (
            f"select * from sources {self._config.collection} where "
            + " and ".join(
                self._yql_condition(key, value)
                for key, value in conditions.items()
            )
        )
        return self._client.count_documents(yql=yql)

    @staticmethod
    def _scope_conditions(scope: VectorStoreScope) -> dict[str, object]:
        conditions: dict[str, object] = {"tenant_id": scope.tenant_id}
        if scope.namespace is not None:
            conditions["namespace"] = scope.namespace
        if scope.workspace_id is not None:
            conditions["workspace_id"] = scope.workspace_id
        return conditions

    @staticmethod
    def _yql_condition(key: str, value: object) -> str:
        if not isinstance(key, str) or not key.replace("_", "").isalnum():
            raise VectorStoreContractError("vespa filter contains an invalid field")
        field = key if key in {"tenant_id", "namespace", "workspace_id"} else f"metadata.{key}"
        if not isinstance(value, (str, int, float, bool)) or value is None:
            raise VectorStoreContractError("vespa filter value is unsupported")
        escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
        return f'{field} contains "{escaped}"'
