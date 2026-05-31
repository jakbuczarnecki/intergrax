# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Vespa vector store adapter implementing ``VectorStore``."""

from __future__ import annotations

from typing import Any, Optional, Sequence

from langchain_core.documents import Document

from intergrax.integrations.contracts.vector_store import MetadataFilter, VectorStore, VectorStoreHit
from intergrax.integrations.providers.vector_store.vespa.client import VespaRestClient
from intergrax.integrations.providers.vector_store.vespa.config import VespaIntegrationConfig


class VespaVectorStore(VectorStore):
    """Feed/query facade over Vespa document API."""

    def __init__(self, config: VespaIntegrationConfig, client: VespaRestClient) -> None:
        self._config = config
        self._client = client
        self._doc_ids: list[str] = []

    @property
    def rest_client(self) -> VespaRestClient:
        return self._client

    def add_documents(
        self,
        documents: Sequence[Document],
        embeddings: Sequence[Sequence[float]],
        *,
        ids: Optional[Sequence[str]] = None,
    ) -> None:
        for index, doc in enumerate(documents):
            doc_id = ids[index] if ids and index < len(ids) else f"{self._config.tenant_id}-{index}"
            embedding = list(embeddings[index]) if index < len(embeddings) else []
            self._client.feed_document(
                doc_id=doc_id,
                fields={
                    "content": doc.page_content,
                    "metadata": dict(doc.metadata or {}),
                    "embedding": {"values": embedding},
                    "tenant_id": self._config.tenant_id,
                },
            )
            self._doc_ids.append(doc_id)

    def query(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int,
        metadata_filter: Optional[MetadataFilter] = None,
        include_embeddings: bool = False,
    ) -> list[VectorStoreHit]:
        _ = metadata_filter, include_embeddings
        yql = f"select * from sources {self._config.collection} where true"
        rows = self._client.query_yql(yql, hits=top_k, ranking="default")
        hits: list[VectorStoreHit] = []
        for rank, row in enumerate(rows[:top_k], start=1):
            fields = row.get("fields") if isinstance(row, dict) else {}
            if not isinstance(fields, dict):
                fields = {}
            content = str(fields.get("content") or "")
            hits.append(
                VectorStoreHit(
                    id=str(row.get("id") or fields.get("id") or rank),
                    content=content,
                    metadata=dict(fields.get("metadata") or {}),
                    similarity_score=1.0 / float(rank),
                    rank=rank,
                )
            )
        return hits

    def delete(self, ids: Sequence[str]) -> None:
        for doc_id in ids:
            self._client.delete_document(doc_id)
            if doc_id in self._doc_ids:
                self._doc_ids.remove(doc_id)

    def count(self) -> int:
        if self._doc_ids:
            return len(self._doc_ids)
        try:
            return self._client.count_documents()
        except Exception:
            return 0
