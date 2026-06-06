# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import json
import math
import uuid
from typing import Any, Optional, Sequence

from langchain_core.documents import Document

from intergrax.integrations.contracts.base import HealthStatus, IntegrationConfigurationError
from intergrax.integrations.contracts.health_probe import IntegrationHealthProbe
from intergrax.integrations.providers.vector_store.inmemory.rag_store import InMemoryVectorStore
from intergrax.rag.embedding.contracts.embedding_metadata_key import EmbeddingMetadataKey
from intergrax.rag.vectorstore.contracts.vector_store import MetadataFilter, VectorStoreHit


class PgVectorRagStore(InMemoryVectorStore, IntegrationHealthProbe):
    """
    pgvector-backed vector store facade.

    Without ``INTERGRAX_PGVECTOR_DSN`` uses in-memory cosine index (tests/local).
    With DSN, persists vectors in PostgreSQL (JSONB embeddings; pgvector extension optional).
    """

    _TABLE = "intergrax_pgvector"

    def __init__(self, tenant_id: str, *, dsn: str | None = None) -> None:
        super().__init__(tenant_id)
        self._dsn = (dsn or "").strip() or None
        self._connection: Any | None = None
        if self._dsn is not None:
            self._connection = self._open_connection()
            self._ensure_schema()

    def add_documents(
        self,
        documents: Sequence[Document],
        embeddings: Sequence[Sequence[float]],
        *,
        ids: Optional[Sequence[str]] = None,
    ) -> None:
        if self._connection is None:
            super().add_documents(documents, embeddings, ids=ids)
            return

        if len(documents) == 0:
            return
        if len(documents) != len(embeddings):
            raise ValueError("documents and embeddings length mismatch")

        id_list = list(ids) if ids else [str(uuid.uuid4()) for _ in range(len(documents))]
        with self._connection.cursor() as cursor:
            for index, doc in enumerate(documents):
                payload = dict(doc.metadata or {})
                if EmbeddingMetadataKey.VECTOR in payload:
                    payload.pop(EmbeddingMetadataKey.VECTOR, None)
                payload["tenant_id"] = self._tenant_id
                payload["text"] = doc.page_content or ""
                vector = [float(v) for v in embeddings[index]]
                cursor.execute(
                    f"""
                    INSERT INTO {self._TABLE} (id, tenant_id, embedding, payload, text_content)
                    VALUES (%s, %s, %s::jsonb, %s::jsonb, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        embedding = EXCLUDED.embedding,
                        payload = EXCLUDED.payload,
                        text_content = EXCLUDED.text_content
                    """,
                    (
                        id_list[index],
                        self._tenant_id,
                        json.dumps(vector),
                        json.dumps(payload),
                        doc.page_content or "",
                    ),
                )
            self._connection.commit()

    def query(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int,
        metadata_filter: Optional[MetadataFilter] = None,
        include_embeddings: bool = False,
    ) -> list[VectorStoreHit]:
        if self._connection is None:
            return super().query(
                query_embedding,
                top_k=top_k,
                metadata_filter=metadata_filter,
                include_embeddings=include_embeddings,
            )

        vector = [float(v) for v in query_embedding]
        effective_where: dict[str, Any] = (
            dict(metadata_filter.conditions) if metadata_filter is not None else {}
        )
        effective_where["tenant_id"] = self._tenant_id

        with self._connection.cursor() as cursor:
            cursor.execute(
                f"SELECT id, embedding, payload, text_content FROM {self._TABLE} WHERE tenant_id = %s",
                (self._tenant_id,),
            )
            rows = cursor.fetchall()

        candidates: list[tuple[str, float, dict[str, Any], str, list[float]]] = []
        for row_id, embedding_raw, payload_raw, text_content in rows:
            payload = payload_raw if isinstance(payload_raw, dict) else json.loads(payload_raw)
            emb = embedding_raw if isinstance(embedding_raw, list) else json.loads(embedding_raw)
            if not self._metadata_matches(payload, effective_where):
                continue
            score = self._cosine_similarity(vector, emb)
            candidates.append((str(row_id), score, payload, str(text_content), emb))

        candidates.sort(key=lambda item: item[1], reverse=True)
        hits: list[VectorStoreHit] = []
        for rank, (row_id, score, payload, text, emb) in enumerate(candidates[:top_k]):
            hits.append(
                VectorStoreHit(
                    id=row_id,
                    content=text or str(payload.get("text", "")),
                    metadata=payload,
                    similarity_score=float(score),
                    rank=rank,
                    embedding=emb if include_embeddings else None,
                )
            )
        return hits

    def delete(self, ids: Sequence[str]) -> None:
        if self._connection is None:
            super().delete(ids)
            return
        with self._connection.cursor() as cursor:
            for row_id in ids:
                cursor.execute(
                    f"DELETE FROM {self._TABLE} WHERE id = %s AND tenant_id = %s",
                    (row_id, self._tenant_id),
                )
            self._connection.commit()

    def count(self) -> int:
        if self._connection is None:
            return super().count()
        with self._connection.cursor() as cursor:
            cursor.execute(
                f"SELECT COUNT(*) FROM {self._TABLE} WHERE tenant_id = %s",
                (self._tenant_id,),
            )
            row = cursor.fetchone()
        return int(row[0]) if row else 0

    def list_collections(self) -> list[str]:
        suffix = "pg" if self._connection is not None else "memory"
        return [f"pgvector:{self._tenant_id}:{suffix}"]

    def health(self) -> HealthStatus:
        if self._connection is None:
            return HealthStatus(slug="pgvector", healthy=True, detail="in-memory fallback")
        try:
            with self._connection.cursor() as cursor:
                cursor.execute("SELECT 1")
            return HealthStatus(slug="pgvector", healthy=True, detail=f"tenant={self._tenant_id}")
        except Exception as exc:  # noqa: BLE001 — health probe surface
            return HealthStatus(slug="pgvector", healthy=False, detail=str(exc))

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def _open_connection(self) -> Any:
        try:
            import psycopg
        except ImportError as exc:
            raise IntegrationConfigurationError("pgvector requires psycopg") from exc
        return psycopg.connect(self._dsn)

    def _ensure_schema(self) -> None:
        assert self._connection is not None
        with self._connection.cursor() as cursor:
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._TABLE} (
                    id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    embedding JSONB NOT NULL,
                    payload JSONB NOT NULL,
                    text_content TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{self._TABLE}_tenant ON {self._TABLE} (tenant_id)"
            )
        self._connection.commit()

    @staticmethod
    def _metadata_matches(payload: dict[str, Any], effective_where: dict[str, Any]) -> bool:
        for key, value in effective_where.items():
            if payload.get(key) != value:
                return False
        return True

    @staticmethod
    def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
