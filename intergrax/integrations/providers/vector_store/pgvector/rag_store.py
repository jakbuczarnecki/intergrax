# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import json
import math
from typing import Any, Optional, Sequence

from intergrax.integrations.contracts.base import HealthStatus, IntegrationConfigurationError
from intergrax.integrations.contracts.health_probe import IntegrationHealthProbe
from intergrax.integrations.providers.vector_store.inmemory.rag_store import InMemoryVectorStore
from intergrax.rag.vectorstore.contracts.native_vectorstore import (
    MetadataFilter,
    VectorStoreHit,
    VectorStoreRecord,
    VectorStoreScope,
)
from intergrax.rag.vectorstore.providers.native_provider_boundary import (
    native_hit,
    provider_metadata,
    validate_query,
    validate_records,
    validate_scope,
)


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

    def add_records(
        self,
        records: Sequence[VectorStoreRecord],
        *,
        scope: VectorStoreScope,
    ) -> Sequence[str]:
        if self._connection is None:
            return super().add_records(records, scope=scope)

        validated = validate_records(records, scope=scope, tenant_id=self._tenant_id)
        id_list = [record.vector_id for record in validated]
        with self._connection.cursor() as cursor:
            for record in validated:
                payload = provider_metadata(record.document, scope=scope)
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
                        record.vector_id,
                        self._tenant_id,
                        json.dumps(record.embedding.tolist()),
                        json.dumps(payload),
                        record.document.content,
                    ),
                )
                self._payloads[record.vector_id] = dict(payload)
                self._documents[record.vector_id] = record.document
            self._connection.commit()
        return id_list

    def query(
        self,
        query_embedding: Sequence[float],
        *,
        scope: VectorStoreScope,
        top_k: int,
        metadata_filter: Optional[MetadataFilter] = None,
        include_embeddings: bool = False,
    ) -> list[VectorStoreHit]:
        if self._connection is None:
            return super().query(
                query_embedding,
                scope=scope,
                top_k=top_k,
                metadata_filter=metadata_filter,
                include_embeddings=include_embeddings,
            )

        vector, limit = validate_query(query_embedding, top_k=top_k)
        validate_scope(scope, tenant_id=self._tenant_id)
        effective_where = dict(
            MetadataFilter.for_scope(scope, metadata_filter).conditions
        )

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
        for rank, (row_id, score, payload, text, emb) in enumerate(candidates[:limit]):
            hits.append(
                native_hit(
                    vector_id=row_id,
                    content=text,
                    metadata=payload,
                    similarity_score=float(score),
                    rank=rank,
                    scope=scope,
                    embedding=emb if include_embeddings else None,
                )
            )
        return hits

    def delete(self, ids: Sequence[str], *, scope: VectorStoreScope) -> None:
        if self._connection is None:
            super().delete(ids, scope=scope)
            return
        validate_scope(scope, tenant_id=self._tenant_id)
        with self._connection.cursor() as cursor:
            for row_id in ids:
                cursor.execute(
                    f"""
                    DELETE FROM {self._TABLE}
                    WHERE id = %s AND tenant_id = %s
                      AND payload->>'namespace' IS NOT DISTINCT FROM %s
                      AND payload->>'workspace_id' IS NOT DISTINCT FROM %s
                    """,
                    (row_id, self._tenant_id, scope.namespace, scope.workspace_id),
                )
            self._connection.commit()

    def count(self, *, scope: VectorStoreScope) -> int:
        if self._connection is None:
            return super().count(scope=scope)
        validate_scope(scope, tenant_id=self._tenant_id)
        with self._connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT COUNT(*) FROM {self._TABLE}
                WHERE tenant_id = %s
                  AND payload->>'namespace' IS NOT DISTINCT FROM %s
                  AND payload->>'workspace_id' IS NOT DISTINCT FROM %s
                """,
                (self._tenant_id, scope.namespace, scope.workspace_id),
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
