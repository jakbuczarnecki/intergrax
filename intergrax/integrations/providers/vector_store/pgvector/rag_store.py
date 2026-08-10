# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from intergrax.integrations.contracts.base import (
    HealthStatus,
    IntegrationConfigurationError,
    IntegrationDependencyError,
)
from intergrax.integrations.contracts.health_probe import IntegrationHealthProbe
from intergrax.rag.vectorstore.contracts.native_vectorstore import (
    MetadataFilter,
    VectorStoreContractError,
    VectorStoreHit,
    VectorStoreRecord,
    VectorStoreScope,
)
from intergrax.rag.vectorstore.contracts.vector_store import VectorStore
from intergrax.rag.vectorstore.providers.native_provider_boundary import (
    effective_filter,
    native_hit,
    provider_metadata,
    validate_query,
    validate_records,
    validate_scope,
)
from intergrax.knowledge.contracts.validation import require_non_empty_str


class PgVectorRagStore(VectorStore, IntegrationHealthProbe):
    """
    Native PostgreSQL + pgvector vector store.

    A real PgVector store is fail-closed: it requires a DSN, an explicit
    embedding dimension, psycopg, the Python pgvector adapter, and the
    PostgreSQL ``vector`` extension.
    """

    _TABLE = "intergrax_pgvector"

    def __init__(
        self,
        tenant_id: str,
        *,
        dsn: str | None = None,
        dimension: int | None = None,
    ) -> None:
        self._tenant_id = require_non_empty_str(tenant_id, field_name="tenant_id")
        self._dsn = (dsn or "").strip() or None
        self._connection: Any | None = None
        self._register_vector: Any | None = None
        if self._dsn is None:
            raise IntegrationConfigurationError(
                "pgvector requires INTERGRAX_PGVECTOR_DSN or connection_string; "
                "use the explicit inmemory provider for local memory storage"
            )
        self._dimension = self._validate_dimension(dimension)
        self._connection = self._open_connection()
        try:
            self._ensure_schema()
        except IntegrationConfigurationError:
            self.close()
            raise
        except IntegrationDependencyError:
            self.close()
            raise
        except Exception as exc:  # noqa: BLE001 — provider boundary
            self.close()
            raise IntegrationDependencyError(
                "pgvector schema preparation failed"
            ) from exc

    def add_records(
        self,
        records: Sequence[VectorStoreRecord],
        *,
        scope: VectorStoreScope,
    ) -> Sequence[str]:
        validated = validate_records(records, scope=scope, tenant_id=self._tenant_id)
        self._validate_record_dimensions(validated)
        id_list = [record.vector_id for record in validated]
        connection = self._require_connection()
        try:
            with connection.cursor() as cursor:
                for record in validated:
                    payload = provider_metadata(record.document, scope=scope)
                    cursor.execute(
                        f"""
                        INSERT INTO {self._TABLE} (
                            logical_id, tenant_id, namespace, workspace_id,
                            source_id, embedding, payload, text_content
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s)
                        ON CONFLICT (tenant_id, namespace, workspace_id, logical_id)
                        DO UPDATE SET
                            source_id = EXCLUDED.source_id,
                            embedding = EXCLUDED.embedding,
                            payload = EXCLUDED.payload,
                            text_content = EXCLUDED.text_content
                        """,
                        (
                            record.vector_id,
                            self._tenant_id,
                            scope.namespace,
                            scope.workspace_id,
                            record.document.provenance.source_id,
                            record.embedding.tolist(),
                            json.dumps(payload),
                            record.document.content,
                        ),
                    )
            connection.commit()
        except Exception as exc:  # noqa: BLE001 — provider boundary
            self._rollback(connection)
            raise IntegrationDependencyError("pgvector insert/upsert failed") from exc
        return id_list

    def query(
        self,
        query_embedding: Sequence[float],
        *,
        scope: VectorStoreScope,
        top_k: int,
        metadata_filter: MetadataFilter | None = None,
        include_embeddings: bool = False,
    ) -> list[VectorStoreHit]:
        vector, limit = validate_query(query_embedding, top_k=top_k)
        self._validate_vector_dimension(vector.size, field_name="query_embedding")
        validate_scope(scope, tenant_id=self._tenant_id)
        where_sql, where_params = self._where_clause(scope, metadata_filter)
        embedding_sql = ", embedding" if include_embeddings else ""
        connection = self._require_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT logical_id,
                           COALESCE(
                               GREATEST(
                                   0.0,
                                   LEAST(1.0, 1.0 - (embedding <=> %s::vector))
                               ),
                               0.0
                           ) AS similarity_score,
                           payload,
                           text_content
                           {embedding_sql}
                    FROM {self._TABLE}
                    WHERE {where_sql}
                    ORDER BY embedding <=> %s::vector ASC NULLS LAST, logical_id
                    LIMIT %s
                    """,
                    (vector.tolist(), *where_params, vector.tolist(), limit),
                )
                rows = cursor.fetchall()
        except Exception as exc:  # noqa: BLE001 — provider boundary
            self._rollback(connection)
            raise IntegrationDependencyError("pgvector query failed") from exc

        hits: list[VectorStoreHit] = []
        for rank, row in enumerate(rows):
            row_id, score, payload, text = row[:4]
            embedding = (
                self._embedding_values(row[4])
                if include_embeddings
                else None
            )
            hits.append(
                native_hit(
                    vector_id=str(row_id),
                    content=text,
                    metadata=self._payload_mapping(payload),
                    similarity_score=float(score),
                    rank=rank,
                    scope=scope,
                    embedding=embedding,
                )
            )
        return hits

    def delete(self, ids: Sequence[str], *, scope: VectorStoreScope) -> None:
        logical_ids = list(ids)
        if not logical_ids:
            return
        validate_scope(scope, tenant_id=self._tenant_id)
        connection = self._require_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    DELETE FROM {self._TABLE}
                    WHERE logical_id = ANY(%s)
                      AND tenant_id = %s
                      AND namespace IS NOT DISTINCT FROM %s
                      AND workspace_id IS NOT DISTINCT FROM %s
                    """,
                    (
                        logical_ids,
                        self._tenant_id,
                        scope.namespace,
                        scope.workspace_id,
                    ),
                )
            connection.commit()
        except Exception as exc:  # noqa: BLE001 — provider boundary
            self._rollback(connection)
            raise IntegrationDependencyError("pgvector scoped delete failed") from exc

    def list_source_record_ids(
        self,
        *,
        source_id: str,
        scope: VectorStoreScope,
    ) -> Sequence[str]:
        canonical_source_id = require_non_empty_str(source_id, field_name="source_id")
        validate_scope(scope, tenant_id=self._tenant_id)
        connection = self._require_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT logical_id FROM {self._TABLE}
                    WHERE tenant_id = %s
                      AND namespace IS NOT DISTINCT FROM %s
                      AND workspace_id IS NOT DISTINCT FROM %s
                      AND source_id = %s
                    ORDER BY logical_id
                    """,
                    (
                        self._tenant_id,
                        scope.namespace,
                        scope.workspace_id,
                        canonical_source_id,
                    ),
                )
                rows = cursor.fetchall()
        except Exception as exc:  # noqa: BLE001 — provider boundary
            self._rollback(connection)
            raise IntegrationDependencyError(
                "pgvector source ownership lookup failed"
            ) from exc
        return [str(row[0]) for row in rows]

    def count(self, *, scope: VectorStoreScope) -> int:
        validate_scope(scope, tenant_id=self._tenant_id)
        connection = self._require_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT COUNT(*) FROM {self._TABLE}
                    WHERE tenant_id = %s
                      AND namespace IS NOT DISTINCT FROM %s
                      AND workspace_id IS NOT DISTINCT FROM %s
                    """,
                    (self._tenant_id, scope.namespace, scope.workspace_id),
                )
                row = cursor.fetchone()
        except Exception as exc:  # noqa: BLE001 — provider boundary
            self._rollback(connection)
            raise IntegrationDependencyError("pgvector count failed") from exc
        return int(row[0]) if row else 0

    def list_collections(self) -> list[str]:
        return [f"pgvector:{self._tenant_id}:native"]

    def health(self) -> HealthStatus:
        if self._connection is None:
            return HealthStatus(
                slug="pgvector",
                healthy=False,
                detail="provider is closed or not configured",
            )
        try:
            with self._connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.execute(
                    "SELECT EXISTS ("
                    "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
                    ")"
                )
                extension_available = bool(cursor.fetchone()[0])
            if not extension_available:
                return HealthStatus(
                    slug="pgvector",
                    healthy=False,
                    detail="PostgreSQL reachable; pgvector extension unavailable",
                )
            return HealthStatus(
                slug="pgvector",
                healthy=True,
                detail=f"PostgreSQL + pgvector ready; tenant={self._tenant_id}",
            )
        except Exception as exc:  # noqa: BLE001 — health probe surface
            return HealthStatus(slug="pgvector", healthy=False, detail=str(exc))

    def close(self) -> None:
        if self._connection is not None:
            connection = self._connection
            self._connection = None
            try:
                connection.close()
            except Exception:
                pass

    def _open_connection(self) -> Any:
        try:
            import psycopg
            from pgvector.psycopg import register_vector
        except ImportError as exc:
            raise IntegrationDependencyError(
                "pgvector requires the 'integrations-pgvector' extra "
                "(psycopg[binary] and pgvector)"
            ) from exc
        connection = None
        try:
            connection = psycopg.connect(self._dsn)
            self._register_vector = register_vector
            return connection
        except Exception as exc:  # noqa: BLE001 — provider boundary
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    pass
            raise IntegrationDependencyError(
                "pgvector could not connect to PostgreSQL or register its vector adapter"
            ) from exc

    def _ensure_schema(self) -> None:
        connection = self._require_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
                if self._register_vector is None:
                    raise IntegrationDependencyError(
                        "pgvector Python adapter is not initialized"
                    )
                self._register_vector(connection)
                cursor.execute(
                    """
                    SELECT column_name, udt_name
                    FROM information_schema.columns
                    WHERE table_schema = current_schema()
                      AND table_name = %s
                    """,
                    (self._TABLE,),
                )
                columns = {str(name): str(udt) for name, udt in cursor.fetchall()}
                if columns:
                    self._validate_existing_schema(cursor, columns)
                else:
                    cursor.execute(
                        f"""
                        CREATE TABLE {self._TABLE} (
                            row_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                            logical_id TEXT NOT NULL,
                            tenant_id TEXT NOT NULL,
                            namespace TEXT,
                            workspace_id TEXT,
                            source_id TEXT NOT NULL,
                            embedding vector({self._dimension}) NOT NULL,
                            payload JSONB NOT NULL,
                            text_content TEXT NOT NULL,
                            CONSTRAINT uq_{self._TABLE}_logical_scope
                                UNIQUE NULLS NOT DISTINCT (
                                    tenant_id, namespace, workspace_id, logical_id
                                )
                        )
                        """
                    )
                cursor.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_{self._TABLE}_scope
                    ON {self._TABLE} (tenant_id, namespace, workspace_id)
                    """
                )
                cursor.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_{self._TABLE}_source
                    ON {self._TABLE} (
                        tenant_id, namespace, workspace_id, source_id
                    )
                    """
                )
            connection.commit()
        except IntegrationConfigurationError:
            self._rollback(connection)
            raise
        except Exception as exc:  # noqa: BLE001 — provider boundary
            self._rollback(connection)
            raise IntegrationDependencyError(
                "pgvector extension or schema preparation failed"
            ) from exc

    def _validate_existing_schema(
        self,
        cursor: Any,
        columns: Mapping[str, str],
    ) -> None:
        required = {
            "row_id",
            "logical_id",
            "tenant_id",
            "namespace",
            "workspace_id",
            "source_id",
            "embedding",
            "payload",
            "text_content",
        }
        if not required.issubset(columns):
            raise IntegrationConfigurationError(
                f"existing {self._TABLE} schema is incompatible with native pgvector; "
                "refusing automatic migration or data loss"
            )
        if columns["embedding"] != "vector":
            raise IntegrationConfigurationError(
                f"existing {self._TABLE}.embedding is {columns['embedding']}, "
                "not a native pgvector vector; refusing automatic migration"
            )
        cursor.execute(
            """
            SELECT format_type(a.atttypid, a.atttypmod)
            FROM pg_attribute AS a
            JOIN pg_class AS c ON c.oid = a.attrelid
            WHERE c.relname = %s
              AND a.attname = 'embedding'
              AND a.attnum > 0
              AND NOT a.attisdropped
            """,
            (self._TABLE,),
        )
        row = cursor.fetchone()
        actual_type = str(row[0]) if row else ""
        expected_type = f"vector({self._dimension})"
        if actual_type != expected_type:
            raise IntegrationConfigurationError(
                f"existing {self._TABLE}.embedding dimension is {actual_type!r}; "
                f"configured dimension is {self._dimension}; refusing automatic migration"
            )

    def _where_clause(
        self,
        scope: VectorStoreScope,
        metadata_filter: MetadataFilter | None,
    ) -> tuple[str, list[Any]]:
        effective = effective_filter(scope, metadata_filter)
        clauses = [
            "tenant_id = %s",
            "namespace IS NOT DISTINCT FROM %s",
            "workspace_id IS NOT DISTINCT FROM %s",
        ]
        params: list[Any] = [
            self._tenant_id,
            scope.namespace,
            scope.workspace_id,
        ]
        routing_keys = {"tenant_id", "namespace", "workspace_id"}
        for key, value in effective.conditions.items():
            if key in routing_keys:
                continue
            clauses.append("payload @> %s::jsonb")
            params.append(json.dumps({key: value}))
        return " AND ".join(clauses), params

    def _require_connection(self) -> Any:
        if self._connection is None:
            raise IntegrationConfigurationError("pgvector provider is closed")
        return self._connection

    @staticmethod
    def _rollback(connection: Any) -> None:
        try:
            connection.rollback()
        except Exception:
            pass

    @staticmethod
    def _payload_mapping(payload: object) -> Mapping[str, object]:
        if not isinstance(payload, Mapping):
            raise VectorStoreContractError("pgvector payload must be a JSON object")
        return payload

    @staticmethod
    def _embedding_values(value: object) -> list[float]:
        to_list = getattr(value, "to_list", None)
        if callable(to_list):
            return list(to_list())
        try:
            return list(value)  # type: ignore[arg-type]
        except TypeError as exc:
            raise VectorStoreContractError(
                "pgvector result embedding is not a vector sequence"
            ) from exc

    def _validate_record_dimensions(
        self,
        records: Sequence[VectorStoreRecord],
    ) -> None:
        for record in records:
            self._validate_vector_dimension(record.embedding.size, field_name="embedding")

    def _validate_vector_dimension(self, value: int, *, field_name: str) -> None:
        if value != self._dimension:
            raise IntegrationConfigurationError(
                f"pgvector {field_name} dimension {value} does not match "
                f"configured dimension {self._dimension}"
            )

    @staticmethod
    def _validate_dimension(value: int | None, *, field_name: str = "dimension") -> int:
        if type(value) is not int or value <= 0:
            raise IntegrationConfigurationError(
                f"pgvector requires a positive integer {field_name}; "
                "set INTERGRAX_PGVECTOR_DIMENSION explicitly"
            )
        return value
