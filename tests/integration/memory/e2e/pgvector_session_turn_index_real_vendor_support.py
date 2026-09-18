# © Artur Czarnecki. All rights reserved.

"""MEM-FINAL-AUDIT-5E — pgvector-backed SessionTurnIndex qualification harness."""

from __future__ import annotations

import os
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from urllib.parse import urlparse, urlunparse

import numpy as np
from langchain_core.documents import Document
from numpy.typing import NDArray

from intergrax.applications._shared.session_turn_index_rag_adapters import (
    build_session_turn_index_creation_context,
)
from intergrax.integrations.contracts.base import IntegrationConfigurationError, IntegrationDependencyError
from intergrax.integrations.providers.vector_store.pgvector.bundle import create_pgvector_vector_store
from intergrax.integrations.providers.vector_store.pgvector.integration import (
    PGVECTOR_VECTOR_STORE_PROVIDER_ID,
    PgvectorVectorStoreIntegration,
)
from intergrax.integrations.providers.vector_store.pgvector.rag_store import PgVectorRagStore
from intergrax.memory.contracts.session_turn_index import SessionTurnIndexStore
from intergrax.memory.session_turn_index_service import VectorSessionTurnIndexStore
from intergrax.rag.bootstrap.rag_stack_bootstrap import RagStack, create_default_rag_stack
from intergrax.rag.embedding.contracts.base_embedding_manager import BaseEmbeddingManager
from intergrax.rag.embedding.contracts.embedding_result import EmbeddingResult
from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.rag.vectorstore.contracts.native_vectorstore import VectorStoreScope
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager

DSN_ENV = "INTERGRAX_PGVECTOR_DSN"
DIMENSION_ENV = "INTERGRAX_PGVECTOR_DIMENSION"
_DEFAULT_DIMENSION = 4
_TABLE = PgVectorRagStore._TABLE

CreateSessionTurnIndexStore = Callable[[], SessionTurnIndexStore]
DisposeSessionTurnIndexStore = Callable[[SessionTurnIndexStore], Awaitable[None]]

_QUALIFICATION_TENANTS = (
    "tenant-5e-a",
    "tenant-5e-b",
)


@dataclass(frozen=True, slots=True)
class PgvectorSessionTurnIndexQualificationEnv:
    qualification_run_id: str
    tenant_id: str


class DeterministicQualificationEmbeddingManager(BaseEmbeddingManager):
    """Stable vectors for contract qualification (not embedding-vendor proof)."""

    def __init__(self, *, dimensions: int) -> None:
        self._dimensions = dimensions

    def embed_one(self, text: str) -> list[float]:
        return self.embed_texts([text])[0].tolist()

    def embed_documents(self, documents: Sequence[Document]) -> EmbeddingResult:
        texts = [doc.page_content for doc in documents]
        matrix = self.embed_texts(texts)
        return EmbeddingResult(embeddings=matrix, texts=list(texts))

    def embed_texts(self, texts: Sequence[str]) -> NDArray[np.float32]:
        rows: list[list[float]] = []
        for text in texts:
            seed = sum(ord(ch) for ch in text) % 997
            rows.append(
                [float((seed + index) % 17) / 17.0 for index in range(self._dimensions)]
            )
        return np.array(rows, dtype=np.float32)


def qualification_embedding_dimension() -> int:
    raw = os.environ.get(DIMENSION_ENV, str(_DEFAULT_DIMENSION)).strip() or str(_DEFAULT_DIMENSION)
    return int(raw)


def unique_qualification_run_id(prefix: str = "mem-5e") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:16]}"


def build_qualification_env(
    *,
    qualification_run_id: str,
    tenant_id: str = "qual-tenant",
) -> PgvectorSessionTurnIndexQualificationEnv:
    return PgvectorSessionTurnIndexQualificationEnv(
        qualification_run_id=qualification_run_id,
        tenant_id=tenant_id,
    )


def pgvector_reachable() -> bool:
    return bool(os.environ.get(DSN_ENV, "").strip())


def ensure_pgvector_available() -> None:
    if not pgvector_reachable():
        raise RuntimeError(f"{DSN_ENV} is not configured")
    env = build_qualification_env(qualification_run_id=unique_qualification_run_id("probe"))
    integration = _open_integration(env)
    try:
        health = integration.rag_store.health()
        if not health.healthy:
            raise RuntimeError(health.detail or "pgvector unhealthy")
        _assert_pgvector_extension(integration)
    finally:
        close_pgvector_integration(integration)


def _assert_pgvector_extension(integration: PgvectorVectorStoreIntegration) -> None:
    inner = integration.rag_store
    if not isinstance(inner, PgVectorRagStore):
        return
    connection = inner._connection
    if connection is None:
        raise RuntimeError("pgvector connection not open")
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        )
        row = cursor.fetchone()
    if not row or not row[0]:
        raise RuntimeError("pgvector extension not installed")


def _open_integration(
    env: PgvectorSessionTurnIndexQualificationEnv,
    *,
    tenant_id: str | None = None,
    dsn_override: str | None = None,
) -> PgvectorVectorStoreIntegration:
    resolved_tenant = tenant_id or env.tenant_id
    overrides: dict[str, object] = {
        "tenant_id": resolved_tenant,
        "dimension": qualification_embedding_dimension(),
    }
    if dsn_override is not None:
        overrides["dsn"] = dsn_override
        overrides["connection_string"] = dsn_override
    integration = create_pgvector_vector_store(**overrides)
    assert isinstance(integration, PgvectorVectorStoreIntegration)
    integration.count(scope=VectorStoreScope(tenant_id=resolved_tenant))
    return integration


def build_pgvector_memory_rag_stack(
    env: PgvectorSessionTurnIndexQualificationEnv,
    *,
    tenant_id: str | None = None,
    dsn_override: str | None = None,
) -> tuple[RagStack, PgvectorVectorStoreIntegration]:
    resolved_tenant = tenant_id or env.tenant_id
    integration = _open_integration(
        env,
        tenant_id=resolved_tenant,
        dsn_override=dsn_override,
    )
    embedding = DeterministicQualificationEmbeddingManager(
        dimensions=qualification_embedding_dimension(),
    )
    vectorstore = VectorstoreManager(
        store=integration.rag_store,
        scope=VectorStoreScope(tenant_id=resolved_tenant),
    )
    stack = create_default_rag_stack(
        tenant_id=resolved_tenant,
        vectorstore_manager=vectorstore,
        embedding_manager=embedding,
        profile=RagProfile(),
    )
    return stack, integration


def build_vector_session_turn_index_store(
    env: PgvectorSessionTurnIndexQualificationEnv,
    *,
    tenant_id: str | None = None,
    dsn_override: str | None = None,
) -> tuple[VectorSessionTurnIndexStore, PgvectorVectorStoreIntegration]:
    resolved_tenant = tenant_id or env.tenant_id
    stack, integration = build_pgvector_memory_rag_stack(
        env,
        tenant_id=resolved_tenant,
        dsn_override=dsn_override,
    )
    creation_context = build_session_turn_index_creation_context(
        tenant_id=resolved_tenant,
        embedding_manager=stack.embedding_manager,
        vectorstore_manager=stack.vectorstore_manager,
    )
    store = VectorSessionTurnIndexStore(
        embedding_port=creation_context.embedding_manager,
        vectorstore_port=creation_context.vectorstore_manager,
        tenant_id=resolved_tenant,
    )
    return store, integration


def close_pgvector_integration(integration: PgvectorVectorStoreIntegration) -> None:
    inner = integration.rag_store
    if isinstance(inner, PgVectorRagStore):
        inner.close()


def _qualification_tenant_ids(env: PgvectorSessionTurnIndexQualificationEnv) -> tuple[str, ...]:
    return (env.tenant_id, "qual-tenant", *_QUALIFICATION_TENANTS)


def drop_qualification_tenant_rows(env: PgvectorSessionTurnIndexQualificationEnv) -> None:
    integration = _open_integration(env)
    inner = integration.rag_store
    if not isinstance(inner, PgVectorRagStore):
        close_pgvector_integration(integration)
        return
    connection = inner._connection
    if connection is None:
        close_pgvector_integration(integration)
        return
    tenants = _qualification_tenant_ids(env)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                f"DELETE FROM {_TABLE} WHERE tenant_id = ANY(%s)",
                (list(tenants),),
            )
        connection.commit()
    finally:
        close_pgvector_integration(integration)


def pgvector_topology_metadata(
    integration: PgvectorVectorStoreIntegration,
) -> dict[str, str | None]:
    inner = integration.rag_store
    postgres_version: str | None = None
    pgvector_version: str | None = None
    driver_version: str | None = None
    index_type: str | None = None
    dimension: str | None = None
    metric = "cosine (<=> operator)"
    dsn_host: str | None = None
    database: str | None = None
    if isinstance(inner, PgVectorRagStore):
        dimension = str(inner._dimension)
        dsn = inner._dsn or ""
        if dsn:
            parsed = urlparse(dsn)
            dsn_host = parsed.hostname
            database = (parsed.path or "").lstrip("/") or None
        connection = inner._connection
        if connection is not None:
            try:
                import psycopg

                driver_version = getattr(psycopg, "__version__", None)
                if driver_version is not None:
                    driver_version = str(driver_version)
            except ImportError:
                pass
            try:
                with connection.cursor() as cursor:
                    cursor.execute("SHOW server_version")
                    postgres_version = str(cursor.fetchone()[0])
                    cursor.execute(
                        "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
                    )
                    ext_row = cursor.fetchone()
                    pgvector_version = str(ext_row[0]) if ext_row and ext_row[0] else None
                    cursor.execute(
                        """
                        SELECT am.amname
                        FROM pg_index i
                        JOIN pg_class c ON c.oid = i.indexrelid
                        JOIN pg_am am ON am.oid = c.relam
                        WHERE c.relname LIKE %s
                        LIMIT 1
                        """,
                        (f"{_TABLE}%",),
                    )
                    idx_row = cursor.fetchone()
                    index_type = str(idx_row[0]) if idx_row and idx_row[0] else "none / table scan"
            except Exception:
                pass
    return {
        "backend_provider_id": PGVECTOR_VECTOR_STORE_PROVIDER_ID,
        "postgres_version": postgres_version,
        "pgvector_extension_version": pgvector_version,
        "driver_version": driver_version,
        "table": _TABLE,
        "database": database,
        "host": dsn_host,
        "metric": metric,
        "dimension": dimension,
        "index_access_method": index_type,
        "topology": "single-node",
        "transport": "postgresql-tcp",
        "deployment": "local-container-or-external",
    }


def session_turn_index_store_factory(
    env: PgvectorSessionTurnIndexQualificationEnv,
    *,
    tenant_id: str | None = None,
) -> tuple[CreateSessionTurnIndexStore, DisposeSessionTurnIndexStore]:
    integrations: list[PgvectorVectorStoreIntegration] = []

    def _create() -> SessionTurnIndexStore:
        store, integration = build_vector_session_turn_index_store(env, tenant_id=tenant_id)
        integrations.append(integration)
        return store

    async def _dispose(instance: SessionTurnIndexStore) -> None:
        _ = instance
        while integrations:
            close_pgvector_integration(integrations.pop())

    return _create, _dispose


def assert_pgvector_backend_unavailable(dsn: str) -> None:
    probe_env = build_qualification_env(qualification_run_id=unique_qualification_run_id("down"))
    try:
        _open_integration(probe_env, dsn_override=dsn)
    except (IntegrationConfigurationError, IntegrationDependencyError, ConnectionError, OSError, ValueError):
        return
    except Exception as exc:
        error_type = type(exc)
        if (
            error_type.__module__.startswith(("psycopg", "psycopg2"))
            and error_type.__name__ in {"OperationalError", "InterfaceError"}
        ):
            return
        raise
    raise AssertionError("expected pgvector backend to be unavailable for probe DSN")


def unavailable_probe_dsn() -> str:
    configured = os.environ.get(DSN_ENV, "").strip()
    if not configured:
        return "postgresql://127.0.0.1:1/unavailable"
    parsed = urlparse(configured)
    netloc = parsed.hostname or "127.0.0.1"
    return urlunparse(
        (
            parsed.scheme or "postgresql",
            f"{netloc}:1",
            parsed.path or "/unavailable",
            "",
            parsed.query,
            "",
        )
    )


def pgvector_backend_side_scope_clauses() -> tuple[str, ...]:
    return (
        "tenant_id = %s",
        "namespace IS NOT DISTINCT FROM %s",
        "workspace_id IS NOT DISTINCT FROM %s",
        "payload @> %s::jsonb",
    )
