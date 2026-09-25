# © Artur Czarnecki. All rights reserved.

"""PostgreSQL document-store helpers for UCA-6C-R6-R5.9-R4 real-durable proofs."""

from __future__ import annotations

import os
import socket
import uuid
from collections.abc import Callable

from intergrax.integrations.contracts.document_store import ConditionalDocumentStore
from intergrax.integrations.providers.document_store.postgresql.bundle import (
    create_postgresql_document_store,
)
from intergrax.integrations.providers.relational_store.postgresql.config import (
    PostgreSQLIntegrationConfig,
    validate_schema_identifier,
)
from intergrax.integrations.providers.relational_store.postgresql.session import (
    PostgreSQLConnectionProvider,
    import_psycopg,
)

CANONICAL_LOCAL_POSTGRESQL_DSN = (
    "postgresql://intergrax:intergrax@localhost:5434/intergrax"
)
SCHEMA_PREFIX = "uca6c_r59_r4_doc_"


def resolve_uca6c_postgresql_document_dsn() -> str:
    for env_name in (
        "INTERGRAX_POSTGRESQL_DSN",
        "INTERGRAX_AUTONOMOUS_WORK_POSTGRESQL_DSN",
    ):
        configured = os.environ.get(env_name, "").strip()
        if configured:
            return configured
    if _canonical_docker_reachable():
        return CANONICAL_LOCAL_POSTGRESQL_DSN
    return CANONICAL_LOCAL_POSTGRESQL_DSN


def _canonical_docker_reachable() -> bool:
    try:
        with socket.create_connection(("localhost", 5434), timeout=1.0):
            return True
    except OSError:
        return False


def open_isolated_postgresql_document_store() -> tuple[
    str, str, ConditionalDocumentStore
]:
    dsn = resolve_uca6c_postgresql_document_dsn()
    schema_name = f"{SCHEMA_PREFIX}{uuid.uuid4().hex}"
    validate_schema_identifier(schema_name)
    store = create_postgresql_document_store(tenant_schema=schema_name, dsn=dsn)
    return schema_name, dsn, store


def fresh_postgresql_document_store_client(
    *,
    tenant_schema: str,
    dsn: str,
) -> ConditionalDocumentStore:
    return create_postgresql_document_store(tenant_schema=tenant_schema, dsn=dsn)


def drop_postgresql_document_schema(schema_name: str, *, dsn: str) -> None:
    schema = validate_schema_identifier(schema_name)
    config = PostgreSQLIntegrationConfig(dsn=dsn)
    provider = PostgreSQLConnectionProvider(config, tenant_schema="public")
    _, _, _, sql = import_psycopg()
    with provider.connection() as session:
        statement = sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
            sql.Identifier(schema)
        )
        session.execute_statement(statement)
        session.commit()


def postgresql_document_store_factory(
    *,
    tenant_schema: str,
    dsn: str,
) -> Callable[[], ConditionalDocumentStore]:
    def _open() -> ConditionalDocumentStore:
        return fresh_postgresql_document_store_client(
            tenant_schema=tenant_schema,
            dsn=dsn,
        )

    return _open


__all__ = [
    "CANONICAL_LOCAL_POSTGRESQL_DSN",
    "drop_postgresql_document_schema",
    "fresh_postgresql_document_store_client",
    "open_isolated_postgresql_document_store",
    "postgresql_document_store_factory",
    "resolve_uca6c_postgresql_document_dsn",
]
