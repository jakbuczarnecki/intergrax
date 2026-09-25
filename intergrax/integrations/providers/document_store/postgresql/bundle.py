# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""PostgreSQL document store composition root."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any

from intergrax.integrations.contracts.document_store import ConditionalDocumentStore
from intergrax.integrations.providers.document_store.postgresql.adapter import (
    _PostgreSQLDocumentStore,
)
from intergrax.integrations.providers.document_store.postgresql.client import (
    PostgreSQLDocumentTableClient,
)
from intergrax.integrations.providers.relational_store.postgresql.config import (
    PostgreSQLIntegrationConfig,
)
from intergrax.integrations.providers.relational_store.postgresql.session import (
    PostgreSQLConnectionProvider,
)

POSTGRESQL_DOCUMENT_STORE_PROVIDER_ID = "postgresql"


def _cursor_secret_for_config(
    config: PostgreSQLIntegrationConfig, tenant_schema: str
) -> bytes:
    material = f"{config.connection_string()}:{tenant_schema}".encode("utf-8")
    return hashlib.sha256(material).digest()


def create_postgresql_document_store(
    *,
    tenant_schema: str,
    dsn: str | None = None,
    connection_factory: Callable[[], Any] | None = None,
    config_overrides: dict[str, object] | None = None,
) -> ConditionalDocumentStore:
    """Open a new document-store client bound to one PostgreSQL schema."""
    overrides = dict(config_overrides or {})
    if dsn is not None:
        overrides["dsn"] = dsn
    config = PostgreSQLIntegrationConfig.from_env(**overrides)
    provider = PostgreSQLConnectionProvider(
        config,
        connection_factory=connection_factory,
        tenant_schema=tenant_schema,
    )
    client = PostgreSQLDocumentTableClient(
        provider,
        cursor_secret=_cursor_secret_for_config(config, tenant_schema),
    )
    return _PostgreSQLDocumentStore(client)


__all__ = [
    "POSTGRESQL_DOCUMENT_STORE_PROVIDER_ID",
    "create_postgresql_document_store",
]
