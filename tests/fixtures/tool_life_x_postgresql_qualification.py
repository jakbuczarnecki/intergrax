# © Artur Czarnecki. All rights reserved.

"""PostgreSQL Docker qualification helpers for TOOL-LIFE-X L3-PHYSICAL."""

from __future__ import annotations

import os
import uuid
from contextlib import contextmanager
from typing import Any, Generator

import pytest

from intergrax.integrations.providers.relational_store.postgresql.config import (
    PostgreSQLIntegrationConfig,
)
from intergrax.integrations.providers.relational_store.postgresql.opens import (
    open_postgresql_relational_store,
)
from intergrax.integrations.providers.relational_store.postgresql.session import (
    PostgreSQLConnectionProvider,
    PostgreSQLSession,
    import_psycopg,
)

_QUAL_DEFAULT_DSN = "postgresql://intergrax:intergrax@localhost:5434/intergrax"
_CONNECT_PROBE_TIMEOUT_S = 3.0


def resolve_tool_life_postgresql_config(
    *,
    tenant_schema: str | None = None,
) -> PostgreSQLIntegrationConfig:
    dsn = os.environ.get("INTERGRAX_POSTGRESQL_DSN", "").strip()
    if dsn:
        return PostgreSQLIntegrationConfig.from_env(tenant_schema=tenant_schema)
    return PostgreSQLIntegrationConfig(
        host=os.environ.get("INTERGRAX_POSTGRESQL_HOST", "localhost"),
        port=int(os.environ.get("INTERGRAX_POSTGRESQL_PORT", "5434")),
        user=os.environ.get("INTERGRAX_POSTGRESQL_USER", "intergrax"),
        password=os.environ.get("INTERGRAX_POSTGRESQL_PASSWORD", "intergrax"),
        database=os.environ.get("INTERGRAX_POSTGRESQL_DATABASE", "intergrax"),
        tenant_schema=tenant_schema,
    )


def require_tool_life_postgresql() -> None:
    psycopg, _, _, _ = import_psycopg()
    try:
        with psycopg.connect(
            _QUAL_DEFAULT_DSN,
            connect_timeout=int(_CONNECT_PROBE_TIMEOUT_S),
        ) as conn:
            conn.execute("SELECT 1")
    except Exception as exc:
        pytest.skip(f"TOOL-LIFE-X L3-PHYSICAL PostgreSQL unavailable: {exc}")


@contextmanager
def tool_life_postgresql_store(
    *,
    schema_prefix: str = "tlx_l3_phys",
) -> Generator[tuple[Any, PostgreSQLIntegrationConfig], None, None]:
    """Open an isolated tenant schema on the qualification PostgreSQL instance."""
    require_tool_life_postgresql()
    schema = f"{schema_prefix}_{uuid.uuid4().hex[:10]}"
    config = resolve_tool_life_postgresql_config(tenant_schema=schema)
    provider = PostgreSQLConnectionProvider(config, tenant_schema=schema)
    connection = provider.open_configured_connection()
    session = PostgreSQLSession(connection)
    provider.ensure_schema_exists(session, schema)
    connection.commit()
    integration = open_postgresql_relational_store(
        config,
        connection_factory=lambda: connection,
    )
    store = integration.client
    assert store is not None
    try:
        yield store, config
    finally:
        store.close()
        cleanup = provider.open_configured_connection()
        try:
            cleanup.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
            cleanup.commit()
        finally:
            cleanup.close()
