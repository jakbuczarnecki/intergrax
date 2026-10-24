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


def _qualification_endpoint_label(config: PostgreSQLIntegrationConfig) -> str:
    return f"{config.host}:{config.port}/{config.database}"


def require_tool_life_postgresql() -> None:
    """Fail closed when mandatory L3-PHYSICAL PostgreSQL is unreachable (not pytest.skip)."""
    config = resolve_tool_life_postgresql_config()
    psycopg, _, _, _ = import_psycopg()
    target = _qualification_endpoint_label(config)
    try:
        with psycopg.connect(
            config.connection_string(),
            connect_timeout=int(_CONNECT_PROBE_TIMEOUT_S),
        ) as conn:
            conn.execute("SELECT 1")
    except Exception as exc:
        pytest.fail(
            "TOOL-LIFE-X L3-PHYSICAL PostgreSQL required but unavailable "
            f"at {target}: {type(exc).__name__}"
        )


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


@contextmanager
def tool_life_postgresql_dual_tenant_stores(
    *,
    schema_prefix_a: str = "tlx_tenant_a",
    schema_prefix_b: str = "tlx_tenant_b",
) -> Generator[
    tuple[tuple[Any, PostgreSQLIntegrationConfig], tuple[Any, PostgreSQLIntegrationConfig]],
    None,
    None,
]:
    """Two tenant schemas on the same instance, both live for the duration of the block."""
    require_tool_life_postgresql()
    schema_a = f"{schema_prefix_a}_{uuid.uuid4().hex[:10]}"
    schema_b = f"{schema_prefix_b}_{uuid.uuid4().hex[:10]}"
    config_a = resolve_tool_life_postgresql_config(tenant_schema=schema_a)
    config_b = resolve_tool_life_postgresql_config(tenant_schema=schema_b)
    provider_a = PostgreSQLConnectionProvider(config_a, tenant_schema=schema_a)
    provider_b = PostgreSQLConnectionProvider(config_b, tenant_schema=schema_b)
    conn_a = provider_a.open_configured_connection()
    conn_b = provider_b.open_configured_connection()
    provider_a.ensure_schema_exists(PostgreSQLSession(conn_a), schema_a)
    provider_b.ensure_schema_exists(PostgreSQLSession(conn_b), schema_b)
    conn_a.commit()
    conn_b.commit()
    store_a = open_postgresql_relational_store(
        config_a,
        connection_factory=lambda: conn_a,
    ).client
    store_b = open_postgresql_relational_store(
        config_b,
        connection_factory=lambda: conn_b,
    ).client
    assert store_a is not None and store_b is not None
    try:
        yield (store_a, config_a), (store_b, config_b)
    finally:
        store_a.close()
        store_b.close()
        for provider, schema in ((provider_a, schema_a), (provider_b, schema_b)):
            cleanup = provider.open_configured_connection()
            try:
                cleanup.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
                cleanup.commit()
            finally:
                cleanup.close()
