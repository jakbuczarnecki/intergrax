# © Artur Czarnecki. All rights reserved.

"""TOOL-LIFE-X L3-PHYSICAL — real Docker PostgreSQL adapter qualification (TL3/TL10/TL13)."""

from __future__ import annotations

import threading
import time

import pytest

from intergrax.integrations.contracts.configured_relational_store_execution import (
    RelationalQueryRequest,
)
from intergrax.integrations.providers.relational_store.postgresql.session import (
    PostgreSQLConnectionProvider,
    PostgreSQLSession,
    import_psycopg,
    set_local_config,
)
from intergrax.integrations.relational_store_execution_adapter import (
    RelationalStoreExecutionAdapter,
)
from intergrax.tools.providers.database.contracts import DatabaseQueryInput
from intergrax.tools.providers.database.service import database_query
from intergrax.tools.registry.wiring import ToolWiringContext
from tests.fixtures.tool_life_x_postgresql_qualification import (
    require_tool_life_postgresql,
    resolve_tool_life_postgresql_config,
    tool_life_postgresql_store,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gate,
    pytest.mark.network,
]


@pytest.fixture(autouse=True)
def _require_postgresql() -> None:
    require_tool_life_postgresql()


def test_pg_fast_query_via_canonical_database_tool() -> None:
    with tool_life_postgresql_store() as (store, _config):
        store.execute("CREATE TABLE tlx_fast (id INTEGER PRIMARY KEY, v TEXT NOT NULL)")
        store.execute("INSERT INTO tlx_fast (id, v) VALUES (1, 'ok')")
        ctx = ToolWiringContext(
            relational_store=store,
            relational_store_execution=RelationalStoreExecutionAdapter(store),
        )
        out = database_query(ctx, DatabaseQueryInput(sql="SELECT id, v FROM tlx_fast"))
    assert out.row_count == 1
    assert out.rows[0]["v"] == "ok"


def test_pg_statement_timeout_on_slow_query() -> None:
    psycopg, errors, _, _ = import_psycopg()
    with tool_life_postgresql_store() as (store, config):
        conn = store._require_connection()  # qualification: physical session control
        session = PostgreSQLSession(conn)
        set_local_config(session, "statement_timeout", "300")
        with pytest.raises(errors.QueryCanceled):
            store.fetch_all("SELECT pg_sleep(2)")


def test_pg_physical_backend_cancel_during_pg_sleep() -> None:
    psycopg, errors, _, _ = import_psycopg()
    with tool_life_postgresql_store() as (store, _config):
        cancel_error: BaseException | None = None

        def _run_sleep() -> None:
            nonlocal cancel_error
            try:
                store.fetch_all("SELECT pg_sleep(30)")
            except BaseException as exc:
                cancel_error = exc

        worker = threading.Thread(target=_run_sleep, daemon=True)
        worker.start()
        time.sleep(0.3)
        store.cancel_backend()
        worker.join(timeout=10.0)
        assert cancel_error is not None
        assert isinstance(cancel_error, errors.QueryCanceled)


def test_pg_lock_wait_timeout_and_rollback_no_side_effect() -> None:
    psycopg, errors, _, _ = import_psycopg()
    with tool_life_postgresql_store() as (store, config):
        store.execute(
            "CREATE TABLE tlx_lock (id INTEGER PRIMARY KEY, counter INTEGER NOT NULL DEFAULT 0)"
        )
        store.execute("INSERT INTO tlx_lock (id, counter) VALUES (1, 0)")
        holder_provider = PostgreSQLConnectionProvider(config, tenant_schema=config.tenant_schema or "public")
        holder_conn = holder_provider.open_configured_connection()
        holder = PostgreSQLSession(holder_conn)
        holder.execute("BEGIN")
        holder.execute("LOCK TABLE tlx_lock IN ACCESS EXCLUSIVE MODE")
        waiter_provider = PostgreSQLConnectionProvider(config, tenant_schema=config.tenant_schema or "public")
        waiter_conn = waiter_provider.open_configured_connection()
        waiter = PostgreSQLSession(waiter_conn)
        set_local_config(waiter, "lock_timeout", "300")
        with pytest.raises(errors.LockNotAvailable):
            waiter.execute("UPDATE tlx_lock SET counter = counter + 1 WHERE id = 1")
        waiter_conn.rollback()
        holder_conn.rollback()
        holder_conn.close()
        waiter_conn.close()
        rows = store.fetch_all("SELECT counter FROM tlx_lock WHERE id = 1")
    assert rows[0]["counter"] == 0


def test_pg_bounded_query_physical_enforcement() -> None:
    with tool_life_postgresql_store() as (store, _config):
        store.execute("CREATE TABLE tlx_big (id INTEGER PRIMARY KEY, payload TEXT NOT NULL)")
        for i in range(80):
            store.execute(
                "INSERT INTO tlx_big (id, payload) VALUES (%s, %s)",
                (i, f"row-{i}"),
            )
        adapter = RelationalStoreExecutionAdapter(store)
        result = adapter.query(
            RelationalQueryRequest(
                sql="SELECT id, payload FROM tlx_big ORDER BY id",
                max_rows=5,
                max_result_bytes=400,
            ),
        )
    assert result.physical_bounds_enforced is True
    assert result.row_count <= 5
    assert result.truncated is True


def test_pg_connection_acquisition_timeout_unreachable_host() -> None:
    psycopg, _, _, _ = import_psycopg()
    bad = resolve_tool_life_postgresql_config().model_copy(
        update={"host": "127.0.0.1", "port": 1, "dsn": None},
    )
    started = time.monotonic()
    with pytest.raises(Exception):
        psycopg.connect(bad.connection_string(), connect_timeout=1)
    elapsed = time.monotonic() - started
    assert elapsed < 5.0


def test_pg_tenant_schema_isolation_between_stores() -> None:
    with tool_life_postgresql_store(schema_prefix="tlx_tenant_a") as (store_a, _):
        store_a.execute("CREATE TABLE tlx_iso (id INTEGER PRIMARY KEY)")
        store_a.execute("INSERT INTO tlx_iso (id) VALUES (1)")
    with tool_life_postgresql_store(schema_prefix="tlx_tenant_b") as (store_b, config_b):
        provider = PostgreSQLConnectionProvider(
            config_b,
            tenant_schema=config_b.tenant_schema or "public",
        )
        conn = provider.open_configured_connection()
        session = PostgreSQLSession(conn)
        with pytest.raises(Exception):
            session.execute("SELECT * FROM tlx_iso")
        conn.close()


def test_pg_sql_error_rolls_back_mutating_transaction() -> None:
    with tool_life_postgresql_store() as (store, _config):
        store.execute("CREATE TABLE tlx_mut (id INTEGER PRIMARY KEY)")
        conn = store._require_connection()
        conn.execute("BEGIN")
        conn.execute("INSERT INTO tlx_mut (id) VALUES (1)")
        with pytest.raises(Exception):
            conn.execute("INSERT INTO tlx_mut (id) VALUES (1)")
        conn.rollback()
        rows = store.fetch_all("SELECT COUNT(*) AS c FROM tlx_mut")
    assert int(rows[0]["c"]) == 0
