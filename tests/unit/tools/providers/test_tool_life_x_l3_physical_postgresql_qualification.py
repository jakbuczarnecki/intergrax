# © Artur Czarnecki. All rights reserved.

"""TOOL-LIFE-X L3-PHYSICAL — real Docker PostgreSQL adapter qualification (TL3/TL10/TL13)."""

from __future__ import annotations

import json
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
from intergrax.integrations.relational_bounded_row_accumulation import (
    RelationalResultByteLimitExceeded,
)
from intergrax.integrations.relational_store_execution_adapter import (
    RelationalStoreExecutionAdapter,
)
from intergrax.integrations.providers.relational_store.postgresql.adapter import (
    _PostgreSQLRelationalStore,
)
from intergrax.tools.providers.database.contracts import DatabaseQueryInput
from intergrax.tools.providers.database.service import database_query
from intergrax.tools.registry.wiring import ToolWiringContext
from tests.fixtures.tool_life_x_postgresql_qualification import (
    require_tool_life_postgresql,
    resolve_tool_life_postgresql_config,
    tool_life_postgresql_dual_tenant_stores,
    tool_life_postgresql_store,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gate,
    pytest.mark.network,
]

# Shared qualification role can read any schema when SQL is fully qualified.
# Isolation under test is search_path / tenant-scoped session configuration, not DB ACL.
_QUALIFIED_CROSS_SCHEMA_READS_EXPECTED_WITH_SHARED_ROLE = True


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


def test_pg_bounded_query_stops_remote_work_early() -> None:
    """Server-side cursor + itersize=1 must not evaluate the full slow series."""
    with tool_life_postgresql_store() as (store, _config):
        started = time.monotonic()
        rows, truncated, reason = store.fetch_query_bounded(
            "SELECT i FROM generate_series(1, 40) AS s(i) "
            "WHERE pg_sleep(0.08) IS NOT NULL",
            max_rows=3,
            max_result_bytes=1_000_000,
        )
        elapsed = time.monotonic() - started
    assert truncated is True
    assert reason == "row_limit"
    assert len(rows) == 3
    assert elapsed < 2.5


def test_pg_bounded_query_byte_limit_before_row_limit() -> None:
    with tool_life_postgresql_store() as (store, _config):
        rows, truncated, reason = store.fetch_query_bounded(
            "SELECT repeat('ż', 40) AS payload FROM generate_series(1, 20) g(i)",
            max_rows=20,
            max_result_bytes=500,
        )
    assert truncated is True
    assert reason == "byte_limit"
    assert len(rows) >= 1
    assert (
        len(json.dumps([dict(r) for r in rows], ensure_ascii=False).encode("utf-8"))
        <= 500
    )


def test_pg_bounded_query_single_row_byte_limit_raises() -> None:
    with tool_life_postgresql_store() as (store, _config):
        with pytest.raises(RelationalResultByteLimitExceeded):
            store.fetch_query_bounded(
                "SELECT repeat('x', 8000) AS payload",
                max_rows=10,
                max_result_bytes=100,
            )


def test_pg_bounded_query_empty_result() -> None:
    with tool_life_postgresql_store() as (store, _config):
        store.execute("CREATE TABLE tlx_empty (id INTEGER PRIMARY KEY)")
        rows, truncated, reason = store.fetch_query_bounded(
            "SELECT id FROM tlx_empty",
            max_rows=5,
            max_result_bytes=500,
        )
    assert rows == ()
    assert truncated is False
    assert reason == ""


def test_pg_bounded_query_exact_row_boundary() -> None:
    with tool_life_postgresql_store() as (store, _config):
        rows, truncated, reason = store.fetch_query_bounded(
            "SELECT i AS id FROM generate_series(1, 4) g(i)",
            max_rows=4,
            max_result_bytes=50_000,
        )
    assert len(rows) == 4
    assert truncated is False
    assert reason == ""


def test_pg_bounded_query_cancel_during_streamed_fetch() -> None:
    psycopg, errors, _, _ = import_psycopg()
    with tool_life_postgresql_store() as (store, _config):
        cancel_error: BaseException | None = None

        def _run_bounded() -> None:
            nonlocal cancel_error
            try:
                store.fetch_query_bounded(
                    "SELECT i FROM generate_series(1, 500) AS s(i) "
                    "WHERE pg_sleep(0.05) IS NOT NULL",
                    max_rows=500,
                    max_result_bytes=10_000_000,
                )
            except BaseException as exc:
                cancel_error = exc

        worker = threading.Thread(target=_run_bounded, daemon=True)
        worker.start()
        time.sleep(0.35)
        store.cancel_backend()
        worker.join(timeout=15.0)
        assert cancel_error is not None
        assert isinstance(cancel_error, errors.QueryCanceled)


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


def test_pg_tenant_schema_isolation_coexistent_tenants() -> None:
    with tool_life_postgresql_dual_tenant_stores() as (
        (store_a, config_a),
        (store_b, config_b),
    ):
        schema_a = config_a.tenant_schema or "public"
        schema_b = config_b.tenant_schema or "public"
        store_a.execute("CREATE TABLE tlx_iso (id INTEGER PRIMARY KEY, tag TEXT NOT NULL)")
        store_b.execute("CREATE TABLE tlx_iso (id INTEGER PRIMARY KEY, tag TEXT NOT NULL)")
        store_a.execute("INSERT INTO tlx_iso (id, tag) VALUES (1, 'tenant-a')")
        store_b.execute("INSERT INTO tlx_iso (id, tag) VALUES (2, 'tenant-b')")

        rows_a = store_a.fetch_all("SELECT id, tag FROM tlx_iso ORDER BY id")
        rows_b = store_b.fetch_all("SELECT id, tag FROM tlx_iso ORDER BY id")
        assert rows_a == [{"id": 1, "tag": "tenant-a"}]
        assert rows_b == [{"id": 2, "tag": "tenant-b"}]
        assert store_a.fetch_all("SELECT id FROM tlx_iso WHERE id = 2") == []
        assert store_b.fetch_all("SELECT id FROM tlx_iso WHERE id = 1") == []

        cross = store_a.fetch_all(f'SELECT id, tag FROM "{schema_b}".tlx_iso ORDER BY id')
        if _QUALIFIED_CROSS_SCHEMA_READS_EXPECTED_WITH_SHARED_ROLE:
            assert cross == [{"id": 2, "tag": "tenant-b"}]
        else:
            assert cross == []

        reverse = store_b.fetch_all(f'SELECT id, tag FROM "{schema_a}".tlx_iso ORDER BY id')
        if _QUALIFIED_CROSS_SCHEMA_READS_EXPECTED_WITH_SHARED_ROLE:
            assert reverse == [{"id": 1, "tag": "tenant-a"}]
        else:
            assert reverse == []


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


def test_pg_bounded_fetch_does_not_commit_caller_owned_write_transaction() -> None:
    with tool_life_postgresql_store() as (store, _config):
        store.execute("CREATE TABLE tlx_tx_own (id INTEGER PRIMARY KEY)")
        conn = store._require_connection()
        conn.execute("BEGIN")
        conn.execute("INSERT INTO tlx_tx_own (id) VALUES (1)")
        rows, truncated, reason = store.fetch_query_bounded(
            "SELECT id FROM tlx_tx_own ORDER BY id",
            max_rows=10,
            max_result_bytes=5_000,
        )
        assert rows == ({"id": 1},)
        assert truncated is False
        conn.execute("ROLLBACK")
        count_rows = store.fetch_all("SELECT COUNT(*) AS c FROM tlx_tx_own")
    assert int(count_rows[0]["c"]) == 0


def test_pg_bounded_fetch_inside_caller_read_only_transaction() -> None:
    with tool_life_postgresql_store() as (store, _config):
        conn = store._require_connection()
        conn.execute("BEGIN READ ONLY")
        rows, truncated, reason = store.fetch_query_bounded(
            "SELECT 1 AS v",
            max_rows=1,
            max_result_bytes=500,
        )
        conn.rollback()
    assert rows == ({"v": 1},)
    assert truncated is False
    assert reason == ""


def test_pg_bounded_fetch_uses_named_server_side_cursor() -> None:
    """Qualification guard: bounded fetch must open a named (server-side) cursor."""
    with tool_life_postgresql_store() as (store, config):
        conn = store._require_connection()
        opened_names: list[str] = []
        original_cursor = conn.cursor

        def _tracking_cursor(name: str | None = None, *args: object, **kwargs: object):
            if name is not None:
                opened_names.append(name)
            return original_cursor(name, *args, **kwargs)

        conn.cursor = _tracking_cursor  # type: ignore[method-assign]
        bounded_store = _PostgreSQLRelationalStore(config, conn)
        bounded_store.fetch_query_bounded(
            "SELECT 1 AS v",
            max_rows=1,
            max_result_bytes=500,
        )
    assert any(name.startswith("igx_bq_") for name in opened_names)
