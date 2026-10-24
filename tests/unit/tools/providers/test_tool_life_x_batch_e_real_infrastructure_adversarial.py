# © Artur Czarnecki. All rights reserved.

"""TOOL-LIFE-X Batch E — real-socket HTTP and file-backed SQLite adversarial qualification."""

from __future__ import annotations

import socket
import sqlite3
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import pytest

from intergrax.integrations.providers.http_client.allowlist.client import AllowlistHttpClient
from intergrax.integrations.providers.relational_store.sqlite.bundle import (
    create_sqlite_relational_store,
)
from intergrax.integrations.relational_store_execution_adapter import (
    RelationalStoreExecutionAdapter,
)
from intergrax.integrations.contracts.configured_relational_store_execution import (
    RelationalQueryRequest,
)
from intergrax.tools.providers.database.contracts import DatabaseQueryInput
from intergrax.tools.providers.database.service import database_query
from intergrax.tools.providers.http.contracts import HttpRequestInput
from intergrax.tools.providers.http.service import http_request
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = [pytest.mark.unit, pytest.mark.gate]


_FAULT_BEHAVIOR: Callable[[BaseHTTPRequestHandler], None] = lambda _handler: None


class _FaultHandler(BaseHTTPRequestHandler):
    """Configurable local fault-injection HTTP handler (real TCP, no mocks)."""

    def log_message(self, format: str, *args: object) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802
        _FAULT_BEHAVIOR(self)

    def do_POST(self) -> None:  # noqa: N802
        _FAULT_BEHAVIOR(self)


@contextmanager
def _local_fault_server(
    behavior: Callable[[BaseHTTPRequestHandler], None],
) -> tuple[str, frozenset[str]]:
    global _FAULT_BEHAVIOR
    previous = _FAULT_BEHAVIOR
    _FAULT_BEHAVIOR = behavior
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FaultHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    base = f"http://{host}:{port}"
    try:
        yield base, frozenset({host})
    finally:
        server.shutdown()
        thread.join(timeout=3.0)
        _FAULT_BEHAVIOR = previous


def _allowlist_client(allowed_hosts: frozenset[str]) -> AllowlistHttpClient:
    return AllowlistHttpClient(allowed_hosts=allowed_hosts, default_timeout_s=30.0)


def _http_ctx(client: AllowlistHttpClient) -> ToolWiringContext:
    return ToolWiringContext(http_client=client)


def test_real_http_normal_success() -> None:
    def ok(handler: BaseHTTPRequestHandler) -> None:
        handler.send_response(200)
        handler.send_header("Content-Type", "text/plain")
        handler.end_headers()
        handler.wfile.write(b"ok")

    with _local_fault_server(ok) as (base, hosts):
        out = http_request(
            _http_ctx(_allowlist_client(hosts)),
            HttpRequestInput(url=f"{base}/", method="GET", timeout_s=5.0),
        )
    assert out.success is True
    assert out.body == "ok"


def test_real_http_fast_5xx_failure() -> None:
    def fail(handler: BaseHTTPRequestHandler) -> None:
        handler.send_response(503)
        handler.end_headers()
        handler.wfile.write(b"unavailable")

    with _local_fault_server(fail) as (base, hosts):
        out = http_request(
            _http_ctx(_allowlist_client(hosts)),
            HttpRequestInput(url=f"{base}/", method="GET", timeout_s=5.0),
        )
    assert out.success is False
    assert out.status_code == 503


def test_real_http_connection_refused() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    client = _allowlist_client(frozenset({"127.0.0.1"}))
    out = http_request(
        _http_ctx(client),
        HttpRequestInput(url=f"http://127.0.0.1:{port}/", method="GET", timeout_s=1.0),
    )
    assert out.success is False
    assert out.error


def test_real_http_read_timeout() -> None:
    def slow(handler: BaseHTTPRequestHandler) -> None:
        time.sleep(2.5)
        handler.send_response(200)
        handler.end_headers()
        handler.wfile.write(b"late")

    with _local_fault_server(slow) as (base, hosts):
        out = http_request(
            _http_ctx(_allowlist_client(hosts)),
            HttpRequestInput(url=f"{base}/", method="GET", timeout_s=1.0),
        )
    assert out.success is False
    assert out.error


def test_real_http_slow_stream_byte_ceiling() -> None:
    def huge(handler: BaseHTTPRequestHandler) -> None:
        handler.send_response(200)
        handler.send_header("Content-Type", "application/octet-stream")
        handler.end_headers()
        handler.wfile.write(b"x" * 65536)

    with _local_fault_server(huge) as (base, hosts):
        out = http_request(
            _http_ctx(_allowlist_client(hosts)),
            HttpRequestInput(
                url=f"{base}/",
                method="GET",
                timeout_s=5.0,
                max_response_bytes=1024,
            ),
        )
    assert out.success is True
    assert out.truncated is True
    assert len(out.body.encode("utf-8")) <= 1024


def test_real_http_connection_reset_mid_body() -> None:
    def reset(handler: BaseHTTPRequestHandler) -> None:
        handler.send_response(200)
        handler.send_header("Content-Type", "application/octet-stream")
        handler.send_header("Content-Length", "100000")
        handler.end_headers()
        handler.wfile.write(b"x" * 2048)
        handler.request.shutdown(socket.SHUT_RDWR)

    with _local_fault_server(reset) as (base, hosts):
        out = http_request(
            _http_ctx(_allowlist_client(hosts)),
            HttpRequestInput(
                url=f"{base}/",
                method="GET",
                timeout_s=5.0,
                max_response_bytes=65536,
            ),
        )
    assert out.success is False
    assert out.error


def test_real_http_post_transport_error_on_reset() -> None:
    def reset_post(handler: BaseHTTPRequestHandler) -> None:
        length = int(handler.headers.get("Content-Length", "0"))
        if length:
            handler.rfile.read(length)
        handler.request.shutdown(socket.SHUT_RDWR)

    with _local_fault_server(reset_post) as (base, hosts):
        with pytest.raises(RuntimeError, match="http_transport_error"):
            http_request(
                _http_ctx(_allowlist_client(hosts)),
                HttpRequestInput(url=f"{base}/", method="POST", body="{}", timeout_s=5.0),
            )


def test_real_http_429_rate_limited() -> None:
    def limited(handler: BaseHTTPRequestHandler) -> None:
        handler.send_response(429)
        handler.end_headers()
        handler.wfile.write(b"rate")

    with _local_fault_server(limited) as (base, hosts):
        out = http_request(
            _http_ctx(_allowlist_client(hosts)),
            HttpRequestInput(url=f"{base}/", method="GET", timeout_s=5.0),
        )
    assert out.success is False
    assert out.status_code == 429


def test_real_sqlite_file_bounded_huge_result() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        integration = create_sqlite_relational_store(data_dir=Path(tmp))
        integration.connect()
        integration.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, payload TEXT NOT NULL)")
        for i in range(200):
            integration.execute(
                "INSERT INTO t (id, payload) VALUES (?, ?)",
                (i, f"row-{i}"),
            )
        store = integration.client
        assert store is not None
        adapter = RelationalStoreExecutionAdapter(store)
        result = adapter.query(
            RelationalQueryRequest(sql="SELECT * FROM t", max_rows=5, max_result_bytes=500),
        )
        integration.close()
    assert result.physical_bounds_enforced is True
    assert result.row_count <= 5
    assert result.truncated is True


def test_real_sqlite_file_query_via_database_tool() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        integration = create_sqlite_relational_store(data_dir=Path(tmp))
        integration.connect()
        integration.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")
        integration.execute("INSERT INTO t (id) VALUES (1)")
        store = integration.client
        assert store is not None
        ctx = ToolWiringContext(
            relational_store=store,
            relational_store_execution=RelationalStoreExecutionAdapter(store),
        )
        out = database_query(ctx, DatabaseQueryInput(sql="SELECT id FROM t"))
        integration.close()
    assert out.row_count == 1


def test_real_sqlite_lock_contention_busy_timeout() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        integration = create_sqlite_relational_store(data_dir=Path(tmp))
        integration.connect()
        integration.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")
        db_path = integration.db_path
        holder = sqlite3.connect(db_path, timeout=30.0)
        holder.execute("BEGIN IMMEDIATE")
        blocker = sqlite3.connect(db_path, timeout=0.05)
        started = time.monotonic()
        with pytest.raises(sqlite3.OperationalError):
            blocker.execute("BEGIN IMMEDIATE")
        elapsed = time.monotonic() - started
        holder.close()
        blocker.close()
        integration.close()
    assert elapsed < 2.0

