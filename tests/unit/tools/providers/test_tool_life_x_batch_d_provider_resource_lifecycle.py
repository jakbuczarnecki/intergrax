# © Artur Czarnecki. All rights reserved.

"""TOOL-LIFE-X Batch D — SQL/HTTP/browser/MCP provider resource & lifecycle bounds."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

import httpx
import pytest

from intergrax.integrations.contracts.browser_automation import PageContent
from intergrax.integrations.contracts.configured_relational_store_execution import (
    RelationalQueryRequest,
)
from intergrax.integrations.contracts.relational_store import RelationalStore
from intergrax.integrations.contracts.browser_automation_navigation import (
    BrowserAutomationNavigationTimeoutPort,
)
from intergrax.integrations.contracts.http_client import HttpResponse
from intergrax.integrations.contracts.relational_store_bounded_query import (
    RelationalStoreBoundedQueryPort,
)
from intergrax.integrations.relational_bounded_row_accumulation import accumulate_bounded_sql_rows
from intergrax.integrations.relational_store_execution_adapter import RelationalStoreExecutionAdapter
from intergrax.integrations.providers.http_client.allowlist.client import AllowlistHttpClient
from intergrax.integrations.providers.browser_automation.playwright.integration import (
    PlaywrightBrowserAutomationIntegration,
)
from intergrax.tools.mcp_result_bounds import (
    McpPayloadByteLimitExceeded,
    bound_mcp_structured_result,
    emit_bounded_mcp_list,
    serialize_bounded_mcp_json,
)
from intergrax.tools.providers.browser.contracts import BrowserFetchPageInput
from intergrax.tools.providers.browser.service import browser_fetch_page
from intergrax.tools.providers.database.contracts import DatabaseQueryInput
from intergrax.tools.providers.database.service import database_query
from intergrax.tools.providers.http.contracts import HttpRequestInput
from intergrax.tools.providers.http.service import http_request
from intergrax.tools.providers.provider_resource_enforcement import (
    ProviderResourceLimitExceeded,
    bound_database_query_rows,
)
from intergrax.tools.providers.provider_transport_capabilities import (
    BROWSER_AUTOMATION_CAPABILITIES,
    HTTP_REST_CAPABILITIES,
    MCP_CATALOG_CAPABILITIES,
    SQL_DATABASE_CAPABILITIES,
)
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = pytest.mark.unit


class _BoundedMemoryStore(RelationalStoreBoundedQueryPort):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows
        self.rows_fetched = 0

    def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        _ = sql, params

    def fetch_all(self, sql: str, params: Sequence[Any] = ()) -> Sequence[Mapping[str, Any]]:
        _ = sql, params
        self.rows_fetched = len(self._rows)
        return self._rows

    def fetch_query_bounded(
        self,
        sql: str,
        params: Sequence[Any] = (),
        *,
        max_rows: int,
        max_result_bytes: int,
    ) -> tuple[tuple[Mapping[str, Any], ...], bool, str]:
        _ = sql, params
        accumulated, truncated, reason = accumulate_bounded_sql_rows(
            self._rows,
            max_rows=max_rows,
            max_result_bytes=max_result_bytes,
        )
        self.rows_fetched = len(accumulated) + (1 if truncated and reason == "row_limit" else 0)
        return tuple(accumulated), truncated, reason


class _FakeHttp:
    def __init__(self, body: str, *, transport_bounded: bool = False) -> None:
        self._body = body
        self._transport_bounded = transport_bounded
        self.last_max_response_bytes: int | None = None

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        body: str = "",
        timeout_s: float = 30.0,
        max_response_bytes: int | None = None,
    ) -> HttpResponse:
        _ = method, url, headers, body, timeout_s
        self.last_max_response_bytes = max_response_bytes
        if self._transport_bounded and max_response_bytes is not None:
            raw = self._body.encode("utf-8")[:max_response_bytes]
            truncated = len(self._body.encode("utf-8")) > max_response_bytes
            return HttpResponse(
                status_code=200,
                body=raw.decode("utf-8", errors="ignore"),
                body_truncated_at_transport=truncated,
                transport_byte_ceiling_enforced=True,
            )
        return HttpResponse(status_code=200, body=self._body)


class _FakeBrowser(BrowserAutomationNavigationTimeoutPort):
    def __init__(self, text: str) -> None:
        self._text = text
        self.last_navigation_timeout_s: float | None = None

    @property
    def navigation_timeout_integration_enforced(self) -> bool:
        return True

    def fetch_page(
        self,
        url: str,
        *,
        wait_until: str = "load",
        navigation_timeout_s: float | None = None,
    ) -> PageContent:
        _ = wait_until
        self.last_navigation_timeout_s = navigation_timeout_s
        return PageContent(url=url, text=self._text, html="<p>x</p>")

    def close(self) -> None:
        return None


class _IncapableBrowser:
    def fetch_page(
        self,
        url: str,
        *,
        wait_until: str = "load",
        navigation_timeout_s: float | None = None,
    ) -> PageContent:
        _ = wait_until, navigation_timeout_s
        return PageContent(url=url, text="x", html="<p>x</p>")

    def close(self) -> None:
        return None


class _UnboundedMemoryStore(RelationalStore):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        _ = sql, params

    def fetch_all(self, sql: str, params: Sequence[Any] = ()) -> Sequence[Mapping[str, Any]]:
        _ = sql, params
        return self._rows


def test_sql_row_limit_truncates() -> None:
    rows = [{"id": i} for i in range(5)]
    bounded, truncated, reason = bound_database_query_rows(
        rows,
        max_rows=2,
        max_result_bytes=1_000_000,
    )
    assert truncated is True
    assert reason == "row_limit"
    assert len(bounded) == 2


def test_sql_byte_limit_raises_on_single_huge_row() -> None:
    rows = [{"blob": "x" * 500}]
    with pytest.raises(ProviderResourceLimitExceeded) as exc_info:
        bound_database_query_rows(rows, max_rows=10, max_result_bytes=32)
    assert exc_info.value.code == "database_result_byte_limit"


def test_database_query_bounded_port_row_ceiling() -> None:
    store = _BoundedMemoryStore([{"id": i} for i in range(10)])
    ctx = ToolWiringContext(relational_store_execution=RelationalStoreExecutionAdapter(store))
    out = database_query(
        ctx,
        DatabaseQueryInput(sql="SELECT 1", max_rows=3, max_result_bytes=1_000_000),
    )
    assert out.row_count == 3
    assert out.truncated is True
    assert out.limit_reason == "row_limit"
    assert store.rows_fetched <= 4


def test_http_passes_max_response_bytes_to_client() -> None:
    client = _FakeHttp("hello", transport_bounded=True)
    ctx = ToolWiringContext(http_client=client)
    http_request(
        ctx,
        HttpRequestInput(method="GET", url="https://example.com/", max_response_bytes=3),
    )
    assert client.last_max_response_bytes == 3


def test_http_response_body_truncated_at_transport() -> None:
    huge = "ą" * 10_000
    ctx = ToolWiringContext(http_client=_FakeHttp(huge, transport_bounded=True))
    out = http_request(
        ctx,
        HttpRequestInput(method="GET", url="https://example.com/", max_response_bytes=100),
    )
    assert out.success is True
    assert out.truncated is True
    assert len(out.body.encode("utf-8")) <= 100


def test_http_post_transport_error_propagates() -> None:
    class _FailingHttp:
        def request(self, *args: object, **kwargs: object) -> HttpResponse:
            raise TimeoutError("read timed out")

    ctx = ToolWiringContext(http_client=_FailingHttp())
    with pytest.raises(RuntimeError, match="http_transport_error"):
        http_request(
            ctx,
            HttpRequestInput(method="POST", url="https://example.com/", body="{}"),
        )


def test_browser_incapable_integration_reports_timeout_not_enforced() -> None:
    ctx = ToolWiringContext(browser_automation=_IncapableBrowser())
    out = browser_fetch_page(
        ctx,
        BrowserFetchPageInput(url="https://example.com", max_text_chars=256, navigation_timeout_s=5.0),
    )
    assert out.navigation_timeout_enforced is False


def test_browser_integration_facade_inherits_timeout_from_client() -> None:
    inner = _FakeBrowser("ok")
    integration = PlaywrightBrowserAutomationIntegration.from_client(inner, enabled=True)
    ctx = ToolWiringContext(browser_automation=integration)
    out = browser_fetch_page(
        ctx,
        BrowserFetchPageInput(url="https://example.com", max_text_chars=256, navigation_timeout_s=3.0),
    )
    assert out.navigation_timeout_enforced is True
    assert inner.last_navigation_timeout_s == 3.0


def test_sql_fallback_marks_physical_bounds_not_enforced() -> None:
    store = _UnboundedMemoryStore([{"id": i} for i in range(8)])
    adapter = RelationalStoreExecutionAdapter(store)
    result = adapter.query(
        RelationalQueryRequest(sql="SELECT 1", max_rows=2, max_result_bytes=1_000_000),
    )
    assert result.physical_bounds_enforced is False
    assert result.row_count == 2
    assert result.truncated is True


def test_allowlist_http_redirect_to_allowed_host() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/a":
            return httpx.Response(302, headers={"Location": "/b"}, request=request)
        return httpx.Response(200, text="done", request=request)

    client = AllowlistHttpClient(
        allowed_hosts=frozenset({"example.com"}),
        transport=httpx.MockTransport(handler),
    )
    response = client.request("GET", "https://example.com/a")
    assert response.status_code == 200
    assert response.body == "done"


def test_allowlist_http_redirect_to_denied_host_blocked() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"Location": "https://evil.test/"}, request=request)

    client = AllowlistHttpClient(
        allowed_hosts=frozenset({"example.com"}),
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(PermissionError, match="host_not_allowed"):
        client.request("GET", "https://example.com/start")


def test_allowlist_http_redirect_strips_authorization_on_host_change() -> None:
    seen_auth: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_auth.append(request.headers.get("Authorization"))
        if request.url.host == "example.com":
            return httpx.Response(
                302,
                headers={"Location": "https://allowed.example.org/ok"},
                request=request,
            )
        return httpx.Response(200, text="ok", request=request)

    client = AllowlistHttpClient(
        allowed_hosts=frozenset({"example.com", "allowed.example.org"}),
        transport=httpx.MockTransport(handler),
    )
    client.request(
        "GET",
        "https://example.com/start",
        headers={"Authorization": "Bearer secret"},
    )
    assert seen_auth[0] == "Bearer secret"
    assert seen_auth[1] is None


def test_mcp_list_overflow_raises_not_empty_list() -> None:
    payload = [{"tool_id": f"t{i}", "data": "x" * 200} for i in range(50)]
    with pytest.raises(McpPayloadByteLimitExceeded):
        emit_bounded_mcp_list(payload, max_bytes=256)


def test_browser_text_truncated_and_timeout_propagated() -> None:
    browser = _FakeBrowser("z" * 1_000)
    ctx = ToolWiringContext(browser_automation=browser)
    out = browser_fetch_page(
        ctx,
        BrowserFetchPageInput(url="https://example.com", max_text_chars=256, navigation_timeout_s=12.0),
    )
    assert len(out.text) == 256
    assert out.truncated is True
    assert browser.last_navigation_timeout_s == 12.0
    assert out.navigation_timeout_enforced is True


def test_mcp_catalog_serialization_bound() -> None:
    payload = {"data": "x" * 50_000}
    serialized, truncated = serialize_bounded_mcp_json(payload, max_bytes=256)
    assert truncated is True
    assert len(serialized.encode("utf-8")) <= 256


def test_mcp_invocation_structured_bound_returns_error_not_invalid_json() -> None:
    payload = {"data": "x" * 50_000}
    bounded, truncated = bound_mcp_structured_result(payload, max_bytes=256)
    assert truncated is True
    assert bounded == {"error": "mcp_payload_byte_limit_exceeded"}
    json.dumps(bounded)


def test_mcp_serialization_within_limit() -> None:
    payload = {"ok": True}
    serialized, truncated = serialize_bounded_mcp_json(payload, max_bytes=1024)
    assert truncated is False
    assert json.loads(serialized) == payload


def test_provider_capability_matrix_declarations() -> None:
    assert SQL_DATABASE_CAPABILITIES.supports_physical_row_ceiling is True
    assert HTTP_REST_CAPABILITIES.supports_physical_result_byte_ceiling is True
    assert MCP_CATALOG_CAPABILITIES.supports_physical_result_byte_ceiling is False
    assert BROWSER_AUTOMATION_CAPABILITIES.supports_physical_result_byte_ceiling is False
