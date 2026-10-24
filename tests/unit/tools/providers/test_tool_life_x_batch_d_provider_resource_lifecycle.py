# © Artur Czarnecki. All rights reserved.

"""TOOL-LIFE-X Batch D — SQL/HTTP/browser/MCP provider resource & lifecycle bounds."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

import pytest

from intergrax.integrations.contracts.browser_automation import PageContent
from intergrax.integrations.contracts.http_client import HttpResponse
from intergrax.integrations.relational_store_execution_adapter import RelationalStoreExecutionAdapter
from intergrax.tools.mcp_result_bounds import serialize_bounded_mcp_json
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


class _MemoryStore:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        _ = sql, params

    def fetch_all(self, sql: str, params: Sequence[Any] = ()) -> Sequence[Mapping[str, Any]]:
        _ = sql, params
        return self._rows


class _FakeHttp:
    def __init__(self, body: str) -> None:
        self._body = body

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        body: str = "",
        timeout_s: float = 30.0,
    ) -> HttpResponse:
        _ = method, url, headers, body, timeout_s
        return HttpResponse(status_code=200, body=self._body)


class _FakeBrowser:
    def __init__(self, text: str) -> None:
        self._text = text

    def fetch_page(self, url: str, *, wait_until: str = "load") -> PageContent:
        _ = wait_until
        return PageContent(url=url, text=self._text, html="<p>x</p>")

    def close(self) -> None:
        return None


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


def test_database_query_applies_row_ceiling() -> None:
    store = _MemoryStore([{"id": i} for i in range(10)])
    ctx = ToolWiringContext(relational_store_execution=RelationalStoreExecutionAdapter(store))
    out = database_query(
        ctx,
        DatabaseQueryInput(sql="SELECT 1", max_rows=3, max_result_bytes=1_000_000),
    )
    assert out.row_count == 3
    assert out.truncated is True
    assert out.limit_reason == "row_limit"


def test_http_response_body_truncated() -> None:
    huge = "ą" * 10_000
    ctx = ToolWiringContext(http_client=_FakeHttp(huge))
    out = http_request(
        ctx,
        HttpRequestInput(method="GET", url="https://example.com/", max_response_bytes=100),
    )
    assert out.success is True
    assert out.truncated is True
    assert len(out.body.encode("utf-8")) <= 100


def test_browser_text_truncated() -> None:
    ctx = ToolWiringContext(browser_automation=_FakeBrowser("z" * 1_000))
    out = browser_fetch_page(
        ctx,
        BrowserFetchPageInput(url="https://example.com", max_text_chars=256),
    )
    assert len(out.text) == 256
    assert out.truncated is True


def test_mcp_serialization_bound() -> None:
    payload = {"data": "x" * 50_000}
    serialized, truncated = serialize_bounded_mcp_json(payload, max_bytes=256)
    assert truncated is True
    assert len(serialized.encode("utf-8")) <= 256


def test_mcp_serialization_within_limit() -> None:
    payload = {"ok": True}
    serialized, truncated = serialize_bounded_mcp_json(payload, max_bytes=1024)
    assert truncated is False
    assert json.loads(serialized) == payload


def test_provider_capability_matrix_declarations() -> None:
    assert SQL_DATABASE_CAPABILITIES.supports_row_ceiling is True
    assert HTTP_REST_CAPABILITIES.supports_result_byte_ceiling is True
    assert MCP_CATALOG_CAPABILITIES.cancellation.value == "cooperative_only"
    assert BROWSER_AUTOMATION_CAPABILITIES.supports_result_byte_ceiling is True
