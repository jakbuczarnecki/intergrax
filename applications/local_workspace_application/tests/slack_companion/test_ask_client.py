# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json
import logging

import httpx
import pytest

from local_workspace_application.slack_companion.ask_client import (
    SlackAskClientConfig,
    WorkspaceAskHttpClient,
)
from local_workspace_application.slack_companion.models import SlackAskClientError

pytestmark = pytest.mark.unit


def _ask_payload(*, status: str = "completed") -> dict[str, object]:
    return {
        "run_id": "run-1",
        "workspace_id": "ws-1",
        "status": status,
        "question": "Q?",
        "answer": "A" if status == "completed" else None,
        "citations": [
            {
                "evidence_id": "e1",
                "document_id": "d1",
                "source_id": "s1",
                "workspace_id": "ws-1",
                "source_path": "/secret/path/policy.pdf",
                "file_name": "policy.pdf",
                "excerpt": "secret excerpt",
                "score": 0.9,
            }
        ],
        "created_at": "2026-07-23T12:00:00Z",
        "completed_at": "2026-07-23T12:00:01Z",
        "error": None,
    }


@pytest.mark.asyncio
async def test_ask_client_url_tenant_body_and_completed() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=_ask_payload())

    transport = httpx.MockTransport(handler)
    client = WorkspaceAskHttpClient(
        SlackAskClientConfig(
            base_url="http://127.0.0.1:8020",
            api_key="secret-key",
            timeout_seconds=5.0,
        ),
        transport=transport,
    )
    result = await client.ask(
        tenant_id="tenant-1",
        workspace_id="ws-1",
        question="What is the policy?",
    )
    assert len(calls) == 1
    request = calls[0]
    assert str(request.url) == "http://127.0.0.1:8020/v1/local_workspace/workspaces/ws-1/ask"
    assert request.headers["X-Tenant-Id"] == "tenant-1"
    assert request.headers["X-API-Key"] == "secret-key"
    body = json.loads(request.content.decode("utf-8"))
    assert body == {"question": "What is the policy?", "limit": 10}
    assert result.status == "completed"
    assert result.answer == "A"
    assert result.citations[0].file_name == "policy.pdf"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status",
    ["insufficient_evidence", "failed"],
)
async def test_ask_client_parses_typed_statuses(status: str) -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json=_ask_payload(status=status))
    )
    client = WorkspaceAskHttpClient(
        SlackAskClientConfig(base_url="http://localhost:8020"),
        transport=transport,
    )
    result = await client.ask(tenant_id="t", workspace_id="ws", question="Q")
    assert result.status == status


@pytest.mark.asyncio
async def test_ask_client_timeout_mapped_safely() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow")

    client = WorkspaceAskHttpClient(
        SlackAskClientConfig(base_url="http://localhost:8020", timeout_seconds=0.1),
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(SlackAskClientError) as exc_info:
        await client.ask(tenant_id="t", workspace_id="ws", question="Q")
    assert exc_info.value.kind == "timeout"


@pytest.mark.asyncio
async def test_ask_client_non_2xx_mapped_safely() -> None:
    transport = httpx.MockTransport(lambda _r: httpx.Response(502, json={"detail": "x"}))
    client = WorkspaceAskHttpClient(
        SlackAskClientConfig(base_url="http://localhost:8020"),
        transport=transport,
    )
    with pytest.raises(SlackAskClientError) as exc_info:
        await client.ask(tenant_id="t", workspace_id="ws", question="Q")
    assert exc_info.value.kind == "http_502"


@pytest.mark.asyncio
async def test_api_key_not_logged(caplog: pytest.LogCaptureFixture) -> None:
    transport = httpx.MockTransport(lambda _r: httpx.Response(500, text="fail"))
    client = WorkspaceAskHttpClient(
        SlackAskClientConfig(
            base_url="http://localhost:8020",
            api_key="super-secret-api-key",
        ),
        transport=transport,
    )
    with caplog.at_level(logging.WARNING):
        with pytest.raises(SlackAskClientError):
            await client.ask(tenant_id="t", workspace_id="ws", question="Q")
    joined = " ".join(record.getMessage() for record in caplog.records)
    assert "super-secret-api-key" not in joined


@pytest.mark.asyncio
async def test_list_workspaces_uses_tenant_and_filters_active() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            200,
            json={
                "workspaces": [
                    {
                        "workspace_id": "ws-1",
                        "tenant_id": "tenant-1",
                        "name": "Active One",
                        "status": "active",
                        "created_at": "2026-07-23T12:00:00Z",
                        "updated_at": "2026-07-23T12:00:00Z",
                    },
                    {
                        "workspace_id": "ws-2",
                        "tenant_id": "tenant-1",
                        "name": "Archived One",
                        "status": "archived",
                        "created_at": "2026-07-23T12:00:00Z",
                        "updated_at": "2026-07-23T12:00:00Z",
                    },
                ]
            },
        )

    client = WorkspaceAskHttpClient(
        SlackAskClientConfig(base_url="http://127.0.0.1:8020", api_key="k"),
        transport=httpx.MockTransport(handler),
    )
    items = await client.list_workspaces(tenant_id="tenant-1")
    assert len(calls) == 1
    assert str(calls[0].url) == "http://127.0.0.1:8020/v1/local_workspace/workspaces"
    assert calls[0].method == "GET"
    assert calls[0].headers["X-Tenant-Id"] == "tenant-1"
    assert [item.workspace_id for item in items] == ["ws-1"]


@pytest.mark.asyncio
async def test_no_automatic_retry() -> None:
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503, text="busy")

    client = WorkspaceAskHttpClient(
        SlackAskClientConfig(base_url="http://localhost:8020"),
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(SlackAskClientError):
        await client.ask(tenant_id="t", workspace_id="ws", question="Q")
    assert calls["n"] == 1
