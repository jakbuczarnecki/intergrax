# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json
import logging

import httpx
import pytest

from intergrax.integrations.contracts.conversation_channel import (
    ConversationAttachmentContent,
)
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


def _attachment(
    *,
    attachment_id: str = "F1",
    file_name: str = "a.pdf",
    content_type: str = "application/pdf",
    body: bytes = b"%PDF",
) -> ConversationAttachmentContent:
    return ConversationAttachmentContent(
        attachment_id=attachment_id,
        file_name=file_name,
        content_type=content_type,
        body=body,
    )


def _batch_payload(
    *,
    status: str = "accepted",
    workspace_id: str = "ws-1",
    accepted_count: int = 1,
    failed_count: int = 0,
) -> dict[str, object]:
    items: list[dict[str, object]] = []
    for i in range(accepted_count):
        items.append(
            {
                "position": i,
                "file_name": f"ok-{i}.pdf",
                "status": "accepted",
                "input_id": f"in-{i}",
                "source_id": f"src-{i}",
                "operation_id": f"op-{i}",
                "operation_status": "queued",
                "error_code": None,
            }
        )
    for i in range(failed_count):
        items.append(
            {
                "position": accepted_count + i,
                "file_name": f"bad-{i}.bin",
                "status": "failed",
                "error_code": "managed_file_empty",
            }
        )
    return {
        "batch_id": "batch-1",
        "workspace_id": workspace_id,
        "status": status,
        "accepted_count": accepted_count,
        "failed_count": failed_count,
        "items": items,
    }


def test_managed_files_url() -> None:
    client = WorkspaceAskHttpClient(
        SlackAskClientConfig(base_url="http://127.0.0.1:8020")
    )
    assert client.build_managed_files_url("ws-1") == (
        "http://127.0.0.1:8020/v1/local_workspace/workspaces/ws-1/knowledge/files"
    )


@pytest.mark.asyncio
async def test_upload_managed_files_multipart_headers_and_order() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(202, json=_batch_payload(accepted_count=2, status="accepted"))

    client = WorkspaceAskHttpClient(
        SlackAskClientConfig(
            base_url="http://127.0.0.1:8020",
            api_key="secret-key",
        ),
        transport=httpx.MockTransport(handler),
    )
    result = await client.upload_managed_files(
        tenant_id="tenant-1",
        workspace_id="ws-1",
        idempotency_key="idem-1",
        attachments=[
            _attachment(attachment_id="F1", file_name="one.pdf", body=b"AAA"),
            _attachment(
                attachment_id="F2",
                file_name="two.txt",
                content_type="text/plain",
                body=b"BBB",
            ),
        ],
    )
    assert len(calls) == 1
    request = calls[0]
    assert (
        str(request.url)
        == "http://127.0.0.1:8020/v1/local_workspace/workspaces/ws-1/knowledge/files"
    )
    assert request.headers["X-Tenant-Id"] == "tenant-1"
    assert request.headers["Idempotency-Key"] == "idem-1"
    assert request.headers["X-API-Key"] == "secret-key"
    content_type = request.headers["Content-Type"]
    assert content_type.startswith("multipart/form-data; boundary=")
    body = request.content
    assert b'name="files"' in body
    assert body.count(b'name="files"') == 2
    assert b"one.pdf" in body
    assert b"two.txt" in body
    assert b"AAA" in body
    assert b"BBB" in body
    assert body.find(b"one.pdf") < body.find(b"two.txt")
    assert result.status == "accepted"
    assert result.accepted_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["accepted", "partial", "failed"])
async def test_upload_managed_files_parses_statuses(status: str) -> None:
    accepted = 2 if status == "partial" else (1 if status == "accepted" else 0)
    failed = 1 if status in {"partial", "failed"} else 0
    client = WorkspaceAskHttpClient(
        SlackAskClientConfig(base_url="http://localhost:8020"),
        transport=httpx.MockTransport(
            lambda _r: httpx.Response(
                202,
                json=_batch_payload(
                    status=status,
                    accepted_count=accepted,
                    failed_count=failed,
                ),
            )
        ),
    )
    result = await client.upload_managed_files(
        tenant_id="t",
        workspace_id="ws-1",
        idempotency_key="k",
        attachments=[_attachment()],
    )
    assert result.status == status


@pytest.mark.asyncio
async def test_upload_workspace_mismatch_parse_error() -> None:
    client = WorkspaceAskHttpClient(
        SlackAskClientConfig(base_url="http://localhost:8020"),
        transport=httpx.MockTransport(
            lambda _r: httpx.Response(
                202,
                json=_batch_payload(workspace_id="ws-other"),
            )
        ),
    )
    with pytest.raises(SlackAskClientError) as exc:
        await client.upload_managed_files(
            tenant_id="t",
            workspace_id="ws-1",
            idempotency_key="k",
            attachments=[_attachment()],
        )
    assert exc.value.kind == "parse_error"
    assert "ws-other" not in str(exc.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [404, 409, 413, 503])
async def test_upload_http_errors_mapped(status: int) -> None:
    client = WorkspaceAskHttpClient(
        SlackAskClientConfig(base_url="http://localhost:8020"),
        transport=httpx.MockTransport(
            lambda _r: httpx.Response(status, json={"detail": "secret-body"})
        ),
    )
    with pytest.raises(SlackAskClientError) as exc:
        await client.upload_managed_files(
            tenant_id="t",
            workspace_id="ws-1",
            idempotency_key="k",
            attachments=[_attachment()],
        )
    assert exc.value.kind == f"http_{status}"
    assert "secret-body" not in str(exc.value)


@pytest.mark.asyncio
async def test_upload_timeout_and_transport_mapped() -> None:
    def timeout_handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow")

    client = WorkspaceAskHttpClient(
        SlackAskClientConfig(base_url="http://localhost:8020", timeout_seconds=0.1),
        transport=httpx.MockTransport(timeout_handler),
    )
    with pytest.raises(SlackAskClientError) as exc:
        await client.upload_managed_files(
            tenant_id="t",
            workspace_id="ws-1",
            idempotency_key="k",
            attachments=[_attachment()],
        )
    assert exc.value.kind == "timeout"

    def transport_handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    client = WorkspaceAskHttpClient(
        SlackAskClientConfig(base_url="http://localhost:8020"),
        transport=httpx.MockTransport(transport_handler),
    )
    with pytest.raises(SlackAskClientError) as exc2:
        await client.upload_managed_files(
            tenant_id="t",
            workspace_id="ws-1",
            idempotency_key="k",
            attachments=[_attachment()],
        )
    assert exc2.value.kind == "transport_error"
