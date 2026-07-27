# © Artur Czarnecki. All rights reserved.

"""LKW-WORKSPACE-CONTENTS-1A — Slack ``sources`` inspection (no Ask)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
from local_workspace_application.slack_companion.ask_client import (
    SlackAskClientConfig,
    WorkspaceAskHttpClient,
)
from local_workspace_application.slack_companion.authorization import (
    SlackCompanionAuthConfig,
)
from local_workspace_application.slack_companion.dedupe_repository import (
    SlackEventDedupeRepository,
    build_slack_dedupe_key,
)
from local_workspace_application.slack_companion.models import (
    SlackAskClientError,
    SlackDedupeStatus,
    SlackSourceListItem,
)
from local_workspace_application.slack_companion.rendering import (
    NO_WORKSPACE_AVAILABLE_TEXT,
    SOURCE_LIST_EMPTY_TEXT,
    SOURCE_LIST_LOAD_FAILED_TEXT,
    SOURCE_WORKSPACE_UNAVAILABLE_TEXT,
    render_source_list,
)
from local_workspace_application.slack_companion.selection_store import (
    InMemorySlackWorkspaceSelectionStore,
    SlackWorkspaceSelection,
    slack_selection_actor_key,
)
from local_workspace_application.slack_companion.workflow import (
    SlackAskWorkflow,
    is_sources_command,
    order_sources_for_listing,
    parse_sources_list_command,
)

from intergrax.integrations._shared.in_memory_document_store import (
    InMemoryDocumentStore,
)
from intergrax.integrations.contracts.conversation_channel import (
    ConversationActor,
    ConversationAddress,
    ConversationDeliveryReceipt,
    ConversationEventKind,
    InboundConversationEvent,
    OutboundConversationMessage,
)

pytestmark = pytest.mark.unit


def _event(
    *,
    event_id: str = "Ev-src-1",
    team_id: str = "T_OK",
    user_id: str = "U_OK",
    text: str = "sources",
) -> InboundConversationEvent:
    return InboundConversationEvent(
        event_id=event_id,
        address=ConversationAddress(
            installation_id=team_id,
            conversation_id="Dchannel",
            thread_id="1712222.000300",
        ),
        actor=ConversationActor(actor_id=user_id, is_bot=False),
        kind=ConversationEventKind.MESSAGE,
        text=text,
    )


def _source_payload(
    *,
    source_id: str,
    label: str,
    source_type: str = "local_folder",
    status: str = "registered",
    recursive: bool = True,
    last_sync_at: str | None = None,
    path: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "source_id": source_id,
        "workspace_id": "ws-active",
        "source_type": source_type,
        "label": label,
        "status": status,
        "recursive": recursive,
        "created_at": "2026-07-24T10:00:00Z",
        "last_sync_at": last_sync_at,
    }
    if path is not None:
        payload["path"] = path
    return payload


def _transport(
    *,
    sources: list[dict[str, object]] | None = None,
    status_code: int = 200,
    ask_calls: list[httpx.Request] | None = None,
    source_calls: list[httpx.Request] | None = None,
    list_calls: list[httpx.Request] | None = None,
    malformed: bool = False,
) -> httpx.MockTransport:
    ask_bucket = ask_calls if ask_calls is not None else []
    source_bucket = source_calls if source_calls is not None else []
    list_bucket = list_calls if list_calls is not None else []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/ask"):
            ask_bucket.append(request)
            return httpx.Response(
                200,
                json={
                    "run_id": "ask-run-1",
                    "workspace_id": "ws-active",
                    "status": "completed",
                    "question": "Q",
                    "answer": "Ask answer",
                    "citations": [],
                    "created_at": "2026-07-23T12:00:00Z",
                },
            )
        if path.rstrip("/").endswith("/workspaces") and request.method == "GET":
            list_bucket.append(request)
            return httpx.Response(200, json={"workspaces": []})
        if "/sources" in path and request.method == "GET":
            source_bucket.append(request)
            if status_code != 200:
                return httpx.Response(status_code, json={"detail": "boom"})
            if malformed:
                return httpx.Response(200, text="not-json")
            return httpx.Response(
                200,
                json={"sources": sources if sources is not None else []},
            )
        return httpx.Response(404, json={"detail": "missing"})

    return httpx.MockTransport(handler)


def _workflow(
    *,
    transport: httpx.MockTransport,
    dedupe: SlackEventDedupeRepository | None = None,
    sent: list[OutboundConversationMessage] | None = None,
    selection_store: InMemorySlackWorkspaceSelectionStore | None = None,
    active_workspace_id: str = "ws-active",
) -> tuple[SlackAskWorkflow, list[OutboundConversationMessage]]:
    outbound = sent if sent is not None else []

    async def send(message: OutboundConversationMessage) -> Any:
        outbound.append(message)
        return ConversationDeliveryReceipt(
            message_id=f"msg-{len(outbound)}",
            address=message.address,
            delivered_at=datetime.now(UTC),
        )

    workflow = SlackAskWorkflow(
        auth_config=SlackCompanionAuthConfig(
            approved_team_id="T_OK",
            approved_user_id="U_OK",
            tenant_id="tenant-a",
            active_workspace_id=active_workspace_id,
        ),
        dedupe=dedupe or SlackEventDedupeRepository(InMemoryDocumentStore()),
        ask_client=WorkspaceAskHttpClient(
            SlackAskClientConfig(base_url="http://ask.test"),
            transport=transport,
        ),
        send=send,
        selection_store=selection_store,
    )
    return workflow, outbound


# --- Parser ---


@pytest.mark.parametrize(
    "text",
    ["sources", "SOURCES", "  sources  ", "Sources"],
)
def test_sources_command_matches(text: str) -> None:
    assert is_sources_command(text) is True
    assert parse_sources_list_command(text) is not None


@pytest.mark.parametrize(
    "text",
    [
        "source",
        "sources 1",
        "sources status",
        "my sources",
        "source list",
        "where are the sources for this matter?",
    ],
)
def test_sources_command_non_matches(text: str) -> None:
    assert is_sources_command(text) is False
    assert parse_sources_list_command(text) is None


# --- Ordering / rendering ---


def test_order_sources_deterministic() -> None:
    items = [
        SlackSourceListItem(
            source_id="b",
            workspace_id="ws",
            source_type="local_folder",
            label="Beta",
            status="registered",
        ),
        SlackSourceListItem(
            source_id="a",
            workspace_id="ws",
            source_type="local_folder",
            label="Alpha",
            status="ready",
        ),
        SlackSourceListItem(
            source_id="c",
            workspace_id="ws",
            source_type="object_storage",
            label="Alpha",
            status="registered",
        ),
    ]
    ordered = order_sources_for_listing(items)
    assert [i.source_id for i in ordered] == ["a", "c", "b"]


def test_render_source_list_empty() -> None:
    assert render_source_list([]) == SOURCE_LIST_EMPTY_TEXT


def test_render_source_list_safe_fields() -> None:
    text = render_source_list(
        [
            SlackSourceListItem(
                source_id="src-secret",
                workspace_id="ws-secret",
                source_type="local_folder",
                label="Contracts",
                status="ready",
                recursive=True,
                last_sync_at=datetime(2026, 7, 24, 10, 15, tzinfo=UTC),
            ),
            SlackSourceListItem(
                source_id="src-2",
                workspace_id="ws-secret",
                source_type="local_folder",
                label="Technical specifications",
                status="registered",
                recursive=True,
                last_sync_at=None,
            ),
        ]
    )
    assert "Sources in the active workspace:" in text
    assert "1. Contracts" in text
    assert "Type: local folder" in text
    assert "Status: ready" in text
    assert "Recursive: yes" in text
    assert "Last sync: 2026-07-24 10:15 UTC" in text
    assert "2. Technical specifications" in text
    assert "Last sync: never" in text
    assert "src-secret" not in text
    assert "ws-secret" not in text
    assert "tenant-a" not in text
    assert r"C:\Users" not in text
    assert "Private" not in text


def test_render_non_folder_source_hides_recursive() -> None:
    text = render_source_list(
        [
            SlackSourceListItem(
                source_id="s1",
                workspace_id="ws",
                source_type="object_storage",
                label="Archive",
                status="registered",
                recursive=True,
            )
        ]
    )

    assert "Type: object storage" in text
    assert "Recursive:" not in text


def test_render_defensive_label_normalization() -> None:
    text = render_source_list(
        [
            SlackSourceListItem(
                source_id="s1",
                workspace_id="ws",
                source_type="local_folder",
                label="  Contracts\n\twith\x00noise  ",
                status="registered",
            )
        ]
    )
    assert "Contracts with noise" in text
    assert "\x00" not in text
    assert "\n\t" not in text.split("1. ", 1)[-1].split("\n", 1)[0]


def test_render_long_label_bounded() -> None:
    long_label = "A" * 200
    text = render_source_list(
        [
            SlackSourceListItem(
                source_id="s1",
                workspace_id="ws",
                source_type="local_folder",
                label=long_label,
                status="registered",
            )
        ]
    )
    assert "A" * 200 not in text
    assert "…" in text


def test_render_does_not_expose_path_like_label_parents() -> None:
    # Renderer must not invent path structure; only show provided safe label.
    text = render_source_list(
        [
            SlackSourceListItem(
                source_id="s1",
                workspace_id="ws",
                source_type="local_folder",
                label="Contracts",
                status="registered",
            )
        ]
    )
    for fragment in ("C:\\", "Users", "Artur", "Private", "Client-X"):
        assert fragment not in text
    assert "Contracts" in text


# --- HTTP client ---


@pytest.mark.asyncio
async def test_list_sources_client_url_tenant_and_api_key() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            200,
            json={
                "sources": [
                    _source_payload(
                        source_id="s1",
                        label="Contracts",
                        path=r"C:\Users\Artur\Private\Client-X\Contracts",
                    )
                ]
            },
        )

    client = WorkspaceAskHttpClient(
        SlackAskClientConfig(
            base_url="http://127.0.0.1:8020",
            api_key="secret-key",
        ),
        transport=httpx.MockTransport(handler),
    )
    items = await client.list_sources(tenant_id="tenant-1", workspace_id="ws-1")
    assert len(calls) == 1
    assert (
        str(calls[0].url)
        == "http://127.0.0.1:8020/v1/local_workspace/workspaces/ws-1/sources"
    )
    assert calls[0].headers["X-Tenant-Id"] == "tenant-1"
    assert calls[0].headers["X-API-Key"] == "secret-key"
    assert len(items) == 1
    assert items[0].label == "Contracts"
    assert not hasattr(items[0], "path")
    assert "path" not in items[0].model_dump()


@pytest.mark.asyncio
async def test_list_sources_api_key_absent_when_not_configured() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"sources": []})

    client = WorkspaceAskHttpClient(
        SlackAskClientConfig(base_url="http://127.0.0.1:8020"),
        transport=httpx.MockTransport(handler),
    )
    await client.list_sources(tenant_id="t", workspace_id="ws")
    assert "X-API-Key" not in calls[0].headers


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("exc", "kind"),
    [
        (httpx.ReadTimeout("slow"), "timeout"),
        (httpx.ConnectError("down"), "transport_error"),
    ],
)
async def test_list_sources_transport_errors(exc: Exception, kind: str) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise exc

    client = WorkspaceAskHttpClient(
        SlackAskClientConfig(base_url="http://localhost:8020", timeout_seconds=0.1),
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(SlackAskClientError) as exc_info:
        await client.list_sources(tenant_id="t", workspace_id="ws")
    assert exc_info.value.kind == kind


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "kind"),
    [(404, "http_404"), (500, "http_500")],
)
async def test_list_sources_http_errors(status_code: int, kind: str) -> None:
    client = WorkspaceAskHttpClient(
        SlackAskClientConfig(base_url="http://localhost:8020"),
        transport=httpx.MockTransport(
            lambda _r: httpx.Response(status_code, json={"detail": "x"})
        ),
    )
    with pytest.raises(SlackAskClientError) as exc_info:
        await client.list_sources(tenant_id="t", workspace_id="ws")
    assert exc_info.value.kind == kind


@pytest.mark.asyncio
async def test_list_sources_malformed_json_parse_error() -> None:
    client = WorkspaceAskHttpClient(
        SlackAskClientConfig(base_url="http://localhost:8020"),
        transport=httpx.MockTransport(lambda _r: httpx.Response(200, text="{bad")),
    )
    with pytest.raises(SlackAskClientError) as exc_info:
        await client.list_sources(tenant_id="t", workspace_id="ws")
    assert exc_info.value.kind == "parse_error"


@pytest.mark.asyncio
async def test_list_sources_invalid_schema_parse_error() -> None:
    client = WorkspaceAskHttpClient(
        SlackAskClientConfig(base_url="http://localhost:8020"),
        transport=httpx.MockTransport(
            lambda _r: httpx.Response(200, json={"sources": "nope"})
        ),
    )
    with pytest.raises(SlackAskClientError) as exc_info:
        await client.list_sources(tenant_id="t", workspace_id="ws")
    assert exc_info.value.kind == "parse_error"


@pytest.mark.asyncio
async def test_list_sources_ignores_extra_path_field() -> None:
    client = WorkspaceAskHttpClient(
        SlackAskClientConfig(base_url="http://localhost:8020"),
        transport=httpx.MockTransport(
            lambda _r: httpx.Response(
                200,
                json={
                    "sources": [
                        _source_payload(
                            source_id="s1",
                            label="Contracts",
                            path=r"C:\Users\Artur\Private\Client-X\Contracts",
                        )
                    ]
                },
            )
        ),
    )
    items = await client.list_sources(tenant_id="t", workspace_id="ws")
    assert items[0].label == "Contracts"
    assert "path" not in items[0].model_fields_set or "path" not in items[0].model_dump()
    dumped = items[0].model_dump()
    assert "path" not in dumped


# --- Workflow ---


@pytest.mark.asyncio
async def test_sources_success_zero_ask_and_safe_response() -> None:
    ask_calls: list[httpx.Request] = []
    source_calls: list[httpx.Request] = []
    list_calls: list[httpx.Request] = []
    transport = _transport(
        sources=[
            _source_payload(source_id="s1", label="Contracts", status="ready"),
            _source_payload(source_id="s2", label="Alpha", status="registered"),
        ],
        ask_calls=ask_calls,
        source_calls=source_calls,
        list_calls=list_calls,
    )
    dedupe = SlackEventDedupeRepository(InMemoryDocumentStore())
    workflow, outbound = _workflow(transport=transport, dedupe=dedupe)
    await workflow.handle(_event())
    assert ask_calls == []
    assert list_calls == []
    assert len(source_calls) == 1
    assert source_calls[0].headers["X-Tenant-Id"] == "tenant-a"
    assert len(outbound) == 1
    text = outbound[0].text
    assert "1. Alpha" in text
    assert "2. Contracts" in text
    assert "s1" not in text
    assert "s2" not in text
    assert "ws-active" not in text
    key = build_slack_dedupe_key(team_id="T_OK", event_id="Ev-src-1")
    record = dedupe._get(key)
    assert record is not None
    assert record.status == SlackDedupeStatus.COMPLETED
    assert record.ask_run_id is None


@pytest.mark.asyncio
async def test_sources_empty_list() -> None:
    source_calls: list[httpx.Request] = []
    workflow, outbound = _workflow(
        transport=_transport(sources=[], source_calls=source_calls)
    )
    await workflow.handle(_event())
    assert len(source_calls) == 1
    assert outbound[0].text == SOURCE_LIST_EMPTY_TEXT


@pytest.mark.asyncio
async def test_sources_404_unavailable() -> None:
    workflow, outbound = _workflow(transport=_transport(status_code=404))
    await workflow.handle(_event())
    assert outbound[0].text == SOURCE_WORKSPACE_UNAVAILABLE_TEXT
    assert "ws-active" not in outbound[0].text


@pytest.mark.asyncio
async def test_sources_timeout_load_failed() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow")

    workflow, outbound = _workflow(transport=httpx.MockTransport(handler))
    await workflow.handle(_event())
    assert outbound[0].text == SOURCE_LIST_LOAD_FAILED_TEXT


@pytest.mark.asyncio
async def test_sources_parse_error_load_failed() -> None:
    workflow, outbound = _workflow(transport=_transport(malformed=True))
    await workflow.handle(_event())
    assert outbound[0].text == SOURCE_LIST_LOAD_FAILED_TEXT


@pytest.mark.asyncio
async def test_sources_no_active_workspace_zero_http() -> None:
    source_calls: list[httpx.Request] = []
    ask_calls: list[httpx.Request] = []
    store = InMemorySlackWorkspaceSelectionStore()
    actor = slack_selection_actor_key(team_id="T_OK", user_id="U_OK")
    store.suppress_configured(actor)
    workflow, outbound = _workflow(
        transport=_transport(source_calls=source_calls, ask_calls=ask_calls),
        selection_store=store,
        active_workspace_id="ws-active",
    )
    await workflow.handle(_event())
    assert source_calls == []
    assert ask_calls == []
    assert outbound[0].text == NO_WORKSPACE_AVAILABLE_TEXT


@pytest.mark.asyncio
async def test_sources_uses_selected_workspace() -> None:
    source_calls: list[httpx.Request] = []
    store = InMemorySlackWorkspaceSelectionStore()
    actor = slack_selection_actor_key(team_id="T_OK", user_id="U_OK")
    store.set(
        actor,
        SlackWorkspaceSelection(workspace_id="ws-selected", workspace_name="Selected"),
    )
    workflow, outbound = _workflow(
        transport=_transport(
            sources=[_source_payload(source_id="s1", label="Docs")],
            source_calls=source_calls,
        ),
        selection_store=store,
        active_workspace_id="ws-configured",
    )
    await workflow.handle(_event())
    assert len(source_calls) == 1
    assert source_calls[0].url.path.endswith("/workspaces/ws-selected/sources")
    assert "Docs" in outbound[0].text


@pytest.mark.asyncio
async def test_sources_uses_configured_fallback() -> None:
    source_calls: list[httpx.Request] = []
    workflow, _ = _workflow(
        transport=_transport(
            sources=[_source_payload(source_id="s1", label="Docs")],
            source_calls=source_calls,
        ),
        active_workspace_id="ws-configured",
    )
    await workflow.handle(_event())
    assert source_calls[0].url.path.endswith("/workspaces/ws-configured/sources")


@pytest.mark.asyncio
async def test_sources_duplicate_event_no_repeat_http() -> None:
    source_calls: list[httpx.Request] = []
    dedupe = SlackEventDedupeRepository(InMemoryDocumentStore())
    workflow, outbound = _workflow(
        transport=_transport(
            sources=[_source_payload(source_id="s1", label="Docs")],
            source_calls=source_calls,
        ),
        dedupe=dedupe,
    )
    event = _event(event_id="Ev-dup")
    await workflow.handle(event)
    await workflow.handle(event)
    assert len(source_calls) == 1
    assert len(outbound) == 1


@pytest.mark.asyncio
async def test_sources_unauthorized_zero_http() -> None:
    source_calls: list[httpx.Request] = []
    ask_calls: list[httpx.Request] = []
    workflow, outbound = _workflow(
        transport=_transport(source_calls=source_calls, ask_calls=ask_calls)
    )
    await workflow.handle(_event(user_id="U_OTHER"))
    assert source_calls == []
    assert ask_calls == []
    assert outbound == []


@pytest.mark.asyncio
async def test_natural_language_with_sources_still_asks() -> None:
    ask_calls: list[httpx.Request] = []
    source_calls: list[httpx.Request] = []
    workflow, outbound = _workflow(
        transport=_transport(ask_calls=ask_calls, source_calls=source_calls)
    )
    await workflow.handle(_event(text="where are the sources for this matter?"))
    assert len(ask_calls) == 1
    assert source_calls == []
    assert "Checking the selected workspace" in outbound[0].text


@pytest.mark.asyncio
async def test_sources_command_discovered_in_help() -> None:
    workflow, outbound = _workflow(transport=_transport())
    await workflow.handle(_event(text="help", event_id="Ev-help"))
    text = outbound[0].text
    assert "`sources`" in text
    assert "List sources in the active workspace." in text
