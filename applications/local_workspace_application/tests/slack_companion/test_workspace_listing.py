# © Artur Czarnecki. All rights reserved.

"""LKW-SLACK-WORKFLOW-1B-1 — exact ``workspaces`` listing (no Ask)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx
import pytest

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.conversation_channel import (
    ConversationActor,
    ConversationAddress,
    ConversationDeliveryReceipt,
    ConversationEventKind,
    InboundConversationEvent,
    OutboundConversationMessage,
)
from local_workspace_application.slack_companion.ask_client import (
    SlackAskClientConfig,
    WorkspaceAskHttpClient,
)
from local_workspace_application.slack_companion.authorization import SlackCompanionAuthConfig
from local_workspace_application.slack_companion.dedupe_repository import (
    SlackEventDedupeRepository,
)
from local_workspace_application.slack_companion.models import SlackWorkspaceListItem
from local_workspace_application.slack_companion.rendering import (
    GENERIC_ERROR_TEXT,
    WORKSPACE_LIST_EMPTY_TEXT,
    render_workspace_list,
)
from local_workspace_application.slack_companion.workflow import (
    SlackAskWorkflow,
    is_workspaces_command,
    order_workspaces_for_listing,
)

pytestmark = pytest.mark.unit


def _event(
    *,
    event_id: str = "Ev-ws-1",
    team_id: str = "T_OK",
    user_id: str = "U_OK",
    text: str = "workspaces",
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


def _workspace_payload(
    *,
    workspace_id: str,
    name: str,
    status: str = "active",
    tenant_id: str = "tenant-a",
) -> dict[str, object]:
    return {
        "workspace_id": workspace_id,
        "tenant_id": tenant_id,
        "name": name,
        "description": "",
        "status": status,
        "created_at": "2026-07-23T12:00:00Z",
        "updated_at": "2026-07-23T12:00:00Z",
    }


def _list_transport(
    *,
    workspaces: list[dict[str, object]] | None = None,
    status_code: int = 200,
    ask_calls: list[httpx.Request] | None = None,
    list_calls: list[httpx.Request] | None = None,
) -> httpx.MockTransport:
    ask_bucket = ask_calls if ask_calls is not None else []
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
        if path.rstrip("/").endswith("/workspaces"):
            list_bucket.append(request)
            if status_code != 200:
                return httpx.Response(status_code, json={"detail": "boom"})
            return httpx.Response(
                200,
                json={"workspaces": workspaces if workspaces is not None else []},
            )
        return httpx.Response(404, json={"detail": "missing"})

    return httpx.MockTransport(handler)


def _workflow(
    *,
    transport: httpx.MockTransport,
    dedupe: SlackEventDedupeRepository | None = None,
    sent: list[OutboundConversationMessage] | None = None,
) -> tuple[SlackAskWorkflow, list[OutboundConversationMessage]]:
    outbound = sent if sent is not None else []

    async def send(message: OutboundConversationMessage) -> Any:
        outbound.append(message)
        return ConversationDeliveryReceipt(
            message_id=f"msg-{len(outbound)}",
            address=message.address,
            delivered_at=datetime.now(timezone.utc),
        )

    workflow = SlackAskWorkflow(
        auth_config=SlackCompanionAuthConfig(
            approved_team_id="T_OK",
            approved_user_id="U_OK",
            tenant_id="tenant-a",
            active_workspace_id="ws-active",
        ),
        dedupe=dedupe or SlackEventDedupeRepository(InMemoryDocumentStore()),
        ask_client=WorkspaceAskHttpClient(
            SlackAskClientConfig(base_url="http://ask.test"),
            transport=transport,
        ),
        send=send,
    )
    return workflow, outbound


@pytest.mark.parametrize(
    "text",
    ["workspaces", "WORKSPACES", "  workspaces  ", "Workspaces"],
)
def test_workspaces_command_recognized_case_insensitive_trim(text: str) -> None:
    assert is_workspaces_command(text) is True


@pytest.mark.parametrize(
    "text",
    ["show workspaces", "workspaces please", "what are my workspaces", "workspace"],
)
def test_non_exact_workspaces_text_is_not_command(text: str) -> None:
    assert is_workspaces_command(text) is False


def test_order_active_first_then_name_then_id() -> None:
    items = [
        SlackWorkspaceListItem(workspace_id="ws-c", name="Beta", status="active"),
        SlackWorkspaceListItem(workspace_id="ws-b", name="Alpha", status="active"),
        SlackWorkspaceListItem(workspace_id="ws-active", name="Zulu", status="active"),
        SlackWorkspaceListItem(workspace_id="ws-a", name="Alpha", status="active"),
    ]
    ordered = order_workspaces_for_listing(items, active_workspace_id="ws-active")
    assert [item.workspace_id for item in ordered] == [
        "ws-active",
        "ws-a",
        "ws-b",
        "ws-c",
    ]


def test_render_marks_active_without_ids() -> None:
    text = render_workspace_list(
        [
            SlackWorkspaceListItem(
                workspace_id="ws-active", name="Workspace Alpha", status="active"
            ),
            SlackWorkspaceListItem(
                workspace_id="ws-beta", name="Workspace Beta", status="active"
            ),
        ],
        active_workspace_id="ws-active",
    )
    assert "Available workspaces:" in text
    assert "1. Workspace Alpha — active" in text
    assert "2. Workspace Beta" in text
    assert "ws-active" not in text
    assert "ws-beta" not in text
    assert "tenant" not in text.casefold()


def test_render_empty_list() -> None:
    assert render_workspace_list([], active_workspace_id="ws-active") == (
        WORKSPACE_LIST_EMPTY_TEXT
    )


def test_render_does_not_mark_when_active_missing() -> None:
    text = render_workspace_list(
        [SlackWorkspaceListItem(workspace_id="ws-other", name="Other", status="active")],
        active_workspace_id="ws-active",
    )
    assert "— active" not in text
    assert "1. Other" in text


@pytest.mark.asyncio
async def test_exact_workspaces_lists_without_ask() -> None:
    ask_calls: list[httpx.Request] = []
    list_calls: list[httpx.Request] = []
    transport = _list_transport(
        workspaces=[
            _workspace_payload(workspace_id="ws-beta", name="Workspace Beta"),
            _workspace_payload(workspace_id="ws-active", name="Workspace Alpha"),
            _workspace_payload(
                workspace_id="ws-archived",
                name="Archived Room",
                status="archived",
            ),
        ],
        ask_calls=ask_calls,
        list_calls=list_calls,
    )
    workflow, sent = _workflow(transport=transport)
    await workflow.handle(_event(text="  WORKSPACES  "))

    assert len(ask_calls) == 0
    assert len(list_calls) == 1
    assert list_calls[0].headers["X-Tenant-Id"] == "tenant-a"
    assert str(list_calls[0].url).endswith("/v1/local_workspace/workspaces")
    assert len(sent) == 1
    assert sent[0].address.thread_id == "1712222.000300"
    text = sent[0].text
    assert "Workspace Alpha — active" in text
    assert "Workspace Beta" in text
    assert "Archived Room" not in text
    assert "ws-active" not in text
    assert "ws-beta" not in text
    assert "tenant-a" not in text
    assert "Checking the selected workspace" not in text


@pytest.mark.asyncio
async def test_workspaces_please_remains_ask() -> None:
    ask_calls: list[httpx.Request] = []
    list_calls: list[httpx.Request] = []
    transport = _list_transport(ask_calls=ask_calls, list_calls=list_calls)
    workflow, sent = _workflow(transport=transport)
    await workflow.handle(_event(text="workspaces please", event_id="Ev-ask-1"))

    assert len(ask_calls) == 1
    assert len(list_calls) == 0
    assert len(sent) == 2
    assert "Ask answer" in sent[1].text


@pytest.mark.asyncio
async def test_empty_list_safe_message() -> None:
    transport = _list_transport(workspaces=[])
    workflow, sent = _workflow(transport=transport)
    await workflow.handle(_event())
    assert len(sent) == 1
    assert sent[0].text == WORKSPACE_LIST_EMPTY_TEXT


@pytest.mark.asyncio
async def test_same_event_id_does_not_list_twice() -> None:
    list_calls: list[httpx.Request] = []
    transport = _list_transport(
        workspaces=[_workspace_payload(workspace_id="ws-active", name="Alpha")],
        list_calls=list_calls,
    )
    workflow, sent = _workflow(transport=transport)
    event = _event(event_id="Ev-dup-ws")
    await workflow.handle(event)
    await workflow.handle(event)
    assert len(list_calls) == 1
    assert len(sent) == 1


@pytest.mark.asyncio
async def test_list_error_safe_message_no_internals() -> None:
    transport = _list_transport(status_code=500)
    workflow, sent = _workflow(transport=transport)
    await workflow.handle(_event(event_id="Ev-err"))
    assert len(sent) == 1
    assert sent[0].text == GENERIC_ERROR_TEXT
    assert "Traceback" not in sent[0].text
    assert "500" not in sent[0].text
    assert "tenant-a" not in sent[0].text


@pytest.mark.asyncio
async def test_ask_happy_path_still_works() -> None:
    ask_calls: list[httpx.Request] = []
    list_calls: list[httpx.Request] = []
    transport = _list_transport(ask_calls=ask_calls, list_calls=list_calls)
    workflow, sent = _workflow(transport=transport)
    await workflow.handle(_event(text="What is leave policy?", event_id="Ev-ask-ok"))
    assert len(ask_calls) == 1
    assert len(list_calls) == 0
    assert len(sent) == 2
    assert "Ask answer" in sent[1].text


@pytest.mark.asyncio
async def test_listing_and_ask_use_identical_tenant_header() -> None:
    """Slack listing and regular Ask must send the same X-Tenant-Id."""
    ask_calls: list[httpx.Request] = []
    list_calls: list[httpx.Request] = []
    transport = _list_transport(
        workspaces=[_workspace_payload(workspace_id="ws-active", name="Alpha")],
        ask_calls=ask_calls,
        list_calls=list_calls,
    )
    workflow, _sent = _workflow(transport=transport)

    await workflow.handle(_event(text="workspaces", event_id="Ev-list-tenant"))
    await workflow.handle(_event(text="What is leave policy?", event_id="Ev-ask-tenant"))

    assert len(list_calls) == 1
    assert len(ask_calls) == 1
    assert list_calls[0].headers["X-Tenant-Id"] == ask_calls[0].headers["X-Tenant-Id"]
    assert list_calls[0].headers["X-Tenant-Id"] == "tenant-a"