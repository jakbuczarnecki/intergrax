# © Artur Czarnecki. All rights reserved.

"""LKW-WORKSPACE-MANAGEMENT-1 — Slack create / delete lifecycle commands."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
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
from local_workspace_application.slack_companion.pending_deletion_store import (
    InMemorySlackPendingDeletionStore,
)
from local_workspace_application.slack_companion.rendering import (
    ACK_TEXT,
    NO_WORKSPACE_AVAILABLE_TEXT,
    WORKSPACE_CREATE_USAGE_TEXT,
    WORKSPACE_CREATED_PREFIX,
    WORKSPACE_DELETE_CANCELLED_TEXT,
    WORKSPACE_DELETE_CONFIRM_HEADER,
    WORKSPACE_DELETE_MISSING_PENDING_TEXT,
    WORKSPACE_DELETE_SUCCESS_PREFIX,
)
from local_workspace_application.slack_companion.selection_store import (
    InMemorySlackWorkspaceSelectionStore,
    SlackWorkspaceSelection,
    slack_selection_actor_key,
)
from local_workspace_application.slack_companion.workflow import (
    SlackAskWorkflow,
    is_workspace_create_attempt,
    normalize_workspace_display_name,
    parse_workspace_create,
    parse_workspace_delete,
)

pytestmark = pytest.mark.unit


def _event(
    *,
    event_id: str = "Ev-mgmt-1",
    team_id: str = "T_OK",
    user_id: str = "U_OK",
    text: str = "workspace create Demo",
) -> InboundConversationEvent:
    return InboundConversationEvent(
        event_id=event_id,
        address=ConversationAddress(
            installation_id=team_id,
            conversation_id="Dchannel",
            thread_id="1713333.000400",
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


def _transport(
    *,
    workspaces: list[dict[str, object]] | None = None,
    create_status: int = 201,
    delete_status: int = 204,
    list_status: int = 200,
    ask_status: int = 200,
    create_calls: list[httpx.Request] | None = None,
    delete_calls: list[httpx.Request] | None = None,
    list_calls: list[httpx.Request] | None = None,
    ask_calls: list[httpx.Request] | None = None,
    created_body: dict[str, object] | None = None,
) -> httpx.MockTransport:
    workspace_bucket = list(workspaces or [])
    create_bucket = create_calls if create_calls is not None else []
    delete_bucket = delete_calls if delete_calls is not None else []
    list_bucket = list_calls if list_calls is not None else []
    ask_bucket = ask_calls if ask_calls is not None else []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/ask"):
            ask_bucket.append(request)
            if ask_status != 200:
                return httpx.Response(ask_status, json={"detail": "boom"})
            workspace_id = path.rstrip("/").split("/")[-2]
            return httpx.Response(
                200,
                json={
                    "run_id": "ask-run-1",
                    "workspace_id": workspace_id,
                    "status": "completed",
                    "question": "Q",
                    "answer": f"Ask answer for {workspace_id}",
                    "citations": [],
                },
            )
        if request.method == "GET" and path.rstrip("/").endswith("/workspaces"):
            list_bucket.append(request)
            if list_status != 200:
                return httpx.Response(list_status, json={"detail": "boom"})
            return httpx.Response(200, json={"workspaces": workspace_bucket})
        if request.method == "POST" and path.rstrip("/").endswith("/workspaces"):
            create_bucket.append(request)
            if create_status != 201:
                return httpx.Response(create_status, json={"detail": "boom"})
            body = created_body or _workspace_payload(
                workspace_id="ws-created",
                name="Demo Case",
            )
            workspace_bucket.insert(0, body)
            return httpx.Response(201, json=body)
        if request.method == "DELETE" and "/workspaces/" in path:
            delete_bucket.append(request)
            if delete_status == 204:
                return httpx.Response(204)
            return httpx.Response(delete_status, json={"detail": "not_found"})
        return httpx.Response(404, json={"detail": "not_found"})

    return httpx.MockTransport(handler)


async def _run(
    text: str,
    *,
    event_id: str = "Ev-mgmt-1",
    team_id: str = "T_OK",
    user_id: str = "U_OK",
    configured_workspace_id: str = "ws-configured",
    tenant_id: str = "tenant-a",
    workspaces: list[dict[str, object]] | None = None,
    create_calls: list[httpx.Request] | None = None,
    delete_calls: list[httpx.Request] | None = None,
    list_calls: list[httpx.Request] | None = None,
    ask_calls: list[httpx.Request] | None = None,
    selection_store: InMemorySlackWorkspaceSelectionStore | None = None,
    pending_store: InMemorySlackPendingDeletionStore | None = None,
    create_status: int = 201,
    delete_status: int = 204,
    created_body: dict[str, object] | None = None,
) -> list[str]:
    outbound: list[str] = []

    async def send(message: OutboundConversationMessage) -> ConversationDeliveryReceipt:
        outbound.append(message.text)
        return ConversationDeliveryReceipt(
            message_id="m1",
            address=message.address,
            delivered_at=datetime.now(UTC),
        )

    client = WorkspaceAskHttpClient(
        SlackAskClientConfig(base_url="http://lkw.test/"),
        transport=_transport(
            workspaces=workspaces,
            create_calls=create_calls,
            delete_calls=delete_calls,
            list_calls=list_calls,
            ask_calls=ask_calls,
            create_status=create_status,
            delete_status=delete_status,
            created_body=created_body,
        ),
    )
    workflow = SlackAskWorkflow(
        auth_config=SlackCompanionAuthConfig(
            approved_team_id="T_OK",
            approved_user_id="U_OK",
            tenant_id=tenant_id,
            active_workspace_id=configured_workspace_id,
        ),
        dedupe=SlackEventDedupeRepository(InMemoryDocumentStore()),
        ask_client=client,
        send=send,
        selection_store=selection_store or InMemorySlackWorkspaceSelectionStore(),
        pending_deletion_store=pending_store or InMemorySlackPendingDeletionStore(),
    )
    await workflow.handle(
        _event(event_id=event_id, team_id=team_id, user_id=user_id, text=text)
    )
    return outbound


def test_create_parser_and_invalid() -> None:
    assert parse_workspace_create("workspace create Demo Case") == "Demo Case"
    assert parse_workspace_create("  WORKSPACE   CREATE   Demo   Case  ") == "Demo Case"
    assert parse_workspace_create("workspace create") is None
    assert parse_workspace_create("workspace create \nBad") is None
    assert normalize_workspace_display_name("x" * 101) is None
    assert is_workspace_create_attempt("workspace create")
    assert parse_workspace_delete("workspace delete 2") == 2
    assert parse_workspace_delete("workspace delete 0") is None


@pytest.mark.asyncio
async def test_create_http_uses_tenant_zero_ask_and_selects() -> None:
    create_calls: list[httpx.Request] = []
    ask_calls: list[httpx.Request] = []
    list_calls: list[httpx.Request] = []
    selections = InMemorySlackWorkspaceSelectionStore()
    outbound = await _run(
        "workspace create Demo Case",
        create_calls=create_calls,
        ask_calls=ask_calls,
        list_calls=list_calls,
        selection_store=selections,
        created_body=_workspace_payload(workspace_id="ws-new", name="Demo Case"),
    )
    assert create_calls and create_calls[0].headers.get("X-Tenant-Id") == "tenant-a"
    assert ask_calls == []
    assert list_calls == []
    assert outbound == [f"{WORKSPACE_CREATED_PREFIX}Demo Case"]
    actor = slack_selection_actor_key(team_id="T_OK", user_id="U_OK")
    selected = selections.get(actor)
    assert selected is not None
    assert selected.workspace_id == "ws-new"

    list_outbound = await _run(
        "workspaces",
        event_id="Ev-list-after-create",
        selection_store=selections,
        workspaces=[
            _workspace_payload(workspace_id="ws-new", name="Demo Case"),
            _workspace_payload(workspace_id="ws-configured", name="Configured"),
        ],
        ask_calls=ask_calls,
    )
    assert "1. Demo Case — active" in list_outbound[0]
    assert ask_calls == []


@pytest.mark.asyncio
async def test_invalid_create_usage() -> None:
    outbound = await _run("workspace create")
    assert outbound == [WORKSPACE_CREATE_USAGE_TEXT]


@pytest.mark.asyncio
async def test_delete_request_zero_delete_uses_fresh_list_then_confirm_uses_stored_id() -> None:
    delete_calls: list[httpx.Request] = []
    list_calls: list[httpx.Request] = []
    selections = InMemorySlackWorkspaceSelectionStore()
    pending = InMemorySlackPendingDeletionStore()
    actor = slack_selection_actor_key(team_id="T_OK", user_id="U_OK")
    selections.set(
        actor,
        SlackWorkspaceSelection(workspace_id="ws-a", workspace_name="Alpha"),
    )
    workspaces = [
        _workspace_payload(workspace_id="ws-a", name="Alpha"),
        _workspace_payload(workspace_id="ws-b", name="Beta"),
    ]
    request_out = await _run(
        "workspace delete 2",
        delete_calls=delete_calls,
        list_calls=list_calls,
        selection_store=selections,
        pending_store=pending,
        workspaces=workspaces,
    )
    assert delete_calls == []
    assert len(list_calls) == 1
    assert WORKSPACE_DELETE_CONFIRM_HEADER in request_out[0]
    assert "Beta" in request_out[0]
    stored = pending.get(actor)
    assert stored is not None
    assert stored.workspace_id == "ws-b"

    confirm_out = await _run(
        "workspace delete confirm",
        event_id="Ev-del-confirm",
        delete_calls=delete_calls,
        list_calls=list_calls,
        selection_store=selections,
        pending_store=pending,
        workspaces=workspaces,
    )
    assert len(delete_calls) == 1
    assert delete_calls[0].url.path.endswith("/workspaces/ws-b")
    assert confirm_out[0].startswith(WORKSPACE_DELETE_SUCCESS_PREFIX)
    assert selections.get(actor) is not None  # non-selected preserved
    assert pending.get(actor) is None


@pytest.mark.asyncio
async def test_expired_pending_does_not_delete() -> None:
    delete_calls: list[httpx.Request] = []
    pending = InMemorySlackPendingDeletionStore(ttl=timedelta(seconds=1))
    actor = slack_selection_actor_key(team_id="T_OK", user_id="U_OK")
    pending.set(actor, workspace_id="ws-b", workspace_name="Beta")
    # Force expiry.
    with pending._lock:  # noqa: SLF001 — test-only expiry injection
        current = pending._by_actor[actor]
        pending._by_actor[actor] = type(current)(
            workspace_id=current.workspace_id,
            workspace_name=current.workspace_name,
            requested_at=current.requested_at,
            expires_at=datetime.now(UTC) - timedelta(seconds=1),
        )
    outbound = await _run(
        "workspace delete confirm",
        delete_calls=delete_calls,
        pending_store=pending,
    )
    assert delete_calls == []
    assert outbound == [WORKSPACE_DELETE_MISSING_PENDING_TEXT]


@pytest.mark.asyncio
async def test_cancel_clears_pending() -> None:
    pending = InMemorySlackPendingDeletionStore()
    actor = slack_selection_actor_key(team_id="T_OK", user_id="U_OK")
    pending.set(actor, workspace_id="ws-b", workspace_name="Beta")
    outbound = await _run("workspace delete cancel", pending_store=pending)
    assert outbound == [WORKSPACE_DELETE_CANCELLED_TEXT]
    assert pending.get(actor) is None


@pytest.mark.asyncio
async def test_unauthorized_cannot_mutate_pending() -> None:
    pending = InMemorySlackPendingDeletionStore()
    outbound = await _run(
        "workspace delete 1",
        user_id="U_OTHER",
        pending_store=pending,
        workspaces=[_workspace_payload(workspace_id="ws-a", name="Alpha")],
    )
    assert outbound == []
    assert pending.get(slack_selection_actor_key(team_id="T_OK", user_id="U_OTHER")) is None


@pytest.mark.asyncio
async def test_duplicate_create_and_delete_confirm_once() -> None:
    create_calls: list[httpx.Request] = []
    delete_calls: list[httpx.Request] = []
    selections = InMemorySlackWorkspaceSelectionStore()
    pending = InMemorySlackPendingDeletionStore()
    store = InMemoryDocumentStore()
    outbound: list[str] = []

    async def send(message: OutboundConversationMessage) -> ConversationDeliveryReceipt:
        outbound.append(message.text)
        return ConversationDeliveryReceipt(
            message_id="m1",
            address=message.address,
            delivered_at=datetime.now(UTC),
        )

    client = WorkspaceAskHttpClient(
        SlackAskClientConfig(base_url="http://lkw.test/"),
        transport=_transport(
            create_calls=create_calls,
            delete_calls=delete_calls,
            workspaces=[_workspace_payload(workspace_id="ws-a", name="Alpha")],
            created_body=_workspace_payload(workspace_id="ws-new", name="Demo"),
        ),
    )
    workflow = SlackAskWorkflow(
        auth_config=SlackCompanionAuthConfig(
            approved_team_id="T_OK",
            approved_user_id="U_OK",
            tenant_id="tenant-a",
            active_workspace_id="ws-configured",
        ),
        dedupe=SlackEventDedupeRepository(store),
        ask_client=client,
        send=send,
        selection_store=selections,
        pending_deletion_store=pending,
    )
    event = _event(event_id="Ev-dup-create", text="workspace create Demo")
    await workflow.handle(event)
    await workflow.handle(event)
    assert len(create_calls) == 1

    await workflow.handle(_event(event_id="Ev-del-req", text="workspace delete 1"))
    confirm = _event(event_id="Ev-del-confirm-dup", text="workspace delete confirm")
    await workflow.handle(confirm)
    await workflow.handle(confirm)
    assert len(delete_calls) == 1


@pytest.mark.asyncio
async def test_delete_selected_clears_selection_nonselected_preserves() -> None:
    selections = InMemorySlackWorkspaceSelectionStore()
    pending = InMemorySlackPendingDeletionStore()
    actor = slack_selection_actor_key(team_id="T_OK", user_id="U_OK")
    selections.set(
        actor,
        SlackWorkspaceSelection(workspace_id="ws-a", workspace_name="Alpha"),
    )
    await _run(
        "workspace delete 1",
        event_id="Ev-del-sel-req",
        selection_store=selections,
        pending_store=pending,
        workspaces=[
            _workspace_payload(workspace_id="ws-a", name="Alpha"),
            _workspace_payload(workspace_id="ws-b", name="Beta"),
        ],
    )
    await _run(
        "workspace delete confirm",
        event_id="Ev-del-sel-confirm",
        selection_store=selections,
        pending_store=pending,
    )
    assert selections.get(actor) is None


@pytest.mark.asyncio
async def test_no_workspace_available_prevents_ask() -> None:
    ask_calls: list[httpx.Request] = []
    selections = InMemorySlackWorkspaceSelectionStore()
    actor = slack_selection_actor_key(team_id="T_OK", user_id="U_OK")
    selections.suppress_configured(actor)
    outbound = await _run(
        "What is in the docs?",
        ask_calls=ask_calls,
        selection_store=selections,
        configured_workspace_id="ws-configured",
    )
    assert ask_calls == []
    assert outbound == [NO_WORKSPACE_AVAILABLE_TEXT]
    assert ACK_TEXT not in outbound
