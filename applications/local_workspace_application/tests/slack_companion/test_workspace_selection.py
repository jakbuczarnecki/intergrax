# © Artur Czarnecki. All rights reserved.

"""LKW-SLACK-WORKFLOW-1B-2 — text ``workspace <n>`` selection (no Ask)."""

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
from local_workspace_application.slack_companion.rendering import (
    ACK_TEXT,
    GENERIC_ERROR_TEXT,
    SELECTED_WORKSPACE_UNAVAILABLE_TEXT,
    WORKSPACE_LIST_EMPTY_TEXT,
    WORKSPACE_LIST_LOAD_FAILED_TEXT,
    WORKSPACE_OUT_OF_RANGE_TEXT,
    WORKSPACE_SELECTION_USAGE_TEXT,
)
from local_workspace_application.slack_companion.selection_store import (
    InMemorySlackWorkspaceSelectionStore,
    SlackWorkspaceSelection,
    slack_selection_actor_key,
)
from local_workspace_application.slack_companion.workflow import (
    SlackAskWorkflow,
    is_workspace_selection_attempt,
    order_workspaces_for_listing,
    parse_workspace_selection,
    resolve_effective_workspace_id,
)
from local_workspace_application.slack_companion.models import SlackWorkspaceListItem

pytestmark = pytest.mark.unit


def _event(
    *,
    event_id: str = "Ev-sel-1",
    team_id: str = "T_OK",
    user_id: str = "U_OK",
    text: str = "workspace 1",
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
    list_status: int = 200,
    ask_status: int = 200,
    ask_calls: list[httpx.Request] | None = None,
    list_calls: list[httpx.Request] | None = None,
) -> httpx.MockTransport:
    ask_bucket = ask_calls if ask_calls is not None else []
    list_bucket = list_calls if list_calls is not None else []

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
                    "created_at": "2026-07-23T12:00:00Z",
                },
            )
        if path.rstrip("/").endswith("/workspaces"):
            list_bucket.append(request)
            if list_status != 200:
                return httpx.Response(list_status, json={"detail": "boom"})
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
    selections: InMemorySlackWorkspaceSelectionStore | None = None,
    sent: list[OutboundConversationMessage] | None = None,
    approved_team_id: str = "T_OK",
    approved_user_id: str = "U_OK",
    active_workspace_id: str = "ws-active",
) -> tuple[SlackAskWorkflow, list[OutboundConversationMessage], InMemorySlackWorkspaceSelectionStore]:
    outbound = sent if sent is not None else []
    store = selections if selections is not None else InMemorySlackWorkspaceSelectionStore()

    async def send(message: OutboundConversationMessage) -> Any:
        outbound.append(message)
        return ConversationDeliveryReceipt(
            message_id=f"msg-{len(outbound)}",
            address=message.address,
            delivered_at=datetime.now(timezone.utc),
        )

    workflow = SlackAskWorkflow(
        auth_config=SlackCompanionAuthConfig(
            approved_team_id=approved_team_id,
            approved_user_id=approved_user_id,
            tenant_id="tenant-a",
            active_workspace_id=active_workspace_id,
        ),
        dedupe=dedupe or SlackEventDedupeRepository(InMemoryDocumentStore()),
        ask_client=WorkspaceAskHttpClient(
            SlackAskClientConfig(base_url="http://ask.test"),
            transport=transport,
        ),
        send=send,
        selection_store=store,
    )
    return workflow, outbound, store


_DEFAULT_WORKSPACES = [
    _workspace_payload(workspace_id="ws-beta", name="Workspace Beta"),
    _workspace_payload(workspace_id="ws-active", name="Workspace Alpha"),
    _workspace_payload(
        workspace_id="ws-archived",
        name="Archived Room",
        status="archived",
    ),
]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("workspace 1", 1),
        ("Workspace 2", 2),
        ("  workspace   3  ", 3),
        ("WORKSPACE 10", 10),
    ],
)
def test_parser_accepts_valid_selection(text: str, expected: int) -> None:
    assert parse_workspace_selection(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "workspace",
        "workspace 0",
        "workspace -1",
        "workspace two",
        "workspace 1 extra",
        "select workspace 1",
        "workspaces",
    ],
)
def test_parser_rejects_invalid_selection(text: str) -> None:
    assert parse_workspace_selection(text) is None


def test_selection_attempt_detection() -> None:
    assert is_workspace_selection_attempt("workspace") is True
    assert is_workspace_selection_attempt("workspace 0") is True
    assert is_workspace_selection_attempt("workspaces") is False
    assert is_workspace_selection_attempt("select workspace 1") is False


def test_selection_ordering_matches_workspaces_command() -> None:
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


@pytest.mark.asyncio
async def test_valid_selection_stores_and_confirms_name_only() -> None:
    ask_calls: list[httpx.Request] = []
    list_calls: list[httpx.Request] = []
    transport = _transport(
        workspaces=_DEFAULT_WORKSPACES,
        ask_calls=ask_calls,
        list_calls=list_calls,
    )
    workflow, sent, store = _workflow(transport=transport)
    await workflow.handle(_event(text="workspace 2", event_id="Ev-sel-ok"))

    assert len(ask_calls) == 0
    assert len(list_calls) == 1
    assert list_calls[0].headers["X-Tenant-Id"] == "tenant-a"
    assert len(sent) == 1
    assert sent[0].text == "Selected workspace: Workspace Beta"
    assert ACK_TEXT not in sent[0].text
    assert "ws-beta" not in sent[0].text
    assert "tenant-a" not in sent[0].text
    key = slack_selection_actor_key(team_id="T_OK", user_id="U_OK")
    selection = store.get(key)
    assert selection is not None
    assert selection.workspace_id == "ws-beta"
    assert selection.workspace_name == "Workspace Beta"


@pytest.mark.asyncio
async def test_selection_uses_same_order_as_list_index() -> None:
    """Index 1 is configured active (Alpha); index 2 is Beta after name sort."""
    ask_calls: list[httpx.Request] = []
    transport = _transport(workspaces=_DEFAULT_WORKSPACES, ask_calls=ask_calls)
    workflow, sent, store = _workflow(transport=transport)
    await workflow.handle(_event(text="workspace 1", event_id="Ev-idx-1"))
    key = slack_selection_actor_key(team_id="T_OK", user_id="U_OK")
    assert store.get(key) is not None
    assert store.get(key).workspace_id == "ws-active"
    assert sent[0].text == "Selected workspace: Workspace Alpha"
    assert len(ask_calls) == 0


@pytest.mark.asyncio
async def test_out_of_range_does_not_mutate_prior_selection() -> None:
    transport = _transport(workspaces=_DEFAULT_WORKSPACES)
    workflow, sent, store = _workflow(transport=transport)
    key = slack_selection_actor_key(team_id="T_OK", user_id="U_OK")
    store.set(
        key,
        SlackWorkspaceSelection(workspace_id="ws-beta", workspace_name="Workspace Beta"),
    )
    await workflow.handle(_event(text="workspace 99", event_id="Ev-oor"))
    assert sent[0].text == WORKSPACE_OUT_OF_RANGE_TEXT
    assert store.get(key) is not None
    assert store.get(key).workspace_id == "ws-beta"


@pytest.mark.asyncio
async def test_empty_list_safe_response_no_mutation() -> None:
    transport = _transport(workspaces=[])
    workflow, sent, store = _workflow(transport=transport)
    key = slack_selection_actor_key(team_id="T_OK", user_id="U_OK")
    store.set(
        key,
        SlackWorkspaceSelection(workspace_id="ws-beta", workspace_name="Workspace Beta"),
    )
    await workflow.handle(_event(text="workspace 1", event_id="Ev-empty"))
    assert sent[0].text == WORKSPACE_LIST_EMPTY_TEXT
    assert store.get(key).workspace_id == "ws-beta"


@pytest.mark.asyncio
async def test_listing_http_failure_safe_error() -> None:
    transport = _transport(workspaces=_DEFAULT_WORKSPACES, list_status=500)
    workflow, sent, _store = _workflow(transport=transport)
    await workflow.handle(_event(text="workspace 1", event_id="Ev-list-fail"))
    assert sent[0].text == WORKSPACE_LIST_LOAD_FAILED_TEXT
    assert "500" not in sent[0].text
    assert "Traceback" not in sent[0].text


@pytest.mark.asyncio
async def test_invalid_selection_usage_no_list_no_ask() -> None:
    ask_calls: list[httpx.Request] = []
    list_calls: list[httpx.Request] = []
    transport = _transport(
        workspaces=_DEFAULT_WORKSPACES,
        ask_calls=ask_calls,
        list_calls=list_calls,
    )
    workflow, sent, store = _workflow(transport=transport)
    await workflow.handle(_event(text="workspace 0", event_id="Ev-usage"))
    assert len(ask_calls) == 0
    assert len(list_calls) == 0
    assert sent[0].text == WORKSPACE_SELECTION_USAGE_TEXT
    assert store.get(slack_selection_actor_key(team_id="T_OK", user_id="U_OK")) is None


@pytest.mark.asyncio
async def test_next_regular_ask_uses_selected_workspace() -> None:
    ask_calls: list[httpx.Request] = []
    list_calls: list[httpx.Request] = []
    transport = _transport(
        workspaces=_DEFAULT_WORKSPACES,
        ask_calls=ask_calls,
        list_calls=list_calls,
    )
    workflow, sent, _store = _workflow(transport=transport)
    await workflow.handle(_event(text="workspace 2", event_id="Ev-sel-then-ask"))
    await workflow.handle(_event(text="What is leave policy?", event_id="Ev-ask-sel"))

    assert len(list_calls) == 1
    assert len(ask_calls) == 1
    assert "/workspaces/ws-beta/ask" in str(ask_calls[0].url)
    assert ask_calls[0].headers["X-Tenant-Id"] == "tenant-a"
    assert list_calls[0].headers["X-Tenant-Id"] == ask_calls[0].headers["X-Tenant-Id"]
    assert ACK_TEXT in sent[1].text
    assert "Ask answer for ws-beta" in sent[2].text


@pytest.mark.asyncio
async def test_regular_ask_without_selection_uses_configured() -> None:
    ask_calls: list[httpx.Request] = []
    transport = _transport(workspaces=_DEFAULT_WORKSPACES, ask_calls=ask_calls)
    workflow, _sent, _store = _workflow(transport=transport)
    await workflow.handle(_event(text="What is leave policy?", event_id="Ev-ask-cfg"))
    assert len(ask_calls) == 1
    assert "/workspaces/ws-active/ask" in str(ask_calls[0].url)


@pytest.mark.asyncio
async def test_selection_scoped_by_team_and_user() -> None:
    transport = _transport(workspaces=_DEFAULT_WORKSPACES)
    shared = InMemorySlackWorkspaceSelectionStore()
    workflow_a, _sent_a, _ = _workflow(transport=transport, selections=shared)
    workflow_b, _sent_b, _ = _workflow(
        transport=transport,
        selections=shared,
        approved_team_id="T_OTHER",
        approved_user_id="U_OTHER",
    )
    await workflow_a.handle(_event(text="workspace 2", event_id="Ev-scope-a"))
    await workflow_b.handle(
        _event(
            text="workspace 1",
            event_id="Ev-scope-b",
            team_id="T_OTHER",
            user_id="U_OTHER",
        )
    )
    key_a = slack_selection_actor_key(team_id="T_OK", user_id="U_OK")
    key_b = slack_selection_actor_key(team_id="T_OTHER", user_id="U_OTHER")
    assert shared.get(key_a).workspace_id == "ws-beta"
    assert shared.get(key_b).workspace_id == "ws-active"


@pytest.mark.asyncio
async def test_unauthorized_does_not_read_or_mutate_selection() -> None:
    list_calls: list[httpx.Request] = []
    ask_calls: list[httpx.Request] = []
    transport = _transport(
        workspaces=_DEFAULT_WORKSPACES,
        ask_calls=ask_calls,
        list_calls=list_calls,
    )
    workflow, sent, store = _workflow(transport=transport)
    key = slack_selection_actor_key(team_id="T_OK", user_id="U_OK")
    store.set(
        key,
        SlackWorkspaceSelection(workspace_id="ws-beta", workspace_name="Workspace Beta"),
    )
    await workflow.handle(
        _event(text="workspace 1", event_id="Ev-unauth", team_id="T_BAD", user_id="U_BAD")
    )
    assert len(list_calls) == 0
    assert len(ask_calls) == 0
    assert len(sent) == 0
    assert store.get(key).workspace_id == "ws-beta"


@pytest.mark.asyncio
async def test_duplicate_event_does_not_repeat_selection() -> None:
    list_calls: list[httpx.Request] = []
    transport = _transport(workspaces=_DEFAULT_WORKSPACES, list_calls=list_calls)
    workflow, sent, store = _workflow(transport=transport)
    event = _event(text="workspace 2", event_id="Ev-dup-sel")
    await workflow.handle(event)
    await workflow.handle(event)
    assert len(list_calls) == 1
    assert len(sent) == 1
    key = slack_selection_actor_key(team_id="T_OK", user_id="U_OK")
    assert store.get(key).workspace_id == "ws-beta"


@pytest.mark.asyncio
async def test_selected_workspace_ask_404_clears_without_fallback_retry() -> None:
    ask_calls: list[httpx.Request] = []
    transport = _transport(
        workspaces=_DEFAULT_WORKSPACES,
        ask_status=404,
        ask_calls=ask_calls,
    )
    workflow, sent, store = _workflow(transport=transport)
    key = slack_selection_actor_key(team_id="T_OK", user_id="U_OK")
    store.set(
        key,
        SlackWorkspaceSelection(workspace_id="ws-beta", workspace_name="Workspace Beta"),
    )
    await workflow.handle(_event(text="What happened?", event_id="Ev-404"))
    assert len(ask_calls) == 1
    assert "/workspaces/ws-beta/ask" in str(ask_calls[0].url)
    assert store.get(key) is None
    assert SELECTED_WORKSPACE_UNAVAILABLE_TEXT in sent[-1].text
    assert GENERIC_ERROR_TEXT not in sent[-1].text


@pytest.mark.asyncio
async def test_timeout_does_not_clear_selection() -> None:
    ask_calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/ask"):
            ask_calls.append(request)
            raise httpx.ReadTimeout("slow")
        return httpx.Response(200, json={"workspaces": _DEFAULT_WORKSPACES})

    transport = httpx.MockTransport(handler)
    workflow, sent, store = _workflow(transport=transport)
    key = slack_selection_actor_key(team_id="T_OK", user_id="U_OK")
    store.set(
        key,
        SlackWorkspaceSelection(workspace_id="ws-beta", workspace_name="Workspace Beta"),
    )
    await workflow.handle(_event(text="What happened?", event_id="Ev-timeout"))
    assert len(ask_calls) == 1
    assert store.get(key) is not None
    assert store.get(key).workspace_id == "ws-beta"
    assert sent[-1].text == GENERIC_ERROR_TEXT


@pytest.mark.asyncio
async def test_http_502_does_not_clear_selection() -> None:
    ask_calls: list[httpx.Request] = []
    transport = _transport(
        workspaces=_DEFAULT_WORKSPACES,
        ask_status=502,
        ask_calls=ask_calls,
    )
    workflow, sent, store = _workflow(transport=transport)
    key = slack_selection_actor_key(team_id="T_OK", user_id="U_OK")
    store.set(
        key,
        SlackWorkspaceSelection(workspace_id="ws-beta", workspace_name="Workspace Beta"),
    )
    await workflow.handle(_event(text="What happened?", event_id="Ev-502"))
    assert len(ask_calls) == 1
    assert store.get(key).workspace_id == "ws-beta"
    assert sent[-1].text == GENERIC_ERROR_TEXT


@pytest.mark.asyncio
async def test_workspaces_command_still_zero_ask() -> None:
    ask_calls: list[httpx.Request] = []
    list_calls: list[httpx.Request] = []
    transport = _transport(
        workspaces=_DEFAULT_WORKSPACES,
        ask_calls=ask_calls,
        list_calls=list_calls,
    )
    workflow, sent, _store = _workflow(transport=transport)
    await workflow.handle(_event(text="workspaces", event_id="Ev-list-still"))
    assert len(ask_calls) == 0
    assert len(list_calls) == 1
    assert "1. Workspace Alpha — active" in sent[0].text
    assert "2. Workspace Beta" in sent[0].text


@pytest.mark.asyncio
async def test_without_selection_configured_is_effective_active() -> None:
    ask_calls: list[httpx.Request] = []
    transport = _transport(workspaces=_DEFAULT_WORKSPACES, ask_calls=ask_calls)
    workflow, sent, store = _workflow(transport=transport)
    key = slack_selection_actor_key(team_id="T_OK", user_id="U_OK")
    assert store.get(key) is None
    assert (
        resolve_effective_workspace_id(None, configured_workspace_id="ws-active")
        == "ws-active"
    )
    await workflow.handle(_event(text="workspaces", event_id="Ev-eff-cfg-list"))
    assert "1. Workspace Alpha — active" in sent[0].text
    assert "2. Workspace Beta" in sent[0].text
    assert "Workspace Beta — active" not in sent[0].text
    await workflow.handle(_event(text="What is leave policy?", event_id="Ev-eff-cfg-ask"))
    assert len(ask_calls) == 1
    assert "/workspaces/ws-active/ask" in str(ask_calls[0].url)


@pytest.mark.asyncio
async def test_after_selection_workspaces_marks_selected_active() -> None:
    ask_calls: list[httpx.Request] = []
    list_calls: list[httpx.Request] = []
    transport = _transport(
        workspaces=_DEFAULT_WORKSPACES,
        ask_calls=ask_calls,
        list_calls=list_calls,
    )
    workflow, sent, store = _workflow(transport=transport)
    await workflow.handle(_event(text="workspace 2", event_id="Ev-eff-sel"))
    await workflow.handle(_event(text="What is leave policy?", event_id="Ev-eff-ask"))
    await workflow.handle(_event(text="workspaces", event_id="Ev-eff-list"))

    key = slack_selection_actor_key(team_id="T_OK", user_id="U_OK")
    selection = store.get(key)
    assert selection is not None
    assert selection.workspace_id == "ws-beta"
    assert len(ask_calls) == 1
    assert "/workspaces/ws-beta/ask" in str(ask_calls[0].url)
    list_text = sent[-1].text
    assert "1. Workspace Beta — active" in list_text
    assert "Workspace Alpha — active" not in list_text
    assert "2. Workspace Alpha" in list_text


@pytest.mark.asyncio
async def test_selected_active_first_ordering_in_workspaces() -> None:
    transport = _transport(workspaces=_DEFAULT_WORKSPACES)
    workflow, sent, store = _workflow(transport=transport)
    key = slack_selection_actor_key(team_id="T_OK", user_id="U_OK")
    store.set(
        key,
        SlackWorkspaceSelection(workspace_id="ws-beta", workspace_name="Workspace Beta"),
    )
    await workflow.handle(_event(text="workspaces", event_id="Ev-eff-order"))
    lines = [line for line in sent[0].text.splitlines() if line and line[0].isdigit()]
    assert lines[0] == "1. Workspace Beta — active"
    assert lines[1] == "2. Workspace Alpha"


@pytest.mark.asyncio
async def test_after_404_clear_configured_becomes_active_again() -> None:
    ask_calls: list[httpx.Request] = []
    transport = _transport(
        workspaces=_DEFAULT_WORKSPACES,
        ask_status=404,
        ask_calls=ask_calls,
    )
    workflow, sent, store = _workflow(transport=transport)
    key = slack_selection_actor_key(team_id="T_OK", user_id="U_OK")
    store.set(
        key,
        SlackWorkspaceSelection(workspace_id="ws-beta", workspace_name="Workspace Beta"),
    )
    await workflow.handle(_event(text="What happened?", event_id="Ev-eff-404"))
    assert store.get(key) is None
    assert SELECTED_WORKSPACE_UNAVAILABLE_TEXT in sent[-1].text

    transport_ok = _transport(workspaces=_DEFAULT_WORKSPACES)
    workflow_ok, sent_ok, store_ok = _workflow(transport=transport_ok, selections=store)
    await workflow_ok.handle(_event(text="workspaces", event_id="Ev-eff-404-list"))
    assert "1. Workspace Alpha — active" in sent_ok[0].text
    assert "Workspace Beta — active" not in sent_ok[0].text
    assert store_ok.get(key) is None


@pytest.mark.asyncio
async def test_restart_semantics_new_store_uses_configured_active() -> None:
    transport = _transport(workspaces=_DEFAULT_WORKSPACES)
    workflow_old, _sent_old, store_old = _workflow(transport=transport)
    await workflow_old.handle(_event(text="workspace 2", event_id="Ev-restart-sel"))
    key = slack_selection_actor_key(team_id="T_OK", user_id="U_OK")
    assert store_old.get(key) is not None

    workflow_new, sent_new, store_new = _workflow(transport=transport)
    assert store_new.get(key) is None
    await workflow_new.handle(_event(text="workspaces", event_id="Ev-restart-list"))
    assert "1. Workspace Alpha — active" in sent_new[0].text
    assert "Workspace Beta — active" not in sent_new[0].text


@pytest.mark.asyncio
async def test_active_marker_isolated_per_actor_key() -> None:
    transport = _transport(workspaces=_DEFAULT_WORKSPACES)
    shared = InMemorySlackWorkspaceSelectionStore()
    workflow_a, sent_a, _ = _workflow(transport=transport, selections=shared)
    workflow_b, sent_b, _ = _workflow(
        transport=transport,
        selections=shared,
        approved_team_id="T_OTHER",
        approved_user_id="U_OTHER",
    )
    await workflow_a.handle(_event(text="workspace 2", event_id="Ev-iso-sel-a"))
    await workflow_a.handle(_event(text="workspaces", event_id="Ev-iso-list-a"))
    await workflow_b.handle(
        _event(
            text="workspaces",
            event_id="Ev-iso-list-b",
            team_id="T_OTHER",
            user_id="U_OTHER",
        )
    )
    assert "1. Workspace Beta — active" in sent_a[-1].text
    assert "Workspace Alpha — active" not in sent_a[-1].text
    assert "1. Workspace Alpha — active" in sent_b[-1].text
    assert "Workspace Beta — active" not in sent_b[-1].text


@pytest.mark.asyncio
async def test_existing_1a_happy_path_remains_green() -> None:
    ask_calls: list[httpx.Request] = []
    list_calls: list[httpx.Request] = []
    transport = _transport(
        workspaces=_DEFAULT_WORKSPACES,
        ask_calls=ask_calls,
        list_calls=list_calls,
    )
    workflow, sent, _store = _workflow(transport=transport)
    await workflow.handle(_event(text="What is leave policy?", event_id="Ev-1a"))
    assert len(ask_calls) == 1
    assert len(list_calls) == 0
    assert len(sent) == 2
    assert sent[0].text == ACK_TEXT
    assert "Ask answer for ws-active" in sent[1].text
