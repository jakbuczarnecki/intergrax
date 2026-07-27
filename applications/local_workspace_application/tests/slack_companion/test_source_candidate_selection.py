# © Artur Czarnecki. All rights reserved.

"""LKW-WORKSPACE-CONTENTS-1B-4-2 — Slack Source Candidate selection."""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.document_store import DocumentRecord
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
    SlackSourceCandidateAcceptResponse,
    SlackSourceCandidateListItem,
)
from local_workspace_application.slack_companion.rendering import (
    MAX_SOURCE_CANDIDATE_ITEMS,
    NO_WORKSPACE_AVAILABLE_TEXT,
    SELECTED_WORKSPACE_UNAVAILABLE_TEXT,
    SOURCE_CANDIDATE_ACCEPTED_FOOTER,
    SOURCE_CANDIDATE_ALREADY_ATTACHED_TEXT,
    SOURCE_CANDIDATE_LIST_EMPTY_TEXT,
    SOURCE_CANDIDATE_LIST_LOAD_FAILED_TEXT,
    SOURCE_CANDIDATE_OUT_OF_RANGE_TEXT,
    SOURCE_CANDIDATE_SERVICE_UNAVAILABLE_TEXT,
    SOURCE_CANDIDATE_UNAVAILABLE_TEXT,
    SOURCE_CANDIDATE_USAGE_TEXT,
    render_source_candidate_accepted,
    render_source_candidate_list,
)
from local_workspace_application.slack_companion.selection_store import (
    InMemorySlackWorkspaceSelectionStore,
    SlackWorkspaceSelection,
    slack_selection_actor_key,
)
from local_workspace_application.slack_companion.workflow import (
    SlackAskWorkflow,
    is_sources_command,
    order_source_candidates_for_listing,
    parse_source_candidate_accept_command,
    parse_source_candidate_accept_invalid_command,
    parse_source_candidates_list_command,
    parse_sources_list_command,
    slack_source_candidate_intake_idempotency_key,
)

pytestmark = pytest.mark.unit


def _event(
    *,
    event_id: str = "Ev-cand-1",
    team_id: str = "T_OK",
    user_id: str = "U_OK",
    text: str = "source candidates",
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


def _candidate_payload(
    *,
    candidate_id: str,
    label: str,
    description: str = "",
    source_type: str = "local_folder",
    available: bool = True,
    path: str | None = None,
    fingerprint: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "candidate_id": candidate_id,
        "label": label,
        "description": description,
        "source_type": source_type,
        "available": available,
    }
    if path is not None:
        payload["path"] = path
    if fingerprint is not None:
        payload["candidate_fingerprint"] = fingerprint
    return payload


def _transport(
    *,
    candidates: list[dict[str, object]] | None = None,
    list_status: int = 200,
    accept_status: int = 202,
    accept_payload: dict[str, object] | None = None,
    ask_calls: list[httpx.Request] | None = None,
    list_calls: list[httpx.Request] | None = None,
    accept_calls: list[httpx.Request] | None = None,
    list_malformed: bool = False,
    accept_malformed: bool = False,
    timeout_on: str | None = None,
    transport_error_on: str | None = None,
) -> httpx.MockTransport:
    ask_bucket = ask_calls if ask_calls is not None else []
    list_bucket = list_calls if list_calls is not None else []
    accept_bucket = accept_calls if accept_calls is not None else []

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
            return httpx.Response(200, json={"workspaces": []})
        if path.endswith("/source-candidates") and request.method == "GET":
            list_bucket.append(request)
            if timeout_on == "list":
                raise httpx.ReadTimeout("timeout", request=request)
            if transport_error_on == "list":
                raise httpx.ConnectError("down", request=request)
            if list_status != 200:
                return httpx.Response(
                    list_status,
                    json={"detail": "boom-secret-path-/srv/private"},
                )
            if list_malformed:
                return httpx.Response(200, text="not-json")
            return httpx.Response(
                200,
                json={
                    "workspace_id": path.split("/")[-2],
                    "candidates": candidates if candidates is not None else [],
                },
            )
        if "/knowledge/source-candidates/" in path and request.method == "POST":
            accept_bucket.append(request)
            if timeout_on == "accept":
                raise httpx.ReadTimeout("timeout", request=request)
            if transport_error_on == "accept":
                raise httpx.ConnectError("down", request=request)
            if accept_status < 200 or accept_status >= 300:
                return httpx.Response(
                    accept_status,
                    json={"detail": "boom-secret-path-/srv/private"},
                )
            if accept_malformed:
                return httpx.Response(202, text="not-json")
            workspace_id = path.split("/")[-4]
            candidate_id = path.rstrip("/").split("/")[-1]
            body = accept_payload or {
                "candidate_id": candidate_id,
                "label": "Contracts",
                "workspace_id": workspace_id,
                "source_id": "src-1",
                "operation_id": "op-1",
                "status": "queued",
            }
            return httpx.Response(accept_status, json=body)
        return httpx.Response(404, json={"detail": "missing"})

    return httpx.MockTransport(handler)


async def _run(
    text: str,
    *,
    transport: httpx.MockTransport | None = None,
    candidates: list[dict[str, object]] | None = None,
    workspace_id: str = "ws-configured",
    selection_store: InMemorySlackWorkspaceSelectionStore | None = None,
    event_id: str = "Ev-cand-1",
    api_key: str | None = None,
    send_error: Exception | None = None,
    suppress_configured: bool = False,
) -> tuple[list[str], SlackAskWorkflow, SlackEventDedupeRepository]:
    outbound: list[str] = []

    async def send(message: OutboundConversationMessage) -> ConversationDeliveryReceipt:
        if send_error is not None:
            raise send_error
        outbound.append(message.text)
        return ConversationDeliveryReceipt(
            message_id="m1",
            address=message.address,
            delivered_at=datetime.now(UTC),
        )

    store = selection_store or InMemorySlackWorkspaceSelectionStore()
    if suppress_configured:
        store.suppress_configured(
            slack_selection_actor_key(team_id="T_OK", user_id="U_OK")
        )
    dedupe = SlackEventDedupeRepository(InMemoryDocumentStore())
    client = WorkspaceAskHttpClient(
        SlackAskClientConfig(
            base_url="http://lkw.test",
            api_key=api_key,
        ),
        transport=transport
        or _transport(candidates=candidates if candidates is not None else []),
    )
    workflow = SlackAskWorkflow(
        auth_config=SlackCompanionAuthConfig(
            approved_team_id="T_OK",
            approved_user_id="U_OK",
            tenant_id="tenant-a",
            active_workspace_id=workspace_id,
        ),
        dedupe=dedupe,
        ask_client=client,
        send=send,
        selection_store=store,
    )
    await workflow.handle(_event(text=text, event_id=event_id))
    return outbound, workflow, dedupe


# --- Parser ---


@pytest.mark.parametrize(
    "text",
    ["source candidates", "SOURCE CANDIDATES", "  Source Candidates  "],
)
def test_parse_source_candidates_list_exact(text: str) -> None:
    assert parse_source_candidates_list_command(text) is not None


@pytest.mark.parametrize(
    "text",
    [
        "source candidate",
        "source candidates 1",
        "show source candidates",
        "what source candidates exist?",
        "sources",
        "source add 1",
    ],
)
def test_parse_source_candidates_list_rejects(text: str) -> None:
    assert parse_source_candidates_list_command(text) is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("source add 1", 1),
        ("SOURCE ADD 12", 12),
        ("  source add 3  ", 3),
    ],
)
def test_parse_source_add_accepts(text: str, expected: int) -> None:
    match = parse_source_candidate_accept_command(text)
    assert match is not None
    assert match.payload == expected


@pytest.mark.parametrize(
    "text",
    [
        "source add",
        "source add 0",
        "source add -1",
        "source add 1.5",
        "source add one",
        "source add 1 now",
        "sources add 1",
        "source add +1",
    ],
)
def test_parse_source_add_rejects(text: str) -> None:
    assert parse_source_candidate_accept_command(text) is None


@pytest.mark.parametrize(
    "text",
    ["source add", "source add 0", "source add now"],
)
def test_parse_source_add_invalid_usage(text: str) -> None:
    assert parse_source_candidate_accept_invalid_command(text) is not None


def test_ordinary_source_questions_are_not_commands() -> None:
    assert parse_source_candidates_list_command("what source is best?") is None
    assert parse_source_candidate_accept_command("please source add later") is None
    assert parse_source_candidate_accept_invalid_command("source of truth?") is None
    assert parse_sources_list_command("sources") is not None
    assert is_sources_command("sources")


# --- Ordering / rendering ---


def test_order_source_candidates_label_then_id() -> None:
    items = [
        SlackSourceCandidateListItem(candidate_id="b", label="Zebra"),
        SlackSourceCandidateListItem(candidate_id="a", label="Apple"),
        SlackSourceCandidateListItem(candidate_id="c", label="apple"),
    ]
    ordered = order_source_candidates_for_listing(items)
    assert [item.candidate_id for item in ordered] == ["a", "c", "b"]


def test_render_safe_list_and_limits() -> None:
    items = [
        SlackSourceCandidateListItem(
            candidate_id="cid-secret",
            label="  Contracts\n",
            description="Current contract documents",
        ),
        SlackSourceCandidateListItem(
            candidate_id="cid-2",
            label="",
            description="",
        ),
    ]
    text = render_source_candidate_list(items)
    assert "Available source candidates:" in text
    assert "1. Contracts" in text
    assert "Current contract documents" in text
    assert "2. Source" in text
    assert "cid-secret" not in text
    assert "Use `source add <number>` to attach a source." in text

    long_label = "L" * 200
    long_desc = "D" * 300
    rendered = render_source_candidate_list(
        [
            SlackSourceCandidateListItem(
                candidate_id="x",
                label=long_label,
                description=long_desc,
            )
        ]
    )
    assert "…" in rendered
    assert long_label not in rendered
    assert long_desc not in rendered

    control = render_source_candidate_list(
        [
            SlackSourceCandidateListItem(
                candidate_id="x",
                label="A\x00B",
                description="C\x01D",
            )
        ]
    )
    assert "\x00" not in control
    assert "\x01" not in control

    success = render_source_candidate_accepted("Product documentation")
    assert "Source accepted: Product documentation" in success
    assert SOURCE_CANDIDATE_ACCEPTED_FOOTER in success
    assert "indexed" not in success.casefold()
    assert "completed" not in success.casefold()
    assert MAX_SOURCE_CANDIDATE_ITEMS == 25


# --- Idempotency ---


def test_idempotency_key_deterministic() -> None:
    key_a = slack_source_candidate_intake_idempotency_key(
        team_id="T1",
        event_id="Ev1",
    )
    key_b = slack_source_candidate_intake_idempotency_key(
        team_id="T1",
        event_id="Ev1",
    )
    key_c = slack_source_candidate_intake_idempotency_key(
        team_id="T1",
        event_id="Ev2",
    )
    assert key_a == key_b
    assert key_a != key_c
    assert key_a.startswith("slack-source-candidate:v1:")
    digest = key_a.rsplit(":", 1)[-1]
    assert len(digest) == 64
    assert digest == digest.lower()
    assert all(ch in "0123456789abcdef" for ch in digest)
    assert "T1" not in key_a
    assert "Ev1" not in key_a


# --- HTTP client ---


@pytest.mark.asyncio
async def test_list_source_candidates_http_contract() -> None:
    calls: list[httpx.Request] = []
    transport = _transport(
        candidates=[
            _candidate_payload(
                candidate_id="contracts",
                label="Contracts",
                path="/secret/path",
                fingerprint="sha256:abc",
            )
        ],
        list_calls=calls,
    )
    client = WorkspaceAskHttpClient(
        SlackAskClientConfig(base_url="http://lkw.test", api_key="k-secret"),
        transport=transport,
    )
    items = await client.list_source_candidates(
        tenant_id="tenant-a",
        workspace_id="ws-1",
    )
    assert len(calls) == 1
    request = calls[0]
    assert str(request.url).endswith(
        "/v1/local_workspace/workspaces/ws-1/source-candidates"
    )
    assert request.headers["X-Tenant-Id"] == "tenant-a"
    assert request.headers["X-API-Key"] == "k-secret"
    assert request.url.params.get("path") is None
    assert len(items) == 1
    dumped = items[0].model_dump()
    assert "path" not in dumped
    assert "candidate_fingerprint" not in dumped
    assert dumped["candidate_id"] == "contracts"


@pytest.mark.asyncio
async def test_list_source_candidates_no_api_key_and_errors() -> None:
    calls: list[httpx.Request] = []
    client = WorkspaceAskHttpClient(
        SlackAskClientConfig(base_url="http://lkw.test"),
        transport=_transport(list_calls=calls, candidates=[]),
    )
    await client.list_source_candidates(tenant_id="t", workspace_id="ws")
    assert "X-API-Key" not in calls[0].headers

    with pytest.raises(SlackAskClientError) as empty:
        await client.list_source_candidates(tenant_id=" ", workspace_id="ws")
    assert empty.value.kind == "parse_error"
    assert "ws" not in str(empty.value)

    for status, kind in ((404, "http_404"), (503, "http_503")):
        err_client = WorkspaceAskHttpClient(
            SlackAskClientConfig(base_url="http://lkw.test"),
            transport=_transport(list_status=status),
        )
        with pytest.raises(SlackAskClientError) as exc:
            await err_client.list_source_candidates(tenant_id="t", workspace_id="ws")
        assert exc.value.kind == kind
        assert "/srv/private" not in str(exc.value)

    timeout_client = WorkspaceAskHttpClient(
        SlackAskClientConfig(base_url="http://lkw.test"),
        transport=_transport(timeout_on="list"),
    )
    with pytest.raises(SlackAskClientError) as timeout_exc:
        await timeout_client.list_source_candidates(tenant_id="t", workspace_id="ws")
    assert timeout_exc.value.kind == "timeout"

    transport_client = WorkspaceAskHttpClient(
        SlackAskClientConfig(base_url="http://lkw.test"),
        transport=_transport(transport_error_on="list"),
    )
    with pytest.raises(SlackAskClientError) as transport_exc:
        await transport_client.list_source_candidates(tenant_id="t", workspace_id="ws")
    assert transport_exc.value.kind == "transport_error"

    malformed = WorkspaceAskHttpClient(
        SlackAskClientConfig(base_url="http://lkw.test"),
        transport=_transport(list_malformed=True),
    )
    with pytest.raises(SlackAskClientError) as parse_exc:
        await malformed.list_source_candidates(tenant_id="t", workspace_id="ws")
    assert parse_exc.value.kind == "parse_error"


@pytest.mark.asyncio
async def test_accept_source_candidate_http_contract() -> None:
    calls: list[httpx.Request] = []
    transport = _transport(accept_calls=calls)
    client = WorkspaceAskHttpClient(
        SlackAskClientConfig(base_url="http://lkw.test", api_key="k"),
        transport=transport,
    )
    parsed = await client.accept_source_candidate(
        tenant_id="tenant-a",
        workspace_id="ws-1",
        candidate_id="contracts",
        idempotency_key="idem-1",
    )
    assert isinstance(parsed, SlackSourceCandidateAcceptResponse)
    assert len(calls) == 1
    request = calls[0]
    assert request.method == "POST"
    assert str(request.url).endswith(
        "/v1/local_workspace/workspaces/ws-1/knowledge/source-candidates/contracts"
    )
    assert request.headers["X-Tenant-Id"] == "tenant-a"
    assert request.headers["Idempotency-Key"] == "idem-1"
    assert request.headers["X-API-Key"] == "k"
    assert b"path" not in (request.content or b"")
    assert b"fingerprint" not in (request.content or b"")
    assert b"label" not in (request.content or b"")

    mismatch = WorkspaceAskHttpClient(
        SlackAskClientConfig(base_url="http://lkw.test"),
        transport=_transport(
            accept_payload={
                "candidate_id": "other",
                "label": "X",
                "workspace_id": "ws-1",
                "source_id": "s",
                "operation_id": "o",
                "status": "queued",
            }
        ),
    )
    with pytest.raises(SlackAskClientError) as cand_mismatch:
        await mismatch.accept_source_candidate(
            tenant_id="t",
            workspace_id="ws-1",
            candidate_id="contracts",
            idempotency_key="idem",
        )
    assert cand_mismatch.value.kind == "parse_error"

    ws_mismatch = WorkspaceAskHttpClient(
        SlackAskClientConfig(base_url="http://lkw.test"),
        transport=_transport(
            accept_payload={
                "candidate_id": "contracts",
                "label": "X",
                "workspace_id": "other-ws",
                "source_id": "s",
                "operation_id": "o",
                "status": "queued",
            }
        ),
    )
    with pytest.raises(SlackAskClientError) as workspace_mismatch:
        await ws_mismatch.accept_source_candidate(
            tenant_id="t",
            workspace_id="ws-1",
            candidate_id="contracts",
            idempotency_key="idem",
        )
    assert workspace_mismatch.value.kind == "parse_error"

    for status, kind in ((404, "http_404"), (409, "http_409"), (503, "http_503")):
        err_client = WorkspaceAskHttpClient(
            SlackAskClientConfig(base_url="http://lkw.test"),
            transport=_transport(accept_status=status),
        )
        with pytest.raises(SlackAskClientError) as exc:
            await err_client.accept_source_candidate(
                tenant_id="t",
                workspace_id="ws",
                candidate_id="c",
                idempotency_key="i",
            )
        assert exc.value.kind == kind
        assert "/srv/private" not in str(exc.value)


# --- Workflow list ---


@pytest.mark.asyncio
async def test_workflow_list_configured_and_empty() -> None:
    ask_calls: list[httpx.Request] = []
    accept_calls: list[httpx.Request] = []
    outbound, _, dedupe = await _run(
        "source candidates",
        transport=_transport(
            candidates=[],
            ask_calls=ask_calls,
            accept_calls=accept_calls,
        ),
    )
    assert outbound == [SOURCE_CANDIDATE_LIST_EMPTY_TEXT]
    assert ask_calls == []
    assert accept_calls == []
    record = dedupe._get(build_slack_dedupe_key(team_id="T_OK", event_id="Ev-cand-1"))
    assert record is not None
    assert record.status is SlackDedupeStatus.COMPLETED


@pytest.mark.asyncio
async def test_workflow_list_numbered_and_selected_priority() -> None:
    store = InMemorySlackWorkspaceSelectionStore()
    store.set(
        slack_selection_actor_key(team_id="T_OK", user_id="U_OK"),
        SlackWorkspaceSelection(workspace_id="ws-selected", workspace_name="Selected"),
    )
    list_calls: list[httpx.Request] = []
    ask_calls: list[httpx.Request] = []
    outbound, _, _ = await _run(
        "source candidates",
        selection_store=store,
        transport=_transport(
            candidates=[
                _candidate_payload(
                    candidate_id="z",
                    label="Zebra",
                    description="Z docs",
                    path="/hidden",
                ),
                _candidate_payload(
                    candidate_id="a",
                    label="Apple",
                    description="A docs",
                    fingerprint="sha256:x",
                ),
            ],
            list_calls=list_calls,
            ask_calls=ask_calls,
        ),
    )
    assert ask_calls == []
    assert str(list_calls[0].url).endswith("/workspaces/ws-selected/source-candidates")
    text = outbound[0]
    assert "1. Apple" in text
    assert "2. Zebra" in text
    assert "A docs" in text
    assert "candidate_id" not in text
    assert "/hidden" not in text
    assert "sha256:" not in text
    assert "ws-selected" not in text
    assert "tenant-a" not in text


@pytest.mark.asyncio
async def test_workflow_list_no_workspace() -> None:
    outbound, _, _ = await _run(
        "source candidates",
        workspace_id="ws-configured",
        suppress_configured=True,
    )
    assert outbound == [NO_WORKSPACE_AVAILABLE_TEXT]


@pytest.mark.asyncio
async def test_workflow_list_404_clears_selected() -> None:
    store = InMemorySlackWorkspaceSelectionStore()
    actor = slack_selection_actor_key(team_id="T_OK", user_id="U_OK")
    store.set(actor, SlackWorkspaceSelection(workspace_id="ws-gone", workspace_name="Gone"))
    outbound, _, dedupe = await _run(
        "source candidates",
        selection_store=store,
        transport=_transport(list_status=404),
    )
    assert outbound == [SELECTED_WORKSPACE_UNAVAILABLE_TEXT]
    assert store.get(actor) is None
    record = dedupe._get(build_slack_dedupe_key(team_id="T_OK", event_id="Ev-cand-1"))
    assert record is not None
    assert record.status is SlackDedupeStatus.COMPLETED


@pytest.mark.asyncio
async def test_workflow_list_failure_safe() -> None:
    outbound, _, _ = await _run(
        "source candidates",
        transport=_transport(list_status=503),
    )
    assert outbound == [SOURCE_CANDIDATE_LIST_LOAD_FAILED_TEXT]
    assert "/srv/private" not in outbound[0]


# --- Workflow selection ---


@pytest.mark.asyncio
async def test_workflow_accept_maps_number_and_idempotency() -> None:
    list_calls: list[httpx.Request] = []
    accept_calls: list[httpx.Request] = []
    ask_calls: list[httpx.Request] = []
    outbound, _, dedupe = await _run(
        "source add 2",
        transport=_transport(
            candidates=[
                _candidate_payload(candidate_id="apple", label="Apple"),
                _candidate_payload(
                    candidate_id="product",
                    label="Product documentation",
                    description="Approved product materials",
                ),
            ],
            list_calls=list_calls,
            accept_calls=accept_calls,
            ask_calls=ask_calls,
        ),
        event_id="Ev-add-2",
    )
    assert ask_calls == []
    assert len(list_calls) == 1
    assert len(accept_calls) == 1
    assert accept_calls[0].url.path.endswith(
        "/knowledge/source-candidates/product"
    )
    expected_key = slack_source_candidate_intake_idempotency_key(
        team_id="T_OK",
        event_id="Ev-add-2",
    )
    assert accept_calls[0].headers["Idempotency-Key"] == expected_key
    text = outbound[0]
    assert "Source accepted: Product documentation" in text
    assert SOURCE_CANDIDATE_ACCEPTED_FOOTER in text
    assert "product" not in text.split("accepted:", 1)[-1].split("\n", 1)[0]
    assert "op-1" not in text
    assert "src-1" not in text
    record = dedupe._get(build_slack_dedupe_key(team_id="T_OK", event_id="Ev-add-2"))
    assert record is not None
    assert record.status is SlackDedupeStatus.COMPLETED


@pytest.mark.asyncio
async def test_workflow_accept_out_of_range_and_hidden() -> None:
    accept_calls: list[httpx.Request] = []
    outbound, _, _ = await _run(
        "source add 3",
        transport=_transport(
            candidates=[
                _candidate_payload(candidate_id="a", label="A"),
                _candidate_payload(candidate_id="b", label="B"),
            ],
            accept_calls=accept_calls,
        ),
    )
    assert outbound == [SOURCE_CANDIDATE_OUT_OF_RANGE_TEXT]
    assert accept_calls == []

    many = [
        _candidate_payload(candidate_id=f"c{i:02d}", label=f"Item {i:02d}")
        for i in range(1, 30)
    ]
    outbound26, _, _ = await _run(
        "source add 26",
        transport=_transport(candidates=many, accept_calls=accept_calls),
        event_id="Ev-26",
    )
    assert outbound26 == [SOURCE_CANDIDATE_OUT_OF_RANGE_TEXT]
    assert accept_calls == []


@pytest.mark.asyncio
async def test_workflow_accept_empty_list_no_post() -> None:
    accept_calls: list[httpx.Request] = []
    outbound, _, _ = await _run(
        "source add 1",
        transport=_transport(candidates=[], accept_calls=accept_calls),
    )
    assert outbound == [SOURCE_CANDIDATE_LIST_EMPTY_TEXT]
    assert accept_calls == []


@pytest.mark.asyncio
async def test_workflow_accept_http_product_errors() -> None:
    for status, expected in (
        (404, SOURCE_CANDIDATE_UNAVAILABLE_TEXT),
        (409, SOURCE_CANDIDATE_ALREADY_ATTACHED_TEXT),
        (503, SOURCE_CANDIDATE_SERVICE_UNAVAILABLE_TEXT),
    ):
        outbound, _, dedupe = await _run(
            "source add 1",
            transport=_transport(
                candidates=[_candidate_payload(candidate_id="c", label="C")],
                accept_status=status,
            ),
            event_id=f"Ev-{status}",
        )
        assert outbound == [expected]
        assert "/srv/private" not in outbound[0]
        record = dedupe._get(
            build_slack_dedupe_key(team_id="T_OK", event_id=f"Ev-{status}")
        )
        assert record is not None
        assert record.status is SlackDedupeStatus.COMPLETED


@pytest.mark.asyncio
async def test_workflow_accept_timeout_marks_failed_same_key_on_retry() -> None:
    from datetime import timedelta

    accept_calls: list[httpx.Request] = []
    store = InMemoryDocumentStore()
    dedupe = SlackEventDedupeRepository(store)
    outbound: list[str] = []

    async def send(message: OutboundConversationMessage) -> ConversationDeliveryReceipt:
        outbound.append(message.text)
        return ConversationDeliveryReceipt(
            message_id="m1",
            address=message.address,
            delivered_at=datetime.now(UTC),
        )

    client = WorkspaceAskHttpClient(
        SlackAskClientConfig(base_url="http://lkw.test"),
        transport=_transport(
            candidates=[_candidate_payload(candidate_id="c", label="C")],
            accept_calls=accept_calls,
            timeout_on="accept",
        ),
    )
    workflow = SlackAskWorkflow(
        auth_config=SlackCompanionAuthConfig(
            approved_team_id="T_OK",
            approved_user_id="U_OK",
            tenant_id="tenant-a",
            active_workspace_id="ws-configured",
        ),
        dedupe=dedupe,
        ask_client=client,
        send=send,
    )
    await workflow.handle(_event(text="source add 1", event_id="Ev-retry"))
    record = dedupe._get(build_slack_dedupe_key(team_id="T_OK", event_id="Ev-retry"))
    assert record is not None
    assert record.status is SlackDedupeStatus.FAILED
    assert outbound[0]
    key = slack_source_candidate_intake_idempotency_key(
        team_id="T_OK",
        event_id="Ev-retry",
    )
    assert accept_calls[0].headers["Idempotency-Key"] == key

    # Expire claim so existing reclaim/redelivery can retry with the same key.
    expired = record.model_copy(
        update={"expires_at": datetime.now(UTC) - timedelta(seconds=1)}
    )
    store.put(
        DocumentRecord(
            partition_key="lkw.slack_companion:dedupe",
            row_key=expired.dedupe_key,
            data=expired.model_dump(mode="json"),
            ttl_seconds=86400,
        )
    )
    await workflow.handle(_event(text="source add 1", event_id="Ev-retry"))
    assert len(accept_calls) == 2
    assert accept_calls[1].headers["Idempotency-Key"] == key

@pytest.mark.asyncio
async def test_workflow_accept_success_send_failure_does_not_retry_mutation() -> None:
    accept_calls: list[httpx.Request] = []
    outbound, _, dedupe = await _run(
        "source add 1",
        transport=_transport(
            candidates=[_candidate_payload(candidate_id="c", label="C")],
            accept_calls=accept_calls,
        ),
        send_error=RuntimeError("slack down"),
        event_id="Ev-send-fail",
    )
    assert outbound == []
    assert len(accept_calls) == 1
    record = dedupe._get(build_slack_dedupe_key(team_id="T_OK", event_id="Ev-send-fail"))
    assert record is not None
    assert record.status is SlackDedupeStatus.COMPLETED


@pytest.mark.asyncio
async def test_invalid_source_add_usage() -> None:
    ask_calls: list[httpx.Request] = []
    outbound, _, _ = await _run(
        "source add",
        transport=_transport(ask_calls=ask_calls),
    )
    assert outbound == [SOURCE_CANDIDATE_USAGE_TEXT]
    assert ask_calls == []


@pytest.mark.asyncio
async def test_same_ordering_for_list_and_selection() -> None:
    candidates = [
        _candidate_payload(candidate_id="z", label="Zebra"),
        _candidate_payload(candidate_id="a", label="Apple"),
        _candidate_payload(candidate_id="p", label="Product"),
    ]
    list_out, _, _ = await _run(
        "source candidates",
        transport=_transport(candidates=candidates),
        event_id="Ev-order-list",
    )
    accept_calls: list[httpx.Request] = []
    await _run(
        "source add 2",
        transport=_transport(candidates=candidates, accept_calls=accept_calls),
        event_id="Ev-order-add",
    )
    assert "1. Apple" in list_out[0]
    assert "2. Product" in list_out[0]
    assert "3. Zebra" in list_out[0]
    assert accept_calls[0].url.path.endswith("/knowledge/source-candidates/p")
