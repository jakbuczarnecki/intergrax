# © Artur Czarnecki. All rights reserved.

"""LKW-SLACK-COMMAND-CATALOG-1 — declarative command registry and dynamic help."""

from __future__ import annotations

from datetime import UTC, datetime
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
from local_workspace_application.slack_companion.commands import (
    SlackCommandDefinition,
    SlackCommandMatch,
    SlackCommandMetadata,
    SlackCommandRegistry,
    discover_slack_commands,
    render_command_help,
    slack_command,
)
from local_workspace_application.slack_companion.dedupe_repository import (
    SlackEventDedupeRepository,
    build_slack_dedupe_key,
)
from local_workspace_application.slack_companion.pending_deletion_store import (
    InMemorySlackPendingDeletionStore,
)
from local_workspace_application.slack_companion.rendering import (
    ACK_TEXT,
    WORKSPACE_CREATE_USAGE_TEXT,
    WORKSPACE_DELETE_USAGE_TEXT,
    WORKSPACE_SELECTION_USAGE_TEXT,
)
from local_workspace_application.slack_companion.selection_store import (
    InMemorySlackWorkspaceSelectionStore,
)
from local_workspace_application.slack_companion.workflow import SlackAskWorkflow

pytestmark = pytest.mark.unit

_ATTR = "__lkw_slack_command__"


def _noop_parser(text: str) -> SlackCommandMatch | None:
    del text
    return None


def _exact_parser(expected: str):
    def parser(text: str) -> SlackCommandMatch | None:
        if (text or "").strip().casefold() == expected.casefold():
            return SlackCommandMatch(payload=expected)
        return None

    return parser


# --- Metadata and decorator ---


def test_decorator_attaches_immutable_metadata() -> None:
    @slack_command(
        command_id="demo.cmd",
        syntax="demo",
        description="Demo command.",
        example="demo",
        priority=1,
        parser=_noop_parser,
    )
    async def handler(self: object, context: object, match: SlackCommandMatch) -> None:
        del self, context, match

    annotation = getattr(handler, _ATTR)
    assert annotation.metadata.command_id == "demo.cmd"
    assert annotation.metadata.syntax == "demo"
    assert annotation.parser is _noop_parser
    with pytest.raises(AttributeError):
        annotation.metadata.command_id = "mutated"  # type: ignore[misc]


@pytest.mark.asyncio
async def test_decorator_preserves_callable_behavior() -> None:
    calls: list[str] = []

    @slack_command(
        command_id="demo.call",
        syntax="demo",
        description="Demo.",
        example="demo",
        priority=1,
        parser=_noop_parser,
    )
    async def handler(self: object, context: object, match: SlackCommandMatch) -> None:
        del self, context, match
        calls.append("ran")

    await handler(object(), object(), SlackCommandMatch())
    assert calls == ["ran"]


def test_invalid_metadata_fails_fast() -> None:
    with pytest.raises(ValueError):
        SlackCommandMetadata(
            command_id="",
            syntax="x",
            description="d",
            example="e",
            priority=1,
        )
    with pytest.raises(ValueError):
        SlackCommandMetadata(
            command_id="id",
            syntax="x",
            description="d",
            example="",
            priority=1,
            visible_in_help=True,
        )
    with pytest.raises(TypeError):
        SlackCommandMetadata(
            command_id="id",
            syntax="x",
            description="d",
            example="e",
            priority=True,  # type: ignore[arg-type]
        )


def test_duplicate_command_id_fails_registry_construction() -> None:
    async def h1(context: object, match: SlackCommandMatch) -> None:
        del context, match

    async def h2(context: object, match: SlackCommandMatch) -> None:
        del context, match

    meta = SlackCommandMetadata(
        command_id="dup",
        syntax="dup",
        description="Dup.",
        example="dup",
        priority=1,
    )
    with pytest.raises(ValueError, match="duplicate"):
        SlackCommandRegistry(
            [
                SlackCommandDefinition(metadata=meta, parser=_noop_parser, handler=h1),
                SlackCommandDefinition(metadata=meta, parser=_noop_parser, handler=h2),
            ]
        )


def test_definitions_ordered_by_priority_then_command_id() -> None:
    async def h(context: object, match: SlackCommandMatch) -> None:
        del context, match

    defs = [
        SlackCommandDefinition(
            metadata=SlackCommandMetadata(
                command_id="b",
                syntax="b",
                description="B.",
                example="b",
                priority=20,
            ),
            parser=_noop_parser,
            handler=h,
        ),
        SlackCommandDefinition(
            metadata=SlackCommandMetadata(
                command_id="a",
                syntax="a",
                description="A.",
                example="a",
                priority=20,
            ),
            parser=_noop_parser,
            handler=h,
        ),
        SlackCommandDefinition(
            metadata=SlackCommandMetadata(
                command_id="z",
                syntax="z",
                description="Z.",
                example="z",
                priority=10,
            ),
            parser=_noop_parser,
            handler=h,
        ),
    ]
    registry = SlackCommandRegistry(defs)
    assert [d.metadata.command_id for d in registry.definitions] == ["z", "a", "b"]


# --- Discovery ---


class _OwnerWithCommands:
    def helper(self) -> str:
        return "helper"

    @slack_command(
        command_id="alpha",
        syntax="alpha",
        description="Alpha.",
        example="alpha",
        priority=20,
        parser=_exact_parser("alpha"),
    )
    async def command_alpha(
        self, context: object, match: SlackCommandMatch
    ) -> None:
        del context
        self.last = ("alpha", match.payload)

    @slack_command(
        command_id="beta",
        syntax="beta",
        description="Beta.",
        example="beta",
        priority=10,
        parser=_exact_parser("beta"),
    )
    async def command_beta(
        self, context: object, match: SlackCommandMatch
    ) -> None:
        del context
        self.last = ("beta", match.payload)


def test_discovery_only_annotated_methods() -> None:
    owner = _OwnerWithCommands()
    registry = discover_slack_commands(owner)
    ids = [d.metadata.command_id for d in registry.definitions]
    assert ids == ["beta", "alpha"]
    assert all(d.metadata.command_id != "helper" for d in registry.definitions)


def test_ordinary_helpers_ignored() -> None:
    owner = _OwnerWithCommands()
    registry = discover_slack_commands(owner)
    assert len(registry.definitions) == 2


def test_discovery_limited_to_supplied_object() -> None:
    class Other:
        @slack_command(
            command_id="other",
            syntax="other",
            description="Other.",
            example="other",
            priority=1,
            parser=_exact_parser("other"),
        )
        async def cmd(self, context: object, match: SlackCommandMatch) -> None:
            del self, context, match

    registry = discover_slack_commands(_OwnerWithCommands())
    assert all(d.metadata.command_id != "other" for d in registry.definitions)
    other_registry = discover_slack_commands(Other())
    assert [d.metadata.command_id for d in other_registry.definitions] == ["other"]


@pytest.mark.asyncio
async def test_bound_handlers_invoke_correct_instance() -> None:
    owner = _OwnerWithCommands()
    registry = discover_slack_commands(owner)
    resolved = registry.match("alpha")
    assert resolved is not None
    await resolved.handler(object(), resolved.match)
    assert owner.last == ("alpha", "alpha")


def test_no_module_global_scan() -> None:
    """Discovery inspects only the supplied owner; empty owner → empty registry."""

    class Empty:
        pass

    assert discover_slack_commands(Empty()).definitions == ()


# --- Workflow helpers ---


def _event(
    *,
    event_id: str = "Ev-cmd-1",
    team_id: str = "T_OK",
    user_id: str = "U_OK",
    text: str = "help",
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
) -> dict[str, object]:
    return {
        "workspace_id": workspace_id,
        "tenant_id": "tenant-a",
        "name": name,
        "description": "",
        "status": status,
        "created_at": "2026-07-23T12:00:00Z",
        "updated_at": "2026-07-23T12:00:00Z",
    }


def _transport(
    *,
    workspaces: list[dict[str, object]] | None = None,
    list_calls: list[httpx.Request] | None = None,
    create_calls: list[httpx.Request] | None = None,
    delete_calls: list[httpx.Request] | None = None,
    ask_calls: list[httpx.Request] | None = None,
) -> httpx.MockTransport:
    workspace_bucket = list(workspaces or [])
    list_bucket = list_calls if list_calls is not None else []
    create_bucket = create_calls if create_calls is not None else []
    delete_bucket = delete_calls if delete_calls is not None else []
    ask_bucket = ask_calls if ask_calls is not None else []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/ask"):
            ask_bucket.append(request)
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
            return httpx.Response(200, json={"workspaces": workspace_bucket})
        if request.method == "POST" and path.rstrip("/").endswith("/workspaces"):
            create_bucket.append(request)
            body = _workspace_payload(workspace_id="ws-created", name="Contracts")
            workspace_bucket.insert(0, body)
            return httpx.Response(201, json=body)
        if request.method == "DELETE" and "/workspaces/" in path:
            delete_bucket.append(request)
            return httpx.Response(204)
        return httpx.Response(404, json={"detail": "not_found"})

    return httpx.MockTransport(handler)


async def _run(
    text: str,
    *,
    event_id: str = "Ev-cmd-1",
    team_id: str = "T_OK",
    user_id: str = "U_OK",
    configured_workspace_id: str = "ws-configured",
    workspaces: list[dict[str, object]] | None = None,
    list_calls: list[httpx.Request] | None = None,
    create_calls: list[httpx.Request] | None = None,
    delete_calls: list[httpx.Request] | None = None,
    ask_calls: list[httpx.Request] | None = None,
    selection_store: InMemorySlackWorkspaceSelectionStore | None = None,
    pending_store: InMemorySlackPendingDeletionStore | None = None,
    dedupe: SlackEventDedupeRepository | None = None,
) -> tuple[list[str], SlackAskWorkflow]:
    outbound: list[str] = []

    async def send(message: OutboundConversationMessage) -> ConversationDeliveryReceipt:
        outbound.append(message.text)
        return ConversationDeliveryReceipt(
            message_id="m1",
            address=message.address,
            delivered_at=datetime.now(UTC),
        )

    workflow = SlackAskWorkflow(
        auth_config=SlackCompanionAuthConfig(
            approved_team_id="T_OK",
            approved_user_id="U_OK",
            tenant_id="tenant-a",
            active_workspace_id=configured_workspace_id,
        ),
        dedupe=dedupe or SlackEventDedupeRepository(InMemoryDocumentStore()),
        ask_client=WorkspaceAskHttpClient(
            SlackAskClientConfig(base_url="http://lkw.test/"),
            transport=_transport(
                workspaces=workspaces,
                list_calls=list_calls,
                create_calls=create_calls,
                delete_calls=delete_calls,
                ask_calls=ask_calls,
            ),
        ),
        send=send,
        selection_store=selection_store or InMemorySlackWorkspaceSelectionStore(),
        pending_deletion_store=pending_store or InMemorySlackPendingDeletionStore(),
    )
    await workflow.handle(
        _event(event_id=event_id, team_id=team_id, user_id=user_id, text=text)
    )
    return outbound, workflow


# --- Matching and dispatch ---


@pytest.mark.asyncio
async def test_help_matches_case_insensitively() -> None:
    for text in ("help", "HELP", "  Help  "):
        ask_calls: list[httpx.Request] = []
        outbound, _ = await _run(text, event_id=f"Ev-{text.strip()}", ask_calls=ask_calls)
        assert len(outbound) == 1
        assert "Available commands:" in outbound[0]
        assert ask_calls == []


@pytest.mark.asyncio
async def test_help_me_does_not_match() -> None:
    ask_calls: list[httpx.Request] = []
    outbound, _ = await _run("help me", ask_calls=ask_calls)
    assert ask_calls
    assert outbound[0] == ACK_TEXT


@pytest.mark.asyncio
async def test_exact_command_maps_to_correct_handler() -> None:
    list_calls: list[httpx.Request] = []
    outbound, _ = await _run(
        "workspaces",
        workspaces=[_workspace_payload(workspace_id="ws-configured", name="Alpha")],
        list_calls=list_calls,
    )
    assert list_calls
    assert "Alpha" in outbound[0]
    assert "— active" in outbound[0]


@pytest.mark.asyncio
async def test_parser_payload_passed_to_handler() -> None:
    create_calls: list[httpx.Request] = []
    outbound, workflow = await _run(
        "workspace create Contracts",
        create_calls=create_calls,
    )
    assert len(create_calls) == 1
    assert "Contracts" in outbound[0]
    actor = workflow._selections.get("T_OK:U_OK")
    assert actor is not None
    assert actor.workspace_id == "ws-created"


@pytest.mark.asyncio
async def test_delete_confirm_wins_before_generic_delete() -> None:
    pending = InMemorySlackPendingDeletionStore()
    pending.set("T_OK:U_OK", workspace_id="ws-a", workspace_name="Alpha")
    delete_calls: list[httpx.Request] = []
    outbound, _ = await _run(
        "workspace delete confirm",
        pending_store=pending,
        delete_calls=delete_calls,
    )
    assert len(delete_calls) == 1
    assert "deleted" in outbound[0].casefold() or "Alpha" in outbound[0]


@pytest.mark.asyncio
async def test_regular_question_produces_no_command_match() -> None:
    outbound, workflow = await _run("what is the deadline?")
    assert workflow._commands.match("what is the deadline?") is None
    assert ACK_TEXT in outbound[0]


@pytest.mark.asyncio
async def test_invalid_workspace_command_is_usage_not_ask() -> None:
    ask_calls: list[httpx.Request] = []
    outbound, _ = await _run("workspace", ask_calls=ask_calls)
    assert ask_calls == []
    assert outbound == [WORKSPACE_SELECTION_USAGE_TEXT]


@pytest.mark.asyncio
async def test_no_formal_command_routed_outside_registry() -> None:
    outbound, workflow = await _run("help")
    public_ids = {
        "help",
        "workspaces.list",
        "sources.list",
        "source_candidates.list",
        "source_candidates.accept",
        "workspace.select",
        "workspace.create",
        "workspace.delete.request",
        "workspace.delete.confirm",
        "workspace.delete.cancel",
    }
    registered = {d.metadata.command_id for d in workflow._commands.definitions}
    assert public_ids <= registered
    for command_id in public_ids:
        assert any(
            d.metadata.command_id == command_id for d in workflow._commands.definitions
        )
    assert "source_candidates.accept.invalid" in registered
    visible_ids = {d.metadata.command_id for d in workflow._commands.visible_commands()}
    assert "source_candidates.accept.invalid" not in visible_ids
    help_text = outbound[0]
    assert "`source candidates`" in help_text
    assert "`source add <number>`" in help_text
    assert "`sources`" in help_text
    assert help_text.index("`source candidates`") < help_text.index("`sources`")
    assert help_text.index("`source add <number>`") < help_text.index("`sources`")
    assert "source_candidates.list" not in help_text
    assert len(registered) == len(workflow._commands.definitions)

# --- Dynamic help ---


@pytest.mark.asyncio
async def test_help_generated_from_registry_metadata() -> None:
    outbound, workflow = await _run("help")
    text = outbound[0]
    for definition in workflow._commands.visible_commands():
        meta = definition.metadata
        assert f"`{meta.syntax}`" in text
        assert meta.description in text
        assert f"Example: `{meta.example}`" in text
        assert meta.command_id not in text or meta.command_id == meta.syntax


@pytest.mark.asyncio
async def test_every_visible_public_command_appears() -> None:
    outbound, workflow = await _run("help")
    text = outbound[0]
    for definition in workflow._commands.visible_commands():
        assert f"`{definition.metadata.syntax}`" in text


@pytest.mark.asyncio
async def test_help_order_follows_registry() -> None:
    outbound, workflow = await _run("help")
    text = outbound[0]
    positions = [
        text.index(f"`{d.metadata.syntax}`")
        for d in workflow._commands.visible_commands()
    ]
    assert positions == sorted(positions)


def test_adding_fake_visible_command_appears_in_help() -> None:
    async def h(context: object, match: SlackCommandMatch) -> None:
        del context, match

    registry = SlackCommandRegistry(
        [
            SlackCommandDefinition(
                metadata=SlackCommandMetadata(
                    command_id="fake.visible",
                    syntax="fake",
                    description="Fake visible.",
                    example="fake",
                    priority=1,
                ),
                parser=_exact_parser("fake"),
                handler=h,
            )
        ]
    )
    text = render_command_help(registry.visible_commands())
    assert "`fake`" in text
    assert "Fake visible." in text


def test_hidden_command_excluded_from_help_but_dispatchable() -> None:
    async def h(context: object, match: SlackCommandMatch) -> None:
        del context, match

    registry = SlackCommandRegistry(
        [
            SlackCommandDefinition(
                metadata=SlackCommandMetadata(
                    command_id="fake.hidden",
                    syntax="hidden",
                    description="Hidden.",
                    example="",
                    priority=1,
                    visible_in_help=False,
                ),
                parser=_exact_parser("hidden"),
                handler=h,
            ),
            SlackCommandDefinition(
                metadata=SlackCommandMetadata(
                    command_id="fake.visible",
                    syntax="visible",
                    description="Visible.",
                    example="visible",
                    priority=2,
                ),
                parser=_exact_parser("visible"),
                handler=h,
            ),
        ]
    )
    help_text = render_command_help(registry.visible_commands())
    assert "`hidden`" not in help_text
    assert "`visible`" in help_text
    assert registry.match("hidden") is not None
    assert registry.match("hidden").command_id == "fake.hidden"


@pytest.mark.asyncio
async def test_help_does_not_expose_ids_or_config() -> None:
    outbound, _ = await _run("help")
    text = outbound[0]
    assert "tenant-a" not in text
    assert "ws-configured" not in text
    assert "command_id" not in text
    assert "/v1/" not in text
    assert "workspaces.list" not in text
    assert "You can also send a normal question" in text


@pytest.mark.asyncio
async def test_help_zero_http_and_no_state_mutation() -> None:
    list_calls: list[httpx.Request] = []
    create_calls: list[httpx.Request] = []
    delete_calls: list[httpx.Request] = []
    ask_calls: list[httpx.Request] = []
    selections = InMemorySlackWorkspaceSelectionStore()
    pending = InMemorySlackPendingDeletionStore()
    outbound, _ = await _run(
        "help",
        list_calls=list_calls,
        create_calls=create_calls,
        delete_calls=delete_calls,
        ask_calls=ask_calls,
        selection_store=selections,
        pending_store=pending,
    )
    assert outbound
    assert list_calls == []
    assert create_calls == []
    assert delete_calls == []
    assert ask_calls == []
    assert selections.get("T_OK:U_OK") is None
    assert pending.get("T_OK:U_OK") is None


# --- Authorization and dedupe ---


@pytest.mark.asyncio
async def test_unauthorized_help_sends_nothing_and_zero_product_calls() -> None:
    ask_calls: list[httpx.Request] = []
    outbound, _ = await _run(
        "help",
        team_id="T_BAD",
        ask_calls=ask_calls,
    )
    assert outbound == []
    assert ask_calls == []


@pytest.mark.asyncio
async def test_duplicate_help_event_sends_one_response() -> None:
    dedupe = SlackEventDedupeRepository(InMemoryDocumentStore())
    outbound1, _ = await _run("help", event_id="Ev-dup-help", dedupe=dedupe)
    outbound2, _ = await _run("help", event_id="Ev-dup-help", dedupe=dedupe)
    assert len(outbound1) == 1
    assert outbound2 == []
    assert build_slack_dedupe_key(team_id="T_OK", event_id="Ev-dup-help") == (
        "T_OK:Ev-dup-help"
    )


# --- Regression ---


@pytest.mark.asyncio
async def test_regression_workspaces() -> None:
    outbound, _ = await _run(
        "workspaces",
        workspaces=[
            _workspace_payload(workspace_id="ws-configured", name="Configured"),
            _workspace_payload(workspace_id="ws-b", name="Beta"),
        ],
    )
    assert "Configured" in outbound[0]
    assert "— active" in outbound[0]


@pytest.mark.asyncio
async def test_regression_workspace_select() -> None:
    outbound, workflow = await _run(
        "workspace 2",
        workspaces=[
            _workspace_payload(workspace_id="ws-configured", name="Configured"),
            _workspace_payload(workspace_id="ws-b", name="Beta"),
        ],
    )
    assert "Beta" in outbound[0]
    sel = workflow._selections.get("T_OK:U_OK")
    assert sel is not None
    assert sel.workspace_id == "ws-b"


@pytest.mark.asyncio
async def test_regression_workspace_create() -> None:
    create_calls: list[httpx.Request] = []
    ask_calls: list[httpx.Request] = []
    outbound, _ = await _run(
        "workspace create Contracts",
        create_calls=create_calls,
        ask_calls=ask_calls,
    )
    assert create_calls
    assert ask_calls == []
    assert "Contracts" in outbound[0]


@pytest.mark.asyncio
async def test_regression_delete_request_confirm_cancel() -> None:
    workspaces = [
        _workspace_payload(workspace_id="ws-configured", name="Configured"),
        _workspace_payload(workspace_id="ws-b", name="Beta"),
    ]
    pending = InMemorySlackPendingDeletionStore()
    selections = InMemorySlackWorkspaceSelectionStore()
    outbound_req, _ = await _run(
        "workspace delete 2",
        workspaces=workspaces,
        pending_store=pending,
        selection_store=selections,
    )
    assert "confirm" in outbound_req[0].casefold() or "delete" in outbound_req[0].casefold()
    assert pending.get("T_OK:U_OK") is not None

    outbound_cancel, _ = await _run(
        "workspace delete cancel",
        pending_store=pending,
        selection_store=selections,
        event_id="Ev-cancel",
    )
    assert outbound_cancel
    assert pending.get("T_OK:U_OK") is None

    await _run(
        "workspace delete 2",
        workspaces=workspaces,
        pending_store=pending,
        selection_store=selections,
        event_id="Ev-req2",
    )
    delete_calls: list[httpx.Request] = []
    outbound_confirm, _ = await _run(
        "workspace delete confirm",
        workspaces=workspaces,
        pending_store=pending,
        selection_store=selections,
        delete_calls=delete_calls,
        event_id="Ev-confirm",
    )
    assert delete_calls
    assert outbound_confirm


@pytest.mark.asyncio
async def test_regression_invalid_create_and_delete_usage() -> None:
    ask_calls: list[httpx.Request] = []
    outbound_create, _ = await _run("workspace create", ask_calls=ask_calls)
    assert outbound_create == [WORKSPACE_CREATE_USAGE_TEXT]
    assert ask_calls == []
    outbound_delete, _ = await _run(
        "workspace delete",
        ask_calls=ask_calls,
        event_id="Ev-del-invalid",
    )
    assert outbound_delete == [WORKSPACE_DELETE_USAGE_TEXT]
    assert ask_calls == []


@pytest.mark.asyncio
async def test_regression_regular_ask_1a() -> None:
    ask_calls: list[httpx.Request] = []
    outbound, _ = await _run("What is in the contract?", ask_calls=ask_calls)
    assert ask_calls
    assert outbound[0] == ACK_TEXT
    assert "Ask answer" in outbound[1]


@pytest.mark.asyncio
async def test_regression_effective_active_and_configured_suppression() -> None:
    selections = InMemorySlackWorkspaceSelectionStore()
    pending = InMemorySlackPendingDeletionStore()
    workspaces = [
        _workspace_payload(workspace_id="ws-configured", name="Configured"),
    ]
    await _run(
        "workspace delete 1",
        workspaces=workspaces,
        selection_store=selections,
        pending_store=pending,
        event_id="Ev-sup-1",
    )
    await _run(
        "workspace delete confirm",
        workspaces=workspaces,
        selection_store=selections,
        pending_store=pending,
        event_id="Ev-sup-2",
    )
    assert selections.is_configured_suppressed("T_OK:U_OK")
    ask_calls: list[httpx.Request] = []
    outbound, _ = await _run(
        "any question",
        selection_store=selections,
        pending_store=pending,
        ask_calls=ask_calls,
        event_id="Ev-sup-3",
        configured_workspace_id="ws-configured",
    )
    assert ask_calls == []
    assert outbound
