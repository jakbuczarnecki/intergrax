# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import asyncio
import threading
from datetime import datetime, timezone
from typing import Any

import httpx
import pytest

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.base import HealthStatus
from intergrax.integrations.contracts.conversation_channel import (
    ConversationActor,
    ConversationAddress,
    ConversationDeliveryReceipt,
    ConversationEventHandler,
    ConversationEventKind,
    InboundConversationEvent,
    OutboundConversationMessage,
)
from intergrax.integrations.providers.conversation_channel.slack.integration import (
    SlackConversationChannelIntegration,
)
from local_workspace_application.slack_companion.ask_client import (
    SlackAskClientConfig,
    WorkspaceAskHttpClient,
)
from local_workspace_application.slack_companion.authorization import SlackCompanionAuthConfig
from local_workspace_application.slack_companion.companion import (
    build_slack_companion,
    resolve_slack_companion_runtime_config,
)
from local_workspace_application.slack_companion.dedupe_repository import (
    SlackEventDedupeRepository,
)
from local_workspace_application.slack_companion.models import SlackAskHttpResponse
from local_workspace_application.slack_companion.rendering import ACK_TEXT
from local_workspace_application.slack_companion.workflow import SlackAskWorkflow
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings

pytestmark = pytest.mark.unit


class FakeConversationChannelBackend:
    def __init__(self) -> None:
        self.handler: ConversationEventHandler | None = None
        self.sent: list[OutboundConversationMessage] = []
        self.started = False
        self.stopped = False

    async def start(self, handler: ConversationEventHandler) -> None:
        self.handler = handler
        self.started = True

    async def stop(self) -> None:
        self.stopped = True
        self.started = False

    async def send(self, message: OutboundConversationMessage) -> ConversationDeliveryReceipt:
        self.sent.append(message)
        return ConversationDeliveryReceipt(
            message_id=f"msg-{len(self.sent)}",
            address=message.address,
            delivered_at=datetime.now(timezone.utc),
        )

    def health(self) -> HealthStatus:
        return HealthStatus(slug="slack", healthy=self.started, detail="fake")


def _event(
    *,
    event_id: str = "Ev-flow-1",
    team_id: str = "T_OK",
    user_id: str = "U_OK",
    text: str = "What is leave policy?",
) -> InboundConversationEvent:
    return InboundConversationEvent(
        event_id=event_id,
        address=ConversationAddress(
            installation_id=team_id,
            conversation_id="Dchannel",
            thread_id="1711111.000200",
        ),
        actor=ConversationActor(actor_id=user_id, is_bot=False),
        kind=ConversationEventKind.MESSAGE,
        text=text,
    )


@pytest.mark.asyncio
async def test_vertical_flow_ack_ask_threaded_answer_and_dedupe() -> None:
    ask_calls: list[httpx.Request] = []

    def ask_handler(request: httpx.Request) -> httpx.Response:
        ask_calls.append(request)
        return httpx.Response(
            200,
            json={
                "run_id": "ask-run-9",
                "workspace_id": "ws-active",
                "status": "completed",
                "question": "What is leave policy?",
                "answer": "Employees receive 20 days.",
                "citations": [
                    {
                        "file_name": "hr-policy.pdf",
                        "source_path": "/local/secret/hr-policy.pdf",
                        "excerpt": "do-not-leak",
                        "evidence_id": "e1",
                        "document_id": "d1",
                        "source_id": "s1",
                        "workspace_id": "ws-active",
                    },
                    {
                        "file_name": "hr-policy.pdf",
                        "source_path": "/other/path.pdf",
                        "excerpt": "dup",
                        "evidence_id": "e2",
                        "document_id": "d2",
                        "source_id": "s2",
                        "workspace_id": "ws-active",
                    },
                ],
                "created_at": "2026-07-23T12:00:00Z",
            },
        )

    backend = FakeConversationChannelBackend()
    ask_client = WorkspaceAskHttpClient(
        SlackAskClientConfig(base_url="http://ask.test"),
        transport=httpx.MockTransport(ask_handler),
    )
    workflow = SlackAskWorkflow(
        auth_config=SlackCompanionAuthConfig(
            approved_team_id="T_OK",
            approved_user_id="U_OK",
            tenant_id="tenant-a",
            active_workspace_id="ws-active",
        ),
        dedupe=SlackEventDedupeRepository(InMemoryDocumentStore()),
        ask_client=ask_client,
        send=backend.send,
    )

    event = _event()
    await workflow.handle(event)
    assert len(ask_calls) == 1
    assert ask_calls[0].headers["X-Tenant-Id"] == "tenant-a"
    assert "ws-active" in str(ask_calls[0].url)
    assert len(backend.sent) == 2
    assert backend.sent[0].text == ACK_TEXT
    assert backend.sent[0].address.thread_id == "1711111.000200"
    assert backend.sent[1].address.thread_id == "1711111.000200"
    assert backend.sent[1].address.conversation_id == "Dchannel"
    assert backend.sent[1].address.installation_id == "T_OK"
    assert "Employees receive 20 days." in backend.sent[1].text
    assert "hr-policy.pdf" in backend.sent[1].text
    assert "/local/secret" not in backend.sent[1].text
    assert "do-not-leak" not in backend.sent[1].text

    await workflow.handle(event)
    assert len(ask_calls) == 1
    assert len(backend.sent) == 2


@pytest.mark.asyncio
async def test_dedupe_occurs_before_acknowledgement_and_ask() -> None:
    order: list[str] = []
    store = InMemoryDocumentStore()
    dedupe = SlackEventDedupeRepository(store)

    async def tracking_send(message: OutboundConversationMessage) -> Any:
        order.append("ack" if message.text == ACK_TEXT else "final")
        return ConversationDeliveryReceipt(
            message_id="m",
            address=message.address,
        )

    def ask_handler(_request: httpx.Request) -> httpx.Response:
        order.append("ask")
        return httpx.Response(
            200,
            json={
                "run_id": "r",
                "workspace_id": "ws",
                "status": "completed",
                "question": "Q",
                "answer": "A",
                "citations": [],
                "created_at": "2026-07-23T12:00:00Z",
            },
        )

    # Pre-claim so workflow sees duplicate before any outbound/Ask.
    assert dedupe.claim(team_id="T_OK", event_id="Ev-dup") is not None
    workflow = SlackAskWorkflow(
        auth_config=SlackCompanionAuthConfig(
            approved_team_id="T_OK",
            approved_user_id="U_OK",
            tenant_id="t",
            active_workspace_id="ws",
        ),
        dedupe=dedupe,
        ask_client=WorkspaceAskHttpClient(
            SlackAskClientConfig(base_url="http://ask.test"),
            transport=httpx.MockTransport(ask_handler),
        ),
        send=tracking_send,
    )
    await workflow.handle(_event(event_id="Ev-dup"))
    assert order == []


@pytest.mark.asyncio
async def test_companion_start_stop_with_fake_integration() -> None:
    backend = FakeConversationChannelBackend()
    integration = SlackConversationChannelIntegration.from_backend(backend, enabled=True)
    settings = LocalWorkspaceBackendSettings(
        slack_companion_enabled=True,
        slack_approved_team_id="T_OK",
        slack_approved_user_id="U_OK",
        slack_tenant_id="tenant-a",
        slack_active_workspace_id="ws-active",
        slack_ask_base_url="http://ask.test",
    )
    runtime = resolve_slack_companion_runtime_config(settings)
    assert runtime is not None
    companion = build_slack_companion(
        runtime=runtime,
        document_store=InMemoryDocumentStore(),
        integration=integration,
        ask_client=WorkspaceAskHttpClient(
            SlackAskClientConfig(base_url="http://ask.test"),
            transport=httpx.MockTransport(
                lambda _r: httpx.Response(
                    200,
                    json={
                        "run_id": "r",
                        "workspace_id": "ws-active",
                        "status": "completed",
                        "question": "Q",
                        "answer": "A",
                        "citations": [],
                        "created_at": "2026-07-23T12:00:00Z",
                    },
                )
            ),
        ),
    )
    await companion.start()
    assert backend.started is True
    assert backend.handler is not None
    await backend.handler(_event())
    assert len(backend.sent) == 2
    await companion.stop()
    assert backend.stopped is True


class _GatedAskClient:
    """Deterministic Ask gate for concurrent workflow.handle races."""

    def __init__(self) -> None:
        self.calls = 0
        self._calls_lock = threading.Lock()
        self._entered = threading.Event()
        self._release = threading.Event()

    async def ask(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        question: str,
    ) -> SlackAskHttpResponse:
        del tenant_id, workspace_id, question
        with self._calls_lock:
            self.calls += 1
        self._entered.set()
        assert self._release.wait(timeout=5.0)
        return SlackAskHttpResponse(
            run_id="ask-concurrent",
            workspace_id="ws-active",
            status="completed",
            question="Q",
            answer="Concurrent answer",
            citations=[],
        )


class _ClaimBarrierDedupe(SlackEventDedupeRepository):
    """Synchronize callers immediately before the shared process claim lock."""

    def __init__(self, document_store: InMemoryDocumentStore, barrier: threading.Barrier) -> None:
        super().__init__(document_store)
        self._barrier = barrier

    def claim(self, *, team_id: str, event_id: str) -> Any:
        self._barrier.wait(timeout=2.0)
        return super().claim(team_id=team_id, event_id=event_id)


@pytest.mark.asyncio
async def test_concurrent_handle_asks_and_replies_once() -> None:
    backend = FakeConversationChannelBackend()
    ask_client = _GatedAskClient()
    claim_barrier = threading.Barrier(2)
    workflow = SlackAskWorkflow(
        auth_config=SlackCompanionAuthConfig(
            approved_team_id="T_OK",
            approved_user_id="U_OK",
            tenant_id="tenant-a",
            active_workspace_id="ws-active",
        ),
        dedupe=_ClaimBarrierDedupe(InMemoryDocumentStore(), claim_barrier),
        ask_client=ask_client,  # type: ignore[arg-type]
        send=backend.send,
    )
    event = _event(event_id="EvConcurrent")

    errors: list[BaseException] = []

    def _run_handle() -> None:
        try:
            asyncio.run(workflow.handle(event))
        except BaseException as exc:  # noqa: BLE001 — surface in parent thread
            errors.append(exc)

    threads = [
        threading.Thread(target=_run_handle),
        threading.Thread(target=_run_handle),
    ]
    for thread in threads:
        thread.start()

    entered = await asyncio.to_thread(ask_client._entered.wait, 2.0)
    assert entered is True
    assert ask_client.calls == 1
    ask_client._release.set()

    for thread in threads:
        thread.join(timeout=5.0)
        assert not thread.is_alive()
    assert errors == []

    assert ask_client.calls == 1
    assert len(backend.sent) == 2
    assert backend.sent[0].text == ACK_TEXT
    assert "Concurrent answer" in backend.sent[1].text
