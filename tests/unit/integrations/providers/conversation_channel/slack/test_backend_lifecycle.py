# © Artur Czarnecki. All rights reserved.

"""Slack backend ack ordering and lifecycle tests."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

from intergrax.integrations.contracts.conversation_channel import (
    ConversationEventKind,
    InboundConversationEvent,
)
from intergrax.integrations.providers.conversation_channel.slack.backend import (
    SlackConversationChannelBackend,
    SlackConversationLifecycleError,
    SlackConversationLifecycleState,
)
from intergrax.integrations.providers.conversation_channel.slack.config import (
    SlackConversationChannelIntegrationConfig,
)

pytestmark = pytest.mark.unit


@dataclass
class FakeSocketModeResponse:
    envelope_id: str

    def to_dict(self) -> dict[str, str]:
        return {"envelope_id": self.envelope_id}


@dataclass
class FakeSocketModeRequest:
    envelope_id: str | None
    type: str
    payload: dict[str, Any]


@dataclass
class FakeSocketClient:
    connected: bool = False
    closed: bool = False
    auto_reconnect_enabled: bool = True
    default_auto_reconnect_enabled: bool = True
    socket_mode_request_listeners: list[Any] = field(default_factory=list)
    on_close_listeners: list[Any] = field(default_factory=list)
    acks: list[str] = field(default_factory=list)
    fail_ack: bool = False

    async def connect(self) -> None:
        self.connected = True

    async def close(self) -> None:
        self.closed = True
        self.connected = False

    async def send_socket_mode_response(self, response: Any) -> None:
        if self.fail_ack:
            raise RuntimeError("ack transport failure")
        if isinstance(response, FakeSocketModeResponse):
            envelope_id = response.envelope_id
        elif isinstance(response, dict):
            envelope_id = response.get("envelope_id")
        else:
            envelope_id = None
        self.acks.append(str(envelope_id))


@dataclass
class FakeWebClient:
    async def chat_postMessage(self, **kwargs: Any) -> dict[str, Any]:
        return {"ok": True, "ts": "1.0"}


def _config() -> SlackConversationChannelIntegrationConfig:
    return SlackConversationChannelIntegrationConfig(
        enabled=True,
        app_token="xapp-test-aaaa",
        bot_token="xoxb-test-bbbb",
    )


def _backend(
    socket: FakeSocketClient | None = None,
    *,
    max_in_flight_handlers: int = 8,
) -> tuple[SlackConversationChannelBackend, FakeSocketClient]:
    client = socket or FakeSocketClient()
    backend = SlackConversationChannelBackend(
        config=_config(),
        web_client=FakeWebClient(),
        socket_client=client,
        max_in_flight_handlers=max_in_flight_handlers,
        socket_mode_response_cls=FakeSocketModeResponse,
    )
    return backend, client


def _message_request(*, envelope_id: str = "env-1") -> FakeSocketModeRequest:
    return FakeSocketModeRequest(
        envelope_id=envelope_id,
        type="events_api",
        payload={
            "event_id": "Ev1",
            "team_id": "T1",
            "event": {
                "type": "message",
                "channel_type": "im",
                "channel": "D1",
                "user": "U1",
                "text": "hello",
                "ts": "1710000000.000100",
            },
        },
    )


def _action_request(*, envelope_id: str = "env-2") -> FakeSocketModeRequest:
    return FakeSocketModeRequest(
        envelope_id=envelope_id,
        type="interactive",
        payload={
            "type": "block_actions",
            "team": {"id": "T1"},
            "user": {"id": "U1"},
            "channel": {"id": "D1"},
            "container": {"message_ts": "1710000000.000200", "thread_ts": "1710000000.000100"},
            "message": {"ts": "1710000000.000200", "thread_ts": "1710000000.000100"},
            "actions": [
                {
                    "type": "static_select",
                    "action_id": "choose",
                    "action_ts": "1710000001.000001",
                    "selected_option": {"value": "a"},
                }
            ],
        },
    )


@pytest.mark.asyncio
async def test_ack_before_handler_for_valid_message() -> None:
    backend, client = _backend()
    order: list[str] = []

    async def handler(event: InboundConversationEvent) -> None:
        order.append(f"handler:{event.kind.value}")

    await backend.start(handler)
    # Patch ack to record ordering relative to handler.
    original_ack = client.send_socket_mode_response

    async def tracked_ack(response: Any) -> None:
        order.append("ack")
        await original_ack(response)

    client.send_socket_mode_response = tracked_ack  # type: ignore[method-assign]
    await client.socket_mode_request_listeners[0](client, _message_request())
    await asyncio.sleep(0)
    assert order[0] == "ack"
    assert order[1] == "handler:message"
    assert order.count("handler:message") == 1


@pytest.mark.asyncio
async def test_ack_before_handler_for_valid_action() -> None:
    backend, client = _backend()
    order: list[str] = []

    async def handler(event: InboundConversationEvent) -> None:
        order.append(f"handler:{event.kind.value}")

    await backend.start(handler)

    async def tracked_ack(response: Any) -> None:
        order.append("ack")
        await FakeSocketClient.send_socket_mode_response(client, response)

    client.send_socket_mode_response = tracked_ack  # type: ignore[method-assign]
    await client.socket_mode_request_listeners[0](client, _action_request())
    await asyncio.sleep(0)
    assert order == ["ack", "handler:action"]


@pytest.mark.asyncio
async def test_unsupported_event_acks_without_handler() -> None:
    backend, client = _backend()
    called = False

    async def handler(_event: InboundConversationEvent) -> None:
        nonlocal called
        called = True

    await backend.start(handler)
    request = FakeSocketModeRequest(
        envelope_id="env-x",
        type="events_api",
        payload={"event_id": "EvX", "team_id": "T1", "event": {"type": "app_mention"}},
    )
    await client.socket_mode_request_listeners[0](client, request)
    await asyncio.sleep(0)
    assert client.acks == ["env-x"]
    assert called is False


@pytest.mark.asyncio
async def test_malformed_event_acks_without_handler() -> None:
    backend, client = _backend()
    called = False

    async def handler(_event: InboundConversationEvent) -> None:
        nonlocal called
        called = True

    await backend.start(handler)
    request = FakeSocketModeRequest(
        envelope_id="env-m",
        type="events_api",
        payload={"team_id": "T1", "event": {"type": "message", "channel_type": "im"}},
    )
    await client.socket_mode_request_listeners[0](client, request)
    await asyncio.sleep(0)
    assert client.acks == ["env-m"]
    assert called is False


@pytest.mark.asyncio
async def test_missing_envelope_id_skips_handler() -> None:
    backend, client = _backend()
    called = False

    async def handler(_event: InboundConversationEvent) -> None:
        nonlocal called
        called = True

    await backend.start(handler)
    await client.socket_mode_request_listeners[0](client, _message_request(envelope_id=""))
    await asyncio.sleep(0)
    assert client.acks == []
    assert called is False


@pytest.mark.asyncio
async def test_handler_exception_does_not_block_next_envelope() -> None:
    backend, client = _backend()
    seen: list[str] = []

    async def handler(event: InboundConversationEvent) -> None:
        seen.append(event.event_id)
        if event.event_id == "Ev1":
            raise RuntimeError("boom")

    await backend.start(handler)
    await client.socket_mode_request_listeners[0](client, _message_request(envelope_id="env-1"))
    second = _message_request(envelope_id="env-2")
    second.payload["event_id"] = "Ev2"
    await client.socket_mode_request_listeners[0](client, second)
    await asyncio.sleep(0.05)
    assert client.acks == ["env-1", "env-2"]
    assert seen == ["Ev1", "Ev2"]


@pytest.mark.asyncio
async def test_lifecycle_start_stop_and_health() -> None:
    backend, client = _backend()
    assert backend.lifecycle_state is SlackConversationLifecycleState.STOPPED
    health = backend.health()
    assert health.healthy is False

    async def handler(_event: InboundConversationEvent) -> None:
        return None

    await backend.start(handler)
    assert backend.lifecycle_state is SlackConversationLifecycleState.READY
    assert backend.health().healthy is True
    assert client.connected is True

    with pytest.raises(SlackConversationLifecycleError):
        await backend.start(handler)

    await backend.stop()
    assert backend.lifecycle_state is SlackConversationLifecycleState.STOPPED
    assert backend.health().healthy is False
    assert client.closed is True

    # Restart supported with injected clients.
    await backend.start(handler)
    assert backend.lifecycle_state is SlackConversationLifecycleState.READY
    await backend.stop()


@pytest.mark.asyncio
async def test_stop_before_start_is_safe() -> None:
    backend, _client = _backend()
    await backend.stop()
    assert backend.lifecycle_state is SlackConversationLifecycleState.STOPPED


@pytest.mark.asyncio
async def test_startup_error_marks_degraded() -> None:
    client = FakeSocketClient()

    async def boom() -> None:
        raise RuntimeError("connect failed")

    client.connect = boom  # type: ignore[method-assign]
    backend, _ = _backend(client)

    async def handler(_event: InboundConversationEvent) -> None:
        return None

    with pytest.raises(SlackConversationLifecycleError):
        await backend.start(handler)
    assert backend.lifecycle_state is SlackConversationLifecycleState.DEGRADED
    assert backend.health().healthy is False


@pytest.mark.asyncio
async def test_pending_handler_tasks_cancelled_on_stop() -> None:
    backend, client = _backend(max_in_flight_handlers=2)
    started = asyncio.Event()
    release = asyncio.Event()

    async def handler(_event: InboundConversationEvent) -> None:
        started.set()
        await release.wait()

    await backend.start(handler)
    await client.socket_mode_request_listeners[0](client, _message_request())
    await started.wait()
    stop_task = asyncio.create_task(backend.stop())
    await asyncio.sleep(0.01)
    release.set()
    await stop_task
    assert backend.lifecycle_state is SlackConversationLifecycleState.STOPPED
    assert len(backend._tasks) == 0  # noqa: SLF001 — lifecycle cleanup contract


@pytest.mark.asyncio
async def test_bounded_in_flight_dispatch() -> None:
    backend, client = _backend(max_in_flight_handlers=1)
    release = asyncio.Event()
    started = 0

    async def handler(_event: InboundConversationEvent) -> None:
        nonlocal started
        started += 1
        await release.wait()

    await backend.start(handler)
    await client.socket_mode_request_listeners[0](client, _message_request(envelope_id="a"))
    await asyncio.sleep(0)
    second = _message_request(envelope_id="b")
    second.payload["event_id"] = "EvDrop"
    await client.socket_mode_request_listeners[0](client, second)
    await asyncio.sleep(0)
    assert started == 1
    release.set()
    await asyncio.sleep(0.01)
    await backend.stop()
