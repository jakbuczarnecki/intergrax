# © Artur Czarnecki. All rights reserved.

"""Slack backend ack ordering and lifecycle tests."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import Any

import httpx
import pytest

from intergrax.integrations.contracts.conversation_channel import (
    ConversationAttachmentFetchError,
    ConversationAttachmentReference,
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
from intergrax.integrations.providers.conversation_channel.slack.integration import (
    SlackConversationChannelIntegration,
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
    files_info_calls: list[str] = field(default_factory=list)
    files_info_response: Any | None = None
    files_info_error: BaseException | None = None

    async def chat_postMessage(self, **kwargs: Any) -> dict[str, Any]:
        return {"ok": True, "ts": "1.0"}

    async def files_info(self, *, file: str) -> Any:
        self.files_info_calls.append(file)
        if self.files_info_error is not None:
            raise self.files_info_error
        if self.files_info_response is not None:
            return self.files_info_response
        return {
            "ok": True,
            "file": {
                "id": file,
                "name": "contract.pdf",
                "mimetype": "application/pdf",
                "size": 4,
                "url_private_download": "https://files.slack.com/files-pri/T-F/download",
                "url_private": "https://files.slack.com/files-pri/T-F/private",
            },
        }


class _DataEnvelope:
    def __init__(self, data: Mapping[str, Any]) -> None:
        self.data = data


class _GetEnvelope:
    def __init__(self, payload: Mapping[str, Any]) -> None:
        self._payload = payload

    def get(self, key: str, default: Any = None) -> Any:
        return self._payload.get(key, default)


def _config() -> SlackConversationChannelIntegrationConfig:
    return SlackConversationChannelIntegrationConfig(
        enabled=True,
        app_token="xapp-test-aaaa",
        bot_token="xoxb-test-bbbb",
    )


def _file_payload(**overrides: object) -> dict[str, Any]:
    file_obj: dict[str, Any] = {
        "id": "F111",
        "name": "contract.pdf",
        "mimetype": "application/pdf",
        "size": 4,
        "url_private_download": "https://files.slack.com/files-pri/T-F/download",
        "url_private": "https://files.slack.com/files-pri/T-F/private",
    }
    file_obj.update(overrides)
    return {"ok": True, "file": file_obj}


def _backend(
    socket: FakeSocketClient | None = None,
    *,
    max_in_flight_handlers: int = 8,
    web_client: FakeWebClient | None = None,
    attachment_transport: httpx.AsyncBaseTransport | None = None,
) -> tuple[SlackConversationChannelBackend, FakeSocketClient, FakeWebClient]:
    client = socket or FakeSocketClient()
    web = web_client or FakeWebClient()
    backend = SlackConversationChannelBackend(
        config=_config(),
        web_client=web,
        socket_client=client,
        max_in_flight_handlers=max_in_flight_handlers,
        socket_mode_response_cls=FakeSocketModeResponse,
        attachment_transport=attachment_transport,
    )
    return backend, client, web


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
    backend, client, _web = _backend()
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
    backend, client, _web = _backend()
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
    backend, client, _web = _backend()
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
    backend, client, _web = _backend()
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
    backend, client, _web = _backend()
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
    backend, client, _web = _backend()
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
    backend, client, _web = _backend()
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
    backend, _client, _web = _backend()
    await backend.stop()
    assert backend.lifecycle_state is SlackConversationLifecycleState.STOPPED


@pytest.mark.asyncio
async def test_startup_error_marks_degraded() -> None:
    client = FakeSocketClient()

    async def boom() -> None:
        raise RuntimeError("connect failed")

    client.connect = boom  # type: ignore[method-assign]
    backend, _, _web = _backend(client)

    async def handler(_event: InboundConversationEvent) -> None:
        return None

    with pytest.raises(SlackConversationLifecycleError):
        await backend.start(handler)
    assert backend.lifecycle_state is SlackConversationLifecycleState.DEGRADED
    assert backend.health().healthy is False


@pytest.mark.asyncio
async def test_pending_handler_tasks_cancelled_on_stop() -> None:
    backend, client, _web = _backend(max_in_flight_handlers=2)
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
    backend, client, _web = _backend(max_in_flight_handlers=1)
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


def _download_transport(
    *,
    body: bytes = b"data",
    status_code: int = 200,
    headers: dict[str, str] | None = None,
    capture: dict[str, Any] | None = None,
) -> httpx.MockTransport:
    captured = capture if capture is not None else {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("Authorization")
        captured["method"] = request.method
        return httpx.Response(
            status_code,
            content=body,
            headers=headers or {"Content-Length": str(len(body))},
        )

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_files_info_called_and_successful_download() -> None:
    capture: dict[str, Any] = {}
    web = FakeWebClient(files_info_response=_file_payload())
    backend, _socket, _web = _backend(
        web_client=web,
        attachment_transport=_download_transport(body=b"data", capture=capture),
    )
    content = await backend.fetch_attachment(
        ConversationAttachmentReference(attachment_id="F111"),
        max_bytes=100,
    )
    assert web.files_info_calls == ["F111"]
    assert content.body == b"data"
    assert content.file_name == "contract.pdf"
    assert content.content_type == "application/pdf"
    assert "xoxb-test-bbbb" not in content.file_name
    assert capture["authorization"] == "Bearer xoxb-test-bbbb"
    assert "files.slack.com" in capture["url"]


@pytest.mark.asyncio
async def test_url_private_download_preferred() -> None:
    capture: dict[str, Any] = {}
    web = FakeWebClient(
        files_info_response=_file_payload(
            url_private_download="https://files.slack.com/download/preferred",
            url_private="https://files.slack.com/private/fallback",
            size=4,
        )
    )
    backend, _, _ = _backend(
        web_client=web,
        attachment_transport=_download_transport(body=b"data", capture=capture),
    )
    await backend.fetch_attachment(
        ConversationAttachmentReference(attachment_id="F111"),
        max_bytes=100,
    )
    assert capture["url"] == "https://files.slack.com/download/preferred"


@pytest.mark.asyncio
async def test_url_private_fallback_accepted() -> None:
    capture: dict[str, Any] = {}
    web = FakeWebClient(
        files_info_response=_file_payload(
            url_private_download="",
            url_private="https://files.slack.com/private/fallback",
            size=4,
        )
    )
    backend, _, _ = _backend(
        web_client=web,
        attachment_transport=_download_transport(body=b"data", capture=capture),
    )
    await backend.fetch_attachment(
        ConversationAttachmentReference(attachment_id="F111"),
        max_bytes=100,
    )
    assert capture["url"] == "https://files.slack.com/private/fallback"


@pytest.mark.asyncio
async def test_declared_size_over_limit_skips_download() -> None:
    capture: dict[str, Any] = {}
    web = FakeWebClient(files_info_response=_file_payload(size=50))
    backend, _, _ = _backend(
        web_client=web,
        attachment_transport=_download_transport(capture=capture),
    )
    with pytest.raises(ConversationAttachmentFetchError) as exc:
        await backend.fetch_attachment(
            ConversationAttachmentReference(attachment_id="F111"),
            max_bytes=10,
        )
    assert exc.value.kind == "attachment_too_large"
    assert capture == {}


@pytest.mark.asyncio
async def test_stream_exceeding_limit_fails_too_large() -> None:
    web = FakeWebClient(files_info_response=_file_payload(size=None))
    # omit size key
    web.files_info_response = {
        "ok": True,
        "file": {
            "id": "F111",
            "name": "big.bin",
            "mimetype": "application/octet-stream",
            "url_private_download": "https://files.slack.com/files-pri/T-F/download",
        },
    }
    backend, _, _ = _backend(
        web_client=web,
        attachment_transport=_download_transport(body=b"0123456789ABCDEF"),
    )
    with pytest.raises(ConversationAttachmentFetchError) as exc:
        await backend.fetch_attachment(
            ConversationAttachmentReference(attachment_id="F111"),
            max_bytes=8,
        )
    assert exc.value.kind == "attachment_too_large"


@pytest.mark.asyncio
async def test_size_mismatch_fails() -> None:
    web = FakeWebClient(files_info_response=_file_payload(size=10))
    backend, _, _ = _backend(
        web_client=web,
        attachment_transport=_download_transport(body=b"data"),
    )
    with pytest.raises(ConversationAttachmentFetchError) as exc:
        await backend.fetch_attachment(
            ConversationAttachmentReference(attachment_id="F111"),
            max_bytes=100,
        )
    assert exc.value.kind == "attachment_size_mismatch"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "kind"),
    [
        (401, "attachment_access_denied"),
        (403, "attachment_access_denied"),
        (404, "attachment_metadata_unavailable"),
        (500, "attachment_download_failed"),
    ],
)
async def test_download_status_mapping(status: int, kind: str) -> None:
    web = FakeWebClient(files_info_response=_file_payload(size=4))
    backend, _, _ = _backend(
        web_client=web,
        attachment_transport=_download_transport(body=b"data", status_code=status),
    )
    with pytest.raises(ConversationAttachmentFetchError) as exc:
        await backend.fetch_attachment(
            ConversationAttachmentReference(attachment_id="F111"),
            max_bytes=100,
        )
    assert exc.value.kind == kind
    assert "xoxb" not in str(exc.value)


@pytest.mark.asyncio
async def test_redirect_not_followed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            302,
            headers={"Location": "https://evil.example/steal"},
        )

    web = FakeWebClient(files_info_response=_file_payload(size=4))
    backend, _, _ = _backend(
        web_client=web,
        attachment_transport=httpx.MockTransport(handler),
    )
    with pytest.raises(ConversationAttachmentFetchError) as exc:
        await backend.fetch_attachment(
            ConversationAttachmentReference(attachment_id="F111"),
            max_bytes=100,
        )
    assert exc.value.kind == "attachment_download_failed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "http://files.slack.com/x",
        "https://user:pass@files.slack.com/x",
        "https://evil.example/x",
    ],
)
async def test_invalid_download_url_rejected(url: str) -> None:
    web = FakeWebClient(
        files_info_response=_file_payload(
            url_private_download=url,
            url_private="",
            size=4,
        )
    )
    capture: dict[str, Any] = {}
    backend, _, _ = _backend(
        web_client=web,
        attachment_transport=_download_transport(capture=capture),
    )
    with pytest.raises(ConversationAttachmentFetchError) as exc:
        await backend.fetch_attachment(
            ConversationAttachmentReference(attachment_id="F111"),
            max_bytes=100,
        )
    assert exc.value.kind == "attachment_metadata_invalid"
    assert capture == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "file_overrides",
    [
        {"is_external": True},
        {"mode": "external"},
        {"mode": "remote"},
        {"remote": {"id": "R1"}},
    ],
)
async def test_external_or_remote_rejected(file_overrides: dict[str, Any]) -> None:
    web = FakeWebClient(files_info_response=_file_payload(**file_overrides))
    backend, _, _ = _backend(web_client=web)
    with pytest.raises(ConversationAttachmentFetchError) as exc:
        await backend.fetch_attachment(
            ConversationAttachmentReference(attachment_id="F111"),
            max_bytes=100,
        )
    assert exc.value.kind == "attachment_unsupported"


@pytest.mark.asyncio
async def test_check_file_info_rejected() -> None:
    web = FakeWebClient(
        files_info_response=_file_payload(file_access="check_file_info")
    )
    backend, _, _ = _backend(web_client=web)
    with pytest.raises(ConversationAttachmentFetchError) as exc:
        await backend.fetch_attachment(
            ConversationAttachmentReference(attachment_id="F111"),
            max_bytes=100,
        )
    assert exc.value.kind == "attachment_access_denied"


@pytest.mark.asyncio
async def test_malformed_files_info_rejected() -> None:
    web = FakeWebClient(files_info_response={"ok": True, "file": "bad"})
    backend, _, _ = _backend(web_client=web)
    with pytest.raises(ConversationAttachmentFetchError) as exc:
        await backend.fetch_attachment(
            ConversationAttachmentReference(attachment_id="F111"),
            max_bytes=100,
        )
    assert exc.value.kind == "attachment_metadata_invalid"


@pytest.mark.asyncio
async def test_files_info_mapping_response_supported() -> None:
    web = FakeWebClient(files_info_response=_file_payload())
    backend, _, _ = _backend(
        web_client=web,
        attachment_transport=_download_transport(body=b"data"),
    )
    content = await backend.fetch_attachment(
        ConversationAttachmentReference(attachment_id="F111"),
        max_bytes=100,
    )
    assert content.body == b"data"


@pytest.mark.asyncio
async def test_files_info_data_envelope_response_supported() -> None:
    web = FakeWebClient(files_info_response=_DataEnvelope(_file_payload()))
    backend, _, _ = _backend(
        web_client=web,
        attachment_transport=_download_transport(body=b"data"),
    )
    content = await backend.fetch_attachment(
        ConversationAttachmentReference(attachment_id="F111"),
        max_bytes=100,
    )
    assert content.body == b"data"
    assert content.file_name == "contract.pdf"


@pytest.mark.asyncio
async def test_files_info_callable_get_response_supported() -> None:
    web = FakeWebClient(files_info_response=_GetEnvelope(_file_payload()))
    backend, _, _ = _backend(
        web_client=web,
        attachment_transport=_download_transport(body=b"data"),
    )
    content = await backend.fetch_attachment(
        ConversationAttachmentReference(attachment_id="F111"),
        max_bytes=100,
    )
    assert content.body == b"data"
    assert content.content_type == "application/pdf"


@pytest.mark.asyncio
async def test_files_info_malformed_non_mapping_response() -> None:
    web = FakeWebClient(files_info_response=object())
    backend, _, _ = _backend(web_client=web)
    with pytest.raises(ConversationAttachmentFetchError) as exc:
        await backend.fetch_attachment(
            ConversationAttachmentReference(attachment_id="F111"),
            max_bytes=100,
        )
    assert exc.value.kind == "attachment_metadata_invalid"


@pytest.mark.asyncio
async def test_provider_exception_text_absent() -> None:
    web = FakeWebClient(
        files_info_error=RuntimeError("secret token xoxb-leaked url https://files.slack.com")
    )
    backend, _, _ = _backend(web_client=web)
    with pytest.raises(ConversationAttachmentFetchError) as exc:
        await backend.fetch_attachment(
            ConversationAttachmentReference(attachment_id="F111"),
            max_bytes=100,
        )
    assert exc.value.kind == "attachment_metadata_unavailable"
    assert "xoxb" not in str(exc.value)
    assert "https://" not in str(exc.value)
    assert "secret" not in str(exc.value)


@pytest.mark.asyncio
async def test_integration_delegates_fetch_attachment() -> None:
    web = FakeWebClient(files_info_response=_file_payload())
    backend, _, _ = _backend(
        web_client=web,
        attachment_transport=_download_transport(body=b"data"),
    )
    integration = SlackConversationChannelIntegration.from_backend(backend, enabled=True)
    content = await integration.fetch_attachment(
        ConversationAttachmentReference(attachment_id="F111"),
        max_bytes=100,
    )
    assert content.body == b"data"


@pytest.mark.asyncio
async def test_integration_unavailable_without_fetcher() -> None:
    class _NoFetchBackend:
        async def start(self, handler: Any) -> None:
            del handler

        async def stop(self) -> None:
            return None

        async def send(self, message: Any) -> Any:
            del message
            raise RuntimeError("unused")

        def health(self) -> bool:
            return True

    integration = SlackConversationChannelIntegration.from_backend(
        _NoFetchBackend(),  # type: ignore[arg-type]
        enabled=True,
    )
    with pytest.raises(ConversationAttachmentFetchError) as exc:
        await integration.fetch_attachment(
            ConversationAttachmentReference(attachment_id="F111"),
            max_bytes=100,
        )
    assert exc.value.kind == "attachment_fetch_unavailable"
