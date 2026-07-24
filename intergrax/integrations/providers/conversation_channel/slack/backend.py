# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Slack Socket Mode + Web API conversation-channel backend."""

from __future__ import annotations

import asyncio
import logging
from enum import Enum
from typing import Any, Callable, Mapping

from intergrax.integrations.contracts.base import (
    HealthStatus,
    IntegrationConfigurationError,
    IntegrationDependencyError,
    IntegrationError,
)
from intergrax.integrations.contracts.conversation_channel import (
    ConversationChannelBackend,
    ConversationDeliveryReceipt,
    ConversationEventHandler,
    InboundConversationEvent,
    OutboundConversationMessage,
)
from intergrax.integrations.providers.conversation_channel.slack.config import (
    SlackConversationChannelIntegrationConfig,
)
from intergrax.integrations.providers.conversation_channel.slack.mapping import (
    map_socket_mode_payload,
    parse_slack_ts,
)
from intergrax.integrations.providers.conversation_channel.slack.rendering import (
    render_chat_post_message_args,
)
from intergrax.utils import attribute_access

_LOG = logging.getLogger(__name__)
_DEFAULT_MAX_IN_FLIGHT = 32
_SLUG = "slack"


class SlackConversationLifecycleState(str, Enum):
    DISABLED = "disabled"
    STOPPED = "stopped"
    STARTING = "starting"
    READY = "ready"
    RECONNECTING = "reconnecting"
    DEGRADED = "degraded"
    STOPPING = "stopping"


class SlackConversationLifecycleError(IntegrationError):
    """Invalid backend lifecycle transition."""


class SlackConversationSendError(IntegrationDependencyError):
    """Slack Web API send failed or returned a malformed success payload."""


class SlackConversationDependencyError(IntegrationConfigurationError):
    """Official Slack SDK is missing or cannot be imported."""


def _redacted_error(exc: BaseException) -> str:
    name = type(exc).__name__
    # Never include exception args that might echo configuration values.
    return f"{name}"


def _import_slack_sdk() -> tuple[Any, Any, Any, Any]:
    try:
        from slack_sdk.errors import SlackApiError
        from slack_sdk.socket_mode.aiohttp import SocketModeClient
        from slack_sdk.socket_mode.response import SocketModeResponse
        from slack_sdk.web.async_client import AsyncWebClient
    except ImportError as exc:
        raise SlackConversationDependencyError(
            "Slack conversation runtime requires slack-sdk and aiohttp. "
            "Install with: uv sync --extra integrations-slack",
        ) from exc
    return SocketModeClient, AsyncWebClient, SocketModeResponse, SlackApiError


class SlackConversationChannelBackend:
    """Production Slack conversation backend (Socket Mode inbound, Web API outbound).

    ``start()`` returns after the Socket Mode client has connected successfully
    and the backend is able to receive envelopes. Reconnect is owned by
    ``slack_sdk`` ``SocketModeClient(auto_reconnect_enabled=True)``.
    """

    def __init__(
        self,
        *,
        config: SlackConversationChannelIntegrationConfig,
        web_client: Any | None = None,
        socket_client: Any | None = None,
        max_in_flight_handlers: int = _DEFAULT_MAX_IN_FLIGHT,
        socket_client_factory: Callable[[str, Any], Any] | None = None,
        web_client_factory: Callable[[str, float], Any] | None = None,
        socket_mode_response_cls: Any | None = None,
        slack_api_error_cls: type[BaseException] | None = None,
    ) -> None:
        if max_in_flight_handlers < 1:
            raise IntegrationConfigurationError("max_in_flight_handlers must be >= 1")
        self._config = config
        self._state = SlackConversationLifecycleState.STOPPED
        self._handler: ConversationEventHandler | None = None
        self._web_client = web_client
        self._socket_client = socket_client
        self._owns_clients = web_client is None and socket_client is None
        self._socket_client_factory = socket_client_factory
        self._web_client_factory = web_client_factory
        self._socket_mode_response_cls = socket_mode_response_cls
        self._slack_api_error_cls = slack_api_error_cls
        self._sdk_available = True
        self._config_valid = True
        self._last_transport_failure: str | None = None
        self._listener_registered = False
        self._tasks: set[asyncio.Task[None]] = set()
        self._semaphore = asyncio.Semaphore(max_in_flight_handlers)
        self._max_in_flight = max_in_flight_handlers
        self._lock = asyncio.Lock()

    @classmethod
    def from_config(
        cls,
        config: SlackConversationChannelIntegrationConfig,
        *,
        max_in_flight_handlers: int = _DEFAULT_MAX_IN_FLIGHT,
    ) -> SlackConversationChannelBackend:
        if not config.enabled:
            raise IntegrationConfigurationError(
                "SlackConversationChannelBackend requires enabled=True configuration",
            )
        # Validate tokens early; do not open network connections here.
        config.validate_for_runtime()
        _import_slack_sdk()
        return cls(config=config, max_in_flight_handlers=max_in_flight_handlers)

    @property
    def lifecycle_state(self) -> SlackConversationLifecycleState:
        return self._state

    def _ensure_clients(self) -> None:
        if self._web_client is not None and self._socket_client is not None:
            return
        SocketModeClient, AsyncWebClient, SocketModeResponse, SlackApiError = _import_slack_sdk()
        app_token, bot_token = self._config.require_runtime_tokens()
        if self._socket_mode_response_cls is None:
            self._socket_mode_response_cls = SocketModeResponse
        if self._slack_api_error_cls is None:
            self._slack_api_error_cls = SlackApiError
        if self._web_client is None:
            if self._web_client_factory is not None:
                self._web_client = self._web_client_factory(
                    bot_token,
                    self._config.api_timeout_seconds,
                )
            else:
                self._web_client = AsyncWebClient(
                    token=bot_token,
                    timeout=self._config.api_timeout_seconds,
                )
        if self._socket_client is None:
            if self._socket_client_factory is not None:
                self._socket_client = self._socket_client_factory(app_token, self._web_client)
            else:
                self._socket_client = SocketModeClient(
                    app_token=app_token,
                    web_client=self._web_client,
                    auto_reconnect_enabled=True,
                )

    def _register_listener(self) -> None:
        if self._listener_registered or self._socket_client is None:
            return
        listeners = attribute_access.optional(
            self._socket_client,
            "socket_mode_request_listeners",
            None,
        )
        if listeners is None:
            raise SlackConversationLifecycleError(
                "Socket Mode client missing socket_mode_request_listeners",
            )
        listeners.append(self._on_socket_mode_request)
        on_close = attribute_access.optional(
            self._socket_client,
            "on_close_listeners",
            None,
        )
        if isinstance(on_close, list):
            on_close.append(self._on_socket_close)
        self._listener_registered = True

    async def _on_socket_close(self, *_args: Any, **_kwargs: Any) -> None:
        if self._state in {
            SlackConversationLifecycleState.STOPPING,
            SlackConversationLifecycleState.STOPPED,
            SlackConversationLifecycleState.STARTING,
        }:
            return
        if attribute_access.optional_bool(
            self._socket_client,
            "auto_reconnect_enabled",
            default=False,
        ):
            self._state = SlackConversationLifecycleState.RECONNECTING
            _LOG.info("slack conversation: transport closed; SDK reconnect in progress")
        else:
            self._state = SlackConversationLifecycleState.DEGRADED
            self._last_transport_failure = "socket_closed"

    async def start(self, handler: ConversationEventHandler) -> None:
        async with self._lock:
            if self._state in {
                SlackConversationLifecycleState.STARTING,
                SlackConversationLifecycleState.READY,
                SlackConversationLifecycleState.RECONNECTING,
            }:
                raise SlackConversationLifecycleError(
                    "Slack conversation backend already started or starting",
                )
            if self._state is SlackConversationLifecycleState.STOPPING:
                raise SlackConversationLifecycleError(
                    "Slack conversation backend is stopping",
                )
            self._state = SlackConversationLifecycleState.STARTING
            self._handler = handler
            try:
                self._ensure_clients()
                if self._socket_client is not None:
                    if hasattr(self._socket_client, "auto_reconnect_enabled"):
                        self._socket_client.auto_reconnect_enabled = True
                    if hasattr(self._socket_client, "default_auto_reconnect_enabled"):
                        self._socket_client.default_auto_reconnect_enabled = True
                self._register_listener()
                connect = attribute_access.optional(
                    self._socket_client,
                    "connect",
                    None,
                )
                if connect is None:
                    raise SlackConversationLifecycleError("Socket Mode client missing connect()")
                result = connect()
                if asyncio.iscoroutine(result) or asyncio.isfuture(result):
                    await result
                # Mark reconnecting→ready when SDK reports connected after start.
                self._state = SlackConversationLifecycleState.READY
                self._last_transport_failure = None
                _LOG.info("slack conversation: Socket Mode started; ready to receive envelopes")
            except Exception as exc:
                self._state = SlackConversationLifecycleState.DEGRADED
                self._last_transport_failure = _redacted_error(exc)
                _LOG.warning(
                    "slack conversation: startup failed (%s)",
                    self._last_transport_failure,
                )
                raise SlackConversationLifecycleError(
                    f"Slack Socket Mode startup failed ({self._last_transport_failure})",
                ) from exc

    async def stop(self) -> None:
        async with self._lock:
            if self._state in {
                SlackConversationLifecycleState.STOPPED,
                SlackConversationLifecycleState.DISABLED,
            }:
                return
            self._state = SlackConversationLifecycleState.STOPPING
            self._handler = None
            # Disable further SDK reconnect attempts before closing.
            if self._socket_client is not None:
                if hasattr(self._socket_client, "auto_reconnect_enabled"):
                    self._socket_client.auto_reconnect_enabled = False
                if hasattr(self._socket_client, "default_auto_reconnect_enabled"):
                    self._socket_client.default_auto_reconnect_enabled = False
            await self._drain_handler_tasks()
            await self._close_socket_client()
            if self._owns_clients:
                await self._close_web_client()
                self._socket_client = None
                self._web_client = None
                self._listener_registered = False
            else:
                # Injected clients remain usable for explicit start→stop→start.
                self._listener_registered = True
            self._state = SlackConversationLifecycleState.STOPPED
            _LOG.info("slack conversation: stopped")

    async def _drain_handler_tasks(self) -> None:
        pending = list(self._tasks)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self._tasks.clear()

    async def _close_socket_client(self) -> None:
        await self._close_client(self._socket_client)

    async def _close_web_client(self) -> None:
        await self._close_client(self._web_client)

    async def _close_client(self, client: Any | None) -> None:
        if client is None:
            return
        for method_name in ("close", "disconnect"):
            method = attribute_access.optional(client, method_name, None)
            if not callable(method):
                continue
            try:
                result = method()
                if asyncio.iscoroutine(result) or asyncio.isfuture(result):
                    await result
                break
            except Exception as exc:  # noqa: BLE001 — best-effort shutdown
                _LOG.warning(
                    "slack conversation: error during %s (%s)",
                    method_name,
                    _redacted_error(exc),
                )

    async def _on_socket_mode_request(self, client: Any, request: Any) -> None:
        envelope_id = attribute_access.optional(
            request,
            "envelope_id",
            None,
        )
        envelope_id_text = str(envelope_id).strip() if envelope_id is not None else ""
        if not envelope_id_text:
            _LOG.warning("slack conversation: malformed transport envelope without envelope_id")
            return

        try:
            await self._acknowledge(client, envelope_id_text)
        except Exception as exc:  # noqa: BLE001 — ack failure is transport-level
            self._state = SlackConversationLifecycleState.DEGRADED
            self._last_transport_failure = f"ack_failed:{_redacted_error(exc)}"
            _LOG.warning(
                "slack conversation: envelope ack failed (%s)",
                self._last_transport_failure,
            )
            return

        if self._state is SlackConversationLifecycleState.RECONNECTING:
            self._state = SlackConversationLifecycleState.READY

        envelope_type = attribute_access.optional(request, "type", None)
        payload = attribute_access.optional(request, "payload", None)
        if not isinstance(payload, Mapping):
            _LOG.info(
                "slack conversation: ignoring envelope type=%s (non-mapping payload)",
                envelope_type,
            )
            return

        event = map_socket_mode_payload(envelope_type=str(envelope_type or ""), payload=payload)
        if event is None:
            _LOG.debug(
                "slack conversation: envelope type=%s acknowledged and ignored",
                envelope_type,
            )
            return
        self._schedule_handler(event)

    async def _acknowledge(self, client: Any, envelope_id: str) -> None:
        response_cls = self._socket_mode_response_cls
        if response_cls is None:
            _, _, SocketModeResponse, _ = _import_slack_sdk()
            response_cls = SocketModeResponse
        send = attribute_access.optional(
            client,
            "send_socket_mode_response",
            None,
        )
        if not callable(send):
            raise SlackConversationLifecycleError("Socket Mode client missing send_socket_mode_response")
        result = send(response_cls(envelope_id=envelope_id))
        if asyncio.iscoroutine(result) or asyncio.isfuture(result):
            await result

    def _schedule_handler(self, event: InboundConversationEvent) -> None:
        handler = self._handler
        if handler is None:
            return
        if self._state in {
            SlackConversationLifecycleState.STOPPING,
            SlackConversationLifecycleState.STOPPED,
            SlackConversationLifecycleState.DISABLED,
        }:
            return
        if len(self._tasks) >= self._max_in_flight:
            _LOG.warning(
                "slack conversation: dropping inbound event; in-flight handler limit reached",
            )
            return

        async def _run() -> None:
            async with self._semaphore:
                try:
                    await handler(event)
                except Exception as exc:  # noqa: BLE001 — handler isolation
                    _LOG.warning(
                        "slack conversation: handler raised (%s) kind=%s",
                        _redacted_error(exc),
                        event.kind.value,
                    )

        task = asyncio.create_task(_run(), name="slack-conversation-handler")
        self._tasks.add(task)

        def _done(done: asyncio.Task[None]) -> None:
            self._tasks.discard(done)
            if done.cancelled():
                return
            exc = done.exception()
            if exc is not None:
                _LOG.debug(
                    "slack conversation: handler task finished with %s",
                    _redacted_error(exc),
                )

        task.add_done_callback(_done)

    async def send(self, message: OutboundConversationMessage) -> ConversationDeliveryReceipt:
        if self._web_client is None:
            self._ensure_clients()
        assert self._web_client is not None
        args = render_chat_post_message_args(message)
        try:
            response = await self._web_client.chat_postMessage(**args)
        except Exception as exc:
            error_cls = self._slack_api_error_cls
            if error_cls is not None and isinstance(exc, error_cls):
                code = ""
                response = attribute_access.optional(exc, "response", None)
                if isinstance(response, Mapping):
                    code = str(response.get("error") or "")
                raise SlackConversationSendError(
                    f"Slack chat.postMessage failed{f': {code}' if code else ''}",
                    integration_name=_SLUG,
                ) from exc
            # Attempt to classify SDK SlackApiError without requiring import at module load.
            try:
                _, _, _, SlackApiError = _import_slack_sdk()
            except SlackConversationDependencyError:
                SlackApiError = ()  # type: ignore[assignment, misc]
            if SlackApiError and isinstance(exc, SlackApiError):
                code = ""
                response = attribute_access.optional(exc, "response", None)
                if isinstance(response, Mapping):
                    code = str(response.get("error") or "")
                raise SlackConversationSendError(
                    f"Slack chat.postMessage failed{f': {code}' if code else ''}",
                    integration_name=_SLUG,
                ) from exc
            raise SlackConversationSendError(
                f"Slack chat.postMessage failed ({_redacted_error(exc)})",
                integration_name=_SLUG,
            ) from exc

        return self._receipt_from_response(message, response)

    def _receipt_from_response(
        self,
        message: OutboundConversationMessage,
        response: Any,
    ) -> ConversationDeliveryReceipt:
        data: Mapping[str, Any]
        if isinstance(response, Mapping):
            data = response
        elif hasattr(response, "data") and isinstance(response.data, Mapping):
            data = response.data
        elif hasattr(response, "get"):
            try:
                data = {
                    "ok": response.get("ok"),
                    "ts": response.get("ts"),
                    "message": response.get("message"),
                }
            except Exception as exc:  # noqa: BLE001
                raise SlackConversationSendError(
                    "Slack chat.postMessage returned an unreadable response",
                    integration_name=_SLUG,
                ) from exc
        else:
            raise SlackConversationSendError(
                "Slack chat.postMessage returned an unreadable response",
                integration_name=_SLUG,
            )

        if data.get("ok") is False:
            code = str(data.get("error") or "unknown_error")
            raise SlackConversationSendError(
                f"Slack chat.postMessage failed: {code}",
                integration_name=_SLUG,
            )

        ts = data.get("ts")
        if not isinstance(ts, str) or not ts.strip():
            message_obj = data.get("message")
            if isinstance(message_obj, Mapping):
                ts = message_obj.get("ts")
        if not isinstance(ts, str) or not ts.strip():
            raise SlackConversationSendError(
                "Slack chat.postMessage success response missing message timestamp",
                integration_name=_SLUG,
            )
        return ConversationDeliveryReceipt(
            message_id=ts.strip(),
            address=message.address,
            delivered_at=parse_slack_ts(ts.strip()),
        )

    def health(self) -> HealthStatus:
        detail_parts = [
            f"lifecycle={self._state.value}",
            f"sdk_available={self._sdk_available}",
            f"config_valid={self._config_valid}",
            f"socket_constructed={self._socket_client is not None}",
            f"transport_started={self._state in {SlackConversationLifecycleState.READY, SlackConversationLifecycleState.RECONNECTING}}",
        ]
        if self._last_transport_failure:
            detail_parts.append(f"last_failure={self._last_transport_failure}")
        healthy = self._state is SlackConversationLifecycleState.READY
        return HealthStatus(slug=_SLUG, healthy=healthy, detail="; ".join(detail_parts))


ConversationChannelBackend.register(SlackConversationChannelBackend)

__all__ = [
    "SlackConversationChannelBackend",
    "SlackConversationDependencyError",
    "SlackConversationLifecycleError",
    "SlackConversationLifecycleState",
    "SlackConversationSendError",
]
