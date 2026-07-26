# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Slack payload → shared conversation-channel models."""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any, Mapping

from intergrax.integrations.contracts.conversation_channel import (
    ConversationActionSelection,
    ConversationActor,
    ConversationAddress,
    ConversationAttachmentReference,
    ConversationEventKind,
    InboundConversationEvent,
)

_LOG = logging.getLogger(__name__)

SUPPORTED_ENVELOPE_TYPES = frozenset({"events_api", "interactive"})
_STATIC_SELECT = "static_select"
_SUPPORTED_MESSAGE_SUBTYPES = frozenset({"file_share"})


def _non_blank(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    return None


def parse_slack_ts(value: str | None) -> datetime | None:
    """Parse Slack ``ts`` (``seconds.microseconds``) into UTC datetime when safe."""
    raw = _non_blank(value)
    if raw is None:
        return None
    try:
        return datetime.fromtimestamp(float(raw), tz=timezone.utc)
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def deterministic_block_action_event_id(
    *,
    team_id: str,
    user_id: str,
    channel_id: str,
    action_ts: str,
    message_ts: str,
    action_id: str,
    selected_value: str,
) -> str:
    """Stable provider-scoped identity for Block Kit static_select actions."""
    canonical = "|".join(
        (
            "v1",
            team_id.strip(),
            user_id.strip(),
            channel_id.strip(),
            action_ts.strip() if action_ts.strip() else "-",
            message_ts.strip() if message_ts.strip() else "-",
            action_id.strip(),
            selected_value.strip(),
        )
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"slack:block_action:v1:{digest}"


def _map_size_bytes(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float):
        if value < 0 or not value.is_integer():
            return None
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or not stripped.isdigit():
            return None
        return int(stripped)
    return None


def _map_attachment_references(
    files_value: Any,
) -> tuple[ConversationAttachmentReference, ...] | None:
    """Map Slack ``files`` array; return None when malformed (fail-closed)."""
    if files_value is None:
        return ()
    if not isinstance(files_value, list):
        return None
    if not files_value:
        return ()
    mapped: list[ConversationAttachmentReference] = []
    for item in files_value:
        entry = _as_mapping(item)
        if entry is None:
            return None
        attachment_id = _non_blank(entry.get("id"))
        if attachment_id is None:
            return None
        size_bytes = _map_size_bytes(entry.get("size"))
        try:
            mapped.append(
                ConversationAttachmentReference(
                    attachment_id=attachment_id,
                    file_name=_non_blank(entry.get("name")),
                    content_type=_non_blank(entry.get("mimetype")),
                    size_bytes=size_bytes,
                )
            )
        except Exception:  # noqa: BLE001 — malformed entry rejects whole event
            return None
    return tuple(mapped)


def map_events_api_message(payload: Mapping[str, Any]) -> InboundConversationEvent | None:
    """Map Events API DM ``message`` payload to ``InboundConversationEvent`` or None."""
    event_id = _non_blank(payload.get("event_id"))
    if event_id is None:
        _LOG.info("slack conversation: events_api missing event_id; ignoring after ack")
        return None

    team_id = _non_blank(payload.get("team_id"))
    if team_id is None:
        _LOG.info("slack conversation: events_api missing team_id; ignoring")
        return None

    event = _as_mapping(payload.get("event"))
    if event is None:
        _LOG.info("slack conversation: events_api missing event object; ignoring")
        return None

    if _non_blank(event.get("type")) != "message":
        return None
    if _non_blank(event.get("channel_type")) != "im":
        _LOG.debug("slack conversation: ignoring non-im message")
        return None
    if _non_blank(event.get("bot_id")) is not None:
        _LOG.debug("slack conversation: ignoring bot-authored message")
        return None

    subtype = _non_blank(event.get("subtype"))
    if subtype is not None and subtype not in _SUPPORTED_MESSAGE_SUBTYPES:
        _LOG.debug("slack conversation: ignoring message subtype")
        return None

    attachments = _map_attachment_references(event.get("files"))
    if attachments is None:
        _LOG.info("slack conversation: events_api malformed files; ignoring")
        return None

    channel = _non_blank(event.get("channel"))
    user = _non_blank(event.get("user"))
    text = _non_blank(event.get("text"))
    ts = _non_blank(event.get("ts"))
    if channel is None or user is None or ts is None:
        _LOG.info("slack conversation: events_api message missing required fields; ignoring")
        return None
    if text is None and not attachments:
        _LOG.info("slack conversation: events_api message missing text/attachments; ignoring")
        return None

    thread_ts = _non_blank(event.get("thread_ts"))
    thread_id = thread_ts if thread_ts is not None else ts

    metadata: dict[str, Any] = {"slack_envelope_type": "events_api"}
    client_msg_id = _non_blank(event.get("client_msg_id"))
    if client_msg_id is not None:
        metadata["client_msg_id"] = client_msg_id

    return InboundConversationEvent(
        event_id=event_id,
        address=ConversationAddress(
            installation_id=team_id,
            conversation_id=channel,
            thread_id=thread_id,
        ),
        actor=ConversationActor(actor_id=user, display_name=None, is_bot=False),
        kind=ConversationEventKind.MESSAGE,
        text=text,
        occurred_at=parse_slack_ts(ts),
        attachments=attachments,
        metadata=metadata,
    )


def _resolve_action_thread_id(payload: Mapping[str, Any]) -> str | None:
    container = _as_mapping(payload.get("container")) or {}
    message = _as_mapping(payload.get("message")) or {}
    for candidate in (
        container.get("thread_ts"),
        message.get("thread_ts"),
        container.get("message_ts"),
        message.get("ts"),
    ):
        resolved = _non_blank(candidate)
        if resolved is not None:
            return resolved
    return None


def map_block_actions(payload: Mapping[str, Any]) -> InboundConversationEvent | None:
    """Map interactive ``block_actions`` static_select to ACTION event or None."""
    if _non_blank(payload.get("type")) != "block_actions":
        return None

    team = _as_mapping(payload.get("team")) or {}
    user = _as_mapping(payload.get("user")) or {}
    channel = _as_mapping(payload.get("channel")) or {}

    team_id = _non_blank(team.get("id"))
    user_id = _non_blank(user.get("id"))
    channel_id = _non_blank(channel.get("id"))
    if team_id is None or user_id is None or channel_id is None:
        _LOG.info("slack conversation: block_actions missing team/user/channel; ignoring")
        return None

    actions = payload.get("actions")
    if not isinstance(actions, list) or len(actions) != 1:
        _LOG.info("slack conversation: block_actions requires exactly one action; ignoring")
        return None

    action = _as_mapping(actions[0])
    if action is None:
        return None
    if _non_blank(action.get("type")) != _STATIC_SELECT:
        _LOG.debug("slack conversation: ignoring unsupported action type")
        return None

    action_id = _non_blank(action.get("action_id"))
    selected = _as_mapping(action.get("selected_option")) or {}
    selected_value = _non_blank(selected.get("value"))
    if action_id is None or selected_value is None:
        _LOG.info("slack conversation: block_actions missing action_id/selected_option; ignoring")
        return None

    thread_id = _resolve_action_thread_id(payload)
    if thread_id is None:
        _LOG.info("slack conversation: block_actions missing thread/message context; ignoring")
        return None

    container = _as_mapping(payload.get("container")) or {}
    message = _as_mapping(payload.get("message")) or {}
    message_ts = _non_blank(container.get("message_ts")) or _non_blank(message.get("ts")) or ""
    action_ts = _non_blank(action.get("action_ts")) or ""

    event_id = deterministic_block_action_event_id(
        team_id=team_id,
        user_id=user_id,
        channel_id=channel_id,
        action_ts=action_ts,
        message_ts=message_ts,
        action_id=action_id,
        selected_value=selected_value,
    )

    display_name = _non_blank(user.get("username"))
    return InboundConversationEvent(
        event_id=event_id,
        address=ConversationAddress(
            installation_id=team_id,
            conversation_id=channel_id,
            thread_id=thread_id,
        ),
        actor=ConversationActor(
            actor_id=user_id,
            display_name=display_name,
            is_bot=False,
        ),
        kind=ConversationEventKind.ACTION,
        action=ConversationActionSelection(
            action_id=action_id,
            selected_value=selected_value,
        ),
        occurred_at=parse_slack_ts(action_ts) if action_ts else None,
        metadata={"slack_envelope_type": "interactive"},
    )


def map_socket_mode_payload(
    *,
    envelope_type: str | None,
    payload: Mapping[str, Any] | None,
) -> InboundConversationEvent | None:
    """Map a Socket Mode envelope type+payload to a shared inbound event."""
    kind = _non_blank(envelope_type)
    if kind is None or kind not in SUPPORTED_ENVELOPE_TYPES:
        _LOG.debug("slack conversation: unsupported envelope type %s", kind)
        return None
    if payload is None:
        return None
    if kind == "events_api":
        return map_events_api_message(payload)
    if kind == "interactive":
        return map_block_actions(payload)
    return None


__all__ = [
    "SUPPORTED_ENVELOPE_TYPES",
    "deterministic_block_action_event_id",
    "map_block_actions",
    "map_events_api_message",
    "map_socket_mode_payload",
    "parse_slack_ts",
]
