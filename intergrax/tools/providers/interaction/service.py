# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.runtime.notifications.models import NotificationMessage
from intergrax.tools.providers.interaction.contracts import (
    InteractionGetLastInputInput,
    InteractionGetLastInputOutput,
    InteractionGetSessionHistoryInput,
    InteractionGetSessionHistoryOutput,
    InteractionHistoryMessageOutput,
    InteractionListSessionsInput,
    InteractionListSessionsOutput,
    InteractionPostReplyInput,
    InteractionPostReplyOutput,
    InteractionSessionOutput,
)
from intergrax.tools.providers.notify.service import _dispatch_notify
from intergrax.tools.registry.runtime_bindings import SessionStorageBinding
from intergrax.tools.registry.wiring import ToolWiringContext

INTERACTION_LIST_SESSIONS_TOOL_ID = "interaction.list_sessions"
INTERACTION_GET_LAST_INPUT_TOOL_ID = "interaction.get_last_input"
INTERACTION_GET_SESSION_HISTORY_TOOL_ID = "interaction.get_session_history"
INTERACTION_POST_REPLY_TOOL_ID = "interaction.post_reply"


def _require_session_storage(ctx: ToolWiringContext) -> SessionStorageBinding:
    storage = ctx.session_storage
    if storage is None:
        raise RuntimeError("session_storage_not_configured")
    if not isinstance(storage, SessionStorageBinding):
        raise RuntimeError("session_storage_invalid_type")
    return storage


def interaction_list_sessions(
    ctx: ToolWiringContext,
    params: InteractionListSessionsInput,
) -> InteractionListSessionsOutput:
    storage = _require_session_storage(ctx)
    raw = storage.list_sessions(
        params.tenant_id.strip(),
        params.user_id.strip(),
        limit=params.limit,
    )
    sessions = [
        InteractionSessionOutput(
            session_id=str(item.get("session_id") or ""),
            user_id=str(item.get("user_id") or params.user_id.strip()),
            tenant_id=str(item.get("tenant_id") or params.tenant_id.strip()),
            updated_at_utc=str(item.get("updated_at_utc") or ""),
        )
        for item in raw
        if str(item.get("session_id") or "")
    ]
    return InteractionListSessionsOutput(
        used=True,
        sessions=sessions,
        total=len(sessions),
        reason="ok",
    )


def interaction_get_last_input(
    ctx: ToolWiringContext,
    params: InteractionGetLastInputInput,
) -> InteractionGetLastInputOutput:
    storage = _require_session_storage(ctx)
    message = storage.get_last_user_input(params.tenant_id.strip(), params.session_id.strip())
    if not message:
        return InteractionGetLastInputOutput(used=True, found=False, reason="input_not_found")
    return InteractionGetLastInputOutput(used=True, found=True, message=message, reason="ok")


def interaction_get_session_history(
    ctx: ToolWiringContext,
    params: InteractionGetSessionHistoryInput,
) -> InteractionGetSessionHistoryOutput:
    storage = _require_session_storage(ctx)
    raw = storage.get_session_history(
        params.tenant_id.strip(),
        params.session_id.strip(),
        limit=params.limit,
    )
    messages = [
        InteractionHistoryMessageOutput(
            role=str(item.get("role") or ""),
            content=str(item.get("content") or ""),
        )
        for item in raw
        if str(item.get("content") or "")
    ]
    return InteractionGetSessionHistoryOutput(
        used=True,
        messages=messages,
        total=len(messages),
        reason="ok",
    )


def interaction_post_reply(ctx: ToolWiringContext, params: InteractionPostReplyInput) -> InteractionPostReplyOutput:
    channel = ctx.notification_channel
    if channel is None:
        return InteractionPostReplyOutput(sent=False, channel=params.channel, detail="notification_channel_not_configured")
    metadata: dict[str, str] = {}
    if params.session_id.strip():
        metadata["session_id"] = params.session_id.strip()
    if params.thread_id.strip():
        metadata["thread_id"] = params.thread_id.strip()
    message = NotificationMessage(
        channel=params.channel.strip(),
        subject=params.subject,
        body=params.body,
        task_id=params.task_id.strip() or "interaction_reply",
        tenant_id=params.tenant_id.strip(),
        metadata=metadata,
    )
    _dispatch_notify(channel, message)
    return InteractionPostReplyOutput(sent=True, channel=params.channel.strip(), detail="ok")
