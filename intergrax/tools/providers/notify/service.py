# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

import asyncio
from typing import Any

from intergrax.runtime.notifications.models import NotificationMessage
from intergrax.tools.providers.notify.contracts import (
    NotifySendBatchInput,
    NotifySendBatchOutput,
    NotifySendInput,
    NotifySendOutput,
)
from intergrax.tools.registry.wiring import ToolWiringContext

NOTIFY_SEND_TOOL_ID = "notify.send"
NOTIFY_SEND_BATCH_TOOL_ID = "notify.send_batch"


def notify_send(ctx: ToolWiringContext, params: NotifySendInput) -> NotifySendOutput:
    channel = ctx.notification_channel
    if channel is None:
        return NotifySendOutput(sent=False, channel=params.channel, detail="notification_channel_not_configured")

    message = NotificationMessage(
        channel=params.channel,
        subject=params.subject,
        body=params.body,
        task_id=params.task_id or "tool",
        tenant_id=params.tenant_id,
        metadata=dict(params.metadata),
    )
    _dispatch_notify(channel, message)
    return NotifySendOutput(sent=True, channel=params.channel, detail="ok")


def notify_send_batch(ctx: ToolWiringContext, params: NotifySendBatchInput) -> NotifySendBatchOutput:
    channel = ctx.notification_channel
    if channel is None:
        return NotifySendBatchOutput(
            sent_count=0,
            failed_count=len(params.messages),
            details=["notification_channel_not_configured"],
        )

    sent_count = 0
    failed_count = 0
    details: list[str] = []
    for item in params.messages:
        message = NotificationMessage(
            channel=item.channel,
            subject=item.subject,
            body=item.body,
            task_id=params.task_id or "tool",
            tenant_id=params.tenant_id,
            metadata=dict(item.metadata),
        )
        try:
            _dispatch_notify(channel, message)
            sent_count += 1
        except Exception as exc:
            failed_count += 1
            details.append(f"{item.channel}:{exc.__class__.__name__}")
    return NotifySendBatchOutput(
        sent_count=sent_count,
        failed_count=failed_count,
        details=details,
    )


def _dispatch_notify(channel: Any, message: NotificationMessage) -> None:
    result = channel.notify(message)
    if asyncio.iscoroutine(result):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(result)
        else:
            loop.run_until_complete(result)
