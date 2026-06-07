# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

import asyncio
from typing import Any

from intergrax.runtime.notifications.models import NotificationMessage
from intergrax.tools.providers.notify.contracts import (
    NotifyCancelScheduledInput,
    NotifyCancelScheduledOutput,
    NotifyListScheduledInput,
    NotifyListScheduledOutput,
    NotifyScheduledItemOutput,
    NotifyScheduleInput,
    NotifyScheduleOutput,
    NotifySendBatchInput,
    NotifySendBatchOutput,
    NotifySendInput,
    NotifySendOutput,
)
from intergrax.tools.registry.runtime_bindings import ScheduledNotificationBinding
from intergrax.tools.registry.wiring import ToolWiringContext

NOTIFY_SEND_TOOL_ID = "notify.send"
NOTIFY_SEND_BATCH_TOOL_ID = "notify.send_batch"
NOTIFY_SCHEDULE_TOOL_ID = "notify.schedule"
NOTIFY_LIST_SCHEDULED_TOOL_ID = "notify.list_scheduled"
NOTIFY_CANCEL_SCHEDULED_TOOL_ID = "notify.cancel_scheduled"


def _require_scheduler(ctx: ToolWiringContext) -> ScheduledNotificationBinding:
    store = ctx.scheduled_notification_store
    if store is None:
        raise RuntimeError("scheduled_notification_store_not_configured")
    return store


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


def notify_schedule(ctx: ToolWiringContext, params: NotifyScheduleInput) -> NotifyScheduleOutput:
    """Record a deferred notification; delivery requires a Tier-3 scheduler/dispatcher."""
    scheduler = _require_scheduler(ctx)
    schedule_id = scheduler.schedule(
        tenant_id=params.tenant_id.strip(),
        channel=params.channel.strip(),
        subject=params.subject.strip(),
        body=params.body.strip(),
        deliver_at_utc=params.deliver_at_utc.strip(),
    )
    return NotifyScheduleOutput(
        scheduled=True,
        schedule_id=schedule_id,
        deliver_at_utc=params.deliver_at_utc.strip(),
        detail="ok",
    )


def notify_list_scheduled(ctx: ToolWiringContext, params: NotifyListScheduledInput) -> NotifyListScheduledOutput:
    scheduler = _require_scheduler(ctx)
    rows = scheduler.list_scheduled(
        params.tenant_id.strip(),
        limit=params.limit,
        status=params.status.strip(),
    )
    schedules = [
        NotifyScheduledItemOutput(
            schedule_id=str(item.get("schedule_id", "")),
            tenant_id=str(item.get("tenant_id", "")),
            channel=str(item.get("channel", "")),
            subject=str(item.get("subject", "")),
            deliver_at_utc=str(item.get("deliver_at_utc", "")),
            status=str(item.get("status", "")),
        )
        for item in rows
    ]
    return NotifyListScheduledOutput(
        used=True,
        schedules=schedules,
        total=len(schedules),
        detail="ok",
    )


def notify_cancel_scheduled(
    ctx: ToolWiringContext,
    params: NotifyCancelScheduledInput,
) -> NotifyCancelScheduledOutput:
    scheduler = _require_scheduler(ctx)
    cancelled = scheduler.cancel_scheduled(params.schedule_id.strip(), params.tenant_id.strip())
    detail = "ok" if cancelled else "schedule_not_found_or_not_pending"
    return NotifyCancelScheduledOutput(
        cancelled=cancelled,
        schedule_id=params.schedule_id.strip(),
        detail=detail,
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
