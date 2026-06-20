# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.applications._shared.notify_tool_wiring import wire_scheduled_notification_tool_binding
from intergrax.tools.providers.notify.contracts import (
    NotifyCancelScheduledInput,
    NotifyDispatchDueInput,
    NotifyListScheduledInput,
    NotifyScheduleInput,
)
from intergrax.tools.providers.notify.service import (
    notify_cancel_scheduled,
    notify_dispatch_due,
    notify_list_scheduled,
    notify_schedule,
)
from intergrax.tools.registry.wiring import ToolWiringContext
from tests.unit.tools.providers.notify.test_notify_send import FakeNotificationChannel

pytestmark = pytest.mark.unit


def test_notify_list_and_cancel_scheduled() -> None:
    ctx = wire_scheduled_notification_tool_binding(
        ToolWiringContext(notification_channel=FakeNotificationChannel())
    )
    scheduled = notify_schedule(
        ctx,
        NotifyScheduleInput(
            subject="maintenance",
            body="window",
            deliver_at_utc="2026-06-08T00:00:00Z",
        ),
    )
    listed = notify_list_scheduled(ctx, NotifyListScheduledInput(status="pending"))
    assert listed.used is True
    assert listed.total == 1
    assert listed.schedules[0].schedule_id == scheduled.schedule_id

    cancelled = notify_cancel_scheduled(
        ctx,
        NotifyCancelScheduledInput(schedule_id=scheduled.schedule_id),
    )
    assert cancelled.cancelled is True

    pending = notify_list_scheduled(ctx, NotifyListScheduledInput(status="pending"))
    assert pending.total == 0


def test_notify_dispatch_due() -> None:
    ctx = wire_scheduled_notification_tool_binding(
        ToolWiringContext(notification_channel=FakeNotificationChannel())
    )
    notify_schedule(
        ctx,
        NotifyScheduleInput(
            subject="due",
            body="send me",
            deliver_at_utc="2026-06-01T00:00:00Z",
        ),
    )
    out = notify_dispatch_due(
        ctx,
        NotifyDispatchDueInput(deliver_before_utc="2026-06-07T00:00:00Z"),
    )
    assert out.dispatched_count == 1
