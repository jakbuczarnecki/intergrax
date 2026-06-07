# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.applications._shared.notify_tool_wiring import wire_scheduled_notification_tool_binding
from intergrax.integrations.registry.catalog_manifests import LOG
from intergrax.tools.providers.notify.contracts import NotifyScheduleInput
from intergrax.tools.providers.notify.service import notify_schedule
from intergrax.tools.registry.runtime_bindings import ScheduledNotificationBinding
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = pytest.mark.unit


class FakeNotificationChannel:
    async def notify(self, message: object) -> None:
        del message


def test_wire_scheduled_notification_tool_binding_attaches_store() -> None:
    ctx = ToolWiringContext(notification_channel=FakeNotificationChannel())
    wired = wire_scheduled_notification_tool_binding(ctx)
    assert isinstance(wired.scheduled_notification_store, ScheduledNotificationBinding)
    out = notify_schedule(
        wired,
        NotifyScheduleInput(
            subject="maintenance",
            body="window",
            deliver_at_utc="2026-06-08T00:00:00Z",
        ),
    )
    assert out.scheduled is True
    assert out.schedule_id.startswith("sched_")


def test_wire_scheduled_notification_skips_without_channel() -> None:
    ctx = wire_scheduled_notification_tool_binding(ToolWiringContext())
    assert ctx.scheduled_notification_store is None


def test_notify_schedule_requires_store_when_channel_present_without_wiring() -> None:
    ctx = ToolWiringContext(notification_channel=LOG)
    with pytest.raises(RuntimeError, match="scheduled_notification_store_not_configured"):
        notify_schedule(
            ctx,
            NotifyScheduleInput(subject="x", body="y", deliver_at_utc="2026-06-08T00:00:00Z"),
        )
