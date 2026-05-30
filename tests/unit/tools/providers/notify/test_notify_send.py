# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import pytest

from intergrax.runtime.notifications.models import NotificationMessage
from intergrax.tools.providers.notify.contracts import NotifySendInput
from intergrax.tools.providers.notify.service import notify_send
from intergrax.tools.registry.bootstrap import register_default_tools, reset_default_tools_bootstrap
from intergrax.tools.registry.catalog import clear_tool_catalog, get_bundle, list_catalog_tool_ids
from intergrax.tools.registry.factory import build_registry_from_profile
from intergrax.tools.registry.profile import ToolProfile
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = pytest.mark.unit


@dataclass
class FakeNotificationChannel:
    messages: List[NotificationMessage] = field(default_factory=list)

    async def notify(self, message: NotificationMessage) -> None:
        self.messages.append(message)


@pytest.fixture(autouse=True)
def _clean_catalog() -> None:
    clear_tool_catalog()
    reset_default_tools_bootstrap()
    yield
    clear_tool_catalog()
    reset_default_tools_bootstrap()


def test_notify_send_success() -> None:
    channel = FakeNotificationChannel()
    ctx = ToolWiringContext(notification_channel=channel)
    out = notify_send(
        ctx,
        NotifySendInput(
            subject="Deploy complete",
            body="Version 1.2.3 is live.",
            channel="slack",
            task_id="task-1",
        ),
    )
    assert out.sent is True
    assert out.channel == "slack"
    assert len(channel.messages) == 1
    assert channel.messages[0].subject == "Deploy complete"


def test_notify_send_not_configured() -> None:
    out = notify_send(
        ToolWiringContext(),
        NotifySendInput(subject="Hi", body="Test"),
    )
    assert out.sent is False
    assert out.detail == "notification_channel_not_configured"


def test_notify_tool_registered_in_catalog() -> None:
    register_default_tools()
    assert "notify.send" in list_catalog_tool_ids()
    assert get_bundle("notify").tool_ids == ("notify.send",)


def test_build_registry_enables_notify_tool() -> None:
    register_default_tools()
    ctx = ToolWiringContext(notification_channel=FakeNotificationChannel())
    registry = build_registry_from_profile(ToolProfile(enabled=["notify.send"]), ctx=ctx)
    assert registry.has("notify.send")
