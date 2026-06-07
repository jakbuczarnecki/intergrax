# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from typing import Sequence

import pytest

from intergrax.integrations.contracts.collaboration_suite import (
    CalendarEvent,
    CalendarEventsResult,
    MailListResult,
    MailMessage,
    UserRecord,
)
from intergrax.tools.providers.collaboration.contracts import (
    CollaborationGetMessageInput,
    CollaborationGetUserInput,
    CollaborationListCalendarInput,
    CollaborationListMessagesInput,
    CollaborationSendMailInput,
)
from intergrax.tools.providers.collaboration.service import (
    collaboration_get_message,
    collaboration_get_user,
    collaboration_list_calendar,
    collaboration_list_messages,
    collaboration_send_mail,
)
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = pytest.mark.unit


class FakeCollaborationSuite:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str, str, tuple[str, ...]]] = []

    def get_message(self, user_id: str, message_id: str) -> MailMessage:
        return MailMessage(id=message_id, subject="Hello", body_preview="Preview", from_address="a@test")

    def list_messages(self, user_id: str, *, folder: str = "inbox", limit: int = 25) -> MailListResult:
        return MailListResult(
            messages=[MailMessage(id="m1", subject="One", body_preview="p1")],
            total=1,
        )

    def send_mail(self, user_id: str, *, subject: str, body: str, to: Sequence[str]) -> None:
        self.sent.append((user_id, subject, body, tuple(to)))

    def list_calendar_events(
        self,
        user_id: str,
        *,
        start: str,
        end: str,
        limit: int = 50,
    ) -> CalendarEventsResult:
        return CalendarEventsResult(
            events=[CalendarEvent(id="e1", subject="Sync", start=start, end=end)],
            total=1,
        )

    def get_user(self, user_id: str) -> UserRecord:
        return UserRecord(id=user_id, display_name="Test User", email="user@test")


def test_collaboration_read_and_send() -> None:
    suite = FakeCollaborationSuite()
    ctx = ToolWiringContext(collaboration_suite=suite)

    listed = collaboration_list_messages(ctx, CollaborationListMessagesInput(user_id="user-1"))
    assert listed.total == 1

    message = collaboration_get_message(ctx, CollaborationGetMessageInput(user_id="user-1", message_id="m1"))
    assert message.message.subject == "Hello"

    events = collaboration_list_calendar(
        ctx,
        CollaborationListCalendarInput(user_id="user-1", start="2026-01-01T00:00:00Z", end="2026-01-02T00:00:00Z"),
    )
    assert events.total == 1

    user = collaboration_get_user(ctx, CollaborationGetUserInput(user_id="user-1"))
    assert user.user.display_name == "Test User"

    collaboration_send_mail(
        ctx,
        CollaborationSendMailInput(user_id="user-1", subject="Hi", body="Body", to=["x@test"]),
    )
    assert suite.sent
