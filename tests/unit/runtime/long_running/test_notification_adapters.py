# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.runtime.long_running.notification import (
    LoggingNotificationAdapter,
    SlackNotificationAdapter,
    TeamsNotificationAdapter,
    resolve_notification_adapter,
)
from intergrax.runtime.long_running.models import NotificationMessage


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.gate
async def test_logging_notification_adapter(caplog):
    adapter = LoggingNotificationAdapter()
    await adapter.notify(
        NotificationMessage(
            channel="log",
            subject="Progress",
            body="50% complete",
            task_id="task_1",
            tenant_id="t1",
        )
    )


@pytest.mark.unit
@pytest.mark.gate
def test_resolve_notification_adapter_channels():
    assert isinstance(resolve_notification_adapter("log"), LoggingNotificationAdapter)
    assert isinstance(resolve_notification_adapter("slack"), SlackNotificationAdapter)
    assert isinstance(resolve_notification_adapter("teams"), TeamsNotificationAdapter)
