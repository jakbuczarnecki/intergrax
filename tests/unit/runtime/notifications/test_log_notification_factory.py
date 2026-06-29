# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tests that LOG notifications delegate to integrations.providers.log."""

from __future__ import annotations

import pytest

from intergrax.integrations.providers.notification_channel.log.integration import LogNotificationChannelIntegration
from intergrax.runtime.notifications.factory import create_notification_adapter, resolve_notification_settings

pytestmark = pytest.mark.unit


def test_notification_factory_log_delegates_to_integration() -> None:
    adapter = create_notification_adapter(
        resolve_notification_settings(backend="log"),
    )
    assert isinstance(adapter, LogNotificationChannelIntegration)
