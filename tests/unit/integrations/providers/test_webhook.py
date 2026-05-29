# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for generic webhook integration provider (Phase M.4)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from intergrax.integrations._shared.conformance import assert_notification_channel
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.webhook.bundle import (
    WebhookIntegrationBundle,
    create_webhook_integration,
    create_webhook_notification_channel,
)
from intergrax.integrations.providers.webhook.register import register_webhook_integration
from intergrax.integrations.registry.bootstrap import register_default_integrations, reset_default_integrations_state
from intergrax.integrations.registry.catalog import clear_catalog
from intergrax.integrations.registry.factory import resolve
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.registry.slugs import IntegrationSlug

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clean_catalog() -> None:
    clear_catalog()
    reset_default_integrations_state()
    yield
    clear_catalog()
    reset_default_integrations_state()


@pytest.fixture
def mock_notification() -> MagicMock:
    adapter = MagicMock()
    adapter.notify = AsyncMock()
    return adapter


def test_create_webhook_integration_bundle(mock_notification: MagicMock) -> None:
    bundle = create_webhook_integration(
        webhook_url="https://hooks.example.com/notify",
        notification_adapter=mock_notification,
    )

    assert isinstance(bundle, WebhookIntegrationBundle)
    assert bundle.notification_channel is mock_notification
    assert bundle.config.webhook_url == "https://hooks.example.com/notify"


def test_create_webhook_notification_channel_injects_adapter(mock_notification: MagicMock) -> None:
    channel = create_webhook_notification_channel(notification_adapter=mock_notification)

    assert channel is mock_notification


def test_register_and_resolve_via_profile(mock_notification: MagicMock) -> None:
    register_webhook_integration()
    profile = IntegrationProfile(notification_channel=IntegrationSlug.WEBHOOK)

    channel = resolve(
        IntegrationCategory.NOTIFICATION_CHANNEL,
        profile=profile,
        config={"notification_adapter": mock_notification},
    )

    assert_notification_channel(channel)
    assert channel is mock_notification


def test_register_default_integrations_includes_webhook(mock_notification: MagicMock) -> None:
    register_default_integrations()
    profile = IntegrationProfile(notification_channel=IntegrationSlug.WEBHOOK)

    channel = resolve(
        IntegrationCategory.NOTIFICATION_CHANNEL,
        profile=profile,
        config={"notification_adapter": mock_notification},
    )

    assert channel is mock_notification


def test_runtime_notification_factory_delegates_webhook_to_integration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INTERGRAX_WEBHOOK_URL", "https://hooks.example.com/generic")
    from intergrax.runtime.notifications.adapters.webhook_adapter import WebhookNotificationAdapter
    from intergrax.runtime.notifications.factory import create_notification_adapter, resolve_notification_settings

    adapter = create_notification_adapter(
        resolve_notification_settings(
            backend="webhook",
            webhook_url="https://hooks.example.com/generic",
        ),
    )
    assert isinstance(adapter, WebhookNotificationAdapter)
    assert adapter.webhook_url == "https://hooks.example.com/generic"


def test_long_running_resolve_notification_delegates_webhook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INTERGRAX_WEBHOOK_URL", "https://hooks.example.com/generic")
    from intergrax.runtime.long_running.notification import resolve_notification_adapter
    from intergrax.runtime.notifications.adapters.webhook_adapter import WebhookNotificationAdapter

    adapter = resolve_notification_adapter("webhook")
    assert isinstance(adapter, WebhookNotificationAdapter)
