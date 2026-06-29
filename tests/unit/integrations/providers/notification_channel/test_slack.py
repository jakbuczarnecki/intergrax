# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for Slack integration provider (Phase M.4)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from intergrax.integrations._shared.conformance import (
    assert_interaction_surface,
    assert_notification_channel,
)
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.notification_channel.slack.adapter import SlackInteractionAdapter
from intergrax.integrations.providers.notification_channel.slack.bundle import (
    SlackIntegrationBundle,
    create_slack_integration,
    create_slack_interaction_surface,
    create_slack_notification_channel,
)
from intergrax.integrations.providers.notification_channel.slack.integration import SlackNotificationChannelIntegration
from intergrax.integrations.providers.notification_channel.slack.register import register_slack_integration
from intergrax.integrations.registry.bootstrap import register_default_integrations, reset_default_integrations_state
from intergrax.integrations.registry.catalog import clear_catalog
from intergrax.integrations.registry.factory import resolve
from intergrax.integrations.registry.profile import IntegrationProfile

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


@pytest.fixture
def mock_interaction() -> SlackInteractionAdapter:
    return SlackInteractionAdapter()


def test_create_slack_integration_bundle(
    mock_notification: MagicMock,
    mock_interaction: SlackInteractionAdapter,
) -> None:
    bundle = create_slack_integration(
        webhook_url="https://hooks.slack.com/test",
        signing_secret="secret",
        notification_adapter=mock_notification,
        interaction_adapter=mock_interaction,
    )

    assert isinstance(bundle, SlackIntegrationBundle)
    assert isinstance(bundle.notification_channel, SlackNotificationChannelIntegration)
    assert bundle.notification_channel._require_runtime() is mock_notification
    assert bundle.interaction_surface is mock_interaction
    assert bundle.config.webhook_url == "https://hooks.slack.com/test"
    assert bundle.config.signing_secret == "secret"


def test_create_slack_notification_channel_injects_adapter(mock_notification: MagicMock) -> None:
    channel = create_slack_notification_channel(notification_adapter=mock_notification)

    assert isinstance(channel, SlackNotificationChannelIntegration)
    assert channel._require_runtime() is mock_notification


def test_create_slack_interaction_surface_uses_slack_channel() -> None:
    surface = create_slack_interaction_surface()

    assert isinstance(surface, SlackInteractionAdapter)
    assert surface.channel == "slack"
    assert surface.can_handle({"command": "/intergrax", "text": "hello"})


def test_register_and_resolve_notification_channel(mock_notification: MagicMock) -> None:
    register_slack_integration()
    profile = IntegrationProfile(
        notification_channel="slack",
        interaction_surface="slack",
    )

    channel = resolve(
        IntegrationCategory.NOTIFICATION_CHANNEL,
        profile=profile,
        config={"notification_adapter": mock_notification},
    )

    assert_notification_channel(channel)
    assert isinstance(channel, SlackNotificationChannelIntegration)
    assert channel._require_runtime() is mock_notification


def test_register_and_resolve_interaction_surface(mock_interaction: SlackInteractionAdapter) -> None:
    register_slack_integration()
    profile = IntegrationProfile(
        notification_channel="slack",
        interaction_surface="slack",
    )

    surface = resolve(
        IntegrationCategory.INTERACTION_SURFACE,
        profile=profile,
        config={"interaction_adapter": mock_interaction},
    )

    assert_interaction_surface(surface)
    assert isinstance(surface, SlackInteractionAdapter)


def test_register_default_integrations_includes_slack(mock_notification: MagicMock) -> None:
    register_default_integrations()
    profile = IntegrationProfile(notification_channel="slack")

    channel = resolve(
        IntegrationCategory.NOTIFICATION_CHANNEL,
        profile=profile,
        config={"notification_adapter": mock_notification},
    )

    assert isinstance(channel, SlackNotificationChannelIntegration)
    assert channel._require_runtime() is mock_notification


def test_runtime_notification_factory_delegates_slack_to_integration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTERGRAX_SLACK_WEBHOOK_URL", "https://hooks.slack.com/services/test")
    from intergrax.runtime.notifications.adapters.webhook_adapter import WebhookNotificationAdapter
    from intergrax.runtime.notifications.factory import create_notification_adapter, resolve_notification_settings

    adapter = create_notification_adapter(
        resolve_notification_settings(backend="slack"),
    )
    assert isinstance(adapter, SlackNotificationChannelIntegration)
    runtime = adapter._require_runtime()
    assert isinstance(runtime, WebhookNotificationAdapter)
    assert runtime.webhook_url == "https://hooks.slack.com/services/test"


def test_long_running_resolve_notification_delegates_slack(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTERGRAX_SLACK_WEBHOOK_URL", "https://hooks.slack.com/services/test")
    from intergrax.runtime.long_running.notification import resolve_notification_adapter
    from intergrax.runtime.notifications.adapters.webhook_adapter import WebhookNotificationAdapter

    adapter = resolve_notification_adapter("slack")
    assert isinstance(adapter, SlackNotificationChannelIntegration)
    runtime = adapter._require_runtime()
    assert isinstance(runtime, WebhookNotificationAdapter)


def test_inbound_verifier_slack_mode_uses_integration() -> None:
    from intergrax.integrations.providers.notification_channel.slack.adapter import SlackInteractionAdapter
    from intergrax.runtime.interactions.verification.factory import create_inbound_verifier, resolve_inbound_verifier_settings
    from intergrax.runtime.interactions.verification.slack_signature import SlackSignatureVerifier

    verifier = create_inbound_verifier(
        resolve_inbound_verifier_settings(mode="slack", slack_signing_secret="secret"),
    )
    assert isinstance(verifier, SlackSignatureVerifier)

    surface = create_slack_interaction_surface()
    assert isinstance(surface, SlackInteractionAdapter)
