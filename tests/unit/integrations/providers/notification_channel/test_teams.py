# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for Teams integration provider (Phase M.4)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from intergrax.integrations._shared.conformance import (
    assert_interaction_surface,
    assert_notification_channel,
)
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.notification_channel.teams.adapter import _TeamsInteractionAdapter
from intergrax.integrations.providers.notification_channel.teams.bundle import (
    TeamsIntegrationBundle,
    create_teams_integration,
    create_teams_interaction_surface,
    create_teams_notification_channel,
)
from intergrax.integrations.providers.notification_channel.teams.integration import TeamsNotificationChannelIntegration
from intergrax.integrations.providers.notification_channel.teams.register import register_teams_integration
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
def mock_interaction() -> _TeamsInteractionAdapter:
    return _TeamsInteractionAdapter()


def test_create_teams_integration_bundle(
    mock_notification: MagicMock,
    mock_interaction: _TeamsInteractionAdapter,
) -> None:
    bundle = create_teams_integration(
        webhook_url="https://outlook.office.com/webhook/test",
        security_token="token",
        notification_adapter=mock_notification,
        interaction_adapter=mock_interaction,
    )

    assert isinstance(bundle, TeamsIntegrationBundle)
    assert isinstance(bundle.notification_channel, TeamsNotificationChannelIntegration)
    assert bundle.notification_channel._require_client() is mock_notification
    assert bundle.interaction_surface is mock_interaction
    assert bundle.config.webhook_url == "https://outlook.office.com/webhook/test"
    assert bundle.config.security_token == "token"


def test_create_teams_notification_channel_injects_adapter(mock_notification: MagicMock) -> None:
    channel = create_teams_notification_channel(notification_adapter=mock_notification)

    assert isinstance(channel, TeamsNotificationChannelIntegration)
    assert channel._require_client() is mock_notification


def test_create_teams_interaction_surface_uses_teams_channel() -> None:
    surface = create_teams_interaction_surface()

    assert isinstance(surface, _TeamsInteractionAdapter)
    assert surface.channel == "teams"
    assert surface.can_handle(
        {
            "type": "message",
            "channelId": "msteams",
            "serviceUrl": "https://smba.trafficmanager.net/teams/",
            "text": "hello",
        }
    )


def test_register_and_resolve_notification_channel(mock_notification: MagicMock) -> None:
    register_teams_integration()
    profile = IntegrationProfile(
        notification_channel="teams",
    )

    channel = resolve(
        IntegrationCategory.NOTIFICATION_CHANNEL,
        profile=profile,
        config={"notification_adapter": mock_notification},
    )

    assert_notification_channel(channel)
    assert isinstance(channel, TeamsNotificationChannelIntegration)
    assert channel._require_client() is mock_notification


def test_register_and_resolve_interaction_surface(mock_interaction: _TeamsInteractionAdapter) -> None:
    from intergrax.integrations.providers.notification_channel.teams.bundle import (
        create_teams_interaction_surface,
    )

    surface = create_teams_interaction_surface(interaction_adapter=mock_interaction)

    assert_interaction_surface(surface)
    assert isinstance(surface, _TeamsInteractionAdapter)


def test_register_default_integrations_includes_teams(mock_notification: MagicMock) -> None:
    register_default_integrations()
    profile = IntegrationProfile(notification_channel="teams")

    channel = resolve(
        IntegrationCategory.NOTIFICATION_CHANNEL,
        profile=profile,
        config={"notification_adapter": mock_notification},
    )

    assert isinstance(channel, TeamsNotificationChannelIntegration)
    assert channel._require_client() is mock_notification


def test_runtime_notification_factory_delegates_teams_to_integration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INTERGRAX_TEAMS_WEBHOOK_URL", "https://outlook.office.com/webhook/test")
    from intergrax.runtime.notifications.factory import create_notification_adapter, resolve_notification_settings

    adapter = create_notification_adapter(
        resolve_notification_settings(backend="teams"),
    )
    assert isinstance(adapter, TeamsNotificationChannelIntegration)
    assert adapter.webhook_url == "https://outlook.office.com/webhook/test"


def test_long_running_resolve_notification_delegates_teams(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTERGRAX_TEAMS_WEBHOOK_URL", "https://outlook.office.com/webhook/test")
    from intergrax.runtime.long_running.notification import resolve_notification_adapter

    adapter = resolve_notification_adapter("teams")
    assert isinstance(adapter, TeamsNotificationChannelIntegration)


def test_inbound_verifier_teams_mode_uses_integration() -> None:
    from intergrax.runtime.interactions.verification.factory import create_inbound_verifier, resolve_inbound_verifier_settings
    from intergrax.runtime.interactions.verification.teams_signature import TeamsSignatureVerifier

    verifier = create_inbound_verifier(
        resolve_inbound_verifier_settings(mode="teams", teams_security_token="secret"),
    )
    assert isinstance(verifier, TeamsSignatureVerifier)

    surface = create_teams_interaction_surface()
    assert isinstance(surface, _TeamsInteractionAdapter)
