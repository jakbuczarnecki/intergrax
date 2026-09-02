# © Artur Czarnecki. All rights reserved.

"""Conversation channel provider matrix, taxonomy, and multi-category identity."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pytest

from intergrax.integrations.contracts.base import HealthStatus, IntegrationConfigurationError
from intergrax.integrations.contracts.conversation_channel import (
    ConversationAddress,
    ConversationChannelBackend,
    ConversationDeliveryReceipt,
    ConversationEventHandler,
    ConversationEventKind,
    InboundConversationEvent,
    OutboundConversationMessage,
)
from intergrax.integrations.providers.conversation_channel.discord.integration import (
    DiscordConversationChannelIntegration,
)
from intergrax.integrations.providers.conversation_channel.google_chat.integration import (
    GoogleChatConversationChannelIntegration,
)
from intergrax.integrations.providers.conversation_channel.mattermost.integration import (
    MattermostConversationChannelIntegration,
)
from intergrax.integrations.providers.conversation_channel.rocket_chat.integration import (
    RocketChatConversationChannelIntegration,
)
from intergrax.integrations.providers.conversation_channel.slack.integration import (
    SlackConversationChannelIntegration,
)
from intergrax.integrations.providers.conversation_channel.teams.integration import (
    TeamsConversationChannelIntegration,
)
from intergrax.integrations.providers.conversation_channel.telegram.integration import (
    TelegramConversationChannelIntegration,
)
from intergrax.integrations.providers.layout import (
    SECONDARY_PROVIDER_CATEGORIES,
    SLUG_CATEGORY,
    categories_for_provider,
    provider_category_keys,
    provider_import_path,
)
from intergrax.integrations.providers.notification_channel.discord.integration import (
    DiscordNotificationChannelIntegration,
)
from intergrax.integrations.providers.notification_channel.slack.integration import (
    SlackNotificationChannelIntegration,
)
from intergrax.integrations.providers.notification_channel.teams.integration import (
    TeamsNotificationChannelIntegration,
)
from intergrax.integrations.providers.notification_channel.telegram.integration import (
    TelegramNotificationChannelIntegration,
)
from intergrax.runtime.integrations.categories import PROVIDER_CATEGORY_CONTRACT_REGISTRY
from intergrax.runtime.integrations.categories.messaging import (
    CONVERSATION_CHANNEL_INTEGRATION_CONTRACT_SCHEMA,
    ConversationChannelIntegrationContract,
    NotificationChannelIntegrationContract,
)
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationContract,
    PlatformIntegrationKind,
)
from intergrax.runtime.integrations.registry_v2 import (
    DuplicateIntegrationRegistrationError,
    IntegrationRegistry,
    build_contract_registry,
    build_integration_registration,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _canonical_catalog_bootstrap() -> None:
    from intergrax.integrations.registry.bootstrap import (
        register_default_integrations,
        reset_default_integrations_state,
    )
    from intergrax.integrations.registry.catalog import clear_catalog

    clear_catalog()
    reset_default_integrations_state()
    register_default_integrations(preset="full")
    from intergrax.integrations.providers.conversation_channel.google_chat.register import (
        register_google_chat_integration,
    )
    from intergrax.integrations.providers.conversation_channel.mattermost.register import (
        register_mattermost_integration,
    )
    from intergrax.integrations.providers.conversation_channel.rocket_chat.register import (
        register_rocket_chat_integration,
    )

    register_mattermost_integration()
    register_rocket_chat_integration()
    register_google_chat_integration()
    yield
    clear_catalog()
    reset_default_integrations_state()


_PROVIDERS = (
    ("slack", SlackConversationChannelIntegration, "create_slack_conversation_channel_integration"),
    ("teams", TeamsConversationChannelIntegration, "create_teams_conversation_channel_integration"),
    ("discord", DiscordConversationChannelIntegration, "create_discord_conversation_channel_integration"),
    ("telegram", TelegramConversationChannelIntegration, "create_telegram_conversation_channel_integration"),
    ("mattermost", MattermostConversationChannelIntegration, "create_mattermost_conversation_channel_integration"),
    ("rocket_chat", RocketChatConversationChannelIntegration, "create_rocket_chat_conversation_channel_integration"),
    ("google_chat", GoogleChatConversationChannelIntegration, "create_google_chat_conversation_channel_integration"),
)

_DUAL = (
    ("slack", SlackNotificationChannelIntegration, SlackConversationChannelIntegration),
    ("teams", TeamsNotificationChannelIntegration, TeamsConversationChannelIntegration),
    ("discord", DiscordNotificationChannelIntegration, DiscordConversationChannelIntegration),
    ("telegram", TelegramNotificationChannelIntegration, TelegramConversationChannelIntegration),
)


@dataclass
class FakeConversationBackend:
    started: bool = False
    stopped: bool = False
    handler: ConversationEventHandler | None = None
    sent: list[OutboundConversationMessage] = field(default_factory=list)

    async def start(self, handler: ConversationEventHandler) -> None:
        self.started = True
        self.handler = handler

    async def stop(self) -> None:
        self.stopped = True

    async def send(self, message: OutboundConversationMessage) -> ConversationDeliveryReceipt:
        self.sent.append(message)
        return ConversationDeliveryReceipt(
            message_id="msg-1",
            address=message.address,
            delivered_at=datetime.now(timezone.utc),
        )

    def health(self) -> HealthStatus | bool:
        return HealthStatus(slug="fake", healthy=True, detail="ok")


@pytest.mark.asyncio
@pytest.mark.parametrize(("slug", "integration_cls", "_factory"), _PROVIDERS)
async def test_conversation_provider_delegates_to_backend(
    slug: str,
    integration_cls: type[ConversationChannelIntegrationContract],
    _factory: str,
) -> None:
    backend = FakeConversationBackend()
    integration = integration_cls.from_backend(backend, enabled=False)
    assert integration.backend is backend

    async def handler(_event: InboundConversationEvent) -> None:
        return None

    await integration.start(handler)
    await integration.stop()
    receipt = await integration.send(
        OutboundConversationMessage(
            address=ConversationAddress(installation_id="i", conversation_id="c"),
            text="hello",
        )
    )
    health = integration.health()

    assert backend.started is True
    assert backend.stopped is True
    assert backend.handler is handler
    assert receipt.message_id == "msg-1"
    assert health.healthy is True


@pytest.mark.parametrize(("slug", "integration_cls", "factory_name"), _PROVIDERS)
def test_conversation_provider_requires_backend(
    slug: str,
    integration_cls: type[ConversationChannelIntegrationContract],
    factory_name: str,
) -> None:
    registration = build_integration_registration(slug, category="conversation_channel")
    integration = registration.factory(enabled=False)
    assert isinstance(integration, integration_cls)
    assert integration.enabled is False
    with pytest.raises(IntegrationConfigurationError):
        integration.health()


@pytest.mark.parametrize(("slug", "integration_cls", "factory_name"), _PROVIDERS)
def test_conversation_provider_registration_metadata(
    slug: str,
    integration_cls: type[ConversationChannelIntegrationContract],
    factory_name: str,
) -> None:
    registration = build_integration_registration(slug, category="conversation_channel")
    assert registration.provider_id == slug
    assert registration.category == "conversation_channel"
    assert registration.integration_kind == PlatformIntegrationKind.CONVERSATION_CHANNEL.value
    assert registration.contract_class is ConversationChannelIntegrationContract
    assert registration.integration_class is integration_cls
    assert registration.default_enabled is False
    runtime_expected = slug == "slack"
    assert registration.supports_runtime_binding is runtime_expected
    assert registration.supports_health_check is runtime_expected
    assert registration.metadata["conversation_features"] == ("text", "single_choice")
    assert registration.metadata["runtime_implemented"] is runtime_expected
    assert registration.factory.__name__ == factory_name
    assert issubclass(registration.integration_class, ConversationChannelIntegrationContract)

    if slug == "slack":
        previous = {
            key: os.environ.pop(key, None)
            for key in (
                "INTERGRAX_SLACK_APP_TOKEN",
                "INTERGRAX_SLACK_BOT_TOKEN",
                "INTERGRAX_SLACK_CONVERSATION_ENABLED",
            )
        }
        try:
            with pytest.raises(IntegrationConfigurationError):
                registration.factory(enabled=True)
        finally:
            for key, value in previous.items():
                if value is not None:
                    os.environ[key] = value
    else:
        with pytest.raises(IntegrationConfigurationError):
            registration.factory(enabled=True)

    disabled = registration.factory(enabled=False)
    assert isinstance(disabled, integration_cls)
    assert disabled.enabled is False


def test_conversation_channel_category_contract() -> None:
    assert "conversation_channel" in PROVIDER_CATEGORY_CONTRACT_REGISTRY
    contract_cls = PROVIDER_CATEGORY_CONTRACT_REGISTRY["conversation_channel"]
    assert contract_cls is ConversationChannelIntegrationContract
    assert issubclass(contract_cls, PlatformIntegrationContract)

    contract = contract_cls.for_provider(provider_id="example")
    assert contract.schema_id == CONVERSATION_CHANNEL_INTEGRATION_CONTRACT_SCHEMA
    assert contract.integration_kind == PlatformIntegrationKind.CONVERSATION_CHANNEL.value
    assert tuple(c.value for c in contract.capabilities) == (
        PlatformIntegrationCapability.CONNECT.value,
        PlatformIntegrationCapability.READ.value,
        PlatformIntegrationCapability.WRITE.value,
        PlatformIntegrationCapability.HEALTH_CHECK.value,
    )


def test_taxonomy_preserves_notification_primary_and_adds_conversation() -> None:
    for slug in ("slack", "teams", "discord", "telegram"):
        assert SLUG_CATEGORY[slug] == "notification_channel"
        assert categories_for_provider(slug) == ("notification_channel", "conversation_channel")
        assert SECONDARY_PROVIDER_CATEGORIES[slug] == ("conversation_channel",)
        assert provider_import_path(slug) == (
            f"intergrax.integrations.providers.notification_channel.{slug}"
        )
        assert provider_import_path(slug, "conversation_channel") == (
            f"intergrax.integrations.providers.conversation_channel.{slug}"
        )

    for slug in ("mattermost", "rocket_chat", "google_chat"):
        assert SLUG_CATEGORY[slug] == "conversation_channel"
        assert categories_for_provider(slug) == ("conversation_channel",)


@pytest.mark.parametrize(("slug", "notification_cls", "conversation_cls"), _DUAL)
def test_multi_category_registry_identities_are_distinct(
    slug: str,
    notification_cls: type[Any],
    conversation_cls: type[Any],
) -> None:
    registry = IntegrationRegistry()
    notification = build_integration_registration(slug, category="notification_channel")
    conversation = build_integration_registration(slug, category="conversation_channel")
    registry.register(notification)
    registry.register(conversation)

    assert registry.get(provider_id=slug, category="notification_channel") is notification
    assert registry.get(provider_id=slug, category="conversation_channel") is conversation
    assert notification.integration_class is notification_cls
    assert conversation.integration_class is conversation_cls
    assert notification.integration_class is not conversation.integration_class
    assert issubclass(notification.integration_class, NotificationChannelIntegrationContract)
    assert issubclass(conversation.integration_class, ConversationChannelIntegrationContract)

    with pytest.raises(DuplicateIntegrationRegistrationError):
        registry.register(conversation)


def test_conversation_only_providers_resolve() -> None:
    registry = build_contract_registry(
        slugs=("mattermost", "rocket_chat", "google_chat"),
        exclude_deferred=False,
    )
    for slug in ("mattermost", "rocket_chat", "google_chat"):
        registration = registry.get(provider_id=slug, category="conversation_channel")
        assert registration.supports_runtime_binding is False
        assert registration.category == "conversation_channel"


def test_provider_category_keys_include_dual_identities() -> None:
    keys = set(provider_category_keys())
    for slug in ("slack", "teams", "discord", "telegram"):
        assert (slug, "notification_channel") in keys
        assert (slug, "conversation_channel") in keys
    for slug in ("mattermost", "rocket_chat", "google_chat"):
        assert (slug, "conversation_channel") in keys


def test_interaction_surface_remains_absent() -> None:
    assert "interaction_surface" not in PROVIDER_CATEGORY_CONTRACT_REGISTRY
    assert "interaction_surface" not in set(SLUG_CATEGORY.values())
    assert "lab_json" not in SLUG_CATEGORY
    assert "slash_command" not in SLUG_CATEGORY
