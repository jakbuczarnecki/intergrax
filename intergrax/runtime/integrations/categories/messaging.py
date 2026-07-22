# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Messaging, notification, and conversation provider category contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from intergrax.runtime.integrations.categories._base import (
    CategoryIntegrationConfig,
    _CONNECT_READ_WRITE_HEALTH,
    _CONNECT_WRITE_HEALTH,
    category_for_provider,
)
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationContract,
    PlatformIntegrationKind,
)

MESSAGE_BUS_INTEGRATION_CONTRACT_SCHEMA = "message_bus_integration_contract.v1"
NOTIFICATION_CHANNEL_INTEGRATION_CONTRACT_SCHEMA = "notification_channel_integration_contract.v1"
CONVERSATION_CHANNEL_INTEGRATION_CONTRACT_SCHEMA = "conversation_channel_integration_contract.v1"


class MessageBusIntegrationContract(PlatformIntegrationContract):
    """Category contract for message_bus providers (kafka, rabbitmq, …)."""

    schema_id: Literal["message_bus_integration_contract.v1"] = MESSAGE_BUS_INTEGRATION_CONTRACT_SCHEMA
    integration_kind: str = PlatformIntegrationKind.MESSAGE_BUS.value
    capabilities: tuple[PlatformIntegrationCapability, ...] = Field(
        default_factory=lambda: _CONNECT_WRITE_HEALTH
    )
    config: CategoryIntegrationConfig = Field(default_factory=CategoryIntegrationConfig)

    @classmethod
    def for_provider(
        cls,
        *,
        provider_id: str,
        capabilities: tuple[PlatformIntegrationCapability, ...] | None = None,
        display_name: str | None = None,
        version: str | None = None,
        config: CategoryIntegrationConfig | None = None,
    ) -> MessageBusIntegrationContract:
        return category_for_provider(
            cls,
            provider_id=provider_id,
            integration_kind=PlatformIntegrationKind.MESSAGE_BUS.value,
            default_capabilities=_CONNECT_WRITE_HEALTH,
            capabilities=capabilities,
            display_name=display_name,
            version=version,
            config=config,
        )


class NotificationChannelIntegrationContract(PlatformIntegrationContract):
    """Category contract for notification_channel providers (slack, email_smtp, …).

    Semantic direction: application → vendor → recipient. Primary operation: notify.
    Capabilities: CONNECT, WRITE, HEALTH_CHECK (no READ).
    """

    schema_id: Literal["notification_channel_integration_contract.v1"] = (
        NOTIFICATION_CHANNEL_INTEGRATION_CONTRACT_SCHEMA
    )
    integration_kind: str = PlatformIntegrationKind.NOTIFICATION_CHANNEL.value
    capabilities: tuple[PlatformIntegrationCapability, ...] = Field(
        default_factory=lambda: _CONNECT_WRITE_HEALTH
    )
    config: CategoryIntegrationConfig = Field(default_factory=CategoryIntegrationConfig)

    @classmethod
    def for_provider(
        cls,
        *,
        provider_id: str,
        capabilities: tuple[PlatformIntegrationCapability, ...] | None = None,
        display_name: str | None = None,
        version: str | None = None,
        config: CategoryIntegrationConfig | None = None,
    ) -> NotificationChannelIntegrationContract:
        return category_for_provider(
            cls,
            provider_id=provider_id,
            integration_kind=PlatformIntegrationKind.NOTIFICATION_CHANNEL.value,
            default_capabilities=_CONNECT_WRITE_HEALTH,
            capabilities=capabilities,
            display_name=display_name,
            version=version,
            config=config,
        )


class ConversationChannelIntegrationContract(PlatformIntegrationContract):
    """Category contract for conversation_channel providers (slack, teams, …).

    A conversation channel is an external near-real-time communication system
    that delivers human-originated conversation events to an application
    and allows the application to reply within the same addressable
    conversation context.

    Semantic direction: human ↔ vendor ↔ application.
    Primary semantics: receive conversation event, reply, receive action,
    maintain provider lifecycle, report health.
    Capabilities: CONNECT, READ, WRITE, HEALTH_CHECK.
    Distinct from notification_channel (outbound notify only).
    """

    schema_id: Literal["conversation_channel_integration_contract.v1"] = (
        CONVERSATION_CHANNEL_INTEGRATION_CONTRACT_SCHEMA
    )
    integration_kind: str = PlatformIntegrationKind.CONVERSATION_CHANNEL.value
    capabilities: tuple[PlatformIntegrationCapability, ...] = Field(
        default_factory=lambda: _CONNECT_READ_WRITE_HEALTH
    )
    config: CategoryIntegrationConfig = Field(default_factory=CategoryIntegrationConfig)

    @classmethod
    def for_provider(
        cls,
        *,
        provider_id: str,
        capabilities: tuple[PlatformIntegrationCapability, ...] | None = None,
        display_name: str | None = None,
        version: str | None = None,
        config: CategoryIntegrationConfig | None = None,
    ) -> ConversationChannelIntegrationContract:
        return category_for_provider(
            cls,
            provider_id=provider_id,
            integration_kind=PlatformIntegrationKind.CONVERSATION_CHANNEL.value,
            default_capabilities=_CONNECT_READ_WRITE_HEALTH,
            capabilities=capabilities,
            display_name=display_name,
            version=version,
            config=config,
        )
