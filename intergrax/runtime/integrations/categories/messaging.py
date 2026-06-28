# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Messaging and notification provider category contracts (INTEGRATIONS-2A)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from intergrax.runtime.integrations.categories._base import (
    CategoryIntegrationConfig,
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
    """Category contract for notification_channel providers (slack, email_smtp, …)."""

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
