# © Artur Czarnecki. All rights reserved.

"""Default provider composition roots for Vendor Knowledge connections."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.conversation_channel.slack.integration import (
    SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
    SlackConversationChannelIntegration,
)
from intergrax.integrations.providers.conversation_channel.slack.tenant_connection_factory import (
    SlackRuntimeBuilder,
    SlackTenantConnectionIntegrationFactory,
)
from intergrax.runtime.vendor_knowledge.tenant_connection_factory_registry import (
    TenantConnectionIntegrationFactoryRegistry,
)


def build_default_vendor_knowledge_connection_factory_registry(
    *,
    slack_runtime_builder: SlackRuntimeBuilder | None = None,
) -> TenantConnectionIntegrationFactoryRegistry:
    """Compose provider-owned connection factories behind a generic registry."""
    registry = TenantConnectionIntegrationFactoryRegistry()
    registry.register(
        provider_id=SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
        integration_kind=IntegrationCategory.CONVERSATION_CHANNEL,
        factory=SlackTenantConnectionIntegrationFactory(
            runtime_builder=slack_runtime_builder,
        ),
    )
    return registry


def build_default_slack_integration_from_env() -> SlackConversationChannelIntegration:
    """Bounded local-development fallback for the legacy Slack env configuration."""
    from intergrax.integrations.providers.conversation_channel.slack.config import (
        SlackConversationChannelIntegrationConfig,
    )

    config = SlackConversationChannelIntegrationConfig.from_env(enabled=True)
    config.validate_for_runtime()
    return SlackConversationChannelIntegration.from_config(config)


__all__ = [
    "build_default_slack_integration_from_env",
    "build_default_vendor_knowledge_connection_factory_registry",
]
