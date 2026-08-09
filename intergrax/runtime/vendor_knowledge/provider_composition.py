# © Artur Czarnecki. All rights reserved.

"""Default provider composition roots for Vendor Knowledge connections."""

from __future__ import annotations

from dataclasses import dataclass

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


@dataclass(frozen=True, slots=True)
class VendorKnowledgeLegacyLocalBootstrap:
    """Provider-neutral contract for the bounded local-development fallback."""

    provider_id: str
    integration_kind: IntegrationCategory
    integration: object


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


def build_default_vendor_knowledge_legacy_local_bootstrap(
    *,
    slack_runtime_builder: SlackRuntimeBuilder | None = None,
) -> VendorKnowledgeLegacyLocalBootstrap | None:
    """Build the legacy local fallback behind the Vendor Knowledge boundary.

    This fallback is intentionally separate from durable tenant connection
    lifecycle. Production startup must use the generic factory registry and
    persisted connection identity instead.
    """
    from intergrax.integrations.providers.conversation_channel.slack.config import (
        SlackConversationChannelIntegrationConfig,
    )

    try:
        config = SlackConversationChannelIntegrationConfig.from_env(enabled=True)
        config.validate_for_runtime()
        integration = (slack_runtime_builder or SlackConversationChannelIntegration.from_config)(
            config
        )
    except Exception:
        return None
    return VendorKnowledgeLegacyLocalBootstrap(
        provider_id=SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
        integration_kind=IntegrationCategory.CONVERSATION_CHANNEL,
        integration=integration,
    )


__all__ = [
    "build_default_vendor_knowledge_connection_factory_registry",
    "build_default_vendor_knowledge_legacy_local_bootstrap",
    "VendorKnowledgeLegacyLocalBootstrap",
]
