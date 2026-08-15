"""Slack Vendor Knowledge contribution."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.conversation_channel.slack.integration import (
    SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
)
from intergrax.runtime.vendor_knowledge.contribution import (
    VendorKnowledgeConnectionFactoryContribution,
    VendorKnowledgeProviderContribution,
)
from intergrax.runtime.vendor_knowledge.contribution_builder import (
    build_adapter,
)
from intergrax.runtime.vendor_knowledge.live.slack.registration import (
    build_slack_live_registration_bundles,
    build_slack_vendor_knowledge_source_plugin,
)
from intergrax.runtime.vendor_knowledge.adapters.slack_conversation import (
    register_slack_conversation_knowledge_adapter,
)
from intergrax.integrations.providers.conversation_channel.slack.tenant_connection_factory import (
    SlackRuntimeBuilder,
    SlackTenantConnectionIntegrationFactory,
)


def build_slack_vendor_knowledge_contribution(
    *,
    runtime_builder: SlackRuntimeBuilder | None = None,
) -> VendorKnowledgeProviderContribution:
    return VendorKnowledgeProviderContribution(
        provider_id=SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
        integration_category=IntegrationCategory.CONVERSATION_CHANNEL,
        adapters=(build_adapter(register_slack_conversation_knowledge_adapter),),
        source_plugins=(build_slack_vendor_knowledge_source_plugin(),),
        connection_factories=(
            VendorKnowledgeConnectionFactoryContribution(
                provider_id=SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
                integration_category=IntegrationCategory.CONVERSATION_CHANNEL,
                factory=SlackTenantConnectionIntegrationFactory(
                    runtime_builder=runtime_builder,
                ),
            ),
        ),
        live_contributions=build_slack_live_registration_bundles(),
    )


__all__ = ["build_slack_vendor_knowledge_contribution"]
