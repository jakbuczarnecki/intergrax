"""Microsoft Graph Vendor Knowledge contribution."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.collaboration_suite.ms365_graph.integration import (
    MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.tenant_connection_factory import (
    Ms365GraphRuntimeBuilder,
    Ms365GraphTenantConnectionIntegrationFactory,
)
from intergrax.runtime.vendor_knowledge.adapters.ms365_graph_calendar import (
    register_msgraph_calendar_knowledge_adapter,
)
from intergrax.runtime.vendor_knowledge.adapters.ms365_graph_drive import (
    register_msgraph_drive_knowledge_adapter,
)
from intergrax.runtime.vendor_knowledge.adapters.ms365_graph_mail import (
    register_msgraph_mail_knowledge_adapter,
)
from intergrax.runtime.vendor_knowledge.adapters.ms365_graph_teams_channel import (
    register_msgraph_teams_channel_knowledge_adapter,
)
from intergrax.runtime.vendor_knowledge.adapters.ms365_graph_teams_chat import (
    register_msgraph_teams_chat_knowledge_adapter,
)
from intergrax.runtime.vendor_knowledge.contribution import (
    VendorKnowledgeConnectionFactoryContribution,
    VendorKnowledgeProviderContribution,
)
from intergrax.runtime.vendor_knowledge.contribution_builder import (
    build_adapter,
)
from intergrax.runtime.vendor_knowledge.live.ms365_graph.registration import (
    build_msgraph_calendar_vendor_knowledge_source_plugin,
    build_msgraph_drive_vendor_knowledge_source_plugin,
    build_msgraph_live_registration_bundles,
    build_msgraph_mail_vendor_knowledge_source_plugin,
    build_msgraph_teams_channel_vendor_knowledge_source_plugin,
    build_msgraph_teams_chat_vendor_knowledge_source_plugin,
)


def build_msgraph_vendor_knowledge_contribution(
    *,
    runtime_builder: Ms365GraphRuntimeBuilder | None = None,
) -> VendorKnowledgeProviderContribution:
    return VendorKnowledgeProviderContribution(
        provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
        integration_category=IntegrationCategory.COLLABORATION_SUITE,
        adapters=(
            build_adapter(register_msgraph_drive_knowledge_adapter),
            build_adapter(register_msgraph_mail_knowledge_adapter),
            build_adapter(register_msgraph_teams_channel_knowledge_adapter),
            build_adapter(register_msgraph_teams_chat_knowledge_adapter),
            build_adapter(register_msgraph_calendar_knowledge_adapter),
        ),
        source_plugins=(
            build_msgraph_drive_vendor_knowledge_source_plugin(),
            build_msgraph_mail_vendor_knowledge_source_plugin(),
            build_msgraph_teams_channel_vendor_knowledge_source_plugin(),
            build_msgraph_teams_chat_vendor_knowledge_source_plugin(),
            build_msgraph_calendar_vendor_knowledge_source_plugin(),
        ),
        connection_factories=(
            VendorKnowledgeConnectionFactoryContribution(
                provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
                integration_category=IntegrationCategory.COLLABORATION_SUITE,
                factory=Ms365GraphTenantConnectionIntegrationFactory(
                    runtime_builder=runtime_builder,
                ),
            ),
        ),
        live_contributions=build_msgraph_live_registration_bundles(),
    )


__all__ = ["build_msgraph_vendor_knowledge_contribution"]
