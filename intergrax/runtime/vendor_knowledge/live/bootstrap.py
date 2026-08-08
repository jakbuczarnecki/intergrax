"""Default provider composition root for Vendor Knowledge Live."""

from __future__ import annotations

from intergrax.runtime.vendor_knowledge.live.ms365_graph.registration import (
    build_msgraph_calendar_vendor_knowledge_source_plugin,
    build_msgraph_drive_vendor_knowledge_source_plugin,
    build_msgraph_live_registration_bundles,
    build_msgraph_mail_vendor_knowledge_source_plugin,
    build_msgraph_teams_channel_vendor_knowledge_source_plugin,
    build_msgraph_teams_chat_vendor_knowledge_source_plugin,
)
from intergrax.runtime.vendor_knowledge.live.registration import (
    VendorKnowledgeLiveRegistrationRegistry,
)
from intergrax.runtime.vendor_knowledge.live.slack.registration import (
    build_slack_live_registration_bundles,
    build_slack_vendor_knowledge_source_plugin,
)


def build_vendor_knowledge_live_registration_registry(
) -> VendorKnowledgeLiveRegistrationRegistry:
    """Build the provider-neutral Live registry from provider contributions."""
    registry = VendorKnowledgeLiveRegistrationRegistry()
    registry.register(build_msgraph_live_registration_bundles())
    registry.register(build_slack_live_registration_bundles())
    registry.register_plugin(build_slack_vendor_knowledge_source_plugin())
    registry.register_plugin(build_msgraph_drive_vendor_knowledge_source_plugin())
    registry.register_plugin(build_msgraph_mail_vendor_knowledge_source_plugin())
    registry.register_plugin(
        build_msgraph_teams_channel_vendor_knowledge_source_plugin()
    )
    registry.register_plugin(build_msgraph_teams_chat_vendor_knowledge_source_plugin())
    registry.register_plugin(build_msgraph_calendar_vendor_knowledge_source_plugin())
    return registry


__all__ = ["build_vendor_knowledge_live_registration_registry"]
