"""Canonical composition of implemented Vendor Knowledge source plugins."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.collaboration_suite.google_workspace.integration import (
    GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.knowledge_read.calendar import (
    GOOGLE_CALENDAR_SOURCE_KIND,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.knowledge_read.docs import (
    GOOGLE_DOCS_SOURCE_KIND,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.knowledge_read.drive import (
    GOOGLE_DRIVE_SOURCE_KIND,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.knowledge_read.sheets import (
    GOOGLE_SHEETS_SOURCE_KIND,
)
from intergrax.integrations.providers.issue_tracker.jira.integration import (
    JIRA_ISSUE_TRACKER_PROVIDER_ID,
)
from intergrax.integrations.providers.issue_tracker.jira.knowledge_read import (
    JIRA_ISSUES_SOURCE_KIND,
)
from intergrax.integrations.providers.wiki_knowledge.confluence.integration import (
    CONFLUENCE_WIKI_KNOWLEDGE_PROVIDER_ID,
)
from intergrax.integrations.providers.wiki_knowledge.confluence.knowledge_read import (
    CONFLUENCE_PAGES_SOURCE_KIND,
)
from intergrax.runtime.vendor_knowledge.live.ms365_graph.registration import (
    build_msgraph_calendar_vendor_knowledge_source_plugin,
    build_msgraph_drive_vendor_knowledge_source_plugin,
    build_msgraph_mail_vendor_knowledge_source_plugin,
    build_msgraph_teams_channel_vendor_knowledge_source_plugin,
    build_msgraph_teams_chat_vendor_knowledge_source_plugin,
)
from intergrax.runtime.vendor_knowledge.live.slack.registration import (
    build_slack_vendor_knowledge_source_plugin,
)
from intergrax.runtime.vendor_knowledge.plugin import (
    VendorKnowledgeMode,
    VendorKnowledgeModeCapability,
    VendorKnowledgeSourceIdentity,
    VendorKnowledgeSourcePlugin,
    VendorKnowledgeSourcePluginRegistry,
)


def _build_durable_adapter_plugin(
    *,
    provider_id: str,
    integration_category: IntegrationCategory,
    source_kind: str,
    runtime_ref: str,
    indexed_runtime_ref: str | None = None,
) -> VendorKnowledgeSourcePlugin:
    capabilities = [
        VendorKnowledgeModeCapability(
            mode=VendorKnowledgeMode.DURABLE,
            contract_version="vendor-knowledge.durable.v1",
            operations=("inventory", "snapshot", "reconciliation", "exact_fetch"),
            runtime_ref=runtime_ref,
            constraints={"application_sink": "platform_foundation"},
        ),
    ]
    if indexed_runtime_ref is not None:
        capabilities.append(
            VendorKnowledgeModeCapability(
                mode=VendorKnowledgeMode.INDEXED,
                contract_version="vendor-knowledge.indexed.v1",
                operations=("eligible", "materialize", "publish", "index"),
                runtime_ref=indexed_runtime_ref,
                constraints={"application_proof": "vk4"},
            )
        )
    return VendorKnowledgeSourcePlugin(
        identity=VendorKnowledgeSourceIdentity(
            provider_id=provider_id,
            integration_category=integration_category,
            source_kind=source_kind,
        ),
        capabilities=tuple(capabilities),
    )


def build_google_workspace_vendor_knowledge_source_plugins() -> (
    tuple[VendorKnowledgeSourcePlugin, ...]
):
    provider_id = GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID
    category = IntegrationCategory.COLLABORATION_SUITE
    return (
        _build_durable_adapter_plugin(
            provider_id=provider_id,
            integration_category=category,
            source_kind=GOOGLE_DRIVE_SOURCE_KIND,
            runtime_ref="knowledge-adapter:google_workspace:collaboration_suite:drive",
        ),
        _build_durable_adapter_plugin(
            provider_id=provider_id,
            integration_category=category,
            source_kind=GOOGLE_DOCS_SOURCE_KIND,
            runtime_ref="knowledge-adapter:google_workspace:collaboration_suite:docs",
            indexed_runtime_ref="indexed-source:google_workspace:docs",
        ),
        _build_durable_adapter_plugin(
            provider_id=provider_id,
            integration_category=category,
            source_kind=GOOGLE_SHEETS_SOURCE_KIND,
            runtime_ref="knowledge-adapter:google_workspace:collaboration_suite:sheets",
            indexed_runtime_ref="indexed-source:google_workspace:sheets",
        ),
        _build_durable_adapter_plugin(
            provider_id=provider_id,
            integration_category=category,
            source_kind=GOOGLE_CALENDAR_SOURCE_KIND,
            runtime_ref="knowledge-adapter:google_workspace:collaboration_suite:calendar",
            indexed_runtime_ref="indexed-source:google_workspace:calendar",
        ),
    )


def build_jira_vendor_knowledge_source_plugin() -> VendorKnowledgeSourcePlugin:
    return _build_durable_adapter_plugin(
        provider_id=JIRA_ISSUE_TRACKER_PROVIDER_ID,
        integration_category=IntegrationCategory.ISSUE_TRACKER,
        source_kind=JIRA_ISSUES_SOURCE_KIND,
        runtime_ref="knowledge-adapter:jira:issue_tracker:issues",
    )


def build_confluence_vendor_knowledge_source_plugin() -> VendorKnowledgeSourcePlugin:
    return _build_durable_adapter_plugin(
        provider_id=CONFLUENCE_WIKI_KNOWLEDGE_PROVIDER_ID,
        integration_category=IntegrationCategory.WIKI_KNOWLEDGE,
        source_kind=CONFLUENCE_PAGES_SOURCE_KIND,
        runtime_ref="knowledge-adapter:confluence:wiki_knowledge:pages",
    )


def build_default_vendor_knowledge_source_plugin_registry() -> (
    VendorKnowledgeSourcePluginRegistry
):
    """Register every implemented source kind without tenant/runtime state."""
    registry = VendorKnowledgeSourcePluginRegistry()
    registry.register(build_slack_vendor_knowledge_source_plugin())
    registry.register(build_msgraph_drive_vendor_knowledge_source_plugin())
    registry.register(build_msgraph_mail_vendor_knowledge_source_plugin())
    registry.register(build_msgraph_teams_channel_vendor_knowledge_source_plugin())
    registry.register(build_msgraph_teams_chat_vendor_knowledge_source_plugin())
    registry.register(build_msgraph_calendar_vendor_knowledge_source_plugin())
    for plugin in build_google_workspace_vendor_knowledge_source_plugins():
        registry.register(plugin)
    registry.register(build_jira_vendor_knowledge_source_plugin())
    registry.register(build_confluence_vendor_knowledge_source_plugin())
    return registry


__all__ = [
    "build_confluence_vendor_knowledge_source_plugin",
    "build_default_vendor_knowledge_source_plugin_registry",
    "build_google_workspace_vendor_knowledge_source_plugins",
    "build_jira_vendor_knowledge_source_plugin",
]
