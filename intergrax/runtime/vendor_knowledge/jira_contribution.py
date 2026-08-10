"""Jira Vendor Knowledge contribution."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.issue_tracker.jira.config import JiraIntegrationConfig
from intergrax.integrations.providers.issue_tracker.jira.integration import (
    JIRA_ISSUE_TRACKER_PROVIDER_ID,
)
from intergrax.integrations.providers.issue_tracker.jira.knowledge_read import (
    JIRA_ISSUES_SOURCE_KIND,
)
from intergrax.integrations.providers.issue_tracker.jira.tenant_connection_factory import (
    JiraTenantConnectionIntegrationFactory,
)
from intergrax.runtime.vendor_knowledge.adapters.jira_issues import (
    register_jira_issues_knowledge_adapter,
)
from intergrax.runtime.vendor_knowledge.contribution import (
    VendorKnowledgeConnectionFactoryContribution,
    VendorKnowledgeProviderContribution,
)
from intergrax.runtime.vendor_knowledge.contribution_builder import (
    build_adapter,
    build_durable_source_plugin,
)


def build_jira_vendor_knowledge_contribution(
    *,
    http_client_factory: Callable[[JiraIntegrationConfig], Any] | None = None,
) -> VendorKnowledgeProviderContribution:
    category = IntegrationCategory.ISSUE_TRACKER
    return VendorKnowledgeProviderContribution(
        provider_id=JIRA_ISSUE_TRACKER_PROVIDER_ID,
        integration_category=category,
        adapters=(build_adapter(register_jira_issues_knowledge_adapter),),
        source_plugins=(
            build_durable_source_plugin(
                provider_id=JIRA_ISSUE_TRACKER_PROVIDER_ID,
                integration_category=category,
                source_kind=JIRA_ISSUES_SOURCE_KIND,
                runtime_ref="knowledge-adapter:jira:issue_tracker:issues",
                indexed_runtime_ref="indexed-source:jira:issues",
            ),
        ),
        connection_factories=(
            VendorKnowledgeConnectionFactoryContribution(
                provider_id=JIRA_ISSUE_TRACKER_PROVIDER_ID,
                integration_category=category,
                factory=JiraTenantConnectionIntegrationFactory(
                    http_client_factory=http_client_factory,
                ),
            ),
        ),
    )


__all__ = ["build_jira_vendor_knowledge_contribution"]
