"""Confluence Vendor Knowledge contribution."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.wiki_knowledge.confluence.config import (
    ConfluenceIntegrationConfig,
)
from intergrax.integrations.providers.wiki_knowledge.confluence.integration import (
    CONFLUENCE_WIKI_KNOWLEDGE_PROVIDER_ID,
)
from intergrax.integrations.providers.wiki_knowledge.confluence.knowledge_read import (
    CONFLUENCE_PAGES_SOURCE_KIND,
)
from intergrax.integrations.providers.wiki_knowledge.confluence.tenant_connection_factory import (
    ConfluenceTenantConnectionIntegrationFactory,
)
from intergrax.runtime.vendor_knowledge.adapters.confluence_pages import (
    register_confluence_pages_knowledge_adapter,
)
from intergrax.runtime.vendor_knowledge.contribution import (
    VendorKnowledgeConnectionFactoryContribution,
    VendorKnowledgeProviderContribution,
)
from intergrax.runtime.vendor_knowledge.contribution_builder import (
    build_adapter,
    build_durable_source_plugin,
)


def build_confluence_vendor_knowledge_contribution(
    *,
    http_client_factory: Callable[[ConfluenceIntegrationConfig], Any] | None = None,
) -> VendorKnowledgeProviderContribution:
    category = IntegrationCategory.WIKI_KNOWLEDGE
    return VendorKnowledgeProviderContribution(
        provider_id=CONFLUENCE_WIKI_KNOWLEDGE_PROVIDER_ID,
        integration_category=category,
        adapters=(build_adapter(register_confluence_pages_knowledge_adapter),),
        source_plugins=(
            build_durable_source_plugin(
                provider_id=CONFLUENCE_WIKI_KNOWLEDGE_PROVIDER_ID,
                integration_category=category,
                source_kind=CONFLUENCE_PAGES_SOURCE_KIND,
                runtime_ref="knowledge-adapter:confluence:wiki_knowledge:pages",
                indexed_runtime_ref="indexed-source:confluence:pages",
            ),
        ),
        connection_factories=(
            VendorKnowledgeConnectionFactoryContribution(
                provider_id=CONFLUENCE_WIKI_KNOWLEDGE_PROVIDER_ID,
                integration_category=category,
                factory=ConfluenceTenantConnectionIntegrationFactory(
                    http_client_factory=http_client_factory,
                ),
            ),
        ),
    )


__all__ = ["build_confluence_vendor_knowledge_contribution"]
