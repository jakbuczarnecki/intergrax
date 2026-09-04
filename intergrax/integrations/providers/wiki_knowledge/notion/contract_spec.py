# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Notion wiki knowledge."""

from __future__ import annotations

from intergrax.integrations.providers.wiki_knowledge.notion.bundle import (
    create_notion_wiki_knowledge_integration,
)
from intergrax.integrations.providers.wiki_knowledge.notion.integration import (
    NOTION_WIKI_KNOWLEDGE_PROVIDER_ID,
    NotionWikiKnowledgeIntegration,
    NotionWikiKnowledgeIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.collaboration import (
    WikiKnowledgeIntegrationContract,
)
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="wiki_knowledge",
    provider_id=NOTION_WIKI_KNOWLEDGE_PROVIDER_ID,
    integration_class=NotionWikiKnowledgeIntegration,
    contract_class=WikiKnowledgeIntegrationContract,
    contract_factory=create_notion_wiki_knowledge_integration,
    display_name="Notion",
    config_class=NotionWikiKnowledgeIntegrationConfig,
    capabilities=(
        PlatformIntegrationCapability.CONNECT,
        PlatformIntegrationCapability.READ,
        PlatformIntegrationCapability.WRITE,
        PlatformIntegrationCapability.HEALTH_CHECK
    ),
    security_posture=PlatformIntegrationSecurityPosture(),
    supports_runtime_binding=True,
    supports_health_check=True,
    metadata={
        "source": "explicit_provider_declaration"
    },
)

CONTRACT_SPECS = (CONTRACT_SPEC,)

__all__ = ["CONTRACT_SPEC", "CONTRACT_SPECS"]
