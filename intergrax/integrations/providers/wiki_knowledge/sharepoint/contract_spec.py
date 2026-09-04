# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Sharepoint wiki knowledge."""

from __future__ import annotations

from intergrax.integrations.providers.wiki_knowledge.sharepoint.bundle import (
    create_sharepoint_wiki_knowledge_integration,
)
from intergrax.integrations.providers.wiki_knowledge.sharepoint.integration import (
    SHAREPOINT_WIKI_KNOWLEDGE_PROVIDER_ID,
    SharepointWikiKnowledgeIntegration,
    SharepointWikiKnowledgeIntegrationConfig,
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
    provider_id=SHAREPOINT_WIKI_KNOWLEDGE_PROVIDER_ID,
    integration_class=SharepointWikiKnowledgeIntegration,
    contract_class=WikiKnowledgeIntegrationContract,
    contract_factory=create_sharepoint_wiki_knowledge_integration,
    display_name="Sharepoint",
    config_class=SharepointWikiKnowledgeIntegrationConfig,
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
