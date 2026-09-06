# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Domain adapters for federated capability catalog read model (Stage 2–5)."""

from __future__ import annotations

from intergrax.capability_catalog.adapters.agent import (
    AgentCatalogCapabilitySource,
    project_agent_catalog_entry,
)
from intergrax.capability_catalog.adapters.agent_ranking import (
    AGENT_STABLE_IDENTITY_RANKER_ID,
    AgentStableIdentityCapabilityRanker,
)
from intergrax.capability_catalog.adapters.skill import (
    SkillBundleCatalogSource,
    project_skill_bundle_entry,
)
from intergrax.capability_catalog.adapters.tool import (
    ToolBundleCatalogSource,
    project_tool_bundle_entry,
)
from intergrax.capability_catalog.adapters.agent_governance import (
    AGENT_TRUST_GOVERNANCE_EVALUATOR_ID,
    AgentTrustGovernanceEvaluator,
)
from intergrax.capability_catalog.adapters.skill_governance import (
    SKILL_PROFILE_GOVERNANCE_EVALUATOR_ID,
    SkillProfileGovernanceEvaluator,
)
from intergrax.capability_catalog.adapters.tool_governance import (
    TOOL_POLICY_GOVERNANCE_EVALUATOR_ID,
    ToolPolicyGovernanceEvaluator,
)
from intergrax.capability_catalog.adapters.private_skill import (
    PrivateSkillCapabilityCatalogSource,
    PrivateSkillCatalogPackage,
    project_private_skill_package,
)
from intergrax.capability_catalog.adapters.private_tool import (
    PrivateToolCapabilityCatalogSource,
    PrivateToolCatalogRecord,
    project_private_tool_record,
)
from intergrax.capability_catalog.adapters.tool_ranking import (
    KEYWORD_OVERLAP_TOOL_RANKER_ID,
    KeywordOverlapToolCapabilityRanker,
)

__all__ = [
    "AGENT_STABLE_IDENTITY_RANKER_ID",
    "AGENT_TRUST_GOVERNANCE_EVALUATOR_ID",
    "AgentTrustGovernanceEvaluator",
    "SKILL_PROFILE_GOVERNANCE_EVALUATOR_ID",
    "SkillProfileGovernanceEvaluator",
    "TOOL_POLICY_GOVERNANCE_EVALUATOR_ID",
    "ToolPolicyGovernanceEvaluator",
    "AgentCatalogCapabilitySource",
    "AgentStableIdentityCapabilityRanker",
    "KEYWORD_OVERLAP_TOOL_RANKER_ID",
    "KeywordOverlapToolCapabilityRanker",
    "PrivateSkillCapabilityCatalogSource",
    "PrivateSkillCatalogPackage",
    "PrivateToolCapabilityCatalogSource",
    "PrivateToolCatalogRecord",
    "SkillBundleCatalogSource",
    "ToolBundleCatalogSource",
    "project_agent_catalog_entry",
    "project_private_skill_package",
    "project_private_tool_record",
    "project_skill_bundle_entry",
    "project_tool_bundle_entry",
]
