# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Domain adapters for federated capability catalog read model (Stage 2)."""

from __future__ import annotations

from intergrax.capability_catalog.adapters.agent import (
    AgentCatalogCapabilitySource,
    project_agent_catalog_entry,
)
from intergrax.capability_catalog.adapters.skill import (
    SkillBundleCatalogSource,
    project_skill_bundle_entry,
)
from intergrax.capability_catalog.adapters.tool import (
    ToolBundleCatalogSource,
    project_tool_bundle_entry,
)

__all__ = [
    "AgentCatalogCapabilitySource",
    "SkillBundleCatalogSource",
    "ToolBundleCatalogSource",
    "project_agent_catalog_entry",
    "project_skill_bundle_entry",
    "project_tool_bundle_entry",
]
