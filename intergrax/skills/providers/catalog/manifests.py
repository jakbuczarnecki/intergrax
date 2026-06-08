# © Artur Czarnecki. All rights reserved.

from intergrax.skills.core.contracts import SkillManifest, SkillRiskTier

CATALOG_TOOL_INTROSPECT = SkillManifest(
    skill_id="catalog.tool_introspect",
    version="1.0.0",
    description="Tool catalog introspection: list tools, describe contracts, resolve skills.",
    tool_ids=("catalog.list_tools", "catalog.describe_tool", "skill.resolve"),
    prompt_instruction_ids=("catalog.tool_introspect.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.LOW,
    tags=("catalog", "introspection", "tools"),
)

