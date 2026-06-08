# © Artur Czarnecki. All rights reserved.

from intergrax.skills.core.contracts import SkillManifest, SkillRiskTier

AGENT_ROSTER_INTROSPECT = SkillManifest(
    skill_id="agent.roster_introspect",
    version="1.0.0",
    description="Agent roster introspection: list registered agents, fetch contracts, and resolve skills.",
    tool_ids=("agent.list_agents", "agent.get_contract", "skill.resolve"),
    prompt_instruction_ids=("agent.roster_introspect.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.LOW,
    tags=("agent", "roster", "introspection"),
)

AGENT_CAPABILITY_MAPPER = SkillManifest(
    skill_id="agent.capability_mapper",
    version="1.0.0",
    description="Map agent contracts to catalog tools and skill packs.",
    tool_ids=("agent.get_contract", "skill.resolve", "catalog.describe_tool"),
    prompt_instruction_ids=("agent.capability_mapper.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.LOW,
    tags=("agent", "capability", "mapper"),
)

