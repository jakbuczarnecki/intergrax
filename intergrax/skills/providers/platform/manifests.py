# © Artur Czarnecki. All rights reserved.

from intergrax.skills.core.contracts import SkillManifest, SkillRiskTier

PLATFORM_CONCIERGE = SkillManifest(
    skill_id="platform.concierge",
    version="1.0.0",
    description=(
        "Intergrax assistant hub: retrieval, web evidence, session memory, and skill introspection."
    ),
    tool_ids=("rag.retrieve", "websearch.query", "memory.read", "skill.resolve"),
    prompt_instruction_ids=("platform.concierge.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("platform", "concierge", "assistant"),
)
